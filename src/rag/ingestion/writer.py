"""Ingestion stage 5: embed chunk texts and persist them to the vector store."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

from rag.embedders.base import Embedder
from rag.retrieval.field_policy import combine_scannable_fields, detect_sensitive_field_ids
from rag.schemas import Chunk, ChunkMetadata, ChunkSpan, RawDocument, VisionCallStats
from rag.vectorstore.base import VectorStore
from rag.vision.base import VisionProvider

logger = logging.getLogger(__name__)

_IMAGE_ALT_TEXT_RE = re.compile(r"^!\[([^\]\[]*)\]")


class Writer:
    """Embeds chunk texts and persists them to the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        vision_provider: VisionProvider | None = None,
    ) -> None:
        """Store the embedder/vectorstore/vision provider this writer will use.

        Parameters
        ----------
        embedder : Embedder
            Embedder used to vectorize chunk texts.
        vectorstore : VectorStore
            Vector store the resulting chunks are persisted to.
        vision_provider : VisionProvider | None, optional
            When set, a vision-generated sibling chunk is added for every
            image span (see `_with_vision_siblings`). `None` (the default,
            and the only mode exercised so far) means text-only image
            handling. No image bytes are ever read, no network call is
            possible.
        """
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._vision_provider = vision_provider
        self.vision_stats = VisionCallStats()

    @property
    def vision_provider(self) -> VisionProvider | None:
        """The configured `VisionProvider`, or `None` for text-only image handling."""
        return self._vision_provider

    def write(
        self,
        document: RawDocument,
        document_id: str,
        chunk_spans: list[ChunkSpan],
        dataset_id: str,
        category: str | None = None,
    ) -> list[Chunk]:
        """Embed `chunk_spans`, build `Chunk`s from `document`'s metadata, and persist them.

        Parameters
        ----------
        document : RawDocument
            The source document these chunks were split from.
        document_id : str
            Stable id of `document`, used as the `Chunk.id`/`chunk_id` prefix.
        chunk_spans : list[ChunkSpan]
            Chunk spans (text plus structural hints), in document order.
        dataset_id : str
            Namespace tag stored on every chunk.
        category : str | None, optional
            Folder-derived category tag, by default None.

        Returns
        -------
        list[Chunk]
            The persisted chunks, each with its embedding set.
        """
        if self._vision_provider is not None:
            chunk_spans = self._with_vision_siblings(chunk_spans, document)
        texts = [span.text for span in chunk_spans]
        embeddings = self._embedder.embed_documents(texts)
        chunk_ids = [f"{document_id}_{i}" for i in range(len(chunk_spans))]
        parent_chunk_ids = self._compute_parent_chunk_ids(chunk_spans, chunk_ids)
        chunks = [
            Chunk(
                id=chunk_ids[i],
                content=span.text,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    chunk_id=chunk_ids[i],
                    source=document.source,
                    source_type=document.source_type,
                    title=document.title,
                    author=document.author,
                    url=document.url,
                    created_at=document.created_at,
                    last_modified=document.last_modified,
                    language=document.language,
                    chunk_index=i,
                    category=category,
                    dataset_id=dataset_id,
                    content_type=span.content_type or "prose",
                    section_path=span.section_path,
                    code_language=span.code_language,
                    table_headers=span.table_headers,
                    attachment_name=span.attachment_name,
                    source_anchor=span.source_anchor,
                    parent_chunk_id=parent_chunk_ids[i],
                    vision_generated=span.vision_generated,
                    vision_description=span.vision_description,
                    sensitive_field_ids=detect_sensitive_field_ids(
                        combine_scannable_fields(
                            span.text,
                            source=document.source,
                            section_path=span.section_path,
                            attachment_name=span.attachment_name,
                            source_anchor=span.source_anchor,
                        )
                    )
                    or None,
                    tenant_id=document.tenant_id,
                    allowed_roles=document.allowed_roles,
                    classification=document.classification,
                    status=document.status,
                    document_version=document.document_version,
                    effective_from=document.effective_from,
                    trust_level=document.trust_level,
                    doc_source_type=document.doc_source_type,
                    supersedes_source=document.supersedes_source,
                    page=span.page,
                ),
                embedding=embedding,
            )
            for i, (span, embedding) in enumerate(zip(chunk_spans, embeddings, strict=True))
        ]
        self._vectorstore.add_chunks(chunks)
        return chunks

    @staticmethod
    def _compute_parent_chunk_ids(
        chunk_spans: list[ChunkSpan], chunk_ids: list[str]
    ) -> list[str | None]:
        """Link each non-prose span to the nearest preceding prose chunk in its section.

        One linear pass over `chunk_spans` in document order (they're
        already ordered that way by the chunker). A table/code/
        configuration/chart/image span's `parent_chunk_id` is the id of the
        most recently seen prose chunk sharing its `section_path`. The
        explanatory paragraph introducing that section, standing in for
        "surrounding context" without needing a full parse tree. Prose
        spans get `None`: reconstructing a section's full prose is already
        just "same document_id + section_path, ordered by chunk_index" (see
        VectorStore.get_chunks_by_section), no pointer chain needed there.
        A non-prose span with no preceding prose in its section (e.g. a
        table opening a section before any paragraph) also gets `None`.

        Parameters
        ----------
        chunk_spans : list[ChunkSpan]
            Chunk spans, in document order.
        chunk_ids : list[str]
            Chunk ids in the same order/length as `chunk_spans`.

        Returns
        -------
        list[str | None]
            One parent chunk id (or None) per span, same order.
        """
        parent_ids: list[str | None] = []
        last_prose_id_by_section: dict[str | None, str] = {}
        for span, chunk_id in zip(chunk_spans, chunk_ids, strict=True):
            if (span.content_type or "prose") == "prose":
                parent_ids.append(None)
                last_prose_id_by_section[span.section_path] = chunk_id
            else:
                parent_ids.append(last_prose_id_by_section.get(span.section_path))
        return parent_ids

    def _with_vision_siblings(
        self, chunk_spans: list[ChunkSpan], document: RawDocument
    ) -> list[ChunkSpan]:
        """Insert a vision-generated sibling span immediately after each image span.

        Only called when a `VisionProvider` is configured. For each
        `content_type="image"` span, resolves its `source_anchor` (e.g.
        "images/x.png") to a real file relative to `document.source`,
        consults `VectorStore`'s image-description cache by the image's own
        sha256 checksum (so an unchanged image is never reprocessed across
        documents/ingestion runs), calls `VisionProvider.describe_image` on
        a cache miss, and appends a second span carrying the description as
        its own embeddable `text`. The original caption/alt-text span is
        never modified.

        A `describe_image` call that raises (Ollama unreachable, timed
        out, or returned something `OllamaVisionProvider` couldn't parse)
        is caught and logged (`vision_provider_call_failed`) rather than
        propagated: that one image is skipped -- no vision sibling, same
        as the missing-asset-file case -- and the rest of the document
        still ingests normally. A vision-model hiccup on one figure should
        never fail an entire document's ingestion.

        Parameters
        ----------
        chunk_spans : list[ChunkSpan]
            Chunk spans, in document order, before vision expansion.
        document : RawDocument
            Source document, used to resolve `source_anchor` to a real path.

        Returns
        -------
        list[ChunkSpan]
            `chunk_spans` with a vision-generated sibling inserted after
            each image span whose asset file was found on disk and whose
            `describe_image` call succeeded.
        """
        assert self._vision_provider is not None  # only called from write() when set
        provider = self._vision_provider
        result: list[ChunkSpan] = []
        for span in chunk_spans:
            result.append(span)
            if span.content_type != "image" or not span.source_anchor:
                continue
            image_path = (Path(document.source).parent / span.source_anchor).resolve()
            if not image_path.is_file():
                continue

            checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
            self.vision_stats.images_processed += 1
            description = self._vectorstore.get_cached_image_description(
                checksum, provider.provider_name, provider.model_name, provider.prompt_version
            )
            if description is None:
                self.vision_stats.cache_misses += 1
                alt_match = _IMAGE_ALT_TEXT_RE.match(span.text)
                alt_text = alt_match.group(1) if alt_match else None
                call_start = time.perf_counter()
                try:
                    description = provider.describe_image(image_path, alt_text=alt_text)
                except Exception:
                    logger.warning(
                        "vision_provider_call_failed",
                        extra={
                            "source_anchor": span.source_anchor,
                            "provider": provider.provider_name,
                        },
                        exc_info=True,
                    )
                    continue
                self.vision_stats.total_latency_ms += (time.perf_counter() - call_start) * 1000
                self._vectorstore.cache_image_description(
                    image_checksum=checksum,
                    source_path=str(image_path),
                    provider=provider.provider_name,
                    model_name=provider.model_name,
                    prompt_version=provider.prompt_version,
                    description=description,
                )
            else:
                self.vision_stats.cache_hits += 1

            result.append(
                ChunkSpan(
                    text=description,
                    content_type="image",
                    section_path=span.section_path,
                    attachment_name=span.attachment_name,
                    source_anchor=span.source_anchor,
                    vision_generated=True,
                    vision_description=description,
                    page=span.page,
                )
            )
        return result
