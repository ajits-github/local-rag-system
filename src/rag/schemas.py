"""Shared data shapes passed between ingestion/retrieval pipeline stages."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Common output shape produced by every loader, before cleaning/chunking."""

    content: str
    source: str
    source_type: str
    title: str | None = None
    author: str | None = None
    url: str | None = None
    created_at: datetime
    last_modified: datetime
    language: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkSpan(BaseModel):
    """One chunk of text plus optional structural hints from a `Chunker`.

    All hint fields default to `None`. `Writer.write` fills in
    `content_type="prose"` for any span that leaves it unset, so every
    persisted `ChunkMetadata.content_type` is always a concrete string,
    never `None`.
    """

    text: str
    content_type: str | None = None
    section_path: str | None = None
    code_language: str | None = None
    table_headers: list[str] | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    # Relationship-aware ingestion (multimodal milestone): id of the nearest
    # preceding prose chunk sharing this span's section_path, set by
    # Writer.write's single pass over chunk_spans (chunkers don't know
    # chunk_id, which is assigned at write time) -- None for prose spans
    # themselves and for a non-prose span with no preceding prose in its
    # section. See structured_markdown.py for how image spans are produced.
    parent_chunk_id: str | None = None
    # Set only for a vision-generated sibling of an image span (see
    # VisionProvider); vision_description is that generated text, stored
    # separately from (never overwriting) the image span's own
    # caption/alt-text-derived `text`.
    vision_generated: bool = False
    vision_description: str | None = None


class ChunkMetadata(BaseModel):
    """Metadata stored alongside a chunk's embedding in the vector store."""

    document_id: str
    chunk_id: str
    source: str
    source_type: str
    title: str | None = None
    author: str | None = None
    url: str | None = None
    created_at: datetime
    last_modified: datetime
    language: str | None = None
    chunk_index: int
    # Relative folder path under the ingested root (e.g. "security"), set by
    # the ingestion pipeline when walking a directory tree. None for
    # single-file ingestion or API uploads, which have no folder context.
    category: str | None = None
    # Namespace tag (e.g. "techfusion", "sample_docs") every chunk must
    # declare at ingestion time. No default: this is deliberate — omitting
    # it silently would defeat the point, which is that two datasets can
    # never accidentally share retrieval results. Required as a filter by
    # eval/run_eval.py; available as an optional POST /query filter too.
    dataset_id: str
    # Structured-content hints from ChunkSpan, carried through per-chunk
    # (not per-document) so a table row and a prose paragraph from the
    # same file can be tagged differently. See ChunkSpan for details.
    content_type: str | None = None
    section_path: str | None = None
    code_language: str | None = None
    table_headers: list[str] | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    # See ChunkSpan for what these mean; carried through per-chunk into
    # persisted metadata the same way the structured-content hints above are.
    parent_chunk_id: str | None = None
    vision_generated: bool = False
    vision_description: str | None = None


class Chunk(BaseModel):
    """A unit of text plus its metadata and (once embedded) its vector."""

    id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """A chunk returned by a vector store search, with its similarity score.

    `origin`/`expanded_from` distinguish directly-retrieved results from
    ones added afterward by relationship expansion (see
    retrieval/pipeline.py). Expanded results keep their originating chunk's
    `score` field meaningless for ranking purposes -- they are appended
    after the ranked list, never interleaved into it or given a fabricated
    comparable score.
    """

    chunk: Chunk
    score: float
    origin: Literal["retrieved", "expanded"] = "retrieved"
    expanded_from: str | None = None


class RetrievalAttribution(BaseModel):
    """Raw, per-retriever rankings from one query, before rerank/expansion.

    Produced by `RetrievalPipeline.retrieve_attribution` (see that
    method's docstring) — purely additive/observability: nothing in the
    production `retrieve()`/`answer()` path constructs or consumes this.
    Each list is independently ranked (`dense`/`bm25` best-first from
    their own retriever; `fused` best-first by RRF score), and none of
    them have been through `Reranker.rerank` (so no `rerank_top_n`
    truncation) or relationship expansion.
    """

    dense: list[SearchResult]
    bm25: list[SearchResult]
    fused: list[SearchResult]
