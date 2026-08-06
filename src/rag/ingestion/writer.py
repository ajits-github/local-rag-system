"""Ingestion stage 5: embed chunk texts and persist them to the vector store."""

from __future__ import annotations

from rag.embedders.base import Embedder
from rag.schemas import Chunk, ChunkMetadata, ChunkSpan, RawDocument
from rag.vectorstore.base import VectorStore


class Writer:
    """Embeds chunk texts and persists them to the vector store."""

    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        """Store the embedder/vectorstore this writer will use.

        Parameters
        ----------
        embedder : Embedder
            Embedder used to vectorize chunk texts.
        vectorstore : VectorStore
            Vector store the resulting chunks are persisted to.
        """
        self._embedder = embedder
        self._vectorstore = vectorstore

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
        texts = [span.text for span in chunk_spans]
        embeddings = self._embedder.embed_documents(texts)
        chunks = [
            Chunk(
                id=f"{document_id}_{i}",
                content=span.text,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    chunk_id=f"{document_id}_{i}",
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
                ),
                embedding=embedding,
            )
            for i, (span, embedding) in enumerate(zip(chunk_spans, embeddings, strict=True))
        ]
        self._vectorstore.add_chunks(chunks)
        return chunks
