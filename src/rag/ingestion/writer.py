"""Ingestion stage 5: embed chunk texts and persist them to the vector store."""

from __future__ import annotations

from rag.embedders.base import Embedder
from rag.schemas import Chunk, ChunkMetadata, RawDocument
from rag.vectorstore.base import VectorStore


class Writer:
    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore

    def write(
        self,
        document: RawDocument,
        document_id: str,
        chunk_texts: list[str],
        dataset_id: str,
        category: str | None = None,
    ) -> list[Chunk]:
        embeddings = self._embedder.embed_documents(chunk_texts)
        chunks = [
            Chunk(
                id=f"{document_id}_{i}",
                content=text,
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
                ),
                embedding=embedding,
            )
            for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings))
        ]
        self._vectorstore.add_chunks(chunks)
        return chunks
