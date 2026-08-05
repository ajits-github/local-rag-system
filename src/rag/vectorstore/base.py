"""Vector store interface. pgvector.py is the v1 implementation — new
backends (Chroma, FAISS, ...) plug in by implementing this class, so
pipeline code never depends on a specific backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag.schemas import Chunk, SearchResult

# Metadata fields that retrieval filters are allowed to touch. An explicit
# whitelist so a filters dict can never be used to inject arbitrary SQL
# identifiers/columns.
ALLOWED_FILTER_FIELDS = {
    "document_id",
    "source",
    "source_type",
    "title",
    "author",
    "url",
    "language",
    "category",
}


class VectorStore(ABC):
    @abstractmethod
    def health_check(self) -> bool:
        """Cheap connectivity check used by GET /health."""

    @abstractmethod
    def get_or_create_document_id(self, source: str, checksum: str) -> tuple[str, bool]:
        """Look up (or register) the stable document_id for `source`.

        Returns (document_id, changed) where `changed` is True when this is
        a new source or its checksum differs from what's on record — i.e.
        the writer stage should replace that document's chunks. document_id
        itself never changes across edits to the same source.
        """

    @abstractmethod
    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Remove all chunks for a document (used before re-writing it)."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks; each chunk must already have its embedding set."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Similarity search, optionally restricted by metadata filters."""
