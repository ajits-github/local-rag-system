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
    "dataset_id",
}


class VectorStore(ABC):
    @abstractmethod
    def health_check(self) -> bool:
        """Cheap connectivity check used by GET /health."""

    @abstractmethod
    def get_or_create_document_id(
        self, source: str, checksum: str, dataset_id: str
    ) -> tuple[str, bool]:
        """Look up (or register) the stable document_id for (source, dataset_id).

        Returns (document_id, changed) where `changed` is True when this is
        a new source or its checksum differs from what's on record — i.e.
        the writer stage should replace that document's chunks. document_id
        itself never changes across edits to the same source. Identity is
        scoped per dataset_id, so the same relative path can exist in two
        different datasets without colliding.
        """

    @abstractmethod
    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Remove all chunks for a document, keeping its `documents` row —
        used mid-re-ingestion, right before writing that document's fresh
        chunks under the same document_id. NOT for full teardown; use
        delete_document for that (see below)."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Fully remove a document's `documents` row (cascading to all its
        chunks). For test teardown / dataset cleanup — never called during
        normal re-ingestion of an unchanged-identity document."""

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None:
        """Remove every document (and cascade its chunks) tagged with
        dataset_id — bulk equivalent of delete_document for clearing/
        re-ingesting a whole namespace."""

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
