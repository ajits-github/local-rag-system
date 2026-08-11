"""Vector store interface: persists embedded chunks and serves similarity search.

pgvector.py is the v1 implementation — new backends (Chroma, FAISS, ...)
plug in by implementing this class, so pipeline code never depends on a
specific backend.
"""

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
    "content_type",
}


class VectorStore(ABC):
    """Persists embedded chunks and serves similarity search over them.

    pgvector.py is the v1 implementation — new backends (Chroma, FAISS,
    ...) plug in by implementing this class, so pipeline code never
    depends on a specific backend.
    """

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap connectivity check used by GET /health.

        Returns
        -------
        bool
            True if the backend is reachable.
        """

    @abstractmethod
    def get_or_create_document_id(
        self, source: str, checksum: str, dataset_id: str
    ) -> tuple[str, bool]:
        """Look up (or register) the stable document_id for (source, dataset_id).

        document_id itself never changes across edits to the same source.
        Identity is scoped per dataset_id, so the same relative path can
        exist in two different datasets without colliding.

        Parameters
        ----------
        source : str
            The document's path, as recorded at ingestion time.
        checksum : str
            sha256 of the current file bytes.
        dataset_id : str
            Namespace this document belongs to.

        Returns
        -------
        tuple[str, bool]
            ``(document_id, changed)`` where `changed` is True when this is
            a new source or its checksum differs from what's on record —
            i.e. the writer stage should replace that document's chunks.
        """

    @abstractmethod
    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Remove all chunks for a document, keeping its `documents` row.

        Used mid-re-ingestion, right before writing that document's fresh
        chunks under the same document_id. NOT for full teardown; use
        `delete_document` for that.

        Parameters
        ----------
        document_id : str
            The document whose chunks should be removed.
        """

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Fully remove a document's `documents` row (cascading to all its chunks).

        For test teardown / dataset cleanup — never called during normal
        re-ingestion of an unchanged-identity document.

        Parameters
        ----------
        document_id : str
            The document to remove.
        """

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None:
        """Remove every document (and cascade its chunks) tagged with dataset_id.

        Bulk equivalent of `delete_document` for clearing/re-ingesting a
        whole namespace.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to clear.
        """

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks; each chunk must already have its embedding set.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks to upsert.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Similarity search, optionally restricted by metadata filters.

        Parameters
        ----------
        query_embedding : list[float]
            The query's embedding vector.
        top_k : int
            Maximum number of results to return.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters; keys must be in
            `ALLOWED_FILTER_FIELDS`.

        Returns
        -------
        list[SearchResult]
            Matching chunks with similarity scores, ranked best-first.

        Raises
        ------
        ValueError
            If `filters` contains a key not in `ALLOWED_FILTER_FIELDS`.
        """

    @abstractmethod
    def search_keyword(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Lexical (BM25) search over chunk content, optionally restricted by metadata filters.

        Parameters
        ----------
        query : str
            The raw query text (tokenized internally; not an embedding).
        top_k : int
            Maximum number of results to return.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters; keys must be in
            `ALLOWED_FILTER_FIELDS`.

        Returns
        -------
        list[SearchResult]
            Matching chunks with BM25 scores, ranked best-first. `.score`
            is the raw BM25 score (unbounded, not comparable across
            queries) -- callers that need a comparable ranking signal
            across dense and keyword results should fuse via
            `rag.retrieval.fusion.reciprocal_rank_fusion`, not compare
            `.score` directly.

        Raises
        ------
        ValueError
            If `filters` contains a key not in `ALLOWED_FILTER_FIELDS`.
        """

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by primary key, e.g. to resolve a `parent_chunk_id` reference.

        Used by relationship expansion (retrieval/pipeline.py) -- callers
        only ever pass ids already sourced from an already-`dataset_id`-
        filtered `SearchResult` (a `parent_chunk_id` on a chunk's own
        metadata), so this never needs its own `dataset_id` check to stay
        isolated: a `chunk_id`/`document_id` is never shared across
        datasets (see CLAUDE.md's "Document identity" section).

        Parameters
        ----------
        chunk_ids : list[str]
            Chunk ids to fetch.

        Returns
        -------
        list[Chunk]
            Matching chunks, in no particular order; ids with no matching
            row are silently omitted (e.g. a stale `parent_chunk_id` from a
            deleted chunk).
        """

    @abstractmethod
    def get_chunks_by_section(self, document_id: str, section_path: str | None) -> list[Chunk]:
        """Fetch every chunk of one document's section, ordered by `chunk_index`.

        Used by relationship expansion to compute a retrieved chunk's
        immediate neighbors (previous/next by `chunk_index`) within the
        same section. `document_id` already uniquely scopes to one
        dataset, so this cannot cross a `dataset_id` boundary.

        Parameters
        ----------
        document_id : str
            The document to fetch chunks from.
        section_path : str | None
            The `section_path` value to match exactly (including `None`,
            for chunks with no section context).

        Returns
        -------
        list[Chunk]
            Matching chunks, ordered by `chunk_index` ascending.
        """

    @abstractmethod
    def get_cached_image_description(self, image_checksum: str) -> str | None:
        """Look up a previously-generated vision description by image checksum.

        Parameters
        ----------
        image_checksum : str
            sha256 of the image file's bytes.

        Returns
        -------
        str | None
            The cached description, or `None` on a cache miss.
        """

    @abstractmethod
    def cache_image_description(
        self,
        image_checksum: str,
        source_path: str,
        provider: str,
        model_name: str,
        description: str,
    ) -> None:
        """Persist a vision-generated description, keyed by image checksum.

        So an unchanged image is never reprocessed by a (cost-incurring)
        hosted VisionProvider call, even across documents or ingestion
        runs. Postgres-backed rather than a separate cache service, per
        this project's "prefer Postgres over Redis absent a concrete need"
        convention.

        Parameters
        ----------
        image_checksum : str
            sha256 of the image file's bytes.
        source_path : str
            Path the image was read from, for provenance/debugging.
        provider : str
            The `VisionProvider.provider_name` that produced this description.
        model_name : str
            The `VisionProvider.model_name` that produced this description.
        description : str
            The generated description text.
        """
