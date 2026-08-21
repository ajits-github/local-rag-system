"""Vector store interface: persists embedded chunks and serves similarity search.

pgvector.py is the v1 implementation. New backends (Chroma, FAISS, ...)
plug in by implementing this class, so pipeline code never depends on a
specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag.retrieval.authorization import AuthorizationContext
from rag.schemas import Chunk, DocumentVersionInfo, SearchResult

# Metadata fields that retrieval filters are allowed to touch. An explicit
# whitelist so a filters dict can never be used to inject arbitrary SQL
# identifiers/columns. These are exact-match convenience filters; tenant,
# role, freshness, and trust enforcement are handled by AuthorizationContext
# and backend-specific authorization predicates.
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
    "tenant_id",
    "classification",
    "status",
    "trust_level",
    "doc_source_type",
}


class VectorStore(ABC):
    """Persists embedded chunks and serves similarity search over them.

    pgvector.py is the v1 implementation. New backends (Chroma, FAISS,
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
            a new source or its checksum differs from what's on record.
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

        For test teardown / dataset cleanup. Never called during normal
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
    def list_document_sources(self, dataset_id: str) -> list[str]:
        """List every currently-persisted document's `source` within a dataset.

        Used by `IngestionPipeline.ingest_path` to detect deleted documents
        (a `source` persisted here but no longer discovered on disk).
        `ingest_path` only ever walks files that currently exist, so this is
        the other half of "new/changed/unchanged/**deleted**" incremental
        ingestion.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to list.

        Returns
        -------
        list[str]
            Every persisted `source` value for this dataset.
        """

    @abstractmethod
    def delete_documents_by_source(self, dataset_id: str, sources: list[str]) -> int:
        """Delete documents (and cascade their chunks) by exact `source` match, within one dataset.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to delete from.
        sources : list[str]
            Exact `source` values to delete.

        Returns
        -------
        int
            Number of documents deleted.
        """

    @abstractmethod
    def count_chunks_by_document(self, dataset_id: str) -> dict[str, int]:
        """Count persisted chunks per `document_id`, within one dataset.

        Used by `IngestionPipeline.ingest_path` to report `chunks_reused`
        (the chunk count of every document skipped this run because its
        checksum was unchanged) without a per-document query.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to count within.

        Returns
        -------
        dict[str, int]
            `{document_id: chunk_count}`, one entry per document that has
            at least one chunk.
        """

    @abstractmethod
    def list_document_versions(self, dataset_id: str) -> list[DocumentVersionInfo]:
        """List one governance-metadata row per document within a dataset.

        Used by `retrieval/freshness.py` to resolve document-version
        families. Governance fields (`status`/`document_version`/
        `effective_from`/`supersedes_source`) are uniform across every
        chunk of a document (see `ingestion/writer.py`), so this is a
        `SELECT DISTINCT` over the chunks table, not a new document-level
        table.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to list.

        Returns
        -------
        list[DocumentVersionInfo]
            One entry per document_id that has at least one chunk in this
            dataset.
        """

    @abstractmethod
    def count_chunks_by_content_type(self, dataset_id: str) -> dict[str, int]:
        """Count persisted chunks per `content_type`, within one dataset.

        Used by `eval/corpus_lineage.py` for the recorded "image count"
        (`counts.get("image", 0)`) and general corpus-composition reporting.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to count within.

        Returns
        -------
        dict[str, int]
            `{content_type: chunk_count}`.
        """

    @abstractmethod
    def get_document_checksums(self, dataset_id: str) -> dict[str, str]:
        """Return every document's `(source, checksum)` pair within a dataset.

        Used by `eval/corpus_lineage.py` to compute a deterministic corpus
        digest (sha256 over sorted `"{source}:{checksum}"` lines) --
        `checksum` lives on the `documents` table (see
        `get_or_create_document_id`), not on `chunks`.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to read.

        Returns
        -------
        dict[str, str]
            `{source: checksum}`, one entry per document.
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
        auth: AuthorizationContext | None = None,
    ) -> list[SearchResult]:
        """Similarity search, optionally restricted by metadata filters and authorization.

        Parameters
        ----------
        query_embedding : list[float]
            The query's embedding vector.
        top_k : int
            Maximum number of results to return.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters; keys must be in
            `ALLOWED_FILTER_FIELDS`.
        auth : AuthorizationContext | None, optional
            Caller identity/date context (see `retrieval/authorization.py`).
            `None` means fully unrestricted. Passing a context can only
            ever narrow results (ANDed with `filters`), never broaden them.
            Applied in SQL, before any row leaves the database.

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
        auth: AuthorizationContext | None = None,
    ) -> list[SearchResult]:
        """Lexical (BM25) search over chunk content, optionally restricted by filters/authorization.

        Parameters
        ----------
        query : str
            The raw query text (tokenized internally; not an embedding).
        top_k : int
            Maximum number of results to return.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters; keys must be in
            `ALLOWED_FILTER_FIELDS`.
        auth : AuthorizationContext | None, optional
            Same semantics as `search`'s `auth` parameter, applied
            to the same underlying SQL fetch before BM25 ranking runs.

        Returns
        -------
        list[SearchResult]
            Matching chunks with BM25 scores, ranked best-first. `.score`
            is the raw BM25 score (unbounded, not comparable across
            queries). Callers that need a comparable ranking signal
            across dense and keyword results should fuse via
            `rag.retrieval.fusion.reciprocal_rank_fusion`, not compare
            `.score` directly.

        Raises
        ------
        ValueError
            If `filters` contains a key not in `ALLOWED_FILTER_FIELDS`.
        """

    @abstractmethod
    def get_chunks_by_ids(
        self, chunk_ids: list[str], auth: AuthorizationContext | None = None
    ) -> list[Chunk]:
        """Fetch chunks by primary key, e.g. to resolve a `parent_chunk_id` reference.

        Used by relationship expansion. Callers pass ids sourced from an
        already dataset-filtered result, such as a `parent_chunk_id` from
        chunk metadata. `auth` is applied defensively so an expansion lookup
        cannot bypass the same authorization predicate used by search.

        Parameters
        ----------
        chunk_ids : list[str]
            Chunk ids to fetch.
        auth : AuthorizationContext | None, optional
            See `search`'s `auth` parameter.

        Returns
        -------
        list[Chunk]
            Matching chunks, in no particular order; ids with no matching
            row (or excluded by `auth`) are silently omitted.
        """

    @abstractmethod
    def get_chunks_by_section(
        self,
        document_id: str,
        section_path: str | None,
        auth: AuthorizationContext | None = None,
    ) -> list[Chunk]:
        """Fetch every chunk of one document's section, ordered by `chunk_index`.

        Used by relationship expansion to compute a retrieved chunk's
        immediate neighbors (previous/next by `chunk_index`) within the
        same section. `document_id` already uniquely scopes to one
        dataset, so this cannot cross a `dataset_id` boundary. `auth` is
        applied defensively (see `get_chunks_by_ids`).

        Parameters
        ----------
        document_id : str
            The document to fetch chunks from.
        section_path : str | None
            The `section_path` value to match exactly (including `None`,
            for chunks with no section context).
        auth : AuthorizationContext | None, optional
            See `search`'s `auth` parameter.

        Returns
        -------
        list[Chunk]
            Matching chunks, ordered by `chunk_index` ascending.
        """

    @abstractmethod
    def get_chunks_by_source(
        self,
        source: str,
        dataset_id: str,
        auth: AuthorizationContext | None = None,
        limit: int | None = None,
    ) -> list[Chunk]:
        """Fetch every chunk of one document by its exact `source` path, ordered by `chunk_index`.

        Used by the agent `get_document`/`get_latest_document` tools
        (`rag/agent/tools.py`) to fetch a specific document's content
        directly, rather than via similarity search. `auth` is applied
        defensively, identically to `get_chunks_by_section`.

        Parameters
        ----------
        source : str
            Exact `source` value to match.
        dataset_id : str
            The dataset namespace to search within.
        auth : AuthorizationContext | None, optional
            See `search`'s `auth` parameter.
        limit : int | None, optional
            Maximum number of chunks to return (a SQL-level `LIMIT`,
            applied before any caller-side selection). Callers needing a
            bounded read regardless of document size (e.g. agent tools)
            should always pass this; `None` returns every chunk.

        Returns
        -------
        list[Chunk]
            Matching chunks, ordered by `chunk_index` ascending.
        """

    @abstractmethod
    def get_cached_image_description(
        self, image_checksum: str, provider: str, model_name: str, prompt_version: str
    ) -> str | None:
        """Look up a previously-generated vision description.

        Keyed by image checksum *and* provider/model/prompt_version (see
        `cache_image_description`), so switching any of the three always
        misses instead of replaying a description generated under
        different instructions.

        Parameters
        ----------
        image_checksum : str
            sha256 of the image file's bytes.
        provider : str
            `VisionProvider.provider_name` of the *current* provider.
        model_name : str
            `VisionProvider.model_name` of the *current* provider.
        prompt_version : str
            `VisionProvider.prompt_version` of the *current* provider.

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
        prompt_version: str,
        description: str,
    ) -> None:
        """Persist a vision-generated description.

        So an unchanged image is never reprocessed by a (cost-incurring)
        VisionProvider call, even across documents or ingestion runs --
        as long as provider/model/prompt_version are also unchanged (the
        cache key includes all three; see `get_cached_image_description`).
        Postgres-backed rather than a separate cache service, per this
        project's "prefer Postgres over Redis absent a concrete need"
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
        prompt_version : str
            The `VisionProvider.prompt_version` that produced this description.
        description : str
            The generated description text.
        """
