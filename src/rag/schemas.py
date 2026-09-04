"""Shared data shapes passed between ingestion/retrieval pipeline stages."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Represent the common output shape produced by every loader, before cleaning/chunking.

    Attributes
    ----------
    content : str
        The document's raw text content.
    source : str
        Path or identifier the document was loaded from.
    source_type : str
        Loader file-type tag, e.g. "markdown" or "text".
    title, author, url : str or None
        Optional descriptive metadata.
    created_at, last_modified : datetime
        Document timestamps.
    language : str or None
        Document language, if known.
    tenant_id : str or None
        Owning tenant, from governance front matter.
    allowed_roles : list[str] or None
        Roles authorized to retrieve this document.
    classification : str or None
        Data-classification label (e.g. "confidential").
    status : str or None
        Version lifecycle status (e.g. "active", "superseded").
    document_version : str or None
        Authored version label.
    effective_from : date or None
        Date this version became effective.
    trust_level : str or None
        Knowledge-source trust label (e.g. "authoritative").
    doc_source_type : str or None
        Trust-provenance label from front matter (e.g.
        "controlled_internal"), distinct from `source_type`.
    supersedes_source : str or None
        Raw filename this document supersedes, as authored; resolved to
        a document by path suffix at query time, not at ingestion.
    extra : dict[str, Any]
        Loader-specific fields not otherwise modeled.

    Notes
    -----
    The governance fields (`tenant_id` through `supersedes_source`) are
    parsed from YAML front matter by `loaders/text_loader.py` and are
    `None`/empty for documents with no such front matter.
    """

    content: str
    source: str
    source_type: str
    title: str | None = None
    author: str | None = None
    url: str | None = None
    created_at: datetime
    last_modified: datetime
    language: str | None = None
    tenant_id: str | None = None
    allowed_roles: list[str] | None = None
    classification: str | None = None
    status: str | None = None
    document_version: str | None = None
    effective_from: date | None = None
    trust_level: str | None = None
    doc_source_type: str | None = None
    supersedes_source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkSpan(BaseModel):
    """Represent one chunk of text plus optional structural hints from a `Chunker`.

    Attributes
    ----------
    text : str
        The chunk's text content.
    content_type : str or None
        Structural content type (e.g. "prose", "table", "code", "image").
        `Writer.write` fills in "prose" for any span left unset.
    section_path : str or None
        Markdown heading breadcrumb, e.g. "Setup > Docker".
    code_language, table_headers, attachment_name, source_anchor : optional
        Structural hints for code, table, and attachment content.
    parent_chunk_id : str or None
        Nearest preceding prose chunk in the same section; set later by
        `Writer.write`, since chunkers don't assign `chunk_id`.
    vision_generated : bool
        Whether `vision_description` was produced by a vision model.
    vision_description : str or None
        Vision-generated description, stored separately from the span's
        own caption/alt-text-derived `text`.
    sensitive_field_ids : list[str] or None
        Ingestion-time tags for sensitive-value policies matched in this
        span's text; does not itself imply a role decision.
    page : int or None
        1-indexed source page number, for loaders that can determine one
        (PDF: the page a span was extracted from; DOCX: a running count
        of explicit manual page breaks encountered so far). `None` for
        Markdown/text/HTML, which have no fixed pagination.
    """

    text: str
    content_type: str | None = None
    section_path: str | None = None
    code_language: str | None = None
    table_headers: list[str] | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    parent_chunk_id: str | None = None
    vision_generated: bool = False
    vision_description: str | None = None
    sensitive_field_ids: list[str] | None = None
    page: int | None = None


class ChunkMetadata(BaseModel):
    """Represent metadata stored alongside a chunk's embedding in the vector store.

    Attributes
    ----------
    document_id, chunk_id : str
        Identifiers for the owning document and this chunk.
    source, source_type, title, author, url : str or None
        Document-level descriptive metadata, copied onto every chunk.
    created_at, last_modified : datetime
        Document timestamps.
    language : str or None
        Document language, if known.
    chunk_index : int
        Position of this chunk within its document.
    category : str or None
        Folder path relative to the ingested root; `None` for
        single-file ingestion or API uploads.
    dataset_id : str
        Namespace every chunk must declare at ingestion time, so
        datasets never share retrieval results.
    content_type : str or None
        Structural content type (e.g. "prose", "table", "code", "image").
    section_path, code_language, table_headers, attachment_name, source_anchor : optional
        Structural hints from the originating `ChunkSpan`.
    parent_chunk_id : str or None
        Nearest preceding prose chunk in the same section, if any.
    vision_generated : bool
        Whether `vision_description` was produced by a vision model.
    vision_description : str or None
        Vision-generated description text, if any.
    sensitive_field_ids : list[str] or None
        Ingestion-time tags for sensitive-value policies matched in this
        chunk's text.
    tenant_id : str or None
        Owning tenant, from document governance metadata.
    allowed_roles : list[str] or None
        Roles authorized to retrieve this chunk's document.
    classification : str or None
        Data-classification label.
    status : str or None
        Version lifecycle status.
    document_version : str or None
        Authored version label.
    effective_from : date or None
        Date this version became effective.
    trust_level : str or None
        Knowledge-source trust label.
    doc_source_type : str or None
        Trust-provenance label from front matter.
    supersedes_source : str or None
        Raw filename this document supersedes, as authored.
    page : int or None
        1-indexed source page number, from the originating `ChunkSpan`.
        `None` for source types with no fixed pagination.

    Notes
    -----
    All governance fields (`tenant_id` through `supersedes_source`) are
    `None` for documents with no governance front matter.
    """

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
    category: str | None = None
    dataset_id: str
    content_type: str | None = None
    section_path: str | None = None
    code_language: str | None = None
    table_headers: list[str] | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    parent_chunk_id: str | None = None
    vision_generated: bool = False
    vision_description: str | None = None
    sensitive_field_ids: list[str] | None = None
    tenant_id: str | None = None
    allowed_roles: list[str] | None = None
    classification: str | None = None
    status: str | None = None
    document_version: str | None = None
    effective_from: date | None = None
    trust_level: str | None = None
    doc_source_type: str | None = None
    supersedes_source: str | None = None
    page: int | None = None


class Chunk(BaseModel):
    """A unit of text plus its metadata and (once embedded) its vector."""

    id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """Represent a chunk returned by a vector store search, with its similarity score.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk.
    score : float
        Similarity or ranking score. Meaningless for ranking on an
        expanded result, which inherits its originating result's score.
    origin : {"retrieved", "expanded", "tool_fetched"}
        Whether this result was directly retrieved, added by relationship
        expansion, or fetched directly by an agent tool (`get_document`/
        `get_latest_document`/`get_related_context`'s seed lookup) rather
        than via similarity search.
    expanded_from : str or None
        `chunk_id` of the originating result, if `origin == "expanded"`.
    injection_suspected : bool
        Whether `detect_injection` flagged this chunk's content.
        Observability only; never gates retrieval.
    redacted_field_ids : list[str]
        `field_id`s whose matched value in this chunk's content was
        replaced with a redaction marker.
    """

    chunk: Chunk
    score: float
    origin: Literal["retrieved", "expanded", "tool_fetched"] = "retrieved"
    expanded_from: str | None = None
    injection_suspected: bool = False
    redacted_field_ids: list[str] = Field(default_factory=list)


class DocumentVersionInfo(BaseModel):
    """Represent one document's governance/versioning metadata, for freshness resolution.

    Attributes
    ----------
    document_id, source : str
        Identifiers for the document.
    status : str or None
        Version lifecycle status (e.g. "active", "superseded").
    document_version : str or None
        Authored version label.
    effective_from : date or None
        Date this version became effective.
    supersedes_source : str or None
        Raw filename this document supersedes, as authored.
    tenant_id : str or None
        Owning tenant.

    Notes
    -----
    Produced by `VectorStore.list_document_versions`, one row per
    `document_id`. Consumed by `retrieval/freshness.py`; never persisted.
    """

    document_id: str
    source: str
    status: str | None = None
    document_version: str | None = None
    effective_from: date | None = None
    supersedes_source: str | None = None
    tenant_id: str | None = None


class IngestionStats(BaseModel):
    """Represent the aggregate and per-file report from `IngestionPipeline.ingest_path`.

    Attributes
    ----------
    discovered : int
        Total files discovered during this ingestion run.
    new, changed, unchanged, deleted : int
        Per-status file counts.
    chunks_embedded : int
        Chunks newly written this run (from "new"/"changed" documents).
    chunks_reused : int
        Chunks belonging to "unchanged" documents, not re-embedded.
    results : list[dict[str, Any]]
        Per-file `{"document_id", "chunks_written", "changed", "status"}`
        records.
    vision_stats : VisionCallStats or None
        `Writer.vision_stats` after this run, or `None` when no
        `VisionProvider` was configured (`config.vision.provider ==
        "none"`): an absent value, not a zeroed one, so "vision wasn't
        used" is distinguishable from "vision ran and touched zero images".
    """

    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    vision_stats: VisionCallStats | None = None


class VisionCallStats(BaseModel):
    """Aggregate `VisionProvider.describe_image` call stats from one `Writer` instance.

    Accumulated by `Writer._with_vision_siblings` across every `write()`
    call the same `Writer` instance handles (one full
    `IngestionPipeline.ingest_path` run). Read off `Writer.vision_stats`
    after a run and folded into `IngestionStats`; not part of
    `eval/run_eval.py`'s report.

    Attributes
    ----------
    images_processed : int
        Total image spans considered (cache hit or miss).
    cache_hits, cache_misses : int
        How many of those were already-cached vs. required a real
        `describe_image` call.
    total_latency_ms : float
        Summed wall-clock time across only the cache-miss (real) calls.
    """

    images_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0


class RetrievalAttribution(BaseModel):
    """Raw, per-retriever rankings from one query, before rerank/expansion.

    Produced by `RetrievalPipeline.retrieve_attribution` for observability;
    production `retrieve()` and `answer()` do not consume it. Each list is
    independently ranked (`dense`/`bm25` best-first from their retriever,
    `fused` best-first by RRF score), before reranking or relationship
    expansion.
    """

    dense: list[SearchResult]
    bm25: list[SearchResult]
    fused: list[SearchResult]
