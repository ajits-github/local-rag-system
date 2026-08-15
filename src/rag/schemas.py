"""Shared data shapes passed between ingestion/retrieval pipeline stages."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Common output shape produced by every loader, before cleaning/chunking.

    `tenant_id`/`allowed_roles`/`classification`/`status`/`document_version`/
    `effective_from`/`trust_level`/`doc_source_type`/`supersedes_source` are
    the safety/freshness milestone's governance fields, parsed from a
    document's YAML front matter by `loaders/text_loader.py` (see that
    module) -- all `None`/empty for documents with no such front matter
    (every pre-existing knowledge-base file), so this is purely additive.
    `doc_source_type` is deliberately not named `source_type` a second time:
    that field already means "markdown"/"text" (the loader's file-type tag)
    here, a different concept from front matter's `source_type`
    (`controlled_internal`/`user_uploaded`, a trust-provenance label) --
    reusing the name would silently overload one column with two meanings.
    `supersedes_source` is the raw `supersedes` filename string as authored
    (e.g. "incident-response-v1.md"), not a resolved document_id -- resolving
    a cross-file reference during single-file ingestion is an
    ordering-fragile chicken/egg problem, so it's matched by path suffix at
    query time instead (see `retrieval/freshness.py`), the same pattern
    already used for gold `relevant_documents`.
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
    # Safety/freshness milestone: document-level governance fields, copied
    # onto every chunk of a document exactly like category/title (see
    # RawDocument for what each means and why doc_source_type isn't named
    # source_type). All None for documents with no governance front matter
    # -- the entire pre-existing corpus is unaffected.
    tenant_id: str | None = None
    allowed_roles: list[str] | None = None
    classification: str | None = None
    status: str | None = None
    document_version: str | None = None
    effective_from: date | None = None
    trust_level: str | None = None
    doc_source_type: str | None = None
    supersedes_source: str | None = None


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
    # Safety milestone: set by RetrievalPipeline from
    # retrieval.injection_detection.detect_injection(chunk.content) --
    # observability/prompt-reinforcement only (see that module's docstring
    # for why this never blocks a result; authorization is the actual gate).
    injection_suspected: bool = False


class DocumentVersionInfo(BaseModel):
    """One document's governance/versioning metadata, for freshness resolution.

    Produced by `VectorStore.list_document_versions` -- one row per
    `document_id` within a dataset (the same governance fields are uniform
    across every chunk of a document, so a plain `SELECT DISTINCT` is
    enough; see `ingestion/writer.py`). Consumed by
    `retrieval/freshness.py` to resolve, per document-version family, which
    version was effective as of a given date -- never persisted itself.
    """

    document_id: str
    source: str
    status: str | None = None
    document_version: str | None = None
    effective_from: date | None = None
    supersedes_source: str | None = None
    tenant_id: str | None = None


class IngestionStats(BaseModel):
    """Aggregate + per-file report from `IngestionPipeline.ingest_path`.

    `results` preserves the pre-existing per-file `{"document_id",
    "chunks_written", "changed"}` shape (plus a new `"status"` key: `"new"`/
    `"changed"`/`"unchanged"`), so existing callers reading individual
    entries are unaffected; the aggregate counts are what's new for the
    safety/freshness milestone's ingestion-observability requirement.
    `chunks_reused` is the total chunk count of every `"unchanged"`
    document (never re-embedded this run, but still real, queryable
    chunks) -- distinct from `chunks_embedded`, which only counts
    newly-written chunks from `"new"`/`"changed"` documents.
    """

    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)


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
