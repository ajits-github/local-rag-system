"""Shared data shapes passed between ingestion/retrieval pipeline stages."""

from __future__ import annotations

from datetime import datetime
from typing import Any

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


class Chunk(BaseModel):
    """A unit of text plus its metadata and (once embedded) its vector."""

    id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """A chunk returned by a vector store search, with its similarity score."""

    chunk: Chunk
    score: float
