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


class ChunkMetadata(BaseModel):
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


class Chunk(BaseModel):
    id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
