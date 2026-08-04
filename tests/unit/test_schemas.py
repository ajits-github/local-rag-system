from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag.schemas import Chunk, ChunkMetadata, RawDocument


def test_raw_document_requires_content_and_timestamps():
    now = datetime.now(timezone.utc)
    doc = RawDocument(
        content="hello",
        source="a.txt",
        source_type="text",
        created_at=now,
        last_modified=now,
    )
    assert doc.title is None
    assert doc.language is None


def test_raw_document_missing_required_field_raises():
    with pytest.raises(ValidationError):
        RawDocument(content="hello", source="a.txt", source_type="text")


def test_chunk_metadata_chunk_id_convention():
    now = datetime.now(timezone.utc)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="doc-1_0",
        source="a.txt",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
    )
    chunk = Chunk(id=metadata.chunk_id, content="text", metadata=metadata, embedding=[0.1, 0.2])
    assert chunk.id == f"{metadata.document_id}_{metadata.chunk_index}"
    assert chunk.embedding == [0.1, 0.2]
