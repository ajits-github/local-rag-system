from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag.schemas import Chunk, ChunkMetadata, RawDocument


def test_raw_document_requires_content_and_timestamps():
    """RawDocument accepts its required fields and defaults optional ones to None."""
    now = datetime.now(UTC)
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
    """Omitting a required timestamp field raises a pydantic ValidationError."""
    with pytest.raises(ValidationError):
        RawDocument(content="hello", source="a.txt", source_type="text")


def test_chunk_metadata_chunk_id_convention():
    """Chunk.id follows the f"{document_id}_{chunk_index}" convention."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="doc-1_0",
        source="a.txt",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    chunk = Chunk(id=metadata.chunk_id, content="text", metadata=metadata, embedding=[0.1, 0.2])
    assert chunk.id == f"{metadata.document_id}_{metadata.chunk_index}"
    assert chunk.embedding == [0.1, 0.2]


def test_chunk_metadata_requires_dataset_id():
    """Omitting dataset_id raises a pydantic ValidationError (no silent default)."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        ChunkMetadata(
            document_id="doc-1",
            chunk_id="doc-1_0",
            source="a.txt",
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
        )
