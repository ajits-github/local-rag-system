from __future__ import annotations

import uuid
from datetime import datetime, timezone

from rag.schemas import Chunk, ChunkMetadata
from rag.vectorstore.pgvector import PgVectorStore


def _store(config) -> PgVectorStore:
    return PgVectorStore(
        dsn=config.database_url(),
        documents_table=config.vectorstore.documents_table,
        chunks_table=config.vectorstore.chunks_table,
        distance_metric=config.vectorstore.distance_metric,
    )


def test_health_check_reports_true_when_reachable(require_postgres, config):
    assert _store(config).health_check() is True


def test_document_id_is_stable_across_edits(require_postgres, config):
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"

    doc_id_1, changed_1 = store.get_or_create_document_id(source, checksum="checksum-a")
    doc_id_2, changed_2 = store.get_or_create_document_id(source, checksum="checksum-a")
    doc_id_3, changed_3 = store.get_or_create_document_id(source, checksum="checksum-b")

    assert doc_id_1 == doc_id_2 == doc_id_3
    assert changed_1 is True
    assert changed_2 is False
    assert changed_3 is True


def test_add_chunks_and_search_round_trip(require_postgres, config):
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    document_id, _ = store.get_or_create_document_id(source, checksum="c1")

    dim = config.embedding.dimension
    now = datetime.now(timezone.utc)
    embedding = [1.0] + [0.0] * (dim - 1)
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_id=f"{document_id}_0",
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
    )
    chunk = Chunk(
        id=metadata.chunk_id, content="hello integration test", metadata=metadata, embedding=embedding
    )

    store.add_chunks([chunk])
    try:
        results = store.search(embedding, top_k=5, filters={"document_id": document_id})
        assert any(r.chunk.id == metadata.chunk_id for r in results)
    finally:
        store.delete_chunks_by_document_id(document_id)
