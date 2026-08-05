from __future__ import annotations

import uuid
from datetime import UTC, datetime

from rag.schemas import Chunk, ChunkMetadata
from rag.vectorstore.pgvector import PgVectorStore

TEST_DATASET_ID = "pytest-integration"


def _store(config) -> PgVectorStore:
    """Build a PgVectorStore from the session config."""
    return PgVectorStore(
        dsn=config.database_url(),
        documents_table=config.vectorstore.documents_table,
        chunks_table=config.vectorstore.chunks_table,
        distance_metric=config.vectorstore.distance_metric,
    )


def test_health_check_reports_true_when_reachable(require_postgres, config):
    """health_check() succeeds against a live, reachable database."""
    assert _store(config).health_check() is True


def test_document_id_is_stable_across_edits(require_postgres, config):
    """document_id stays constant across checksum changes; only `changed` flips."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"

    doc_id_1, changed_1 = store.get_or_create_document_id(source, "checksum-a", TEST_DATASET_ID)
    doc_id_2, changed_2 = store.get_or_create_document_id(source, "checksum-a", TEST_DATASET_ID)
    doc_id_3, changed_3 = store.get_or_create_document_id(source, "checksum-b", TEST_DATASET_ID)

    assert doc_id_1 == doc_id_2 == doc_id_3
    assert changed_1 is True
    assert changed_2 is False
    assert changed_3 is True

    store.delete_document(doc_id_1)


def test_same_source_in_different_datasets_does_not_collide(require_postgres, config):
    """The same source path in two datasets gets two distinct document_ids."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"

    doc_id_a, changed_a = store.get_or_create_document_id(source, "checksum-a", "pytest-ns-a")
    doc_id_b, changed_b = store.get_or_create_document_id(source, "checksum-a", "pytest-ns-b")

    try:
        assert doc_id_a != doc_id_b
        assert changed_a is True
        assert changed_b is True
    finally:
        store.delete_document(doc_id_a)
        store.delete_document(doc_id_b)


def test_add_chunks_and_search_round_trip(require_postgres, config):
    """A chunk written via add_chunks is found again by search()."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    document_id, _ = store.get_or_create_document_id(source, "c1", TEST_DATASET_ID)

    dim = config.embedding.dimension
    now = datetime.now(UTC)
    embedding = [1.0] + [0.0] * (dim - 1)
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_id=f"{document_id}_0",
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id=TEST_DATASET_ID,
    )
    chunk = Chunk(
        id=metadata.chunk_id,
        content="hello integration test",
        metadata=metadata,
        embedding=embedding,
    )

    store.add_chunks([chunk])
    try:
        results = store.search(embedding, top_k=5, filters={"document_id": document_id})
        assert any(r.chunk.id == metadata.chunk_id for r in results)
    finally:
        store.delete_document(document_id)
