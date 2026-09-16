from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
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
    """document_id stays constant across checksum changes; only `changed` flips.

    `get_or_create_document_id` no longer durably commits a checksum on
    its own (see its docstring): a document only stops reading `changed`
    on its next call once `replace_document_chunks` has actually
    committed that checksum, atomically with its chunks. This is the
    fix for the CRITICAL ingestion-atomicity bug -- a standalone checksum
    commit ahead of a chunk write that might still fail is exactly the
    corruption window that used to exist.
    """
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"

    doc_id_1, changed_1 = store.get_or_create_document_id(source, "checksum-a", TEST_DATASET_ID)
    store.replace_document_chunks(doc_id_1, "checksum-a", [])
    doc_id_2, changed_2 = store.get_or_create_document_id(source, "checksum-a", TEST_DATASET_ID)
    doc_id_3, changed_3 = store.get_or_create_document_id(source, "checksum-b", TEST_DATASET_ID)
    store.replace_document_chunks(doc_id_3, "checksum-b", [])
    doc_id_4, changed_4 = store.get_or_create_document_id(source, "checksum-b", TEST_DATASET_ID)

    assert doc_id_1 == doc_id_2 == doc_id_3 == doc_id_4
    assert changed_1 is True
    assert changed_2 is False
    assert changed_3 is True
    assert changed_4 is False

    store.delete_document(doc_id_1)


def test_concurrent_first_time_inserts_of_the_same_new_document_do_not_raise(
    require_postgres, config
):
    """Two racing first-time get_or_create_document_id calls for the same new document.

    Both must succeed and agree on document_id, rather than one raising an
    unhandled `UniqueViolation` from a bare SELECT-then-INSERT. Regression
    test for the concurrent-insert race half of the CRITICAL
    ingestion-atomicity fix.
    """
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    barrier = threading.Barrier(2)

    def _attempt() -> tuple[str, bool]:
        barrier.wait(timeout=5)
        return store.get_or_create_document_id(source, "same-checksum", TEST_DATASET_ID)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result() for future in [executor.submit(_attempt) for _ in range(2)]]

    document_ids = {document_id for document_id, _changed in results}
    try:
        assert len(document_ids) == 1  # both races resolved to the same, canonical document_id
        assert all(changed is True for _document_id, changed in results)
    finally:
        store.delete_document(document_ids.pop())


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


def test_structured_content_fields_round_trip(require_postgres, config):
    """All 6 structured-content metadata fields (incl. table_headers: list[str]) round-trip."""
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
        source_type="markdown",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id=TEST_DATASET_ID,
        content_type="table",
        section_path="Top > Sub",
        code_language=None,
        table_headers=["Name", "Value"],
        attachment_name=None,
        source_anchor="rows 1-20",
    )
    chunk = Chunk(id=metadata.chunk_id, content="| a | 1 |", metadata=metadata, embedding=embedding)

    store.add_chunks([chunk])
    try:
        results = store.search(embedding, top_k=5, filters={"document_id": document_id})
        found = next(r for r in results if r.chunk.id == metadata.chunk_id)
        assert found.chunk.metadata.content_type == "table"
        assert found.chunk.metadata.section_path == "Top > Sub"
        assert found.chunk.metadata.table_headers == ["Name", "Value"]
        assert found.chunk.metadata.source_anchor == "rows 1-20"
        assert found.chunk.metadata.code_language is None
        assert found.chunk.metadata.attachment_name is None
    finally:
        store.delete_document(document_id)


def test_structured_content_fields_refresh_on_reingestion(require_postgres, config):
    """ON CONFLICT DO UPDATE refreshes the 6 new fields too, not just content/embedding."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    document_id, _ = store.get_or_create_document_id(source, "c1", TEST_DATASET_ID)

    dim = config.embedding.dimension
    now = datetime.now(UTC)
    embedding = [1.0] + [0.0] * (dim - 1)
    chunk_id = f"{document_id}_0"

    first = Chunk(
        id=chunk_id,
        content="original prose",
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source=source,
            source_type="markdown",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id=TEST_DATASET_ID,
            content_type="prose",
        ),
        embedding=embedding,
    )
    second = Chunk(
        id=chunk_id,
        content="| a | 1 |",
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source=source,
            source_type="markdown",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id=TEST_DATASET_ID,
            content_type="table",
            table_headers=["Name", "Value"],
        ),
        embedding=embedding,
    )

    store.add_chunks([first])
    store.add_chunks([second])
    try:
        results = store.search(embedding, top_k=5, filters={"document_id": document_id})
        found = next(r for r in results if r.chunk.id == chunk_id)
        assert found.chunk.metadata.content_type == "table"
        assert found.chunk.metadata.table_headers == ["Name", "Value"]
    finally:
        store.delete_document(document_id)


def test_search_keyword_finds_lexical_match(require_postgres, config):
    """search_keyword() finds a chunk via an exact-token match, using a placeholder embedding."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    document_id, _ = store.get_or_create_document_id(source, "c1", TEST_DATASET_ID)

    dim = config.embedding.dimension
    now = datetime.now(UTC)
    chunk_id = f"{document_id}_0"
    chunk = Chunk(
        id=chunk_id,
        content="The retry_transient helper retries TimeoutError and ConnectionError.",
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source=source,
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id=TEST_DATASET_ID,
        ),
        embedding=[0.0] * dim,
    )

    store.add_chunks([chunk])
    try:
        results = store.search_keyword(
            "retry_transient", top_k=5, filters={"document_id": document_id}
        )
        assert any(r.chunk.id == chunk_id for r in results)
    finally:
        store.delete_document(document_id)


def test_search_keyword_respects_filters_and_rejects_disallowed_key(require_postgres, config):
    """search_keyword() applies metadata filters and rejects a non-whitelisted filter key."""
    store = _store(config)
    source = f"test-source-{uuid.uuid4()}"
    document_id, _ = store.get_or_create_document_id(source, "c1", TEST_DATASET_ID)

    dim = config.embedding.dimension
    now = datetime.now(UTC)
    chunk_id = f"{document_id}_0"
    chunk = Chunk(
        id=chunk_id,
        content="A distinctive keyword-search phrase for filter testing.",
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source=source,
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id=TEST_DATASET_ID,
        ),
        embedding=[0.0] * dim,
    )

    store.add_chunks([chunk])
    try:
        matching = store.search_keyword(
            "distinctive", top_k=5, filters={"document_id": document_id}
        )
        assert any(r.chunk.id == chunk_id for r in matching)

        other_doc_only = store.search_keyword(
            "distinctive", top_k=5, filters={"document_id": str(uuid.uuid4())}
        )
        assert not any(r.chunk.id == chunk_id for r in other_doc_only)

        try:
            store.search_keyword("distinctive", top_k=5, filters={"chunk_id": chunk_id})
            raise AssertionError("expected ValueError for a disallowed filter key")
        except ValueError as exc:
            assert "not allowed" in str(exc)
    finally:
        store.delete_document(document_id)


def test_search_keyword_empty_corpus_returns_empty_list(require_postgres, config):
    """search_keyword() returns [] (not an error) when no chunks match the filters."""
    store = _store(config)
    results = store.search_keyword("anything", top_k=5, filters={"document_id": str(uuid.uuid4())})
    assert results == []
