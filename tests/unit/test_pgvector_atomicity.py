"""Regression tests for the CRITICAL ingestion-atomicity fix.

Root cause under test: `get_or_create_document_id`, `delete_chunks_by_document_id`,
and `add_chunks` used to each commit their own, independent transaction, with
no shared transaction and no rollback across the sequence. A failure between
the checksum commit and the chunk write (embedder OOM, DB blip, killed
process) left `documents.checksum` advanced with stale/zero chunks; because
the checksum already matched, every future re-ingestion attempt (including a
deliberate retry) silently reported "unchanged" and never repaired the
corruption.

The fix: `PgVectorStore.get_or_create_document_id` never durably commits the
real checksum on its own (a brand-new row gets the `_PENDING_CHECKSUM`
sentinel; an existing row's checksum is only read). The real checksum is
only ever committed by `replace_document_chunks`, in the same transaction as
the chunk delete + insert. These tests fake the psycopg2 connection/cursor
(mirroring test_pgvector_pool.py's established convention) to prove that
transaction's all-or-nothing behavior without a real Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg2
import psycopg2.extras
import pytest

from rag.schemas import Chunk, ChunkMetadata
from rag.vectorstore.pgvector import PgVectorStore


class _FaultCursor:
    """Cursor stand-in that mutates its connection's *pending* (uncommitted) state.

    `execute()` classifies each statement by its leading SQL keyword and
    updates `conn.pending_checksum`/`pending_chunk_count` accordingly.
    `fail_on`, when set, raises immediately after classifying (but before
    "applying") the matching statement, simulating a mid-transaction
    failure at that exact step.
    """

    def __init__(self, conn: _FaultConnection) -> None:
        self._conn = conn

    def __enter__(self) -> _FaultCursor:
        """Support `with conn.cursor() as cur:`."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Never suppress exceptions raised inside the `with` block."""
        return False

    def execute(self, sql: str, params: object = None) -> None:
        """Classify `sql` by its leading keyword and apply it to pending state."""
        keyword = sql.strip().split(None, 1)[0].upper()
        if keyword == self._conn.fail_on:
            raise psycopg2.OperationalError(f"simulated failure during {keyword}")
        if keyword == "UPDATE":
            self._conn.pending_checksum = params[0]  # type: ignore[index]
        elif keyword == "DELETE":
            self._conn.pending_chunk_count = 0
        elif keyword == "INSERT_CHUNKS":
            self._conn.pending_chunk_count = len(params)  # type: ignore[arg-type]


class _FaultConnection:
    """Connection stand-in that models real commit/rollback semantics.

    `committed_checksum`/`committed_chunk_count` are only updated when the
    `with conn:` block exits without an exception (mirroring psycopg2's own
    commit-on-success/rollback-on-exception behavior); on exception, the
    pending values are discarded and reset back to the last committed ones.
    """

    def __init__(self, initial_checksum: str, initial_chunk_count: int) -> None:
        self.committed_checksum = initial_checksum
        self.committed_chunk_count = initial_chunk_count
        self.pending_checksum = initial_checksum
        self.pending_chunk_count = initial_chunk_count
        self.fail_on: str | None = None
        self.closed = False

    def cursor(self) -> _FaultCursor:
        """Return a fresh fault-injecting cursor bound to this connection."""
        return _FaultCursor(self)

    def __enter__(self) -> _FaultConnection:
        """Support `with conn:` transaction-scope usage."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc_info: object) -> bool:
        """Commit pending state on success, discard it on any exception."""
        if exc_type is None:
            self.committed_checksum = self.pending_checksum
            self.committed_chunk_count = self.pending_chunk_count
        else:
            self.pending_checksum = self.committed_checksum
            self.pending_chunk_count = self.committed_chunk_count
        return False


class _FixedPool:
    """Pool stand-in that always hands back the same connection.

    Needed so a single `replace_document_chunks` call's `UPDATE`/`DELETE`/
    insert all observe and mutate the same connection's pending state, the
    way one real pooled connection backs one method's whole transaction.
    """

    def __init__(
        self, minconn: int, maxconn: int, dsn: str, *, connection: _FaultConnection
    ) -> None:
        self._connection = connection

    def getconn(self) -> _FaultConnection:
        """Return the fixed connection."""
        return self._connection

    def putconn(self, conn: _FaultConnection) -> None:
        """No-op: nothing to return to a single-connection pool."""


def _fake_execute_values(
    cur: _FaultCursor, _sql: str, rows: list[object], **_kwargs: object
) -> None:
    """Stand-in for `execute_values`: route through `cur.execute` as one statement."""
    cur.execute("INSERT_CHUNKS", rows)


def _store(monkeypatch: pytest.MonkeyPatch, conn: _FaultConnection) -> PgVectorStore:
    """Build a PgVectorStore wired to a fixed fault-injecting connection."""
    monkeypatch.setattr(
        "rag.vectorstore.pgvector.ThreadedConnectionPool",
        lambda minconn, maxconn, dsn, **_kwargs: _FixedPool(minconn, maxconn, dsn, connection=conn),
    )
    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", lambda c: None)
    # Patched via the real module object (not a dotted string target) so this
    # doesn't depend on monkeypatch's string-path import-fallback resolving
    # `psycopg2.extras` correctly through pgvector.py's plain `import psycopg2`.
    monkeypatch.setattr(psycopg2.extras, "execute_values", _fake_execute_values)
    return PgVectorStore(dsn="fake-dsn")


def _chunk(document_id: str, index: int) -> Chunk:
    now = datetime.now(UTC)
    chunk_id = f"{document_id}_{index}"
    return Chunk(
        id=chunk_id,
        content=f"chunk {index}",
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source="doc.md",
            source_type="markdown",
            created_at=now,
            last_modified=now,
            chunk_index=index,
            dataset_id="test-dataset",
        ),
        embedding=[0.0],
    )


def test_chunk_insert_failure_rolls_back_the_checksum_and_chunk_delete_too(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failure during the chunk insert leaves checksum and chunk count fully unchanged.

    Fault-injection repro of the exact CRITICAL bug: before the fix, a
    failure at this point would still have left the checksum committed
    (from a prior, separate `get_or_create_document_id` transaction) with
    stale/zero chunks. With the fix, the checksum UPDATE and chunk DELETE
    in this same transaction roll back together with the failed INSERT,
    so nothing is left corrupted, and a retry will correctly see
    `changed=True` again.
    """
    conn = _FaultConnection(initial_checksum="old-checksum", initial_chunk_count=3)
    conn.fail_on = "INSERT_CHUNKS"
    store = _store(monkeypatch, conn)

    with pytest.raises(psycopg2.OperationalError):
        store.replace_document_chunks("doc-1", "new-checksum", [_chunk("doc-1", 0)])

    assert conn.committed_checksum == "old-checksum"
    assert conn.committed_chunk_count == 3


def test_chunk_delete_failure_rolls_back_the_checksum_update_too(monkeypatch: pytest.MonkeyPatch):
    """A failure during the chunk delete leaves the checksum unchanged as well.

    Proves the checksum UPDATE (issued first, earlier in the same
    transaction) is not left committed just because it happened to run
    before the step that actually failed.
    """
    conn = _FaultConnection(initial_checksum="old-checksum", initial_chunk_count=2)
    conn.fail_on = "DELETE"
    store = _store(monkeypatch, conn)

    with pytest.raises(psycopg2.OperationalError):
        store.replace_document_chunks("doc-1", "new-checksum", [_chunk("doc-1", 0)])

    assert conn.committed_checksum == "old-checksum"
    assert conn.committed_chunk_count == 2


def test_successful_replace_commits_checksum_and_chunks_together(monkeypatch: pytest.MonkeyPatch):
    """Sanity check: with no injected fault, both the checksum and chunks do commit."""
    conn = _FaultConnection(initial_checksum="old-checksum", initial_chunk_count=1)
    store = _store(monkeypatch, conn)

    store.replace_document_chunks("doc-1", "new-checksum", [_chunk("doc-1", 0), _chunk("doc-1", 1)])

    assert conn.committed_checksum == "new-checksum"
    assert conn.committed_chunk_count == 2


def test_replace_with_no_chunks_still_commits_the_checksum_and_clears_old_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty `chunks` list is a valid replace (e.g. a document that chunked to nothing)."""
    conn = _FaultConnection(initial_checksum="old-checksum", initial_chunk_count=5)
    store = _store(monkeypatch, conn)

    store.replace_document_chunks("doc-1", "new-checksum", [])

    assert conn.committed_checksum == "new-checksum"
    assert conn.committed_chunk_count == 0
