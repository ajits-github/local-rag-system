"""Unit tests for FeedbackStore's connection-pool lifecycle and SQL construction.

Fakes the pool and the psycopg2 connection/cursor rather than requiring a
real Postgres, per this repo's `test_pgvector_pool.py` precedent: mock at
the narrowest point that avoids real I/O. See
`tests/integration/test_feedback_persistence.py` for the real-Postgres
round-trip coverage (row contents, dedup/update semantics).
"""

from __future__ import annotations

from typing import Any

import psycopg2
import pytest

from rag.feedback.store import FeedbackStore, FeedbackWriteInput


class FakeCursor:
    """Minimal cursor stand-in: context-manager protocol, execute/fetch, and call recording."""

    def __init__(
        self,
        fetchone_result: object = None,
        fetchall_result: list[Any] | None = None,
        fail: Exception | None = None,
    ) -> None:
        self._fetchone_result = fetchone_result
        self._fetchall_result = fetchall_result if fetchall_result is not None else []
        self._fail = fail
        self.executed: list[tuple[str, object]] = []

    def __enter__(self) -> FakeCursor:
        """Support `with conn.cursor() as cur:`."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Never suppress exceptions raised inside the `with` block."""
        return False

    def execute(self, sql: str, params: object = None) -> None:
        """Record the call, then raise `self._fail` if configured."""
        self.executed.append((sql, params))
        if self._fail is not None:
            raise self._fail

    def fetchone(self) -> object:
        """Return the configured single-row result."""
        return self._fetchone_result

    def fetchall(self) -> list[Any]:
        """Return the configured multi-row result."""
        return self._fetchall_result


class FakeConnection:
    """Minimal connection stand-in: a cursor factory plus transaction context-manager."""

    def __init__(self, cursor_factory: Any = None) -> None:
        self._cursor_factory = cursor_factory or (lambda: FakeCursor())

    def cursor(self) -> FakeCursor:
        """Return a fresh cursor from this connection's factory."""
        return self._cursor_factory()

    def __enter__(self) -> FakeConnection:
        """Support `with conn:` transaction-scope usage."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Never suppress exceptions raised inside the `with` block."""
        return False


class FakePool:
    """Fixed-capacity stand-in for `psycopg2.pool.ThreadedConnectionPool`."""

    def __init__(self, minconn: int, maxconn: int, dsn: str) -> None:
        self.maxconn = maxconn
        self._available = [FakeConnection() for _ in range(maxconn)]

    def getconn(self) -> FakeConnection:
        """Check out a connection, or raise PoolError if none remain."""
        if not self._available:
            raise psycopg2.pool.PoolError("connection pool exhausted")
        return self._available.pop()

    def putconn(self, conn: FakeConnection) -> None:
        """Return a connection to the available pool."""
        self._available.append(conn)


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch `ThreadedConnectionPool` to the in-memory `FakePool`."""
    monkeypatch.setattr("rag.feedback.store.ThreadedConnectionPool", FakePool)


def _store(maxconn: int = 2) -> FeedbackStore:
    return FeedbackStore(dsn="fake-dsn", minconn=1, maxconn=maxconn)


def _write_input(**overrides: Any) -> FeedbackWriteInput:
    base: dict[str, Any] = {
        "tenant_id": None,
        "caller_key": "anonymous",
        "request_id": "req-1",
        "rating": "positive",
        "reason": None,
        "comment": None,
        "route": None,
        "dataset_id": None,
        "query_text": None,
        "answer_text": None,
        "cited_source_ids": [],
        "tool_calls": [],
        "generation_model": None,
        "prompt_id": None,
        "prompt_version": None,
        "retrieval_provider": None,
        "reranker_provider": None,
    }
    base.update(overrides)
    return FeedbackWriteInput(**base)


def test_submit_stores_none_tenant_id_as_empty_string(fake_pool: None) -> None:
    """A None tenant_id is stored as '' so it can participate in the UNIQUE constraint."""
    store = _store(maxconn=1)
    cursor = FakeCursor(fetchone_result=("fb-1", True))
    store._pool._available = [FakeConnection(cursor_factory=lambda: cursor)]  # noqa: SLF001

    store.submit(_write_input(tenant_id=None))

    _sql, params = cursor.executed[0]
    assert params[1] == ""  # tenant_id is the second positional value in the INSERT


def test_repeated_submits_do_not_leak_connections(fake_pool: None) -> None:
    """Many more submit() calls than maxconn all return their connection."""
    store = _store(maxconn=2)
    for conn in store._pool._available:  # noqa: SLF001
        conn._cursor_factory = lambda: FakeCursor(fetchone_result=("fb-1", True))  # noqa: SLF001

    for _ in range(10):
        store.submit(_write_input())

    assert len(store._pool._available) == 2  # noqa: SLF001


def test_submit_returns_connection_when_execute_fails(fake_pool: None) -> None:
    """A cursor.execute() failure inside submit() still returns the connection."""
    store = _store(maxconn=2)
    store._pool._available = [  # noqa: SLF001
        FakeConnection(cursor_factory=lambda: FakeCursor(fail=psycopg2.OperationalError("reset")))
        for _ in range(2)
    ]

    for _ in range(6):
        with pytest.raises(psycopg2.OperationalError):
            store.submit(_write_input())

    assert len(store._pool._available) == 2  # noqa: SLF001


def test_export_applies_filters_as_where_clauses(fake_pool: None) -> None:
    """export() builds a WHERE clause covering exactly the supplied filters."""
    store = _store(maxconn=1)
    cursor = FakeCursor(fetchall_result=[])
    store._pool._available = [FakeConnection(cursor_factory=lambda: cursor)]  # noqa: SLF001

    store.export(tenant_id="tenant_alpha", rating="negative", limit=5)

    sql, params = cursor.executed[0]
    assert "tenant_id = %s" in sql
    assert "rating = %s" in sql
    assert "LIMIT %s" in sql
    assert params == ["tenant_alpha", "negative", 5]


def test_export_with_no_filters_omits_where_clause(fake_pool: None) -> None:
    """No filters at all produces an unfiltered SELECT, not a WHERE with no clauses."""
    store = _store(maxconn=1)
    cursor = FakeCursor(fetchall_result=[])
    store._pool._available = [FakeConnection(cursor_factory=lambda: cursor)]  # noqa: SLF001

    store.export()

    sql, params = cursor.executed[0]
    assert "WHERE" not in sql
    assert params == []


def test_export_returns_connection_on_failure(fake_pool: None) -> None:
    """A cursor.execute() failure inside export() still returns the connection."""
    store = _store(maxconn=2)
    store._pool._available = [  # noqa: SLF001
        FakeConnection(cursor_factory=lambda: FakeCursor(fail=psycopg2.OperationalError("reset")))
        for _ in range(2)
    ]

    for _ in range(4):
        with pytest.raises(psycopg2.OperationalError):
            store.export()

    assert len(store._pool._available) == 2  # noqa: SLF001
