"""Regression tests for PgVectorStore's connection-pool lifecycle.

Root cause under test: the old `_conn()` helper called `pool.getconn()`
and `register_vector(conn)` as two separate statements *before* returning
to the caller's own `try/finally`, so a `register_vector` failure (e.g.
the `vector` extension not yet installed on a fresh database) checked out
a pooled connection that no `finally` block ever saw, let alone returned.
Enough leaked connections exhausted the pool
(`psycopg2.pool.PoolError: connection pool exhausted`), reproduced here by
GET /health being polled repeatedly against a database whose schema
hadn't been initialized yet. exactly the containerized Docker smoke
test's failure mode.

These tests fake the pool and the psycopg2 connection/cursor rather than
requiring a real Postgres, per this repo's unit-test convention of
mocking at the narrowest point that avoids real I/O.
"""

from __future__ import annotations

import psycopg2
import pytest

from rag.vectorstore.pgvector import PgVectorStore


class FakeCursor:
    """Minimal cursor stand-in: supports the context-manager protocol and execute/fetch."""

    def __init__(self, fail: Exception | None = None) -> None:
        self._fail = fail

    def __enter__(self) -> FakeCursor:
        """Support `with conn.cursor() as cur:`."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Never suppress exceptions raised inside the `with` block."""
        return False

    def execute(self, *args: object, **kwargs: object) -> None:
        """Raise `self._fail` if configured, otherwise do nothing."""
        if self._fail is not None:
            raise self._fail

    def fetchone(self) -> tuple[int] | None:
        """Return a harmless one-row result."""
        return (1,)

    def fetchall(self) -> list[tuple[int]]:
        """Return no rows."""
        return []


class FakeConnection:
    """Minimal connection stand-in: a cursor factory plus transaction context-manager."""

    def __init__(self, cursor_fail: Exception | None = None) -> None:
        self.closed = False
        self._cursor_fail = cursor_fail

    def cursor(self) -> FakeCursor:
        """Return a fresh fake cursor."""
        return FakeCursor(fail=self._cursor_fail)

    def __enter__(self) -> FakeConnection:
        """Support `with conn:` transaction-scope usage."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Never suppress exceptions raised inside the `with` block."""
        return False


class FakePool:
    """Fixed-capacity stand-in for `psycopg2.pool.ThreadedConnectionPool`.

    Raises the same `psycopg2.pool.PoolError` the real pool raises once
    every connection is checked out and none remain. so a test that
    leaks connections fails the same way the real bug did in CI.
    """

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
    monkeypatch.setattr("rag.vectorstore.pgvector.ThreadedConnectionPool", FakePool)


def _store(maxconn: int = 3) -> PgVectorStore:
    return PgVectorStore(dsn="fake-dsn", minconn=1, maxconn=maxconn)


def test_repeated_health_checks_do_not_exhaust_pool(fake_pool, monkeypatch):
    """Many more health_check() calls than maxconn all succeed without exhausting the pool."""
    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", lambda conn: None)
    store = _store(maxconn=2)

    for _ in range(20):
        assert store.health_check() is True

    assert len(store._pool._available) == 2  # noqa: SLF001


def test_health_check_returns_connection_when_register_vector_fails(fake_pool, monkeypatch):
    """The exact CI failure mode: register_vector fails on every call (missing extension).

    Before the fix, each of these calls would have leaked its checked-out
    connection; by the `maxconn`-th call, `pool.getconn()` itself would
    start raising `PoolError`. With the fix, `health_check()` still
    reports False (it's genuinely unhealthy), but every connection comes
    back to the pool.
    """

    def always_fails(conn: FakeConnection) -> None:
        raise psycopg2.ProgrammingError("vector type not found in the database")

    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", always_fails)
    store = _store(maxconn=3)

    for _ in range(10):  # far more calls than maxconn
        assert store.health_check() is False

    assert len(store._pool._available) == 3  # noqa: SLF001


def test_docker_style_repeated_health_polling_never_raises_pool_error(fake_pool, monkeypatch):
    """Simulates the Docker smoke test's polling loop hitting /health repeatedly.

    Reproduces the reported scenario end to end: a fresh container whose
    database hasn't been initialized yet (register_vector keeps failing)
    gets polled well past `maxconn` times, exactly as
    `docker-build`'s smoke test step does. This must never raise
    `PoolError`, even though every individual health check is unhealthy.
    """

    def always_fails(conn: FakeConnection) -> None:
        raise psycopg2.ProgrammingError("vector type not found in the database")

    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", always_fails)
    store = _store(maxconn=5)

    results = [store.health_check() for _ in range(30)]

    assert all(result is False for result in results)
    assert len(store._pool._available) == 5  # noqa: SLF001


def test_repeated_search_calls_do_not_leak_connections(fake_pool, monkeypatch):
    """Many more search() calls than maxconn all return their connection."""
    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", lambda conn: None)
    store = _store(maxconn=2)

    for _ in range(10):
        assert store.search(query_embedding=[0.1, 0.2], top_k=5) == []

    assert len(store._pool._available) == 2  # noqa: SLF001


def test_search_returns_connection_when_query_execution_fails(fake_pool, monkeypatch):
    """A cursor.execute() failure inside search() still returns the connection."""
    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", lambda conn: None)
    store = _store(maxconn=2)
    # Replace one of the pooled connections with one whose cursor always fails.
    store._pool._available = [  # noqa: SLF001
        FakeConnection(cursor_fail=psycopg2.OperationalError("connection reset")) for _ in range(2)
    ]

    for _ in range(6):  # more attempts than maxconn
        with pytest.raises(psycopg2.OperationalError):
            store.search(query_embedding=[0.1, 0.2], top_k=5)

    assert len(store._pool._available) == 2  # noqa: SLF001


def test_get_or_create_document_id_returns_connection_on_failure(fake_pool, monkeypatch):
    """A write-path failure (get_or_create_document_id) still returns the connection."""
    monkeypatch.setattr("rag.vectorstore.pgvector.register_vector", lambda conn: None)
    store = _store(maxconn=2)
    store._pool._available = [  # noqa: SLF001
        FakeConnection(cursor_fail=psycopg2.OperationalError("connection reset")) for _ in range(2)
    ]

    for _ in range(6):
        with pytest.raises(psycopg2.OperationalError):
            store.get_or_create_document_id("some/source.md", "checksum", "dataset")

    assert len(store._pool._available) == 2  # noqa: SLF001
