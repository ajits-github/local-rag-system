"""DB reliability hardening: config-driven pool sizing/timeouts, and clean 503s.

Covers three related gaps closed together:

1. `VectorStoreConfig`/`FeedbackConfig` now carry `minconn`/`maxconn` (and
   `VectorStoreConfig` also carries `connect_timeout_seconds`/
   `statement_timeout_ms`), threaded through `factory.build_vectorstore`/
   `api.deps.get_feedback_store` into the real `ThreadedConnectionPool`
   construction -- previously hardcoded class defaults with no config
   surface at all.
2. `psycopg2.pool.PoolError` (pool exhaustion) and
   `psycopg2.errors.IntegrityError` (a DB constraint violation) previously
   had no registered FastAPI exception handler, so either surfaced as an
   unhandled 500 with an internal exception message in the response body.
   Both now map to a clean 503 with no internal detail leaked, matching
   the existing `RateLimitExceeded` handler's own style.
"""

from __future__ import annotations

from typing import Any

import psycopg2.errors
import pytest
from fastapi.testclient import TestClient
from psycopg2.pool import PoolError

from rag.api.deps import get_config, get_feedback_store, get_retrieval_pipeline
from rag.api.main import app
from rag.config import load_config
from rag.factory import build_vectorstore
from rag.feedback.store import FeedbackStore
from rag.vectorstore.pgvector import PgVectorStore


class _CapturingPool:
    """Stand-in for `psycopg2.pool.ThreadedConnectionPool` that records its constructor args."""

    def __init__(self, minconn: int, maxconn: int, dsn: str, **kwargs: Any) -> None:
        self.minconn = minconn
        self.maxconn = maxconn
        self.dsn = dsn
        self.kwargs = kwargs

    def getconn(self):  # pragma: no cover - not exercised by these tests
        """Unused by these tests; present only so a real connection attempt fails loudly."""
        raise AssertionError("getconn() should not be called by a config-threading test")

    def putconn(self, conn):  # pragma: no cover - not exercised by these tests
        """Unused by these tests."""


def test_vectorstore_config_defaults_match_previous_hardcoded_values():
    """The new config fields default to (at least) the previous hardcoded pool size."""
    config = load_config()
    assert config.vectorstore.minconn == 1
    assert config.vectorstore.maxconn >= 5
    assert config.vectorstore.connect_timeout_seconds > 0
    assert config.vectorstore.statement_timeout_ms > 0


def test_feedback_config_defaults_match_previous_hardcoded_values():
    """FeedbackConfig's new pool-size fields default to the previous hardcoded values."""
    config = load_config()
    assert config.feedback.minconn == 1
    assert config.feedback.maxconn == 3


def test_build_vectorstore_threads_pool_size_and_timeouts_into_pgvectorstore(monkeypatch):
    """factory.build_vectorstore passes minconn/maxconn/timeouts through to the real pool."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    monkeypatch.setattr("rag.vectorstore.pgvector.ThreadedConnectionPool", _CapturingPool)
    config = load_config().model_copy(deep=True)
    config.vectorstore.minconn = 2
    config.vectorstore.maxconn = 17
    config.vectorstore.connect_timeout_seconds = 9
    config.vectorstore.statement_timeout_ms = 12_345

    store = build_vectorstore(config)

    assert isinstance(store, PgVectorStore)
    pool = store._pool  # noqa: SLF001
    assert pool.minconn == 2
    assert pool.maxconn == 17
    assert pool.kwargs["connect_timeout"] == 9
    assert pool.kwargs["options"] == "-c statement_timeout=12345"


def test_get_feedback_store_threads_pool_size_and_shared_timeouts(monkeypatch):
    """FeedbackStore threads its own pool size and the shared connect/statement timeouts through."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    monkeypatch.setattr("rag.feedback.store.ThreadedConnectionPool", _CapturingPool)

    store = FeedbackStore(
        "postgresql://rag:rag@localhost:15987/ragdb",
        minconn=1,
        maxconn=3,
        connect_timeout_seconds=5,
        statement_timeout_ms=30_000,
    )

    pool = store._pool  # noqa: SLF001
    assert pool.minconn == 1
    assert pool.maxconn == 3
    assert pool.kwargs["connect_timeout"] == 5
    assert pool.kwargs["options"] == "-c statement_timeout=30000"


class _PoolErrorPipeline:
    """RetrievalPipeline double whose `answer()` raises `PoolError`, simulating exhaustion."""

    def answer(self, query: str, filters=None, candidate_k=None, auth=None):
        """Raise PoolError, always, as the real pool does once every connection is checked out."""
        raise PoolError("connection pool exhausted")


class _IntegrityErrorPipeline:
    """RetrievalPipeline double whose `answer()` raises `IntegrityError`."""

    def answer(self, query: str, filters=None, candidate_k=None, auth=None):
        """Raise IntegrityError, always, simulating an uncaught DB constraint violation."""
        raise psycopg2.errors.IntegrityError("duplicate key value violates unique constraint")


@pytest.fixture
def client():
    """Build a TestClient against the real app, cleaning up dependency overrides after."""
    yield TestClient(app)
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_retrieval_pipeline, None)


def test_pool_error_returns_clean_503_not_an_unhandled_500(client):
    """A mid-request PoolError is caught and returns a clean 503, no internal detail leaked."""
    config = load_config()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: _PoolErrorPipeline()

    response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 503
    body = response.json()
    assert body == {"detail": "Service temporarily unavailable"}
    # Never leak the internal exception's class name or message.
    assert "PoolError" not in response.text
    assert "connection pool exhausted" not in response.text


def test_integrity_error_returns_clean_503_not_an_unhandled_500(client):
    """An uncaught IntegrityError is caught and returns a clean 503, no internal detail leaked."""
    config = load_config()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: _IntegrityErrorPipeline()

    response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 503
    body = response.json()
    assert body == {"detail": "Service temporarily unavailable"}
    assert "IntegrityError" not in response.text
    assert "unique constraint" not in response.text


def test_feedback_store_singleton_reads_pool_and_timeout_config_from_app_config(monkeypatch):
    """get_feedback_store() (the real DI singleton) threads config.feedback/vectorstore through."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    monkeypatch.setattr("rag.feedback.store.ThreadedConnectionPool", _CapturingPool)
    get_feedback_store.cache_clear()
    from rag.api import deps as deps_module

    original_get_config = deps_module.get_config
    config = load_config().model_copy(deep=True)
    config.feedback.minconn = 2
    config.feedback.maxconn = 9
    config.vectorstore.connect_timeout_seconds = 7
    config.vectorstore.statement_timeout_ms = 15_000
    deps_module.get_config.cache_clear()
    monkeypatch.setattr(deps_module, "get_config", lambda: config)
    try:
        store = deps_module.get_feedback_store()
    finally:
        get_feedback_store.cache_clear()
        monkeypatch.setattr(deps_module, "get_config", original_get_config)

    pool = store._pool  # noqa: SLF001
    assert pool.minconn == 2
    assert pool.maxconn == 9
    assert pool.kwargs["connect_timeout"] == 7
    assert pool.kwargs["options"] == "-c statement_timeout=15000"
