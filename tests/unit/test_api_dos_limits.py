from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.deps import (
    get_config,
    get_embedder,
    get_llm,
    get_retrieval_pipeline,
    get_vectorstore,
)
from rag.api.main import app
from rag.config import load_config


class _RecordingPipeline:
    """RetrievalPipeline double. should never be reached by an oversized/invalid request."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        """Record the call and return a minimal QueryResponse-shaped dict."""
        self.calls.append({"query": query})
        return {
            "answer": "stub",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }


@pytest.fixture
def client_and_pipeline():
    """Build a TestClient with the default config's DoS limits and a recording pipeline."""
    config = load_config()
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    yield TestClient(app), pipeline, config
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_retrieval_pipeline, None)


def test_oversized_query_string_returns_422(client_and_pipeline):
    """A query longer than dos_limits.max_query_length is rejected with 422."""
    client, pipeline, config = client_and_pipeline
    oversized_query = "x" * (config.security.dos_limits.max_query_length + 1)

    response = client.post("/query", json={"query": oversized_query})

    assert response.status_code == 422
    assert pipeline.calls == []


def test_top_k_over_max_returns_422(client_and_pipeline):
    """A top_k greater than dos_limits.max_top_k is rejected with 422."""
    client, pipeline, config = client_and_pipeline

    response = client.post(
        "/query", json={"query": "hi", "top_k": config.security.dos_limits.max_top_k + 1}
    )

    assert response.status_code == 422
    assert pipeline.calls == []


def test_top_k_zero_returns_422(client_and_pipeline):
    """top_k=0 is rejected with 422, not silently treated as 'use the default'."""
    client, pipeline, _config = client_and_pipeline

    response = client.post("/query", json={"query": "hi", "top_k": 0})

    assert response.status_code == 422
    assert pipeline.calls == []


def test_top_k_negative_returns_422(client_and_pipeline):
    """A negative top_k is rejected with 422 before it can reach retrieval/DB work."""
    client, pipeline, _config = client_and_pipeline

    response = client.post("/query", json={"query": "hi", "top_k": -1})

    assert response.status_code == 422
    assert pipeline.calls == []


def test_oversized_filters_dict_returns_422(client_and_pipeline):
    """A filters dict serializing larger than max_filters_bytes is rejected with 422."""
    client, pipeline, config = client_and_pipeline
    huge_value = "v" * config.security.dos_limits.max_filters_bytes

    response = client.post("/query", json={"query": "hi", "filters": {"k": huge_value}})

    assert response.status_code == 422
    assert pipeline.calls == []


def test_query_within_limits_is_accepted(client_and_pipeline):
    """A normal, within-bounds request is not rejected by DoS limits."""
    client, pipeline, _config = client_and_pipeline

    response = client.post("/query", json={"query": "a perfectly normal question", "top_k": 5})

    assert response.status_code == 200
    assert len(pipeline.calls) == 1


def test_oversized_upload_returns_413(tmp_path, monkeypatch):
    """An upload larger than dos_limits.max_upload_bytes is rejected with 413."""
    from rag.api.deps import get_ingestion_pipeline

    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)

    class _NeverCalledIngestionPipeline:
        def ingest_file(self, path, dataset_id):
            raise AssertionError("ingest_file should never be reached for an oversized upload")

    config = load_config()
    small_limit_config = config.model_copy(
        update={
            "security": config.security.model_copy(
                update={
                    "dos_limits": config.security.dos_limits.model_copy(
                        update={"max_upload_bytes": 10}
                    )
                }
            )
        }
    )
    app.dependency_overrides[get_config] = lambda: small_limit_config
    app.dependency_overrides[get_ingestion_pipeline] = lambda: _NeverCalledIngestionPipeline()
    try:
        client = TestClient(app)
        response = client.post(
            "/ingest",
            data={"dataset_id": "test"},
            files={"files": ("big.txt", b"x" * 1000, "text/plain")},
        )
        assert response.status_code == 413
        assert list(tmp_path.glob("*")) == []  # partial upload cleaned up, not left on disk
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_ingestion_pipeline, None)


def test_oversized_dataset_id_returns_422(tmp_path, monkeypatch):
    """A dataset_id form field longer than 200 characters is rejected with 422.

    `dataset_id` previously had no `max_length` at all, unlike
    `feedback.py`'s identifier fields (capped at 200 chars).
    """
    from rag.api.deps import get_ingestion_pipeline

    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)

    class _NeverCalledIngestionPipeline:
        def ingest_file(self, path, dataset_id, caller=None, source_override=None):
            raise AssertionError("ingest_file should never be reached for an oversized dataset_id")

    app.dependency_overrides[get_ingestion_pipeline] = lambda: _NeverCalledIngestionPipeline()
    try:
        client = TestClient(app)
        response = client.post(
            "/ingest",
            data={"dataset_id": "x" * 201},
            files={"files": ("a.md", b"# hello", "text/markdown")},
        )
        assert response.status_code == 422
        assert list(tmp_path.glob("*")) == []
    finally:
        app.dependency_overrides.pop(get_ingestion_pipeline, None)


def test_unknown_filter_key_returns_422_not_500(client_and_pipeline):
    """An unrecognized filters key is rejected with a clear 422, not an unhandled 500.

    Before this fix, `PgVectorStore._build_where_clause` raised a bare
    `ValueError` for a filter key outside `ALLOWED_FILTER_FIELDS`, which
    nothing caught before it reached FastAPI's default unhandled-exception
    path (an internal-error 500). This is now validated up front, at the
    request boundary, before retrieval ever runs.
    """
    client, pipeline, _config = client_and_pipeline

    response = client.post("/query", json={"query": "hi", "filters": {"bogus_field": "x"}})

    assert response.status_code == 422
    assert "bogus_field" in response.text
    assert pipeline.calls == []


class _FakeVectorStore:
    """Minimal VectorStore double; never reached by a request rejected at the boundary."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class _FakeEmbedder:
    """Minimal Embedder double; never reached by a request rejected at the boundary."""

    def embed_query(self, text):
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]


class _FakeLLM:
    """Minimal LLM double; never reached by a request rejected at the boundary."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def generate(self, system: str, user: str) -> str:
        """Never actually called in this test; present only to satisfy the LLM interface."""
        raise AssertionError("generate() should never be reached for a rejected request")


def test_agent_query_unknown_filter_key_returns_422_not_500():
    """POST /agent/query rejects an unrecognized filters key with 422, matching /query."""
    config = load_config()
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: _FakeVectorStore()
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    app.dependency_overrides[get_llm] = lambda: _FakeLLM()
    try:
        client = TestClient(app)

        response = client.post(
            "/agent/query", json={"query": "hi", "filters": {"bogus_field": "x"}}
        )

        assert response.status_code == 422
        assert "bogus_field" in response.text
        assert pipeline.calls == []
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        app.dependency_overrides.pop(get_vectorstore, None)
        app.dependency_overrides.pop(get_embedder, None)
        app.dependency_overrides.pop(get_llm, None)
