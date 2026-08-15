from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_retrieval_pipeline
from rag.api.main import app
from rag.config import load_config


class _RecordingPipeline:
    """RetrievalPipeline double -- should never be reached by an oversized/invalid request."""

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
