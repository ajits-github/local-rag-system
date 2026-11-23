"""`POST /query` applies `retrieval.dataset_scope` before retrieval runs."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_retrieval_pipeline
from rag.api.main import app
from rag.config import load_config


class _RecordingPipeline:
    """RetrievalPipeline double recording each `answer()` call's filters."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer(self, query, filters=None, candidate_k=None, auth=None, history_turns=None):
        """Record the filters and return a minimal QueryResponse-shaped dict."""
        self.calls.append({"filters": filters})
        return {
            "answer": "stub",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }


@pytest.fixture
def client_with():
    """Yield a factory installing a config and a recording pipeline into the real app."""

    def _build(default=None, policy="all_datasets"):
        config = load_config().model_copy(deep=True)
        config.retrieval.dataset_scope.default_dataset_id = default
        config.retrieval.dataset_scope.missing_dataset_policy = policy
        pipeline = _RecordingPipeline()
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
        return TestClient(app), pipeline

    yield _build
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_retrieval_pipeline, None)


def test_reject_policy_returns_422_without_running_retrieval(client_with):
    """Multi-dataset deployment with `reject`: an unscoped query never reaches the pipeline."""
    client, pipeline = client_with(policy="reject")
    response = client.post("/query", json={"query": "hello"})
    assert response.status_code == 422
    assert "dataset_id" in response.json()["detail"]
    assert pipeline.calls == []
    ok = client.post("/query", json={"query": "hello", "filters": {"dataset_id": "ds1"}})
    assert ok.status_code == 200 and pipeline.calls == [{"filters": {"dataset_id": "ds1"}}]


def test_single_dataset_default_is_filled_in(client_with):
    """A configured default scopes an unscoped query; an explicit dataset is kept."""
    client, pipeline = client_with(default="techfusion", policy="reject")
    assert client.post("/query", json={"query": "hello"}).status_code == 200
    client.post("/query", json={"query": "hello", "filters": {"dataset_id": "other"}})
    assert pipeline.calls == [
        {"filters": {"dataset_id": "techfusion"}},
        {"filters": {"dataset_id": "other"}},
    ]


def test_all_datasets_policy_passes_the_query_through_unscoped(client_with):
    """The default policy keeps unscoped queries working (freshness then spans datasets)."""
    client, pipeline = client_with()
    assert client.post("/query", json={"query": "hello"}).status_code == 200
    assert pipeline.calls == [{"filters": None}]


def test_version_scope_reaches_the_authorization_context(client_with):
    """`version_scope` is a validated request field passed into the pipeline's auth context."""
    client, pipeline = client_with()
    seen = []
    original = pipeline.answer

    def answer(query, filters=None, candidate_k=None, auth=None, history_turns=None):
        seen.append(auth)
        return original(query, filters, candidate_k, auth, history_turns)

    pipeline.answer = answer
    ok = client.post("/query", json={"query": "q", "version_scope": "history"})
    assert ok.status_code == 200 and seen[0].version_scope == "history"
    assert client.post("/query", json={"query": "q", "version_scope": "all"}).status_code == 422
    client.post("/query", json={"query": "q"})
    assert seen[-1] is None
