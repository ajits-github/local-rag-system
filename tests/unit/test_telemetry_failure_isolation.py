"""A broken telemetry backend must never surface as a failed `/query` or `/agent/query` request.

Uses the same dependency-override doubles as
`test_api_agent_query_auth_boundary.py` for the HTTP-layer assertions,
plus direct calls into `rag.observability.metrics`/`tracing` for the
narrower unit-level proof.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_embedder, get_llm, get_retrieval_pipeline, get_vectorstore
from rag.api.main import app
from rag.config import load_config
from rag.observability import metrics, tracing


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        self.calls.append({"query": query})
        return {
            "answer": "stub answer",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }


class _StubVectorStore:
    def health_check(self) -> bool:
        return True


class _StubEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]


class _StubLLM:
    """Always classifies as 'simple', regardless of `config.agent.enabled`.

    Whether the agent is enabled (it calls the LLM once, for classify) or
    disabled (it never calls the LLM at all), these tests only care that
    the request survives broken telemetry, not which config toggle
    produced the classic_rag route.
    """

    def generate(self, system: str, user: str) -> str:
        """Classify as simple, so the run always routes to classic_rag."""
        return '{"query_type": "simple"}'

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def test_broken_metric_object_does_not_raise_through_observe_functions(monkeypatch):
    """A metric object whose `.labels()` raises never propagates out of `observe_http_request`."""

    def _broken_labels(*args, **kwargs):
        raise RuntimeError("simulated broken metric object")

    monkeypatch.setattr(metrics.HTTP_REQUESTS_TOTAL, "labels", _broken_labels)

    # Must not raise, despite the underlying Counter being broken.
    metrics.observe_http_request("GET", "/health", 200, 0.01)


def test_broken_tracer_does_not_raise_through_start_span(monkeypatch):
    """A tracer whose `start_as_current_span` raises never propagates out of `start_span`."""

    def _broken_start(*args, **kwargs):
        raise RuntimeError("simulated broken tracer")

    monkeypatch.setattr(tracing._tracer, "start_as_current_span", _broken_start)

    with tracing.start_span("classify"):
        pass  # must reach here without raising


def _no_auth_config():
    """`load_config()` with JWT auth explicitly disabled, isolating these tests from that toggle."""
    config = load_config()
    auth = config.security.auth.model_copy(update={"enabled": False})
    security = config.security.model_copy(update={"auth": auth})
    return config.model_copy(update={"security": security})


def _client_with(config, monkeypatch):
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: _StubVectorStore()
    app.dependency_overrides[get_embedder] = lambda: _StubEmbedder()
    app.dependency_overrides[get_llm] = lambda: _StubLLM()
    return TestClient(app), pipeline


def test_query_endpoint_survives_a_broken_metric_object(monkeypatch):
    """`POST /query` still returns 200 even when a Prometheus metric object is broken."""

    def _broken_labels(*args, **kwargs):
        raise RuntimeError("simulated broken metric object")

    monkeypatch.setattr(metrics.HTTP_REQUESTS_TOTAL, "labels", _broken_labels)
    client, pipeline = _client_with(_no_auth_config(), monkeypatch)
    try:
        response = client.post("/query", json={"query": "hello"})
        assert response.status_code == 200
        assert len(pipeline.calls) == 1
    finally:
        for dep in (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm):
            app.dependency_overrides.pop(dep, None)


def test_agent_query_endpoint_survives_a_broken_tracer(monkeypatch):
    """`POST /agent/query` still returns 200 even when the OTel tracer is broken."""

    def _broken_start(*args, **kwargs):
        raise RuntimeError("simulated broken tracer")

    monkeypatch.setattr(tracing._tracer, "start_as_current_span", _broken_start)
    client, pipeline = _client_with(_no_auth_config(), monkeypatch)
    try:
        response = client.post("/agent/query", json={"query": "hello"})
        assert response.status_code == 200
        assert len(pipeline.calls) == 1
    finally:
        for dep in (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm):
            app.dependency_overrides.pop(dep, None)
