"""`POST /agent/query/stream`: SSE shape, event ordering, and the enabled/disabled toggle.

Same dependency-override doubles as
`test_api_agent_query_auth_boundary.py`; fast (mocked pipeline/LLM), no
real Postgres/Ollama. `tests/integration/test_agent_query_stream.py`
covers the real-stack case.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_embedder, get_llm, get_retrieval_pipeline, get_vectorstore
from rag.api.main import app
from rag.config import load_config


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
    disabled (it never calls the LLM at all), these tests only care about
    the resulting classic_rag route, not which config toggle produced it.
    """

    def generate(self, system: str, user: str) -> str:
        """Classify as simple, so the run always routes to classic_rag."""
        return '{"query_type": "simple"}'

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def _client_with(config):
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: _StubVectorStore()
    app.dependency_overrides[get_embedder] = lambda: _StubEmbedder()
    app.dependency_overrides[get_llm] = lambda: _StubLLM()
    return TestClient(app), pipeline


def _teardown():
    for dep in (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm):
        app.dependency_overrides.pop(dep, None)


def _no_auth_config():
    """`load_config()` with JWT auth explicitly disabled, isolating these tests from that toggle."""
    config = load_config()
    auth = config.security.auth.model_copy(update={"enabled": False})
    security = config.security.model_copy(update={"auth": auth})
    return config.model_copy(update={"security": security})


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse `event:`/`data:` pairs out of a raw SSE response body."""
    events = []
    event_type = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: ") and event_type is not None:
            events.append((event_type, json.loads(line[len("data: ") :])))
            event_type = None
    return events


def test_stream_endpoint_emits_query_received_route_selected_and_completed_for_classic_route():
    """The classic_rag route streams exactly 3 events, ending in a full response payload."""
    config = _no_auth_config()
    assert config.observability.live_events.enabled is True
    client, pipeline = _client_with(config)
    try:
        with client.stream("POST", "/agent/query/stream", json={"query": "hello"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())
    finally:
        _teardown()

    events = _parse_sse(body)
    event_types = [e for e, _ in events]
    assert event_types == ["query_received", "route_selected", "completed"]
    assert events[1][1]["route"] == "classic_rag"
    final_payload = events[2][1]
    assert final_payload["answer"] == "stub answer"
    assert final_payload["route"] == "classic_rag"
    assert len(pipeline.calls) == 1


def test_stream_endpoint_returns_404_when_live_events_disabled():
    """`POST /agent/query/stream` returns 404 when `observability.live_events.enabled` is off."""
    config = _no_auth_config()
    live_events = config.observability.live_events.model_copy(update={"enabled": False})
    observability = config.observability.model_copy(update={"live_events": live_events})
    patched = config.model_copy(update={"observability": observability})
    client, _pipeline = _client_with(patched)
    try:
        response = client.post("/agent/query/stream", json={"query": "hello"})
        assert response.status_code == 404
    finally:
        _teardown()
