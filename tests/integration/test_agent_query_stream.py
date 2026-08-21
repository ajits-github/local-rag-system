"""Full-stack proof for `POST /agent/query/stream`: real Postgres + real Ollama.

Builds a small, isolated FastAPI app mounting only `agent_stream.router`,
with DI singletons built directly via `rag.factory` against a config copy
with the agent enabled. the same workaround
`tests/unit/test_rate_limiting.py` documents for any config-derived
singleton bound at import time: `rag.api.main.app`'s own singletons are
already bound to the process-default (agent-disabled) config, so a fresh
app is the only way to exercise the agent-enabled path through the real
HTTP/SSE layer without mutating global state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter

from rag.api.deps import (
    get_config,
    get_current_identity,
    get_embedder,
    get_llm,
    get_rate_limiter,
    get_retrieval_pipeline,
    get_vectorstore,
)
from rag.api.routers import agent_stream
from rag.factory import build_embedder, build_llm, build_vectorstore
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline


def _agentic_config(config):
    """Agent enabled, matching `test_agent_end_to_end.py`'s model choice for JSON reliability."""
    agentic = config.model_copy(deep=True)
    agentic.agent.enabled = True
    agentic.generation.model_name = "qwen2.5:3b"
    return agentic


def _write_doc(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def streaming_client(require_postgres, require_ollama, config, tmp_path: Path):
    """Build a small isolated app + TestClient wired to real singletons, agent enabled."""
    agentic = _agentic_config(config)
    dataset_id = f"agent-stream-it-{tmp_path.name}"

    vectorstore = build_vectorstore(agentic)
    embedder = build_embedder(agentic)
    llm = build_llm(agentic)
    pipeline = RetrievalPipeline(agentic, vectorstore=vectorstore, embedder=embedder, llm=llm)

    doc_path = _write_doc(
        tmp_path,
        "rollback.md",
        "# Deployment Rollback\n\nTo roll back a failed deployment, run "
        "`deploy rollback --service checkout` and monitor the health dashboard.\n",
    )
    ingestion = IngestionPipeline(agentic, vectorstore=vectorstore, embedder=embedder)
    ingestion.ingest_path(doc_path.parent, dataset_id=dataset_id)

    app = FastAPI()
    app.state.limiter = Limiter(key_func=lambda request: "test", enabled=False)
    app.include_router(agent_stream.router)
    app.dependency_overrides[get_config] = lambda: agentic
    app.dependency_overrides[get_current_identity] = lambda: None
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: vectorstore
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_rate_limiter] = lambda: app.state.limiter

    yield TestClient(app), dataset_id


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    event_type = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: ") and event_type is not None:
            events.append((event_type, json.loads(line[len("data: ") :])))
            event_type = None
    return events


def test_streamed_events_end_in_completed_or_terminated_with_a_grounded_answer(streaming_client):
    """A real multi-hop-capable query streams only documented safe events and a grounded answer."""
    client, dataset_id = streaming_client

    with client.stream(
        "POST",
        "/agent/query/stream",
        json={
            "query": "How do I roll back a failed deployment for the checkout service?",
            "filters": {"dataset_id": dataset_id},
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    assert events, "expected at least one SSE event"
    final_event_type, final_payload = events[-1]
    assert final_event_type in ("completed", "terminated")
    assert final_payload["answer"]
    # Every intermediate event is one of the documented safe operational types.
    safe_types = {
        "query_received",
        "route_selected",
        "decomposition_started",
        "decomposition_completed",
        "tool_selected",
        "tool_started",
        "tool_completed",
        "evidence_evaluated",
        "retry_started",
        "synthesis_started",
        "completed",
        "terminated",
    }
    assert {event_type for event_type, _ in events} <= safe_types


def test_client_disconnecting_mid_stream_does_not_hang_or_raise(streaming_client):
    """Closing the connection after the first chunk never hangs or raises in the test process."""
    client, dataset_id = streaming_client

    with client.stream(
        "POST",
        "/agent/query/stream",
        json={"query": "How do I roll back a deployment?", "filters": {"dataset_id": dataset_id}},
    ) as response:
        assert response.status_code == 200
        # Read only the first chunk, then let the `with` block close the
        # connection early. must not hang the test process or raise.
        next(response.iter_text(), None)
