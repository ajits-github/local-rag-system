"""Structured `agent_request_completed`/`tool_call_completed` log content.

Calls `run_agent` directly (same `ScriptedLLM`/`FakePipeline`/
`FakeVectorStore`/`FakeEmbedder` double shapes as
`test_agent_graph_tool_failure.py`/`test_agent_node_timing.py`) plus
`caplog` on the `rag.agent.graph` logger, and separately exercises the
classic `/query` route's own completion log via `TestClient`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.api.deps import get_config, get_retrieval_pipeline
from rag.api.main import app
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str = "doc-1_0") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content="content", metadata=metadata)


class ScriptedLLM:
    """Returns one queued response per call, in order."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses this double's generate() will pop from."""
        self._responses = list(responses)

    def generate(self, system: str, user: str) -> str:
        """Return the next queued response."""
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double; `retrieve` can be scripted to raise."""

    def __init__(self, retrieve_results=None, retrieve_error: Exception | None = None) -> None:
        """Store the fixed results (or error) this double's retrieve() will produce."""
        self.retrieve_results = retrieve_results or []
        self._retrieve_error = retrieve_error

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Return the fixed results, or raise the scripted error."""
        if self._retrieve_error is not None:
            raise self._retrieve_error
        return self.retrieve_results

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization parity is tested elsewhere."""
        return auth

    def sanitize_evidence(self, results, auth):
        """Return results unchanged."""
        return results


class FakeVectorStore:
    """Minimal VectorStore double that only answers health checks."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text):
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


def _agent_config(**overrides):
    """Return `load_config()` with the agent enabled and `overrides` applied."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True, **overrides})
    return config.model_copy(update={"agent": agent})


_SECRET_QUERY = "SECRET_QUERY_MARKER_XYZ what is the policy"
_SECRET_ANSWER = "SECRET_ANSWER_MARKER_ABC final synthesized answer"


def test_agent_request_completed_logs_run_shape_not_content(caplog):
    """One `agent_request_completed` record carries counters/timing, never query/answer text."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            '{"sufficient": true}',
            _SECRET_ANSWER,
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    state = AgentState(original_query=_SECRET_QUERY)

    with caplog.at_level(logging.INFO, logger="rag.agent.graph"):
        result = run_agent(
            state,
            pipeline=pipeline,
            vectorstore=FakeVectorStore(),
            embedder=FakeEmbedder(),
            llm=llm,
            config=_agent_config(),
        )

    assert result.state.final_answer == _SECRET_ANSWER

    records = [r for r in caplog.records if r.getMessage() == "agent_request_completed"]
    assert len(records) == 1
    record = records[0]
    assert record.route == "agent"
    assert record.termination_reason == "synthesized"
    assert record.step_count == result.state.step_count
    assert record.tool_call_count == 1
    assert record.retrieval_attempts == 1
    assert record.evidence_sufficient is True
    assert record.duration_ms >= 0.0
    assert record.success is True

    record_text = str(vars(record))
    assert _SECRET_QUERY not in record_text
    assert _SECRET_ANSWER not in record_text
    assert "SECRET_QUERY_MARKER_XYZ" not in record_text
    assert "SECRET_ANSWER_MARKER_ABC" not in record_text


def test_tool_call_completed_logs_a_successful_local_tool_call(caplog):
    """A successful `search_knowledge_base` dispatch logs one safe completion line."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            '{"sufficient": true}',
            "answer",
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    state = AgentState(original_query="a question")

    with caplog.at_level(logging.INFO, logger="rag.agent.graph"):
        run_agent(
            state,
            pipeline=pipeline,
            vectorstore=FakeVectorStore(),
            embedder=FakeEmbedder(),
            llm=llm,
            config=_agent_config(),
        )

    records = [r for r in caplog.records if r.getMessage() == "tool_call_completed"]
    assert len(records) == 1
    record = records[0]
    assert record.tool_name == "search_knowledge_base"
    assert record.success is True
    assert record.result_count == 1
    assert record.origin == "local"
    assert record.error_type is None
    assert record.duration_ms >= 0.0


def test_tool_call_completed_logs_invalid_arguments_as_a_bounded_category(caplog):
    """A rejected (smuggled-field) tool call logs `error_type="invalid_arguments"`, nothing raw."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", '
            '"tool_args": {"query": "q1", "roles": ["security_admin"]}}',
            '{"sufficient": true}',
            "answer",
        ]
    )
    pipeline = FakePipeline()
    state = AgentState(original_query="a question")

    with caplog.at_level(logging.INFO, logger="rag.agent.graph"):
        run_agent(
            state,
            pipeline=pipeline,
            vectorstore=FakeVectorStore(),
            embedder=FakeEmbedder(),
            llm=llm,
            config=_agent_config(),
        )

    records = [r for r in caplog.records if r.getMessage() == "tool_call_completed"]
    assert len(records) == 1
    record = records[0]
    assert record.success is False
    assert record.error_type == "invalid_arguments"
    assert "security_admin" not in str(vars(record))


def test_tool_call_completed_logs_exception_type_not_raw_message(caplog):
    """A tool-execution exception logs the exception's class name, never its raw message."""
    sensitive_message = "connection failed: password=hunter2 host=db.internal"
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            '{"sufficient": false}',
        ]
    )
    pipeline = FakePipeline(retrieve_error=RuntimeError(sensitive_message))
    state = AgentState(original_query="a question")

    with caplog.at_level(logging.INFO, logger="rag.agent.graph"):
        run_agent(
            state,
            pipeline=pipeline,
            vectorstore=FakeVectorStore(),
            embedder=FakeEmbedder(),
            llm=llm,
            config=_agent_config(max_tool_calls=1),
        )

    records = [r for r in caplog.records if r.getMessage() == "tool_call_completed"]
    assert len(records) == 1
    record = records[0]
    assert record.success is False
    assert record.error_type == "RuntimeError"
    record_text = str(vars(record))
    assert sensitive_message not in record_text
    assert "hunter2" not in record_text


class _RecordingPipeline:
    """`/query` router double returning a fixed answer, recording every call."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        """Record the call and return a fixed classic-RAG-shaped result."""
        self.calls.append({"query": query})
        return {
            "answer": _SECRET_ANSWER,
            "sources": [],
            "retrieval_ms": 1.0,
            "generation_ms": 2.0,
            "total_ms": 3.0,
        }


def _no_auth_config():
    """`load_config()` with JWT auth explicitly disabled, isolating this test from that toggle."""
    config = load_config()
    auth = config.security.auth.model_copy(update={"enabled": False})
    security = config.security.model_copy(update={"auth": auth})
    return config.model_copy(update={"security": security})


def test_classic_query_endpoint_logs_completion_without_content(caplog):
    """`POST /query` logs one `agent_request_completed` line, never the query/answer text."""
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = _no_auth_config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    try:
        with caplog.at_level(logging.INFO, logger="rag.api.routers.query"):
            response = TestClient(app).post("/query", json={"query": _SECRET_QUERY})
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)

    assert response.status_code == 200
    records = [r for r in caplog.records if r.getMessage() == "agent_request_completed"]
    assert len(records) == 1
    record = records[0]
    assert record.route == "classic_rag"
    assert record.success is True
    assert record.duration_ms >= 0.0

    record_text = str(vars(record))
    assert _SECRET_QUERY not in record_text
    assert _SECRET_ANSWER not in record_text
