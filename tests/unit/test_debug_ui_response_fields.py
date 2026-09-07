"""Covers the additive response fields the web UI's debug/observability polish added.

`SourceItem.origin` and `AgentQueryResponse.tool_call_details` are both
purely additive (see `21.0-UI-improvement`): they let the frontend
distinguish a RAG/local source from an MCP remote one, and show a safe
per-tool-call card (execution/status/duration), without changing any
retrieval/agent decision logic. This file checks the mapping is wired
correctly end to end, since no pre-existing test asserted on `origin` at
all and `tool_calls` was previously just a bare name list.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rag.agent.state import ToolCallRecord
from rag.api.deps import get_config, get_embedder, get_llm, get_retrieval_pipeline, get_vectorstore
from rag.api.main import app
from rag.api.routers.agent_query import _tool_call_details
from rag.config import load_config


def test_tool_call_details_marks_business_tools_as_mcp_remote_and_rag_tools_as_local():
    """`_tool_call_details` derives `execution` from the static REMOTE_MCP_TOOL_NAMES lookup."""
    records = [
        ToolCallRecord(
            tool_name="search_knowledge_base", result_count=3, latency_ms=42.0, success=True
        ),
        ToolCallRecord(tool_name="get_case_status", result_count=1, latency_ms=83.0, success=True),
        ToolCallRecord(
            tool_name="update_case_status", result_count=0, latency_ms=12.0, success=False
        ),
    ]

    details = _tool_call_details(records)

    assert [d.execution for d in details] == ["local", "mcp_remote", "mcp_remote"]
    assert [d.tool_name for d in details] == [
        "search_knowledge_base",
        "get_case_status",
        "update_case_status",
    ]
    assert [d.success for d in details] == [True, True, False]
    assert [d.latency_ms for d in details] == [42.0, 83.0, 12.0]
    assert [d.result_count for d in details] == [3, 1, 0]


def test_tool_call_details_is_empty_for_no_tool_calls():
    """An empty tool_call_history maps to an empty, never-null list."""
    assert _tool_call_details([]) == []


class _RecordingPipeline:
    """RetrievalPipeline double whose `answer()` returns a source with a non-default origin."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        self.calls.append({"query": query, "filters": filters, "auth": auth})
        return {
            "answer": "stub answer",
            "sources": [
                {
                    "chunk_id": "doc_0",
                    "document_id": "doc",
                    "source": "a.md",
                    "category": None,
                    "score": 0.9,
                    "content_type": "prose",
                    "section_path": None,
                    "page": None,
                    "attachment_name": None,
                    "source_anchor": None,
                    "vision_generated": False,
                    "origin": "expanded",
                }
            ],
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
    """Always classifies as 'simple', so the run always takes the classic_rag route."""

    def generate(self, system: str, user: str) -> str:
        return '{"query_type": "simple"}'

    def health_check(self) -> bool:
        return True


def test_query_endpoint_preserves_origin_per_source():
    """POST /query's sources carry `origin`, previously silently dropped at the API boundary."""
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: load_config()
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        response = client.post("/query", json={"query": "hello"})
        assert response.status_code == 200
        assert response.json()["sources"][0]["origin"] == "expanded"
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)


def test_agent_query_classic_route_preserves_origin_per_source():
    """POST /agent/query's classic_rag route also carries `origin` through Citation."""
    pipeline = _RecordingPipeline()
    app.dependency_overrides[get_config] = lambda: load_config()
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: _StubVectorStore()
    app.dependency_overrides[get_embedder] = lambda: _StubEmbedder()
    app.dependency_overrides[get_llm] = lambda: _StubLLM()
    try:
        client = TestClient(app)
        response = client.post("/agent/query", json={"query": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "classic_rag"
        assert body["sources"][0]["origin"] == "expanded"
        assert body["tool_call_details"] == []
    finally:
        for dep in (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm):
            app.dependency_overrides.pop(dep, None)
