"""Unit tests for the graph-level MCP Stage 2 wiring in `rag.agent.graph`.

No real MCP server/transport here (see
`tests/integration/test_agent_mcp_client_stage2.py` for the real-wire-
protocol dispatch proof) -- these prove the *graph driver's* side of the
contract: the fail-closed branch when `mcp.client.enabled=False`, that
`max_tool_calls` bounds a mix of local and remote tool calls identically,
and that the local tool dispatch table/schemas are untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.agent.tool_schemas import (
    REMOTE_MCP_TOOL_NAMES,
    TOOL_ARG_MODELS,
    GetCaseStatusArgs,
    GetCustomerCaseArgs,
)
from rag.agent.tools import get_related_context
from rag.config import load_config
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk() -> Chunk:
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="doc-1_0",
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id="doc-1_0", content="some evidence", metadata=metadata)


class RemoteToolAlwaysLLM:
    """Always classifies complex, decomposes once, and always picks get_customer_case.

    evaluate_evidence always reports insufficient (bounded loop, exactly
    like test_agent_graph_step_bound.py's InfiniteInsufficientLLM), so
    the only way the run terminates is max_tool_calls.
    """

    def generate(self, system: str, user: str) -> str:
        """Return the scripted response for whichever decision component asked."""
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return '{"tool_name": "get_customer_case", "tool_args": {"case_id": "CASE-1001"}}'
        if "evidence-sufficiency component" in system:
            return '{"sufficient": false, "reformulated_query": "q1 again"}'
        return "final best-effort answer"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double; retrieve() is never expected to be called by these tests."""

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Fail loudly: these tests never expect a local search dispatch."""
        raise AssertionError("retrieve() should not be called by a remote-tool-only run")

    def resolve_auth(self, auth, filters=None, versions=None):
        """Return `auth` unchanged; authorization resolution is tested elsewhere."""
        return auth

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
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
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(
        update={
            "enabled": True,
            "max_retrieval_attempts": 1000,
            "max_tool_calls": 1000,
            **{k: v for k, v in overrides.items() if k in type(config.agent).model_fields},
        }
    )
    mcp = config.mcp.model_copy(
        update={
            "enabled": overrides.get("mcp_enabled", False),
            "client": config.mcp.client.model_copy(
                update={"enabled": overrides.get("mcp_client_enabled", False)}
            ),
        }
    )
    return config.model_copy(update={"agent": agent, "mcp": mcp})


# --- fail-closed when mcp.client.enabled=False -------------------------------


def test_remote_tool_decision_fails_closed_when_mcp_client_disabled():
    """A decision naming a remote tool is rejected as a recorded failure, never dispatched.

    mcp_app=None is passed deliberately: if the fail-closed guard were
    missing or misordered, _dispatch_mcp_tool would instead raise for
    lacking an mcp_app, producing a *different* error string -- this
    test's exact-match on "mcp_client_disabled" only passes if the guard
    fires before any dispatch attempt.
    """
    llm = RemoteToolAlwaysLLM()
    state = AgentState(
        original_query="what's the status of case CASE-1001",
        authorization_context=AuthorizationContext(tenant_id="tenant_alpha", roles=["op"]),
    )

    result = run_agent(
        state,
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_agent_steps=6, mcp_client_enabled=False),
        mcp_app=None,
    )

    remote_calls = [r for r in result.state.tool_call_history if r.tool_name == "get_customer_case"]
    assert remote_calls, "expected at least one recorded get_customer_case attempt"
    assert all(r.success is False for r in remote_calls)
    assert all(r.error == "mcp_client_disabled" for r in remote_calls)


def test_remote_tool_decision_still_offered_literal_is_valid_but_gated_at_dispatch():
    """The decision schema itself accepts the remote tool name regardless of config.

    tool_name selection is purely a name; config-gating happens at
    dispatch (see the previous test), not by narrowing the Literal type
    itself -- this documents that split explicitly.
    """
    assert "get_customer_case" in REMOTE_MCP_TOOL_NAMES
    assert "get_case_status" in REMOTE_MCP_TOOL_NAMES
    assert REMOTE_MCP_TOOL_NAMES.isdisjoint(
        {"search_knowledge_base", "get_document", "get_latest_document", "get_related_context"}
    )


# --- max_tool_calls bounds local and remote calls identically ----------------


def test_max_tool_calls_bounds_remote_tool_dispatch_attempts_too():
    """max_tool_calls counts a (fail-closed) remote-tool attempt exactly like a local one.

    mcp.client stays disabled here (see the previous test for why every
    attempt fails closed rather than dispatching) -- the point of this
    test is purely the counter, not a successful remote call.
    """
    llm = RemoteToolAlwaysLLM()
    state = AgentState(
        original_query="a question about case CASE-1001",
        authorization_context=AuthorizationContext(tenant_id="tenant_alpha", roles=["op"]),
    )

    result = run_agent(
        state,
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_agent_steps=1000, max_tool_calls=3, mcp_client_enabled=False),
        mcp_app=None,
    )

    assert result.state.termination_reason == "max_tool_calls"
    assert result.state.tool_call_count == 3
    assert len(result.state.tool_call_history) == 3


# --- local tool schemas/dispatch table are untouched --------------------------


def test_tool_arg_models_cover_all_six_tools_with_extra_forbid():
    """The 4 local plus 2 remote MCP tools are the only entries, all extra='forbid'."""
    assert set(TOOL_ARG_MODELS) == {
        "search_knowledge_base",
        "get_document",
        "get_latest_document",
        "get_related_context",
        "get_customer_case",
        "get_case_status",
    }
    for tool_name, model in TOOL_ARG_MODELS.items():
        assert model.model_config.get("extra") == "forbid", tool_name


def test_no_remote_tool_arg_model_accepts_an_auth_or_identity_field():
    """Mirrors test_agent_tools_authorization_parity.py's local-tool test, for the 2 new tools."""
    import pydantic

    for field in ("auth", "tenant_id", "roles"):
        assert field not in GetCustomerCaseArgs.model_fields
        assert field not in GetCaseStatusArgs.model_fields

    with pytest.raises(pydantic.ValidationError):
        GetCustomerCaseArgs(case_id="CASE-1001", tenant_id="tenant_evil")
    with pytest.raises(pydantic.ValidationError):
        GetCaseStatusArgs(case_id="CASE-1001", roles=["security_admin"])


# --- mcp_remote evidence never leaks into document-specific tool behavior ----


def test_get_related_context_given_a_synthetic_mcp_chunk_id_returns_empty_not_a_crash():
    """A model that (mis)copies a synthetic mcp:... chunk_id into get_related_context is safe.

    No real VectorStore row exists at that id, so the seed-chunk lookup
    simply misses -- the existing, unmodified "not found = empty"
    behavior get_related_context already has for any unknown chunk_id.
    Proves this local tool needs (and gets) no special-case guard against
    a synthetic MCP identifier: the natural miss is already safe.
    """

    class EmptyVectorStore:
        def get_chunks_by_ids(self, chunk_ids, auth=None):
            return []

        def list_document_versions(self, dataset_id):
            return []

    config = load_config()
    pipeline = RetrievalPipeline(config, vectorstore=EmptyVectorStore(), embedder=FakeEmbedder())
    from rag.agent.tool_schemas import GetRelatedContextArgs

    chunks = get_related_context(
        GetRelatedContextArgs(chunk_id="mcp:get_customer_case:CASE-1001"),
        pipeline,
        EmptyVectorStore(),
        AuthorizationContext(tenant_id="tenant_alpha", roles=["op"]),
        "some-dataset",
    )
    assert chunks == []


def test_mcp_remote_search_result_is_excluded_by_synthesis_trust_ordering_untouched():
    """_order_evidence_for_synthesis's trust sort is unaffected by an mcp_remote result.

    trust_level is unset (None) on synthetic evidence (see
    test_mcp_client_stage2.py's field-level proof); this test confirms
    the *graph*-level sort function itself treats that exactly like any
    other untagged/authoritative chunk -- no special mcp_remote branch
    exists in _order_evidence_for_synthesis, none is needed.
    """
    from rag.agent.graph import _order_evidence_for_synthesis
    from rag.agent.mcp_client import _business_result_to_search_result

    mcp_result = _business_result_to_search_result(
        "get_customer_case",
        "CASE-1001",
        {
            "case_id": "CASE-1001",
            "tenant_id": "tenant_alpha",
            "customer_name": "Acme",
            "subject": "s",
            "description": "d",
            "status": "open",
            "priority": "low",
            "assigned_team": "Team",
            "created_at": "2026-08-21T09:15:00Z",
            "updated_at": "2026-08-27T14:02:00Z",
        },
    )
    untrusted = SearchResult(chunk=_chunk(), score=0.5)
    untrusted.chunk.metadata.trust_level = "untrusted"

    ordered = _order_evidence_for_synthesis([untrusted, mcp_result])
    assert ordered[0] is mcp_result  # untagged/mcp_remote sorts ahead of untrusted
    assert ordered[1] is untrusted
