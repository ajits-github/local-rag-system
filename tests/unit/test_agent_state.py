from __future__ import annotations

from rag.agent.state import AgentState, Citation, ToolCallRecord
from rag.retrieval.authorization import AuthorizationContext


def test_agent_state_defaults_are_empty_and_unstarted():
    """A freshly constructed AgentState has empty collections and zeroed/unset counters."""
    state = AgentState(original_query="hello")
    assert state.authorization_context is None
    assert state.filters is None
    assert state.query_type is None
    assert state.subquestions == []
    assert state.retrieved_evidence == []
    assert state.tool_call_history == []
    assert state.retrieval_attempts == 0
    assert state.tool_call_count == 0
    assert state.step_count == 0
    assert state.evidence_sufficient is None
    assert state.final_answer is None
    assert state.citations == []
    assert state.termination_reason is None


def test_agent_state_holds_authorization_context_by_reference():
    """AuthorizationContext carries no raw credential -- safe to embed directly in state.

    See docs/architecture.md's "Agentic RAG" section: VerifiedIdentity
    (which does carry the raw JWT `sub`) must never enter AgentState;
    AuthorizationContext is the verified-claims-only shape that is safe.
    """
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])
    state = AgentState(original_query="hello", authorization_context=auth)
    assert state.authorization_context is auth
    assert not hasattr(auth, "subject")  # never carries a raw JWT subject/token


def test_tool_call_record_defaults():
    """A ToolCallRecord constructed with only tool_name defaults to a successful, empty call."""
    record = ToolCallRecord(tool_name="search_knowledge_base")
    assert record.args == {}
    assert record.result_count == 0
    assert record.success is True
    assert record.error is None


def test_citation_mirrors_source_item_shape():
    """A Citation holds the same identifying fields as answer()'s sources list entries."""
    citation = Citation(
        chunk_id="c1", document_id="doc-1", source="a.md", category="security", score=0.9
    )
    assert citation.chunk_id == "c1"
    assert citation.score == 0.9
