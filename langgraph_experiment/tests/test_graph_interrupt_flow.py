"""The write-action branch's interrupt/approve/reject flow -- the key learning scenario.

CASE-2001 (tenant_beta) seeds as `status="resolved"`, so "close CASE-2001"
names the one transition `rag.mcp.business.store._SENSITIVE_TRANSITIONS`
actually gates (`resolved -> closed`) -- a realistic scenario where the
business layer's own approval requirement and this graph's own
always-pause-on-write design agree. See `nodes.validate_write_request`'s
docstring for why the graph pauses on *every* write request regardless.
"""

from __future__ import annotations

from langgraph.types import Command

from rag.mcp.business import store as case_store

_WRITE_STATE = {
    "original_query": "Please close CASE-2001",
    "tenant_id": "tenant_beta",
    "roles": ["tenant_beta_operator"],
}


def test_write_request_pauses_at_approval_interrupt(graph, thread_config):
    result = graph.invoke(_WRITE_STATE, thread_config)

    assert "__interrupt__" in result
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("wait_for_approval",)
    assert len(snapshot.interrupts) == 1
    payload = snapshot.interrupts[0].value
    assert payload["case_id"] == "CASE-2001"
    assert payload["new_status"] == "closed"

    # No mutation happened yet.
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_approval_by_role_holder_resumes_and_executes(graph, thread_config):
    graph.invoke(_WRITE_STATE, thread_config)

    result = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
        thread_config,
    )

    assert result["termination_reason"] == "executed"
    assert "closed" in result["final_answer"]
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"
    assert graph.get_state(thread_config).next == ()


def test_rejection_causes_no_mutation(graph, thread_config):
    graph.invoke(_WRITE_STATE, thread_config)

    result = graph.invoke(Command(resume={"decision": "reject"}), thread_config)

    assert result["termination_reason"] == "rejected"
    assert "rejected" in result["final_answer"]
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_approval_claim_without_approver_role_is_denied(graph, thread_config):
    """Human approval must be trusted runtime state, never accepted on the claim alone."""
    graph.invoke(_WRITE_STATE, thread_config)

    result = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["not_an_approver_role"]}),
        thread_config,
    )

    assert result["termination_reason"] == "approval_denied_insufficient_role"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_malformed_case_write_request_never_reaches_interrupt(graph, thread_config):
    """A mutation verb with no extractable case id -> a graceful refusal, no pause at all."""
    result = graph.invoke(
        {"original_query": "please close this case", "tenant_id": "tenant_beta", "roles": []},
        thread_config,
    )
    assert "__interrupt__" not in result
    assert result["termination_reason"] == "invalid_request"
    assert result["case_id"] is None
