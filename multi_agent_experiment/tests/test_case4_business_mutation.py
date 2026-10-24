"""CASE 4 (required scenario): business mutation -> Business Agent -> the existing MCP action path.

The Business Agent must not invent approval: `wait_for_approval` (reused
unmodified from `langgraph_experiment.nodes`) durably pauses via
`interrupt()`, and only a resume payload carrying a real approval role is
honored -- see that function's own docstring. This test is the
multi-agent graph's equivalent of `langgraph_experiment/tests/
test_graph_interrupt_flow.py`, proving the same guarantees hold when the
write branch is reached through this package's coordinator/`Send`-based
routing rather than the single-agent experiment's direct `classify` node.
"""

from __future__ import annotations

from langgraph.types import Command


def test_mutation_request_pauses_for_approval_and_never_dispatches_a_specialist(
    graph, thread_config
):
    """A mutation request pauses for approval without running either specialist branch."""
    result = graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["case_id"] == "CASE-2001"
    assert interrupt_payload["new_status"] == "closed"

    state = graph.get_state(thread_config)
    # Only the write branch ran -- the coordinator's mutation route bypasses
    # knowledge_agent/business_select_tool/merge/evidence_critic entirely.
    assert state.values.get("knowledge_evidence") in (None, [])
    assert state.values.get("case_result") is None  # no read-tool call was made
    assert state.values.get("tool_call_log") in (None, [])


def test_approved_mutation_executes_through_the_unmodified_business_store(graph, thread_config):
    """An approved mutation executes through the reused, unmodified business-store call."""
    graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    result = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
        thread_config,
    )
    assert "__interrupt__" not in result
    assert result["termination_reason"] == "executed"
    assert "closed" in result["final_answer"]
    assert result["citations"] == [
        {"chunk_id": None, "case_id": "CASE-2001", "source": "case:CASE-2001", "score": None}
    ]


def test_rejected_mutation_makes_no_change(graph, thread_config):
    """A rejected mutation leaves the case untouched."""
    graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    result = graph.invoke(Command(resume={"decision": "reject"}), thread_config)
    assert result["termination_reason"] == "rejected"
    assert "no change was made" in result["final_answer"]


def test_approval_claim_without_a_real_approver_role_is_denied(graph, thread_config):
    """The LLM/caller cannot invent approval by simply claiming decision='approve'."""
    graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    result = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["not_an_approval_role"]}),
        thread_config,
    )
    assert result["termination_reason"] == "approval_denied_insufficient_role"


def test_mutation_request_with_no_extractable_case_id_is_rejected_before_any_pause(
    graph, thread_config
):
    """A query matching the mutation-verb heuristic but naming no CASE-<id> must be rejected."""
    result = graph.invoke(
        {
            "original_query": "Please close this ticket",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert "__interrupt__" not in result
    assert result["is_case_mutation"] is True
    assert result["case_id"] is None
    assert result["termination_reason"] == "invalid_request"
    assert "no action was taken" in result["final_answer"]
