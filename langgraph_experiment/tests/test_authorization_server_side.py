"""Authorization is always re-checked server-side; nothing in the resume payload or the graph
state itself is ever trusted as proof of access.
"""

from __future__ import annotations

from langgraph.types import Command

from rag.mcp.business import store as case_store


def test_resume_with_no_approver_roles_at_all_is_denied(graph, thread_config):
    graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    result = graph.invoke(Command(resume={"decision": "approve"}), thread_config)
    assert result["termination_reason"] == "approval_denied_insufficient_role"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_unauthorized_write_request_still_pauses_but_executes_as_denied(graph, thread_config):
    """CASE-1002 is admin-only within tenant_alpha; an operator role cannot mutate it.

    The graph still pauses for human approval (`validate_write_request`
    deliberately never checks case authorization -- see its docstring),
    but `execute_write_action`'s outcome is `None` once
    `update_case_status` itself runs its own authorization check --
    identical to how an unauthorized *read* is indistinguishable from a
    nonexistent case from the caller's own point of view.
    """
    graph.invoke(
        {
            "original_query": "Please close CASE-1002",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert graph.get_state(thread_config).next == ("wait_for_approval",)

    result = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
        thread_config,
    )
    assert result["termination_reason"] == "case_not_found_or_denied"
    assert case_store._SYNTHETIC_CASES["CASE-1002"].status == "open"


def test_cross_tenant_support_role_can_read_a_permitted_case(graph, thread_config):
    """CASE-2002 (tenant_beta) explicitly lists techfusion_support in allowed_roles."""
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-2002?",
            "tenant_id": "tenant_alpha",
            "roles": ["techfusion_support"],
        },
        thread_config,
    )
    assert result["case_found"] is True
    assert result["termination_reason"] == "synthesized"


def test_cross_tenant_without_support_role_cannot_read(graph, thread_config):
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-2002?",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["case_found"] is False
