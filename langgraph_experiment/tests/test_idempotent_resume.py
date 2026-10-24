"""Duplicate resume must not cause an unintended duplicate mutation.

Two layers cooperate here (see README.md's "Idempotency and safety"
section for the full reasoning): the checkpointer-side guard
(`approval_cli._resume`'s "no pending interrupt" check, reproduced
directly against the graph here) and the business-layer guard
(`update_case_status`'s own `already_in_status` outcome, which fires
regardless of whether the graph-side guard is ever bypassed).
"""

from __future__ import annotations

from langgraph.types import Command

from rag.mcp.business import store as case_store

_WRITE_STATE = {
    "original_query": "Please close CASE-2001",
    "tenant_id": "tenant_beta",
    "roles": ["tenant_beta_operator"],
}
_APPROVE = {"decision": "approve", "approver_roles": ["case_status_approver"]}


def test_second_resume_after_completion_is_a_no_op(graph, thread_config):
    graph.invoke(_WRITE_STATE, thread_config)
    first = graph.invoke(Command(resume=_APPROVE), thread_config)
    assert first["termination_reason"] == "executed"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"

    # The graph-level guard an approval CLI is expected to apply (see
    # approval_cli._resume): once graph.get_state(config).next is empty,
    # do not invoke(Command(resume=...)) again. Reproduced here directly
    # against the graph, not the CLI, to test the mechanism itself.
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ()
    assert not snapshot.interrupts


def test_business_layer_is_independently_idempotent_even_if_graph_guard_is_bypassed(
    graph, thread_config
):
    """Belt-and-suspenders: `update_case_status` itself never double-mutates.

    Simulates a caller bypassing the graph-side "already completed" guard
    (e.g. a buggy or malicious approval client that invokes resume twice
    anyway) by calling `execute_write_action`'s underlying function
    directly a second time for the same case/status pair, proving the
    *business* layer -- not just the workflow layer -- is the real
    idempotency guarantee. See README.md's "Idempotency and safety"
    section for why neither guarantee alone would be enough.
    """
    graph.invoke(_WRITE_STATE, thread_config)
    graph.invoke(Command(resume=_APPROVE), thread_config)
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"

    from rag.api.auth import VerifiedIdentity
    from rag.mcp.business.schemas import CaseApproval

    identity = VerifiedIdentity(
        subject="demo-caller", tenant_id="tenant_beta", roles=["tenant_beta_operator"]
    )
    approval = CaseApproval(case_id="CASE-2001", new_status="closed")
    outcome = case_store.update_case_status(
        "CASE-2001", "closed", identity, ["techfusion_support"], [approval]
    )
    assert outcome is not None
    assert outcome.outcome == "already_in_status"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"
