"""Approval expiration, per-workflow timeout, and cancellation -- all enforced server-side.

None of these three checks trust a CLI-side pre-check (the same posture
already applied to `approver_roles` in `test_authorization_server_side.py`):
`nodes.wait_for_approval` re-checks all three itself, against state fields
that were durably checkpointed *before* the interrupt paused, so a resume
sent after either window has passed is refused regardless of what the
resume payload claims.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import GraphDeps
from rag.mcp.business import store as case_store

_WRITE_STATE = {
    "original_query": "Please close CASE-2001",
    "tenant_id": "tenant_beta",
    "roles": ["tenant_beta_operator"],
}
_APPROVE = {"decision": "approve", "approver_roles": ["case_status_approver"]}


def _deps_with(base: GraphDeps, **overrides) -> GraphDeps:
    return GraphDeps(
        config=base.config,
        pipeline=base.pipeline,
        llm=base.llm,
        rag_prompt_template=base.rag_prompt_template,
        approval_roles=base.approval_roles,
        cross_tenant_support_roles=base.cross_tenant_support_roles,
        get_customer_case_fn=base.get_customer_case_fn,
        get_case_status_fn=base.get_case_status_fn,
        update_case_status_fn=base.update_case_status_fn,
        action_ledger=base.action_ledger,
        approval_timeout_seconds=overrides.get(
            "approval_timeout_seconds", base.approval_timeout_seconds
        ),
        max_workflow_duration_seconds=overrides.get(
            "max_workflow_duration_seconds", base.max_workflow_duration_seconds
        ),
    )


def test_approval_expires_after_its_own_timeout_window(deps, thread_config):
    # -1, not 0: see the identical comment on the workflow-timeout test
    # below -- a 0-second window ties (rather than expires) if `now` and
    # `expires_at` land in the same clock tick; -1 can't tie.
    short_deps = _deps_with(deps, approval_timeout_seconds=-1)
    graph = build_graph(short_deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume=_APPROVE), thread_config)

    assert result["termination_reason"] == "approval_expired"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_workflow_timeout_takes_precedence_over_a_well_formed_approval(deps, thread_config):
    # -1, not 0: two back-to-back in-memory invoke() calls can complete
    # within the same clock tick on some platforms, so an elapsed-time-
    # over-0 threshold is racy (elapsed == 0.0 is a real, observed
    # outcome, not hypothetical). -1 makes the check unconditionally true
    # regardless of clock granularity, without weakening what production
    # code actually compares (a plain `>`, unchanged).
    short_deps = _deps_with(deps, max_workflow_duration_seconds=-1)
    graph = build_graph(short_deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume=_APPROVE), thread_config)

    assert result["termination_reason"] == "workflow_timed_out"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_a_well_formed_approval_within_both_windows_still_executes(deps, thread_config):
    """Sanity check: expiration/timeout checks don't fire on an ordinary, prompt resume."""
    graph = build_graph(deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume=_APPROVE), thread_config)

    assert result["termination_reason"] == "executed"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"


def test_cancel_decision_terminates_without_mutation(graph, thread_config):
    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(
        Command(resume={"decision": "cancel", "reason": "superseded"}), thread_config
    )

    assert result["termination_reason"] == "cancelled"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_cancel_after_expiration_still_reports_expiration_not_cancellation(deps, thread_config):
    """Expiration is checked first, so a stale cancel reports why it's stale, not as a cancel."""
    short_deps = _deps_with(deps, approval_timeout_seconds=-1)
    graph = build_graph(short_deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume={"decision": "cancel"}), thread_config)

    assert result["termination_reason"] == "approval_expired"
