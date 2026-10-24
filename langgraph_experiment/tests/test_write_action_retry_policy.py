"""`execute_write_action`'s own `RetryPolicy` -- and why it's only safe with the ledger guarding it.

A `RetryPolicy` on a node that has a real side effect is a double-edged
tool: it recovers from a transient failure automatically, but an
automatic retry of a *write* is exactly the shape of bug the idempotency
ledger exists to prevent (see `idempotency.py`, `graph.py`'s
`_WRITE_ACTION_RETRY_POLICY` comment). This file proves both halves: the
retry recovers (with or without the ledger), and -- the ledger-backed
case, needing real Postgres -- a retry that fires *after* the underlying
mutation already genuinely succeeded does not apply it twice.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import GraphDeps, TransientCaseStoreError
from rag.mcp.business import store as case_store

_WRITE_STATE = {
    "original_query": "Please close CASE-2001",
    "tenant_id": "tenant_beta",
    "roles": ["tenant_beta_operator"],
}
_APPROVE = {"decision": "approve", "approver_roles": ["case_status_approver"]}


def _flaky_update_case_status(fail_times: int, real_fn):
    calls = {"count": 0}

    def fn(case_id, new_status, identity, cross_tenant_support_roles, approved_transitions):
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise TransientCaseStoreError(f"simulated transient failure #{calls['count']}")
        return real_fn(
            case_id, new_status, identity, cross_tenant_support_roles, approved_transitions
        )

    fn.calls = calls
    return fn


def test_retry_policy_recovers_a_transient_write_failure_without_ledger(deps, thread_config):
    """No ledger involved (deps.action_ledger is None): RetryPolicy alone recovers.

    Safe here specifically because the failure happens *before* the real
    mutation runs (the fake raises before delegating) -- no attempt
    actually reaches the case store until the one that succeeds.
    """
    deps.update_case_status_fn = _flaky_update_case_status(
        fail_times=2, real_fn=case_store.update_case_status
    )
    graph = build_graph(deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume=_APPROVE), thread_config)

    assert result["termination_reason"] == "executed"
    assert deps.update_case_status_fn.calls["count"] == 3
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"


def test_retry_after_the_real_mutation_already_committed_does_not_double_transition(
    postgres_deps, thread_config
):
    """The harder case: the fake raises *after* delegating to the real store.

    Simulates a network blip discovered only after the remote call
    already committed -- exactly the scenario `idempotency.py`'s module
    docstring describes as the real crash window. The ledger's
    `mark_started` row (written before `mutate()` runs) is what lets the
    automatic retry recognize "already attempted" instead of blindly
    trusting a second full mutation call is harmless by coincidence.
    """
    real_fn = case_store.update_case_status
    calls = {"count": 0}

    def flaky_after_commit(
        case_id, new_status, identity, cross_tenant_support_roles, approved_transitions
    ):
        calls["count"] += 1
        result = real_fn(
            case_id, new_status, identity, cross_tenant_support_roles, approved_transitions
        )
        if calls["count"] == 1:
            raise TransientCaseStoreError("simulated failure discovered after the remote call")
        return result

    deps = GraphDeps(
        config=postgres_deps.config,
        action_ledger=postgres_deps.action_ledger,
        approval_roles=postgres_deps.approval_roles,
        cross_tenant_support_roles=postgres_deps.cross_tenant_support_roles,
        update_case_status_fn=flaky_after_commit,
    )
    graph = build_graph(deps, InMemorySaver())

    graph.invoke(_WRITE_STATE, thread_config)
    result = graph.invoke(Command(resume=_APPROVE), thread_config)

    assert calls["count"] == 2, "RetryPolicy retried once after the simulated post-commit failure"
    # Not "executed" a second time: attempt 1's real call already
    # transitioned the case before raising; attempt 2's real call (the
    # retry) correctly reports the business layer's own
    # already-in-status idempotency, not a fabricated second transition.
    assert result["termination_reason"] == "already_in_status"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"

    operation_id = result["pending_action"]["operation_id"]
    record = postgres_deps.action_ledger.get(operation_id)
    assert record is not None and record.completed is True
    assert record.outcome == "already_in_status"
