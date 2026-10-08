"""A failed node produces an understandable, resumable workflow state -- not a crashed process.

Two scenarios: a *transient* failure the node's `RetryPolicy` absorbs
automatically (graph.py's `_CASE_READ_RETRY_POLICY`), and a *persistent*
failure that exhausts every retry, propagates out of `.invoke()`, and
still leaves a checkpoint an operator can inspect
(`get_state(config).tasks[i].error`) and resume from once the underlying
problem is fixed -- without losing any state gathered before the failure.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import TransientCaseStoreError


def _flaky_get_customer_case(fail_times: int):
    """Return a fake `get_customer_case_fn` that raises `fail_times` times, then succeeds."""
    calls = {"count": 0}

    def fn(case_id, identity, cross_tenant_support_roles):
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise TransientCaseStoreError(f"simulated transient failure #{calls['count']}")
        from rag.mcp.business import store as case_store

        return case_store.get_customer_case(case_id, identity, cross_tenant_support_roles)

    fn.calls = calls
    return fn


def test_retry_policy_absorbs_a_transient_failure_within_the_attempt_budget(deps, thread_config):
    # RetryPolicy allows 3 attempts.
    deps.get_customer_case_fn = _flaky_get_customer_case(fail_times=2)
    graph = build_graph(deps, InMemorySaver())

    result = graph.invoke(
        {
            "original_query": "Give me details on CASE-1001",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )

    assert result["termination_reason"] == "synthesized"
    assert deps.get_customer_case_fn.calls["count"] == 3


def test_persistent_failure_propagates_with_an_inspectable_checkpoint(deps, thread_config):
    deps.get_customer_case_fn = _flaky_get_customer_case(fail_times=999)  # never recovers
    checkpointer = InMemorySaver()
    graph = build_graph(deps, checkpointer)

    with pytest.raises(TransientCaseStoreError):
        graph.invoke(
            {
                "original_query": "Give me details on CASE-1001",
                "tenant_id": "tenant_alpha",
                "roles": ["tenant_alpha_operator"],
            },
            thread_config,
        )

    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("execute_case_read_tool",)
    assert len(snapshot.tasks) == 1
    assert "TransientCaseStoreError" in snapshot.tasks[0].error
    # State gathered before the failure (classify's/select_case_tool's
    # output) is not lost.
    assert snapshot.values["case_id"] == "CASE-1001"
    assert snapshot.values["case_tool_name"] == "get_customer_case"


def test_failure_recovery_resumes_from_the_failed_node_once_fixed(deps, thread_config):
    """Standing in for 'fix the bug, redeploy, resume': same checkpointer, corrected deps."""
    deps.get_customer_case_fn = _flaky_get_customer_case(fail_times=999)
    checkpointer = InMemorySaver()
    graph = build_graph(deps, checkpointer)

    with pytest.raises(TransientCaseStoreError):
        graph.invoke(
            {
                "original_query": "Give me details on CASE-1001",
                "tenant_id": "tenant_alpha",
                "roles": ["tenant_alpha_operator"],
            },
            thread_config,
        )

    # "Redeploy": a fresh GraphDeps with a working case-read function,
    # same checkpointer/thread.
    from rag.mcp.business import store as case_store

    deps.get_customer_case_fn = case_store.get_customer_case
    fixed_graph = build_graph(deps, checkpointer)

    result = fixed_graph.invoke(None, thread_config)
    assert result["termination_reason"] == "synthesized"
    assert "CASE-1001" in result["final_answer"]
