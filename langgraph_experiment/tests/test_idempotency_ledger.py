"""`ActionLedger`: proving resume/retry cannot double-execute the one write action.

Needs real Postgres (`require_postgres`/`postgres_deps`, self-skips
cleanly without it) -- `ActionLedger` is Postgres-backed by design (see
`idempotency.py`'s module docstring for why: `langgraph-checkpoint-
postgres` is the preferred checkpointer, and this ledger is a second,
independent Postgres-backed mechanism on top of it).
"""

from __future__ import annotations

import uuid

from rag.api.auth import VerifiedIdentity
from rag.mcp.business import store as case_store
from rag.mcp.business.schemas import CaseApproval


def _identity() -> VerifiedIdentity:
    return VerifiedIdentity(subject="test", tenant_id="tenant_beta", roles=["tenant_beta_operator"])


def test_fresh_operation_executes_mutate_exactly_once(postgres_deps):
    ledger = postgres_deps.action_ledger
    operation_id = str(uuid.uuid4())
    calls = {"n": 0}

    def mutate():
        calls["n"] += 1
        return case_store.update_case_status(
            "CASE-2001",
            "closed",
            _identity(),
            [],
            [CaseApproval(case_id="CASE-2001", new_status="closed")],
        )

    result = ledger.record_attempt(
        operation_id, thread_id="t1", case_id="CASE-2001", new_status="closed", mutate=mutate
    )

    assert calls["n"] == 1
    assert result.outcome == "executed"
    record = ledger.get(operation_id)
    assert record is not None
    assert record.completed is True
    assert record.outcome == "executed"


def test_already_completed_operation_is_replayed_without_remutating(postgres_deps):
    ledger = postgres_deps.action_ledger
    operation_id = str(uuid.uuid4())
    calls = {"n": 0}

    def mutate():
        calls["n"] += 1
        return case_store.update_case_status(
            "CASE-2001",
            "closed",
            _identity(),
            [],
            [CaseApproval(case_id="CASE-2001", new_status="closed")],
        )

    first = ledger.record_attempt(
        operation_id, thread_id="t1", case_id="CASE-2001", new_status="closed", mutate=mutate
    )
    second = ledger.record_attempt(
        operation_id, thread_id="t1", case_id="CASE-2001", new_status="closed", mutate=mutate
    )

    assert calls["n"] == 1, "mutate() must not run a second time for an already-completed operation"
    assert second.outcome == first.outcome == "executed"


def test_crash_between_mutation_and_ledger_completion_does_not_double_mutate(postgres_deps):
    """Reproduces the exact scenario the prompt spec's production addendum asks for.

    Simulates "process crashes after mutation but before checkpoint is
    marked complete" directly: `mark_started` + the real mutation are run
    by hand (standing in for the first attempt, up to the crash point),
    deliberately *skipping* `mark_completed` -- exactly what a crash
    between the mutation returning and the ledger being updated would
    leave behind. `record_attempt` is then called as if this were the
    resumed/retried attempt.
    """
    ledger = postgres_deps.action_ledger
    operation_id = str(uuid.uuid4())
    identity = _identity()
    approval = [CaseApproval(case_id="CASE-2001", new_status="closed")]

    # The "crashed" first attempt: started, mutated for real, never completed.
    ledger.mark_started(operation_id, thread_id="t1", case_id="CASE-2001", new_status="closed")
    case_store.update_case_status("CASE-2001", "closed", identity, [], approval)
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"
    record = ledger.get(operation_id)
    assert record is not None and record.completed is False

    # The "resumed" attempt: record_attempt sees a started-but-incomplete
    # row and, per its documented recovery strategy, calls mutate() again.
    # This is safe only because update_case_status is itself naturally
    # idempotent (already_in_status), not because the ledger prevented a
    # second call.
    calls = {"n": 0}

    def mutate():
        calls["n"] += 1
        return case_store.update_case_status("CASE-2001", "closed", identity, [], approval)

    result = ledger.record_attempt(
        operation_id, thread_id="t1", case_id="CASE-2001", new_status="closed", mutate=mutate
    )

    assert calls["n"] == 1, "the recovery attempt does call mutate() again"
    assert result.outcome == "already_in_status", (
        "the business layer's own idempotency, not the ledger alone, is what "
        "prevents a real second transition"
    )
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"

    final_record = ledger.get(operation_id)
    assert final_record is not None and final_record.completed is True


def test_graph_write_action_records_a_completed_ledger_row(postgres_deps, graph, thread_config):
    """End-to-end through the real graph, not `ActionLedger` directly.

    `graph`/`deps` come from the standard (no-ledger) fixtures in
    `conftest.py`; this test swaps in `postgres_deps.action_ledger`
    directly onto that same `deps` object before building its own graph,
    since `graph`'s own fixture already compiled against a ledger-less
    `deps`.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from langgraph_experiment.graph import build_graph as _build_graph
    from langgraph_experiment.wiring import GraphDeps

    deps_with_ledger = GraphDeps(
        config=postgres_deps.config,
        action_ledger=postgres_deps.action_ledger,
        approval_roles=postgres_deps.approval_roles,
        cross_tenant_support_roles=postgres_deps.cross_tenant_support_roles,
    )
    write_graph = _build_graph(deps_with_ledger, InMemorySaver())

    write_graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "thread_id": thread_config["configurable"]["thread_id"],
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    result = write_graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
        thread_config,
    )
    assert result["termination_reason"] == "executed"

    operation_id = result["pending_action"]["operation_id"]
    record = postgres_deps.action_ledger.get(operation_id)
    assert record is not None
    assert record.completed is True
    assert record.outcome == "executed"
