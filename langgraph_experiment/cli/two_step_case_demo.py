"""The CASE-1001 two-step scenario (resolve, then close) in one process.

CASE-1001 seeds as `status="in_progress"`, and `in_progress -> closed` is
not a valid direct transition (`rag.mcp.business.store._VALID_TRANSITIONS`)
-- reaching `closed` needs `in_progress -> resolved` first, then
`resolved -> closed` (the one sensitive transition). Both mutations pause
for approval in this graph (see `nodes.validate_write_request`'s
docstring), so "close CASE-1001" from the prompt spec's own example needs
two separate approved write actions, run here as four `graph.invoke()`
calls in sequence.

**Deliberately not run as four separate `run_demo.py`/`approval_cli.py`
process invocations.** `rag.mcp.business.store._SYNTHETIC_CASES` is an
in-memory, per-process Python dict with no persistence of its own --
correct for its real purpose (a synthetic stand-in backend living inside
one continuously-running `rag-api` server process), but it means a
second CLI process launched after the first one exits starts from the
*original* seed data again, not whatever the first process mutated. This
was discovered empirically while building this demo (see README.md's
"A real limitation, found by actually running this" section). It does
not affect the LangGraph checkpoint itself (that genuinely is durable
across process restarts; see `run_demo.py`/`approval_cli.py` and
`test_restart_resume.py` for the CASE-2001 scenario, which needs only one
mutation and does demonstrate a real cross-process restart).
"""

from __future__ import annotations

import uuid

from langgraph.types import Command

from langgraph_experiment.checkpointer import sqlite_checkpointer
from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import build_default_deps

_IDENTITY = {"tenant_id": "tenant_alpha", "roles": ["tenant_alpha_operator"]}
_APPROVAL = {"decision": "approve", "approver_roles": ["case_status_approver"]}


def _run_step(graph, thread_id: str, query: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "original_query": query,
        "thread_id": thread_id,
        "caller_subject": "demo-caller",
        **_IDENTITY,
    }
    paused = graph.invoke(initial_state, config)
    assert "__interrupt__" in paused, f"expected a pause for: {query!r}, got {paused}"
    print(f"[{thread_id}] paused: {paused['__interrupt__'][0].value['prompt']}")
    result = graph.invoke(Command(resume=_APPROVAL), config)
    print(f"[{thread_id}] {result['termination_reason']}: {result['final_answer']}")


def main() -> int:
    """Run both CASE-1001 mutations, in-process, so the sensitive transition can be reached."""
    # with_ledger=False: keeps this script's original "zero external
    # services" property (in-memory case store + a throwaway SQLite
    # checkpoint file). See cli/run_demo.py for the ledger-guarded,
    # Postgres-backed path this script deliberately doesn't need.
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with sqlite_checkpointer() as checkpointer:
        graph = build_graph(deps, checkpointer)
        print("Step 1: resolve CASE-1001 (in_progress -> resolved, not sensitive, still pauses)")
        _run_step(graph, f"two-step-resolve-{uuid.uuid4()}", "Please resolve CASE-1001")
        print("\nStep 2: close CASE-1001 (resolved -> closed, the sensitive transition)")
        _run_step(graph, f"two-step-close-{uuid.uuid4()}", "Please close CASE-1001")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
