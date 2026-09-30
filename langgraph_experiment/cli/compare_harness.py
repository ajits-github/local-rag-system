"""Run a comparable scenario through both the custom harness and this LangGraph experiment.

Usage
-----
    python -m langgraph_experiment.cli.compare_harness rag "<query>" [--dataset-id ID]
    python -m langgraph_experiment.cli.compare_harness write-action

`rag` runs the same read-only question through both
`rag.agent.graph.run_agent` (route="classic_rag", the production custom
harness) and this package's `read_only` branch, and prints latency/step
counts for both -- needs a running Postgres (`make up`) and native Ollama,
same as any other integration-shaped script in this repo.

`write-action` does not run the custom harness at all for the
approval/interrupt scenario: `run_agent` has no pause/resume primitive to
run, so there is nothing to time or compare quantitatively. It instead
prints the qualitative comparison from `README.md`'s "Compare with
current custom harness" section (kept here as one, reasoned table so
both places agree; see the module docstring below the table).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from langgraph_experiment.checkpointer import sqlite_checkpointer
from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import build_default_deps
from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.retrieval.authorization import AuthorizationContext

#: One row per axis from the prompt spec's "Compare with current custom
#: harness" section. `custom` and `langgraph` are short, factual
#: observations, not a verdict -- see README.md's own comparison section
#: for the full reasoning behind each row (this table is the condensed,
#: printable version of the same conclusions, kept in sync by hand since
#: one is Python and the other is Markdown).
_QUALITATIVE_COMPARISON: list[tuple[str, str, str]] = [
    (
        "Code complexity",
        "~150 lines for the write-action path incl. one Python while-loop; "
        "everything is inline and directly steppable in a debugger.",
        "~110 lines across nodes.py/graph.py for the same path, but "
        "understanding a run means reading graph.py's edges *and* every "
        "node function; harder to step through in a debugger (control "
        "flow lives in the Pregel runtime, not a visible loop).",
    ),
    (
        "State visibility",
        "AgentState is one pydantic object, inspectable at any breakpoint; "
        "no built-in history of prior states within a run.",
        "GraphState is checkpointed after every node -- `get_state_history()` "
        "gives the full sequence of states a run passed through, not just "
        "the current one.",
    ),
    (
        "Branching",
        "Plain if/elif in Python; trivial to read, but the shape of the "
        "graph is not queryable at runtime.",
        "add_conditional_edges() -- the shape of the graph (which nodes "
        "exist, how they connect) is introspectable via get_graph(), which "
        "is how README.md's architecture diagram commands were produced.",
    ),
    (
        "Retry handling",
        "None built in; a tool failure is caught and recorded as a failed "
        "ToolCallRecord, never re-attempted automatically.",
        "RetryPolicy on a node is a one-line, declarative retry with "
        "backoff -- see graph.py's _CASE_READ_RETRY_POLICY.",
    ),
    (
        "Checkpointing",
        "None. A run lives entirely in one Python call stack/process; "
        "nothing survives a crash or restart.",
        "Every node transition is durably checkpointed (SQLite here) -- "
        "a run can be inspected, resumed, or replayed after a full "
        "process restart.",
    ),
    (
        "Pause/resume",
        "Not supported at all -- there is no mechanism to stop mid-run and "
        "continue later; a run either completes or is cancelled "
        "(cancel_event) and abandoned.",
        "interrupt()/Command(resume=...) is the core primitive this "
        "package exists to demonstrate -- see nodes.wait_for_approval.",
    ),
    (
        "Human approval",
        "Exists in production (case_approvals), but must be supplied "
        "*before* the run starts (a router-resolved field on the request) "
        "-- there is no way to pause an in-flight run to ask.",
        "A run can request approval mid-flight and durably wait, "
        "potentially across a process restart, for a human response.",
    ),
    (
        "Latency",
        "One process, one call stack -- no checkpoint-write overhead per step.",
        "Each node transition writes a checkpoint row (SQLite here); "
        "measurable but small overhead per step (see `rag` subcommand "
        "for an actual number on this machine).",
    ),
    (
        "Debugging",
        "Standard Python debugger works end to end; a stack trace points "
        "straight at the failing line.",
        "get_state(config).tasks[i].error surfaces a failed node's "
        "exception without a live debugger attached -- valuable after the "
        "fact or across a process boundary, less immediate than a live pdb "
        "session.",
    ),
    (
        "Testability",
        "Tests call run_agent() directly with fake pipeline/vectorstore/"
        "llm; fully synchronous, no extra test infra.",
        "Tests can inject fakes the same way (GraphDeps), plus assert on "
        "checkpoint state directly (get_state(config).next, .interrupts) "
        "-- see tests/test_graph_interrupt_flow.py.",
    ),
]


def _print_qualitative_comparison() -> None:
    print(f"{'Axis':<20} | {'Custom harness (rag.agent.graph)':<70} | LangGraph experiment")
    print("-" * 160)
    for axis, custom, langgraph in _QUALITATIVE_COMPARISON:
        print(f"{axis:<20} | {custom:<70} | {langgraph}")


def cmd_write_action(_: argparse.Namespace) -> None:
    """Print the qualitative comparison table for the write-action/approval scenario."""
    print(
        "The custom harness has no pause/resume primitive, so there is nothing to run for "
        "a direct timing comparison on the write-action/approval scenario. Qualitative "
        "comparison (see README.md for the full reasoning behind each row):\n"
    )
    _print_qualitative_comparison()


def cmd_rag(args: argparse.Namespace) -> None:
    """Time a read-only RAG query through both the custom harness and this experiment."""
    deps = build_default_deps(with_retrieval=True)
    auth = AuthorizationContext(tenant_id=args.tenant, roles=args.role or [])
    filters = {"dataset_id": args.dataset_id} if args.dataset_id else None

    print("--- Custom harness (rag.agent.graph.run_agent, classic_rag route) ---")
    state = AgentState(original_query=args.query, authorization_context=auth, filters=filters)
    t0 = time.perf_counter()
    custom_result = run_agent(
        state,
        pipeline=deps.pipeline,
        vectorstore=deps.pipeline._vectorstore,  # noqa: SLF001 -- read-only introspection for this comparison script only
        embedder=deps.pipeline._embedder,  # noqa: SLF001
        llm=deps.llm,
        config=deps.config,
    )
    custom_ms = (time.perf_counter() - t0) * 1000
    print(f"answer: {custom_result.state.final_answer}")
    print(f"total_ms: {custom_ms:.1f} (reported: {custom_result.total_ms:.1f})")

    print("\n--- LangGraph experiment (read_only branch) ---")
    thread_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    with sqlite_checkpointer() as checkpointer:
        graph = build_graph(deps, checkpointer)
        result = graph.invoke(
            {
                "original_query": args.query,
                "caller_subject": "compare-harness",
                "tenant_id": args.tenant,
                "roles": args.role or [],
                "dataset_id": args.dataset_id,
            },
            {"configurable": {"thread_id": thread_id}},
        )
    langgraph_ms = (time.perf_counter() - t0) * 1000
    print(f"answer: {result.get('final_answer')}")
    print(f"total_ms: {langgraph_ms:.1f}")

    print(f"\ncheckpoint overhead (approx): {langgraph_ms - custom_ms:+.1f} ms for this one run")


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rag_cmd = sub.add_parser("rag", help="Time a read-only RAG query through both harnesses")
    rag_cmd.add_argument("query")
    rag_cmd.add_argument("--tenant", default=None)
    rag_cmd.add_argument("--role", action="append", default=None)
    rag_cmd.add_argument("--dataset-id", default=None)
    rag_cmd.set_defaults(func=cmd_rag)

    write_cmd = sub.add_parser(
        "write-action", help="Print the qualitative comparison for the approval/interrupt scenario"
    )
    write_cmd.set_defaults(func=cmd_write_action)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
