"""Start or inspect a run of the multi-agent LangGraph experiment.

Usage
-----
    python -m multi_agent_experiment.cli.run_demo start "<query>" [options]
    python -m multi_agent_experiment.cli.run_demo state <thread_id>

Mirrors `langgraph_experiment.cli.run_demo` exactly (same `--db`/
`--thread`/`--subject`/`--tenant`/`--role`/`--dataset-id`/`--no-retrieval`
flags), pointed at this package's own graph/checkpointer -- see
README.md's "Run commands" section for worked examples covering all five
required scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph_experiment.wiring import build_default_deps
from multi_agent_experiment.checkpointer import DEFAULT_DB_PATH, sqlite_checkpointer
from multi_agent_experiment.graph import build_graph


def _print_result(thread_id: str, result: dict) -> None:
    print(f"thread_id: {thread_id}")
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        print("status: PAUSED (awaiting approval)")
        print("interrupt payload:")
        print(json.dumps(interrupt_obj.value, indent=2, default=str))
        print(
            f"\nResume with:\n  python -m multi_agent_experiment.cli.approval_cli approve "
            f"{thread_id} --approver-subject <name> --approver-roles case_status_approver"
        )
        return
    print(f"status: {result.get('termination_reason')}")
    print(f"needs_knowledge: {result.get('needs_knowledge')}")
    print(f"needs_business: {result.get('needs_business')}")
    print(f"selected_specialists: {result.get('selected_specialists')}")
    print(f"answer: {result.get('final_answer')}")
    citations = result.get("citations") or []
    if citations:
        print("citations:")
        for c in citations:
            print(f"  - {c}")
    tool_calls = result.get("tool_call_log") or []
    if tool_calls:
        print("tool_call_log:")
        for call in tool_calls:
            print(f"  - {call}")
    notes = result.get("critic_notes") or []
    if notes:
        print("critic_notes:")
        for note in notes:
            print(f"  - {note}")


def cmd_start(args: argparse.Namespace) -> None:
    """Start a new run on a fresh (or named) thread and print its result."""
    thread_id = args.thread or str(uuid.uuid4())
    # with_ledger mirrors --no-retrieval: both name "is Postgres available,"
    # so a --no-retrieval run needs neither the RAG pipeline nor the
    # production-hardening idempotency ledger's own Postgres connection.
    deps = build_default_deps(
        with_retrieval=not args.no_retrieval, with_ledger=not args.no_retrieval
    )
    with sqlite_checkpointer(args.db) as checkpointer:
        graph = build_graph(deps, checkpointer)
        initial_state = {
            "original_query": args.query,
            "caller_subject": args.subject,
            "tenant_id": args.tenant,
            "roles": args.role or [],
            "dataset_id": args.dataset_id,
        }
        result = graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
        _print_result(thread_id, result)


def cmd_state(args: argparse.Namespace) -> None:
    """Print a thread's current checkpointed state and any pending interrupt."""
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with sqlite_checkpointer(args.db) as checkpointer:
        graph = build_graph(deps, checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": args.thread_id}})
        print(f"thread_id: {args.thread_id}")
        print(f"next: {snapshot.next}")
        print("values:")
        print(json.dumps(snapshot.values, indent=2, default=str))
        if snapshot.interrupts:
            print("pending interrupts:")
            for i in snapshot.interrupts:
                print(json.dumps(i.value, indent=2, default=str))
        if snapshot.tasks:
            for task in snapshot.tasks:
                if task.error is not None:
                    print(f"task '{task.name}' error: {task.error}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: dispatch to start/state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite checkpoint db path")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start a new run on a fresh thread")
    start.add_argument("query")
    start.add_argument("--thread", default=None, help="Thread id (default: a new uuid4)")
    start.add_argument("--subject", default="demo-caller")
    start.add_argument("--tenant", default=None)
    start.add_argument("--role", action="append", default=None, help="Repeatable")
    start.add_argument("--dataset-id", default=None)
    start.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip building the RetrievalPipeline/LLM (no Postgres/Ollama needed; "
        "only the business-only branch works)",
    )
    start.set_defaults(func=cmd_start)

    state = sub.add_parser("state", help="Print a thread's current checkpointed state")
    state.add_argument("thread_id")
    state.set_defaults(func=cmd_state)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
