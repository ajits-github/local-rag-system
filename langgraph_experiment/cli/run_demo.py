"""Start or inspect a run of the experimental LangGraph agent.

Usage
-----
    python -m langgraph_experiment.cli.run_demo start "<query>" [options]
    python -m langgraph_experiment.cli.run_demo state <thread_id>

`--backend postgres` (the default: "prefer PostgreSQL-backed checkpoint
persistence... if available", per the prompt spec) needs `make up`'s
Postgres running; `--backend sqlite` needs nothing but a local file. Both
are real, durable checkpointers. See `checkpointer.py`'s module
docstring. `state`/`approval_cli.py` must be given the same `--backend`
(and, for sqlite, the same `--db` path) a `start` used, since the two
backends are entirely separate stores.

See `README.md`'s "Demo sequence" section for full worked examples,
including the human-in-the-loop scenario this package exists to
demonstrate (`start "Resolve CASE-1001" ...` then `start "Close CASE-1001"
... --thread <same-thread>` is *not* how that works. Each `start` opens
a new thread; see `README.md` for the two-step CASE-1001 sequence and why
it needs two separate threads).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph_experiment import observability
from langgraph_experiment.checkpointer import (
    DEFAULT_DB_PATH,
    postgres_checkpointer,
    sqlite_checkpointer,
)
from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import GraphDeps, build_default_deps


@contextmanager
def _checkpointer(args: argparse.Namespace, deps: GraphDeps) -> Iterator[object]:
    if args.backend == "postgres":
        with postgres_checkpointer(deps.config.database_url()) as saver:
            yield saver
    else:
        with sqlite_checkpointer(args.db) as saver:
            yield saver


def _print_result(thread_id: str, result: dict) -> None:
    print(f"thread_id: {thread_id}")
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        print("status: PAUSED (awaiting approval)")
        print("interrupt payload:")
        print(json.dumps(interrupt_obj.value, indent=2, default=str))
        print(
            f"\nResume with:\n  python -m langgraph_experiment.cli.approval_cli approve "
            f"{thread_id} --approver-subject <name> --approver-roles case_status_approver"
        )
        return
    print(f"status: {result.get('termination_reason')}")
    print(f"route: {result.get('route')}")
    print(f"answer: {result.get('final_answer')}")
    citations = result.get("citations") or []
    if citations:
        print("citations:")
        for c in citations:
            print(f"  - {c}")


def cmd_start(args: argparse.Namespace) -> None:
    """Start a new run on a fresh thread and print its outcome (or interrupt payload)."""
    thread_id = args.thread or str(uuid.uuid4())
    deps = build_default_deps(with_retrieval=not args.no_retrieval, with_ledger=not args.no_ledger)
    with _checkpointer(args, deps) as checkpointer:
        graph = build_graph(deps, checkpointer)
        initial_state = {
            "original_query": args.query,
            "thread_id": thread_id,
            "caller_subject": args.subject,
            "tenant_id": args.tenant,
            "roles": args.role or [],
            "dataset_id": args.dataset_id,
        }
        result = graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
        observability.record_run_outcome(result)
        _print_result(thread_id, result)


def cmd_state(args: argparse.Namespace) -> None:
    """Print a thread's current checkpointed state, including any pending interrupt."""
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with _checkpointer(args, deps) as checkpointer:
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


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("postgres", "sqlite"),
        default="postgres",
        help="Checkpointer backend (default: postgres; needs `make up`)",
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH), help="SQLite db path (--backend sqlite only)"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "only the case-store branch works)",
    )
    start.add_argument(
        "--no-ledger",
        action="store_true",
        help="Skip the Postgres idempotency ledger (falls back to update_case_status's own "
        "natural idempotency alone; see idempotency.py)",
    )
    _add_backend_args(start)
    start.set_defaults(func=cmd_start)

    state = sub.add_parser("state", help="Print a thread's current checkpointed state")
    state.add_argument("thread_id")
    _add_backend_args(state)
    state.set_defaults(func=cmd_state)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
