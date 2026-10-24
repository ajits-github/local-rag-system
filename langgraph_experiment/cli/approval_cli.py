r"""Approve, reject, or cancel a paused write-action run; list runs currently awaiting approval.

Usage
-----
    python -m langgraph_experiment.cli.approval_cli list
    python -m langgraph_experiment.cli.approval_cli approve <thread_id> \
        --approver-subject alice --approver-roles case_status_approver
    python -m langgraph_experiment.cli.approval_cli reject <thread_id> --reason "not yet"
    python -m langgraph_experiment.cli.approval_cli cancel <thread_id>

Stands in for a real approval UI/API endpoint: a human operator invoking
this script *is* the trusted boundary that supplies the resume payload
`nodes.wait_for_approval` reads back from `interrupt()`. Nothing here (or
in the graph) ever accepts an "approved" claim that didn't come through
this path. See that node's docstring for the role check that still runs
even so, and for the server-side (never CLI-side-only) expiration/
workflow-timeout checks a resume can still be refused by even with a
well-formed `approve`.

Pass the same `--backend`/`--db` a `run_demo.py start` used. The two
checkpointer backends are entirely separate stores.

Each invocation of this script is its own process, opening its own
connection to whatever `run_demo.py` wrote to. See `README.md`'s
"Restart/resume experiment" section for why running `start` and `approve`
as two separate commands (rather than one script holding a connection
open) is the point, not an inconvenience.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from langgraph.types import Command

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


def _thread_ids(checkpointer) -> list[str]:
    """Distinct thread ids with at least one checkpoint, most recently written first."""
    seen: dict[str, None] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        seen.setdefault(thread_id, None)
    return list(seen)


def cmd_list(args: argparse.Namespace) -> None:
    """Print every thread currently paused at the approval interrupt."""
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with _checkpointer(args, deps) as checkpointer:
        graph = build_graph(deps, checkpointer)
        found_any = False
        for thread_id in _thread_ids(checkpointer):
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            if not snapshot.interrupts:
                continue
            found_any = True
            pending = (snapshot.values or {}).get("pending_action") or {}
            expires_at = pending.get("expires_at")
            expired_hint = ""
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now(UTC):
                expired_hint = "  [approval window expired; resume will be refused server-side]"
            print(
                f"thread_id={thread_id} case_id={pending.get('case_id')} "
                f"new_status={pending.get('new_status')} "
                f"requested_at={pending.get('requested_at')} "
                f"expires_at={expires_at}{expired_hint}"
            )
        if not found_any:
            print("No runs currently awaiting approval.")


def _resume(args: argparse.Namespace, resume_payload: dict) -> None:
    deps = build_default_deps(with_retrieval=False, with_ledger=not args.no_ledger)
    with _checkpointer(args, deps) as checkpointer:
        graph = build_graph(deps, checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            # Idempotency guard: a second `approve`/`reject`/`cancel` call
            # against a thread that already resumed (or was never paused)
            # must not re-run the mutation. `snapshot.interrupts` being
            # empty means the graph reached END (or never paused); report
            # the outcome already on record instead of invoking again.
            outcome = (snapshot.values or {}).get("case_action_outcome")
            answer = (snapshot.values or {}).get("final_answer")
            print(f"thread_id={args.thread_id} has no pending approval; nothing to do.")
            if answer is not None:
                print(f"Already-recorded outcome: {answer}")
            elif outcome is not None:
                print(f"Already-recorded outcome: {outcome}")
            return
        pending = (snapshot.values or {}).get("pending_action") or {}
        expires_at = pending.get("expires_at")
        if expires_at and datetime.fromisoformat(expires_at) < datetime.now(UTC):
            # A friendly, CLI-side heads-up only. The graph re-checks
            # this server-side regardless (see nodes.wait_for_approval),
            # so this early return is a UX nicety, never the actual
            # enforcement point.
            print(
                f"thread_id={args.thread_id}: the approval window expired at {expires_at}. "
                "Resuming anyway to record the expiry outcome server-side."
            )
        result = graph.invoke(Command(resume=resume_payload), config)
        observability.record_run_outcome(result)
        print(f"thread_id: {args.thread_id}")
        print(f"status: {result.get('termination_reason')}")
        print(f"answer: {result.get('final_answer')}")


def cmd_approve(args: argparse.Namespace) -> None:
    """Resume a paused thread with an approval decision."""
    _resume(
        args,
        {
            "decision": "approve",
            "approver_subject": args.approver_subject,
            "approver_roles": args.approver_roles or [],
        },
    )


def cmd_reject(args: argparse.Namespace) -> None:
    """Resume a paused thread with a rejection decision."""
    _resume(args, {"decision": "reject", "reason": args.reason})


def cmd_cancel(args: argparse.Namespace) -> None:
    """Resume a paused thread with a cancellation decision."""
    _resume(args, {"decision": "cancel", "reason": args.reason})


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("postgres", "sqlite"),
        default="postgres",
        help="Checkpointer backend (default: postgres; must match the `start` that created it)",
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH), help="SQLite db path (--backend sqlite only)"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List threads currently paused awaiting approval")
    _add_backend_args(list_cmd)
    list_cmd.set_defaults(func=cmd_list)

    approve = sub.add_parser("approve", help="Approve a paused write-action run")
    approve.add_argument("thread_id")
    approve.add_argument("--approver-subject", default="demo-approver")
    approve.add_argument("--approver-roles", action="append", default=None, help="Repeatable")
    approve.add_argument("--no-ledger", action="store_true", help="See run_demo.py's --no-ledger")
    _add_backend_args(approve)
    approve.set_defaults(func=cmd_approve)

    reject = sub.add_parser("reject", help="Reject a paused write-action run")
    reject.add_argument("thread_id")
    reject.add_argument("--reason", default=None)
    reject.add_argument("--no-ledger", action="store_true")
    _add_backend_args(reject)
    reject.set_defaults(func=cmd_reject)

    cancel = sub.add_parser(
        "cancel", help="Cancel a paused write-action run (distinct from a human rejection)"
    )
    cancel.add_argument("thread_id")
    cancel.add_argument("--reason", default=None)
    cancel.add_argument("--no-ledger", action="store_true")
    _add_backend_args(cancel)
    cancel.set_defaults(func=cmd_cancel)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
