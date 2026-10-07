r"""Approve or reject a paused write-action run; list runs currently awaiting approval.

Usage
-----
    python -m multi_agent_experiment.cli.approval_cli list
    python -m multi_agent_experiment.cli.approval_cli approve <thread_id> \\
        --approver-subject alice --approver-roles case_status_approver
    python -m multi_agent_experiment.cli.approval_cli reject <thread_id> --reason "not yet"

Identical in every behavior to `langgraph_experiment.cli.approval_cli`
(same idempotent-resume guard, same trust boundary: a human operator
invoking this script *is* what supplies the resume payload
`business_agent.make_wait_for_approval_node`'s reused
`langgraph_experiment.nodes.wait_for_approval` reads back from
`interrupt()`), pointed at this package's own graph/checkpointer.

`with_ledger=False`: keeps this CLI's own "zero external services"
guarantee. `GraphDeps.action_ledger` (the production-hardening
idempotency ledger `execute_write_action` can optionally guard mutations
through) needs its own Postgres connection when enabled, independent of
`with_retrieval`'s RAG-only Postgres/Ollama need. A caller with Postgres
available can pass `with_ledger=True` to `build_default_deps` directly
for the stronger crash-recovery guarantee; not exercised by this demo.
"""

from __future__ import annotations

import argparse
import sys

from langgraph.types import Command

from langgraph_experiment.wiring import build_default_deps
from multi_agent_experiment.checkpointer import DEFAULT_DB_PATH, sqlite_checkpointer
from multi_agent_experiment.graph import build_graph


def _thread_ids(checkpointer) -> list[str]:
    """Distinct thread ids with at least one checkpoint, most recently written first."""
    seen: dict[str, None] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        seen.setdefault(thread_id, None)
    return list(seen)


def cmd_list(args: argparse.Namespace) -> None:
    """Print every thread currently paused awaiting approval."""
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with sqlite_checkpointer(args.db) as checkpointer:
        graph = build_graph(deps, checkpointer)
        found_any = False
        for thread_id in _thread_ids(checkpointer):
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            if not snapshot.interrupts:
                continue
            found_any = True
            pending = (snapshot.values or {}).get("pending_action") or {}
            print(
                f"thread_id={thread_id} case_id={pending.get('case_id')} "
                f"new_status={pending.get('new_status')} "
                f"requested_at={pending.get('requested_at')}"
            )
        if not found_any:
            print("No runs currently awaiting approval.")


def _resume(args: argparse.Namespace, resume_payload: dict) -> None:
    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    with sqlite_checkpointer(args.db) as checkpointer:
        graph = build_graph(deps, checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            outcome = (snapshot.values or {}).get("case_action_outcome")
            answer = (snapshot.values or {}).get("final_answer")
            print(f"thread_id={args.thread_id} has no pending approval; nothing to do.")
            if answer is not None:
                print(f"Already-recorded outcome: {answer}")
            elif outcome is not None:
                print(f"Already-recorded outcome: {outcome}")
            return
        result = graph.invoke(Command(resume=resume_payload), config)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: dispatch to list/approve/reject."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List threads currently paused awaiting approval")
    list_cmd.set_defaults(func=cmd_list)

    approve = sub.add_parser("approve", help="Approve a paused write-action run")
    approve.add_argument("thread_id")
    approve.add_argument("--approver-subject", default="demo-approver")
    approve.add_argument("--approver-roles", action="append", default=None, help="Repeatable")
    approve.set_defaults(func=cmd_approve)

    reject = sub.add_parser("reject", help="Reject a paused write-action run")
    reject.add_argument("thread_id")
    reject.add_argument("--reason", default=None)
    reject.set_defaults(func=cmd_reject)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
