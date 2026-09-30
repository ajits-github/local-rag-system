"""Checkpoint cleanup/retention: delete completed threads' checkpoints older than N days.

Usage
-----
    python -m langgraph_experiment.cli.retention_cleanup --older-than-days 7 [--dry-run]

A durable checkpointer (Postgres or SQLite, see `checkpointer.py`) keeps
every checkpoint for every thread forever unless something deletes them --
LangGraph itself has no built-in retention policy, the same way a
database table has no built-in row-expiry unless something adds one. This
script is that something: a small, explicit, operator-run sweep, not a
background job (no scheduler exists in this experimental package, and
adding one would be exactly the "extra infrastructure for appearance" the
prompt spec says to avoid).

Only ever deletes a thread whose run has **already reached a terminal
state** (`snapshot.next == ()` -- no pending interrupt, nothing left to
resume) *and* whose most recent checkpoint is older than the cutoff.
A thread still paused awaiting approval, no matter how old, is never
touched -- deleting its checkpoint would silently destroy a real pending
human decision, not just tidy up disk space.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from langgraph_experiment.checkpointer import (
    DEFAULT_DB_PATH,
    postgres_checkpointer,
    sqlite_checkpointer,
)
from langgraph_experiment.graph import build_graph
from langgraph_experiment.wiring import build_default_deps


def _most_recent_checkpoint_time(checkpointer, thread_id: str) -> datetime | None:
    """Return the most recent checkpoint's timestamp for one thread, or `None` if it has none."""
    latest: str | None = None
    for tup in checkpointer.list({"configurable": {"thread_id": thread_id}}, limit=1):
        latest = tup.checkpoint.get("ts")
    if latest is None:
        return None
    return datetime.fromisoformat(latest.replace("Z", "+00:00"))


def main(argv: list[str] | None = None) -> int:
    """Parse args and sweep terminal-state threads' checkpoints past the cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("postgres", "sqlite"), default="postgres")
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH), help="SQLite db path (--backend sqlite only)"
    )
    parser.add_argument("--older-than-days", type=float, default=7.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be deleted without deleting it"
    )
    args = parser.parse_args(argv)

    deps = build_default_deps(with_retrieval=False, with_ledger=False)
    cutoff = datetime.now(UTC) - timedelta(days=args.older_than_days)

    checkpointer_cm = (
        postgres_checkpointer(deps.config.database_url())
        if args.backend == "postgres"
        else sqlite_checkpointer(args.db)
    )
    with checkpointer_cm as checkpointer:
        graph = build_graph(deps, checkpointer)
        seen: dict[str, None] = {}
        for tup in checkpointer.list(None):
            seen.setdefault(tup.config["configurable"]["thread_id"], None)

        deleted, skipped_pending, skipped_recent = 0, 0, 0
        for thread_id in seen:
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            if snapshot.next:
                skipped_pending += 1
                continue
            checkpoint_time = _most_recent_checkpoint_time(checkpointer, thread_id)
            if checkpoint_time is None or checkpoint_time >= cutoff:
                skipped_recent += 1
                continue
            print(f"{'[dry-run] would delete' if args.dry_run else 'deleting'} thread {thread_id}")
            if not args.dry_run:
                checkpointer.delete_thread(thread_id)
            deleted += 1

    print(
        f"\ndone: {deleted} deleted, {skipped_pending} skipped (still paused/pending), "
        f"{skipped_recent} skipped (newer than cutoff)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
