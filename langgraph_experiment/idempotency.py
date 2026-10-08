"""A durable, Postgres-backed idempotency ledger for the one write action this graph has.

Answers the prompt spec's production-hardening question directly: "prove
resume cannot execute the mutation twice" after a crash between the
mutation executing and the workflow marking it complete. This is
deliberately a *second*, independent mechanism from LangGraph's own
checkpointer. See this module's "Three distinct guarantees" note below,
and README.md's "Workflow durability vs. business-operation idempotency
vs. transaction atomicity" section for the full writeup.

Uses `psycopg2` (this project's own established Postgres driver, already
a dependency of `rag.vectorstore.pgvector`), not the `psycopg` (v3) driver
`langgraph-checkpoint-postgres` itself requires: a deliberate split:
the checkpointer's own storage engine needs `psycopg` because that's what
the LangGraph library was built against, but this experiment's own
application-level ledger has no reason to add a second driver dependency
just to match it.

Three distinct guarantees, and where each one actually lives
--------------------------------------------------------------
- **Workflow durability** (LangGraph + its checkpointer): the *workflow*
  reliably resumes at the right node after a restart. Proven by
  `test_restart_resume.py`. Says nothing about whether a node's own side
  effect ran zero, one, or more times.
- **Business-operation idempotency** (`rag.mcp.business.store.
  update_case_status`'s own `already_in_status` branch, *and* this
  ledger): calling the same logical mutation twice must be safe. The
  ledger adds a second, independent layer on top of `update_case_status`'s
  own natural idempotency (see `ActionLedger.record_attempt`'s docstring
  for why *both* layers matter, not just one).
- **Transaction atomicity**: there is no single atomic transaction
  spanning "call `update_case_status`" and "mark the ledger row
  complete", that gap is real, and is exactly the crash window
  `test_idempotency_ledger.py::
  test_crash_between_mutation_and_ledger_completion_does_not_double_mutate`
  reproduces. Postgres gives atomicity *within* one SQL statement, never
  across two separate calls into two different systems (the case store
  and the ledger table) with a process crash in between. This module
  does not, and cannot, remove that gap. It only makes what happens
  *if* a crash lands in it safe, by relying on the underlying mutation's
  own idempotency for the recovery step (see `record_attempt`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg2

logger = logging.getLogger(__name__)

_TABLE = "langgraph_experiment_action_ledger"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    operation_id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    case_id TEXT NOT NULL,
    new_status TEXT NOT NULL,
    outcome TEXT,
    previous_status TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
"""


@dataclass(frozen=True)
class LedgerRecord:
    """One ledger row's state, as read back by `ActionLedger.get`."""

    operation_id: str
    outcome: str | None
    previous_status: str | None
    completed: bool


class ActionLedger:
    """A durable `operation_id -> outcome` record, independent of the LangGraph checkpoint.

    Not a general-purpose idempotency library. Scoped narrowly to this
    graph's one write action, mirroring how
    `rag.mcp.business.store`'s own `_CASE_MUTATION_LOCK` is scoped
    narrowly to its one mutator rather than a generic locking framework.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        """Create the ledger table if it doesn't exist yet. Idempotent, safe to call every run."""
        with psycopg2.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def get(self, operation_id: str) -> LedgerRecord | None:
        """Return this operation's current ledger row, or `None` if never attempted."""
        with psycopg2.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT outcome, previous_status, completed_at FROM {_TABLE} "  # noqa: S608
                "WHERE operation_id = %s",
                (operation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        outcome, previous_status, completed_at = row
        return LedgerRecord(
            operation_id=operation_id,
            outcome=outcome,
            previous_status=previous_status,
            completed=completed_at is not None,
        )

    def mark_started(
        self, operation_id: str, *, thread_id: str, case_id: str, new_status: str
    ) -> None:
        """Insert a "started" row before the mutation is attempted.

        `ON CONFLICT DO NOTHING`: a retry (automatic `RetryPolicy` retry,
        or a resumed crashed run) calling this again for the same
        `operation_id` never overwrites an already-recorded `started_at`.
        The row's existence alone is what matters for
        `test_crash_between_mutation_and_ledger_completion_does_not_
        double_mutate` to reconstruct "was an attempt already in flight."
        """
        with psycopg2.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {_TABLE} "  # noqa: S608
                "(operation_id, thread_id, action_type, case_id, new_status) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (operation_id) DO NOTHING",
                (operation_id, thread_id, "update_case_status", case_id, new_status),
            )
            conn.commit()

    def mark_completed(self, operation_id: str, *, outcome: str, previous_status: str) -> None:
        """Record the mutation's real outcome and mark this operation durably complete."""
        with psycopg2.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {_TABLE} SET outcome = %s, previous_status = %s, "  # noqa: S608
                "completed_at = %s WHERE operation_id = %s",
                (outcome, previous_status, datetime.now(UTC), operation_id),
            )
            conn.commit()

    def record_attempt(
        self,
        operation_id: str,
        *,
        thread_id: str,
        case_id: str,
        new_status: str,
        mutate: Any,
    ) -> Any:
        """Run `mutate()` at most as many times as actually needed, ledger-guarded.

        Three cases, in order:

        1. **Already completed** (`get()` returns a row with `completed=True`):
           `mutate()` is never called again. Returns the recorded outcome
           directly. This is the ordinary duplicate-resume/duplicate-
           retry path.
        2. **Started, not completed** (a row exists but `completed_at IS
           NULL`): the exact crash window this module exists to handle --
           some earlier attempt called `mark_started`, and then either
           crashed before calling `mutate()`, crashed *during* `mutate()`,
           or crashed after `mutate()` returned but before
           `mark_completed` ran. This method cannot distinguish those
           three sub-cases from the ledger alone (that information was
           lost when the process crashed), so it does the only safe
           thing available: calls `mutate()` again, exactly as if this
           were a fresh attempt. That is only safe because `mutate()`
           itself (`rag.mcp.business.store.update_case_status`) is
           naturally idempotent for this business operation. A repeat
           call for a case already in its target status returns
           `already_in_status`, never a second real transition. **The
           ledger does not manufacture that safety; it exists on top of
           an operation that already had it.** A non-idempotent mutation
           (e.g. "charge $10") would need a genuinely different recovery
           strategy here (query the external system for the real
           outcome, or refuse and require manual reconciliation). See
           this module's docstring.
        3. **No row at all**: a genuinely new attempt. Inserts the
           "started" row, calls `mutate()`, then marks it completed.

        Parameters
        ----------
        operation_id, thread_id, case_id, new_status
            Identify this ledger row.
        mutate : Callable[[], CaseActionOutcome | None]
            The real mutation call (a closure over
            `deps.update_case_status_fn` and its arguments).

        Returns
        -------
        Any
            Whatever `mutate()` returned (a `CaseActionOutcome | None`),
            either freshly computed or replayed from the ledger's own
            recorded fields when the operation was already complete.
        """
        existing = self.get(operation_id)
        if existing is not None and existing.completed:
            logger.info(
                "action_ledger_replay",
                extra={"operation_id": operation_id, "outcome": existing.outcome},
            )
            return _ReplayedOutcome(
                outcome=existing.outcome, previous_status=existing.previous_status
            )

        self.mark_started(operation_id, thread_id=thread_id, case_id=case_id, new_status=new_status)
        result = mutate()
        if result is not None:
            self.mark_completed(
                operation_id, outcome=result.outcome, previous_status=result.previous_status
            )
        return result


@dataclass(frozen=True)
class _ReplayedOutcome:
    """A ledger-replayed stand-in for `CaseActionOutcome`, exposing the same two field names.

    `nodes.execute_write_action` only ever reads `.outcome`/
    `.previous_status`/`.case_id`/`.new_status`/`.updated_at` off
    whatever `record_attempt` returns before calling `.model_dump(mode=
    "json")` on a real `CaseActionOutcome`. A replay never reconstructs
    a fake `CaseActionOutcome`, since `updated_at` genuinely isn't known
    from the ledger alone. `nodes.py` checks `isinstance(result,
    _ReplayedOutcome)` and renders a slightly different (still accurate)
    answer for that case; see its docstring.
    """

    outcome: str | None
    previous_status: str | None
