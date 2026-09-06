"""Postgres persistence for user feedback.

Uses its own small `ThreadedConnectionPool`, following the exact
connection-lifecycle pattern `vectorstore.pgvector.PgVectorStore`
establishes (`_connection()`: checkout, guaranteed `putconn` in
`finally`), but this is deliberately not a `VectorStore` implementation:
feedback has no embedding/similarity-search/authorization-predicate
concept, just a plain, narrow write/read table. A separate pool (rather
than sharing `PgVectorStore`'s) keeps feedback persistence decoupled from
the retrieval path -- the two can fail, scale, or be reconfigured
independently.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import ThreadedConnectionPool

from rag.feedback.schemas import FeedbackRating, FeedbackRecord

_COLUMNS = (
    "feedback_id",
    "created_at",
    "updated_at",
    "tenant_id",
    "caller_key",
    "request_id",
    "rating",
    "reason",
    "comment",
    "route",
    "dataset_id",
    "query_text",
    "answer_text",
    "cited_source_ids",
    "tool_calls",
    "generation_model",
    "prompt_id",
    "prompt_version",
    "retrieval_provider",
    "reranker_provider",
)


@dataclass
class FeedbackWriteInput:
    """Everything `FeedbackStore.submit` needs, already validated/trust-resolved by the router.

    `tenant_id`/`caller_key` are the router's already-trusted values
    (`api/routers/feedback.py`'s `resolve_trusted_tenant_id`/
    `_resolve_caller_key`), never the client's own unverified claim.
    Nothing here carries a JWT, an `AuthorizationContext`, chain-of-
    thought, or raw prompts -- only the fields this dataclass declares.
    """

    tenant_id: str | None
    caller_key: str
    request_id: str
    rating: FeedbackRating
    reason: str | None
    comment: str | None
    route: str | None
    dataset_id: str | None
    query_text: str | None
    answer_text: str | None
    cited_source_ids: list[str]
    tool_calls: list[str]
    generation_model: str | None
    prompt_id: str | None
    prompt_version: str | None
    retrieval_provider: str | None
    reranker_provider: str | None


def _row_to_record(row: tuple[object, ...]) -> FeedbackRecord:
    """Map one raw DB row (in `_COLUMNS` order) to a `FeedbackRecord`."""
    values: dict[str, Any] = dict(zip(_COLUMNS, row, strict=True))
    values["tenant_id"] = values["tenant_id"] or None
    values["cited_source_ids"] = list(values["cited_source_ids"] or [])
    values["tool_calls"] = list(values["tool_calls"] or [])
    return FeedbackRecord(**values)


class FeedbackStore:
    """Small, dedicated Postgres store for the `feedback` table."""

    def __init__(
        self, dsn: str, table: str = "feedback", minconn: int = 1, maxconn: int = 3
    ) -> None:
        """Open a threaded connection pool against `dsn`.

        Parameters
        ----------
        dsn : str
            Postgres connection string (the same DSN `PgVectorStore` uses).
        table : str, optional
            Name of the feedback table, by default ``"feedback"``.
        minconn, maxconn : int, optional
            Pool bounds. Small on purpose: feedback writes are low-volume
            and never share a pool with the vectorstore/embedding path.
        """
        self._table = table
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)

    @contextmanager
    def _connection(self) -> Iterator[PgConnection]:
        """Check out a pooled connection, guaranteeing it's always returned."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def submit(self, data: FeedbackWriteInput) -> tuple[str, bool]:
        """Insert a new feedback row, or update the caller's existing one for the same run.

        Dedup/update key: `(tenant_id, caller_key, request_id)`, backed
        by a database `UNIQUE` constraint plus `INSERT ... ON CONFLICT
        DO UPDATE` -- atomic under concurrent submissions, unlike a
        SELECT-then-INSERT/UPDATE app-level check. `tenant_id` is stored
        as `''` rather than `NULL` specifically so this key can
        participate in that constraint: Postgres treats every `NULL` as
        distinct from every other `NULL` in a `UNIQUE` constraint, so a
        `NULL`-inclusive key would never actually deduplicate two
        anonymous/no-tenant submissions.

        Parameters
        ----------
        data : FeedbackWriteInput
            Already-validated, already-trust-resolved feedback fields.

        Returns
        -------
        tuple[str, bool]
            `(feedback_id, created)`; `created` is `True` for a brand new
            row, `False` when an existing row was updated instead.
        """
        new_id = str(uuid.uuid4())
        tenant_key = data.tenant_id or ""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table} (
                        feedback_id, tenant_id, caller_key, request_id, rating, reason,
                        comment, route, dataset_id, query_text, answer_text,
                        cited_source_ids, tool_calls, generation_model, prompt_id,
                        prompt_version, retrieval_provider, reranker_provider,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        now(), now()
                    )
                    ON CONFLICT (tenant_id, caller_key, request_id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        reason = EXCLUDED.reason,
                        comment = EXCLUDED.comment,
                        route = EXCLUDED.route,
                        dataset_id = EXCLUDED.dataset_id,
                        query_text = EXCLUDED.query_text,
                        answer_text = EXCLUDED.answer_text,
                        cited_source_ids = EXCLUDED.cited_source_ids,
                        tool_calls = EXCLUDED.tool_calls,
                        generation_model = EXCLUDED.generation_model,
                        prompt_id = EXCLUDED.prompt_id,
                        prompt_version = EXCLUDED.prompt_version,
                        retrieval_provider = EXCLUDED.retrieval_provider,
                        reranker_provider = EXCLUDED.reranker_provider,
                        updated_at = now()
                    RETURNING feedback_id, (xmax = 0) AS inserted
                    """,
                    (
                        new_id,
                        tenant_key,
                        data.caller_key,
                        data.request_id,
                        data.rating,
                        data.reason,
                        data.comment,
                        data.route,
                        data.dataset_id,
                        data.query_text,
                        data.answer_text,
                        data.cited_source_ids,
                        data.tool_calls,
                        data.generation_model,
                        data.prompt_id,
                        data.prompt_version,
                        data.retrieval_provider,
                        data.reranker_provider,
                    ),
                )
                row = cur.fetchone()
        assert row is not None  # RETURNING always yields exactly one row for this statement
        feedback_id, inserted = row
        return str(feedback_id), bool(inserted)

    def export(
        self,
        *,
        tenant_id: str | None = None,
        rating: FeedbackRating | None = None,
        dataset_id: str | None = None,
        route: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[FeedbackRecord]:
        """Read feedback rows for eval-curation export, newest first.

        Not exposed as a public API endpoint by design -- this milestone
        deliberately does not add a broadly-accessible `GET /feedback`;
        see `scripts/export_feedback.py`.

        Parameters
        ----------
        tenant_id, rating, dataset_id, route : str | None
            Exact-match filters; `None` means "no filter on this field."
        start_date, end_date : date | None
            Inclusive `created_at` date bounds.
        limit : int | None
            Maximum rows to return, most recent first.

        Returns
        -------
        list[FeedbackRecord]
        """
        clauses: list[str] = []
        params: list[object] = []
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            params.append(tenant_id)
        if rating is not None:
            clauses.append("rating = %s")
            params.append(rating)
        if dataset_id is not None:
            clauses.append("dataset_id = %s")
            params.append(dataset_id)
        if route is not None:
            clauses.append("route = %s")
            params.append(route)
        if start_date is not None:
            clauses.append("created_at >= %s")
            params.append(start_date)
        if end_date is not None:
            clauses.append("created_at < %s")
            params.append(end_date + timedelta(days=1))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(limit)

        columns_sql = ", ".join(_COLUMNS)
        query = (
            f"SELECT {columns_sql} FROM {self._table} {where} ORDER BY created_at DESC {limit_sql}"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    def delete(self, feedback_id: str) -> None:
        """Delete one feedback row by id. Test/admin cleanup use only."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table} WHERE feedback_id = %s", (feedback_id,))

    def health_check(self) -> bool:
        """Return whether the feedback table is reachable."""
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {self._table} LIMIT 1")
                cur.fetchone()
            return True
        except Exception:
            return False
