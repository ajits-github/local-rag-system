"""Shared feedback vocabulary and the persisted-row representation.

The rating/reason literals live here (not in `api/routers/feedback.py`)
so `FeedbackStore`/`scripts/export_feedback.py` can reuse the exact same
values without importing from the API layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

FeedbackRating = Literal["positive", "negative"]

NEGATIVE_FEEDBACK_REASONS: tuple[str, ...] = (
    "incorrect_answer",
    "outdated_information",
    "unsupported_answer",
    "wrong_or_missing_citation",
    "incomplete_answer",
    "irrelevant_answer",
    "other",
)

POSITIVE_FEEDBACK_REASONS: tuple[str, ...] = (
    "accurate_and_helpful",
    "well_cited",
    "other",
)


class FeedbackRecord(BaseModel):
    """One persisted feedback row, as read back for export/curation.

    Mirrors the `feedback` table's columns
    (`scripts/init_db.py:build_feedback_schema_sql`). `query_text`/
    `answer_text` are the caller-echoed text the original caller already
    received (see `FeedbackConfig.store_query_text`/`store_answer_text`);
    this is never treated as ground truth -- see `docs/architecture.md`'s
    "Feedback loop" section and `scripts/export_feedback.py`'s own
    docstring for the deliberate human-review step before any promotion
    into gold eval data.

    Attributes
    ----------
    feedback_id : str
        Stable row identifier; unchanged across an update (see
        `FeedbackStore.submit`).
    created_at, updated_at : datetime
        First-submission and most-recent-update timestamps.
    tenant_id : str | None
        The trusted tenant the caller belonged to, or `None` when no
        tenant was asserted (`security.auth.enabled=False` and no
        caller-supplied tenant either).
    caller_key : str
        Pseudonymous caller identifier (see
        `api/routers/feedback.py:_resolve_caller_key`); never a raw JWT
        subject.
    request_id : str
        The correlation id from the original answer's response (see
        `QueryResponse.request_id`/`AgentQueryResponse.request_id`).
    rating : FeedbackRating
        `"positive"` or `"negative"`.
    reason : str | None
        One of `NEGATIVE_FEEDBACK_REASONS`/`POSITIVE_FEEDBACK_REASONS`,
        matching `rating`, or `None`.
    comment : str | None
        Bounded free-text comment.
    route : str | None
        `"classic_rag"` or `"agent"`, as reported by the original answer.
    dataset_id : str | None
        The dataset the original query was scoped to, when known.
    query_text, answer_text : str | None
        Caller-echoed text of the original query/answer; `None` when the
        corresponding `FeedbackConfig.store_*_text` toggle is off.
    cited_source_ids : list[str]
        Chunk ids the caller's answer cited, as displayed to them.
    tool_calls : list[str]
        Tool names dispatched for an agentic answer (public metadata
        already returned by `AgentQueryResponse.tool_calls`; never
        reasoning or raw tool arguments).
    generation_model, prompt_id, prompt_version, retrieval_provider,
    reranker_provider : str | None
        A best-effort lineage snapshot of `AppConfig` at the moment this
        feedback was submitted (not necessarily identical to the config
        active when the original answer was generated, since no
        per-answer metadata store exists yet -- see
        `docs/architecture.md`'s "Feedback loop" section).
    """

    feedback_id: str
    created_at: datetime
    updated_at: datetime
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
