"""`POST /feedback`: persist a caller's rating of a prior answer.

Adds a closed feedback loop on top of the existing query/agent routes:

    answer -> thumbs up/down (+ optional structured reason/comment)
    -> this endpoint -> Postgres (`rag.feedback.store.FeedbackStore`)
    -> a script-driven export for human review
    -> optional, deliberate promotion into gold eval data

See `scripts/export_feedback.py` and `docs/architecture.md`'s "Feedback
loop" section for the read/curation side. Feedback is never auto-promoted
to ground truth by this endpoint or anything it calls.

Identity/tenant handling mirrors `routers/query.py`: a verified JWT
identity (when `security.auth.enabled=True`) always wins over a
body-supplied `tenant_id`; a mismatch is logged as `forged_claim_attempt`
but never used (`rag.api.request_auth.resolve_trusted_tenant_id`).
`request_id` is the correlation id every query/agent response already
carries (`QueryResponse.request_id`/`AgentQueryResponse.request_id`,
sourced from the same value as the `x-request-id` response header) --
reused rather than inventing a new "answer id" concept. This endpoint
does not validate that `request_id` refers to a real prior request: no
server-side request log exists to check against, and adding one solely
for this purpose would be a much larger change than this milestone's
scope. `request_id` is therefore treated as an opaque, format-bounded
correlation string a caller asserts, not an existence-verified reference
-- a deliberate, documented choice, not an oversight.

Duplicate/update semantics: one feedback row per
`(tenant_id, caller_key, request_id)`, enforced by a database `UNIQUE`
constraint and an `INSERT ... ON CONFLICT DO UPDATE`
(`FeedbackStore.submit`). `caller_key` is the verified JWT subject
(pseudonymized, never the raw claim) when an identity is present, or the
fixed literal `"anonymous"` otherwise -- there is no session/cookie
concept in this stateless API to key on instead, so two different
anonymous callers submitting feedback on the very same `request_id`
(a per-response UUID never shown to anyone but the original caller)
would share one row. This is a deliberate, documented limitation of
operating with `security.auth.enabled=False`, not an oversight: a
stable, distinguishable caller identity requires a verified identity,
exactly like every other identity-dependent guarantee in this codebase.
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_feedback_store
from rag.api.request_auth import resolve_trusted_tenant_id
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig
from rag.feedback.schemas import (
    NEGATIVE_FEEDBACK_REASONS,
    POSITIVE_FEEDBACK_REASONS,
    FeedbackRating,
)
from rag.feedback.store import FeedbackStore, FeedbackWriteInput
from rag.observability.metrics import observe_error, observe_feedback_submission

router = APIRouter()

_ALL_REASONS = set(NEGATIVE_FEEDBACK_REASONS) | set(POSITIVE_FEEDBACK_REASONS)

# A hard structural bound for identifier-shaped fields (tenant/dataset ids,
# chunk ids, tool names) -- unlike comment/query/answer, these have no
# config-driven runtime limit, so without this a caller could still send an
# unbounded string into an indexed Postgres column via one of these fields.
_IdentifierStr = Annotated[str, Field(max_length=200)]


class FeedbackRequest(BaseModel):
    """Request body for `POST /feedback`.

    `extra="forbid"` (a deviation from `QueryRequest`'s permissive
    default): this is a narrow, security-relevant write path rather than
    a flexible query surface, so an unrecognized field is rejected
    instead of silently ignored.

    `tenant_id` follows the same precedence as `QueryRequest.tenant_id`
    (see `rag.api.request_auth.resolve_trusted_tenant_id`): only trusted
    when no verified JWT identity is present.

    Nothing here carries chain-of-thought, raw prompts, hidden tool
    reasoning, a JWT, or an `AuthorizationContext` -- only what a caller
    already saw in a query/agent response (`answer`, `query`,
    `cited_source_ids`, `tool_calls` -- the latter is just tool *names*,
    already public on `AgentQueryResponse.tool_calls`, never reasoning or
    tool arguments).
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=100)
    rating: FeedbackRating
    reason: str | None = None
    comment: str | None = None
    query: str | None = None
    answer: str | None = None
    route: Literal["classic_rag", "agent"] | None = None
    dataset_id: _IdentifierStr | None = None
    cited_source_ids: list[_IdentifierStr] | None = None
    tool_calls: list[_IdentifierStr] | None = None
    tenant_id: _IdentifierStr | None = None

    @model_validator(mode="after")
    def _reason_matches_rating(self) -> FeedbackRequest:
        """Reject a reason from the wrong rating's enum, or one outside the fixed vocabulary.

        A small, stable, closed vocabulary per this milestone's own
        instruction: a positive rating cannot carry a negative-only
        reason like `incorrect_answer`, and vice versa.
        """
        if self.reason is None:
            return self
        if self.reason not in _ALL_REASONS:
            raise ValueError(f"Unrecognized feedback reason: {self.reason!r}")
        if self.rating == "positive" and self.reason not in POSITIVE_FEEDBACK_REASONS:
            raise ValueError(f"Reason {self.reason!r} is not valid for positive feedback")
        if self.rating == "negative" and self.reason not in NEGATIVE_FEEDBACK_REASONS:
            raise ValueError(f"Reason {self.reason!r} is not valid for negative feedback")
        return self


class FeedbackAck(BaseModel):
    """Response body for `POST /feedback`."""

    feedback_id: str
    status: Literal["created", "updated"]


def _enforce_feedback_limits(body: FeedbackRequest, config: AppConfig) -> None:
    """Reject an oversized feedback submission with a 422.

    Mirrors `rag.api.request_auth.enforce_dos_limits`'s own reasoning:
    the bounds must read from runtime config, not a value baked into the
    Pydantic model at class-definition time.
    """
    limits = config.feedback
    if body.comment is not None and len(body.comment) > limits.max_comment_length:
        log_audit_event(
            "oversized_request_rejected", field="feedback_comment", limit=limits.max_comment_length
        )
        raise HTTPException(
            status_code=422,
            detail=f"comment exceeds maximum length of {limits.max_comment_length} characters",
        )
    if body.query is not None and len(body.query) > limits.max_query_text_length:
        log_audit_event(
            "oversized_request_rejected", field="feedback_query", limit=limits.max_query_text_length
        )
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds maximum length of {limits.max_query_text_length} characters",
        )
    if body.answer is not None and len(body.answer) > limits.max_answer_text_length:
        log_audit_event(
            "oversized_request_rejected",
            field="feedback_answer",
            limit=limits.max_answer_text_length,
        )
        raise HTTPException(
            status_code=422,
            detail=f"answer exceeds maximum length of {limits.max_answer_text_length} characters",
        )
    if body.cited_source_ids is not None and len(body.cited_source_ids) > limits.max_cited_sources:
        log_audit_event(
            "oversized_request_rejected", field="cited_source_ids", limit=limits.max_cited_sources
        )
        raise HTTPException(
            status_code=422,
            detail=f"cited_source_ids exceeds maximum of {limits.max_cited_sources} entries",
        )
    if body.tool_calls is not None and len(body.tool_calls) > limits.max_cited_sources:
        log_audit_event(
            "oversized_request_rejected", field="tool_calls", limit=limits.max_cited_sources
        )
        raise HTTPException(
            status_code=422,
            detail=f"tool_calls exceeds maximum of {limits.max_cited_sources} entries",
        )


def _require_feedback_enabled(config: AppConfig = Depends(get_config)) -> AppConfig:
    """Reject with 404 before any other dependency resolves, when feedback is disabled.

    Declared as `submit_feedback`'s first non-body dependency, mirroring
    `agent_stream.py`'s `_build_validated_agent_state` precedent: FastAPI
    resolves `Depends()` parameters in signature order and stops at the
    first one that raises. Without this, `get_feedback_store()` (an
    `lru_cache`d singleton that eagerly opens a Postgres connection pool in
    `FeedbackStore.__init__`) would still be resolved on every call even
    when `config.feedback.enabled=False`, so a deployment that intends
    feedback to be fully absent could still fail on an unreachable database
    instead of cleanly 404ing -- the same "disabled means never touched"
    guarantee `mcp.enabled=False` already gives `/mcp`.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    AppConfig
        The same config, passed through so the route doesn't need a second
        `Depends(get_config)`.

    Raises
    ------
    HTTPException
        404, when `config.feedback.enabled` is `False`.
    """
    if not config.feedback.enabled:
        raise HTTPException(status_code=404, detail="Feedback collection is disabled")
    return config


def _resolve_caller_key(identity: VerifiedIdentity | None) -> str:
    """Return a stable, non-reversible key identifying the caller, for dedup purposes.

    The verified JWT subject's pseudonymous hash when an identity is
    present, otherwise the fixed literal `"anonymous"` -- see this
    module's docstring for why an unauthenticated caller cannot get a
    stronger guarantee than that in this stateless API.
    """
    if identity is None:
        return "anonymous"
    return pseudonymous_subject(identity.subject)


@router.post("/feedback", response_model=FeedbackAck)
def submit_feedback(
    body: FeedbackRequest,
    config: AppConfig = Depends(_require_feedback_enabled),
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackAck:
    """Persist one rating of a prior answer, tied to its `request_id`.

    Parameters
    ----------
    body : FeedbackRequest
        The rating, plus optional reason/comment/lineage fields.
    config : AppConfig
        Application configuration, resolved (and enabled-checked) by
        `_require_feedback_enabled` ahead of `identity`/`store`.
    identity : VerifiedIdentity | None
        The verified caller identity (see `get_current_identity`), or
        `None` when JWT auth is disabled.
    store : FeedbackStore
        Injected feedback-store singleton.

    Returns
    -------
    FeedbackAck
        The stored feedback row's id and whether it was newly created or
        replaced an earlier submission for the same run.

    Raises
    ------
    HTTPException
        404, when `config.feedback.enabled` is `False` (raised by
        `_require_feedback_enabled` before `identity`/`store` ever
        resolve); 422, when a field exceeds its configured bound; 503, on
        a feedback-storage failure.
    """
    _enforce_feedback_limits(body, config)

    tenant_id = resolve_trusted_tenant_id(identity, body.tenant_id)
    caller_key = _resolve_caller_key(identity)
    generation = config.generation
    write_input = FeedbackWriteInput(
        tenant_id=tenant_id,
        caller_key=caller_key,
        request_id=body.request_id,
        rating=body.rating,
        reason=body.reason,
        comment=body.comment,
        route=body.route,
        dataset_id=body.dataset_id,
        query_text=body.query if config.feedback.store_query_text else None,
        answer_text=body.answer if config.feedback.store_answer_text else None,
        cited_source_ids=body.cited_source_ids or [],
        tool_calls=body.tool_calls or [],
        generation_model=generation.model_name,
        prompt_id=generation.prompt.id,
        prompt_version=generation.prompt.version,
        retrieval_provider=config.retrieval.provider,
        reranker_provider=config.reranker.provider,
    )

    start = time.monotonic()
    try:
        feedback_id, created = store.submit(write_input)
    except Exception:
        observe_feedback_submission(body.rating, "failed", time.monotonic() - start)
        observe_error("feedback")
        raise HTTPException(
            status_code=503, detail="Feedback storage is temporarily unavailable"
        ) from None

    latency = time.monotonic() - start
    observe_feedback_submission(body.rating, "created" if created else "updated", latency)
    log_audit_event(
        "feedback_submitted",
        tenant_id=tenant_id,
        rating=body.rating,
        route=body.route,
        has_comment=bool(body.comment),
        outcome="created" if created else "updated",
    )
    return FeedbackAck(feedback_id=feedback_id, status="created" if created else "updated")
