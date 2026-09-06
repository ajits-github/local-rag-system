"""`POST /query`: answer a question via `RetrievalPipeline.answer`."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_rate_limiter, get_retrieval_pipeline
from rag.api.request_auth import build_authorization_context, enforce_dos_limits
from rag.config import AppConfig
from rag.logging_config import get_request_id
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

router = APIRouter()
_limiter = get_rate_limiter()


class QueryRequest(BaseModel):
    """Request body for `POST /query`.

    `tenant_id`/`roles` are only trusted as caller-identity claims when
    `config.security.auth.enabled` is `False` (the system default) or when
    `security.auth.insecure_dev_mode` is explicitly `True` and no JWT was
    supplied. Whenever a verified `Authorization: Bearer <jwt>` identity is
    present, these two fields are ignored for authorization; the verified
    JWT claims are used instead. `as_of` and `require_trust_level` are
    caller-supplied query parameters rather than identity claims, so they
    are honored either way.
    """

    query: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None
    tenant_id: str | None = None
    roles: list[str] | None = None
    as_of: date | None = None
    require_trust_level: str | None = None


class SourceItem(BaseModel):
    """One retrieved-chunk citation in a `QueryResponse`.

    `content_type`/`section_path`/`page`/`attachment_name`/`source_anchor`/
    `vision_generated` are derived, non-sensitive structural metadata
    already stored on the chunk. `source`/`attachment_name`/`source_anchor`
    are relative, dataset-root-scoped paths, never absolute filesystem paths.
    """

    chunk_id: str
    document_id: str
    source: str
    category: str | None = None
    score: float
    content_type: str | None = None
    section_path: str | None = None
    page: int | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    vision_generated: bool = False


class QueryResponse(BaseModel):
    """Response body for `POST /query`.

    `request_id` is the same correlation id already echoed on the
    `x-request-id` response header (see `api/middleware.py`), sourced
    from the identical per-request contextvar
    (`rag.logging_config.get_request_id`). Exposed here too so a caller
    (the web UI's `POST /feedback` flow) can tie a feedback submission to
    the exact answer/run it rated without a separate lookup.
    """

    answer: str
    sources: list[SourceItem]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    request_id: str | None = None


def _build_authorization_context(
    body: QueryRequest, identity: VerifiedIdentity | None, config: AppConfig
) -> AuthorizationContext | None:
    """Build the `AuthorizationContext` for this request.

    Thin wrapper over `rag.api.request_auth.build_authorization_context`
    (shared with `routers/agent_query.py`) that unpacks `QueryRequest`'s
    fields; see that function for the full precedence/forged-claim
    behavior.
    """
    return build_authorization_context(
        identity, body.tenant_id, body.roles, body.as_of, body.require_trust_level
    )


def _enforce_dos_limits(body: QueryRequest, config: AppConfig) -> None:
    """Reject oversized requests with a 422, per `security.dos_limits`.

    Thin wrapper over `rag.api.request_auth.enforce_dos_limits` (shared
    with `routers/agent_query.py`) that unpacks `QueryRequest`'s fields.
    """
    enforce_dos_limits(body.query, body.top_k, body.filters, config)


def _query_rate_limit_string() -> str:
    """Return the current `requests_per_minute` config value as a slowapi limit string."""
    return f"{get_config().security.rate_limit.requests_per_minute}/minute"


@router.post("/query", response_model=QueryResponse)
@_limiter.limit(_query_rate_limit_string)
def query(
    request: Request,
    body: QueryRequest,
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    config: AppConfig = Depends(get_config),
) -> QueryResponse:
    """Run `body.query` through the retrieval pipeline and return the answer.

    Parameters
    ----------
    request : Request
        The raw HTTP request, required by `slowapi`'s rate-limit
        decorator and passed to `get_current_identity`.
    body : QueryRequest
        The query, plus optional `top_k`/`filters` overrides. `top_k`
        maps onto `RetrievalPipeline.answer`'s `candidate_k`; the public
        API field name remains stable.
    identity : VerifiedIdentity | None
        The verified caller identity (see `get_current_identity`), or
        `None` when JWT auth is disabled.
    pipeline : RetrievalPipeline
        Injected retrieval pipeline singleton.
    config : AppConfig
        Application configuration.

    Returns
    -------
    QueryResponse
        The generated answer, its sources, and stage timings.
    """
    _enforce_dos_limits(body, config)
    auth = _build_authorization_context(body, identity, config)
    result = pipeline.answer(body.query, filters=body.filters, candidate_k=body.top_k, auth=auth)
    # Same event name/shape as the agent routes' run-completion log
    # (rag.agent.graph._finish); never includes query/answer text.
    logger.info(
        "agent_request_completed",
        extra={
            "route": "classic_rag",
            "termination_reason": "synthesized",
            "step_count": 1,
            "tool_call_count": 0,
            "retrieval_attempts": None,
            "evidence_sufficient": None,
            "duration_ms": round(result["total_ms"], 2),
            "success": True,
        },
    )
    return QueryResponse(**result, request_id=get_request_id())
