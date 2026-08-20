"""`POST /query`: answer a question via `RetrievalPipeline.answer`."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_rate_limiter, get_retrieval_pipeline
from rag.api.request_auth import build_authorization_context, enforce_dos_limits
from rag.config import AppConfig
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline

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
    """One retrieved-chunk citation in a `QueryResponse`."""

    chunk_id: str
    document_id: str
    source: str
    category: str | None = None
    score: float


class QueryResponse(BaseModel):
    """Response body for `POST /query`."""

    answer: str
    sources: list[SourceItem]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


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
    return QueryResponse(**result)
