"""`POST /query`: answer a question via `RetrievalPipeline.answer`."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_rate_limiter, get_retrieval_pipeline
from rag.audit import log_audit_event, pseudonymous_subject
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

    When `identity` is present, tenant and role claims come strictly from
    the verified JWT identity. If the request body claims different
    values, this logs a `forged_claim_attempt` event but does not use
    those body fields for authorization.

    When no verified identity is present, the function falls back to body
    fields. That path is used when JWT auth is disabled, or when insecure
    development mode allows a request without a token.

    Parameters
    ----------
    body : QueryRequest
        The parsed request body.
    identity : VerifiedIdentity | None
        The verified caller identity, or `None`.
    config : AppConfig
        Application configuration.

    Returns
    -------
    AuthorizationContext | None
        `None` means fully unrestricted retrieval.
    """
    if identity is not None:
        body_tenant_differs = body.tenant_id is not None and body.tenant_id != identity.tenant_id
        body_roles_differ = bool(body.roles) and set(body.roles or []) != set(identity.roles)
        if body_tenant_differs or body_roles_differ:
            log_audit_event(
                "forged_claim_attempt",
                subject=pseudonymous_subject(identity.subject),
                verified_tenant_id=identity.tenant_id,
                body_tenant_id=body.tenant_id,
            )
        return AuthorizationContext(
            tenant_id=identity.tenant_id,
            roles=identity.roles,
            as_of=body.as_of,
            require_trust_level=body.require_trust_level,
        )

    has_auth_fields = (
        body.tenant_id is not None
        or body.roles
        or body.as_of is not None
        or body.require_trust_level is not None
    )
    return (
        AuthorizationContext(
            tenant_id=body.tenant_id,
            roles=body.roles or [],
            as_of=body.as_of,
            require_trust_level=body.require_trust_level,
        )
        if has_auth_fields
        else None
    )


def _enforce_dos_limits(body: QueryRequest, config: AppConfig) -> None:
    """Reject oversized requests with a 422, per `security.dos_limits`.

    Parameters
    ----------
    body : QueryRequest
        The parsed request body.
    config : AppConfig
        Application configuration.

    Raises
    ------
    HTTPException
        422, when `query`/`top_k`/`filters` exceed their configured bound.
    """
    limits = config.security.dos_limits
    if len(body.query) > limits.max_query_length:
        log_audit_event("oversized_request_rejected", field="query", limit=limits.max_query_length)
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds maximum length of {limits.max_query_length} characters",
        )
    if body.top_k is not None and body.top_k > limits.max_top_k:
        log_audit_event("oversized_request_rejected", field="top_k", limit=limits.max_top_k)
        raise HTTPException(status_code=422, detail=f"top_k exceeds maximum of {limits.max_top_k}")
    if body.filters is not None:
        filters_bytes = len(json.dumps(body.filters).encode("utf-8"))
        if filters_bytes > limits.max_filters_bytes:
            log_audit_event(
                "oversized_request_rejected", field="filters", limit=limits.max_filters_bytes
            )
            raise HTTPException(
                status_code=422,
                detail=f"filters exceeds maximum size of {limits.max_filters_bytes} bytes",
            )


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
