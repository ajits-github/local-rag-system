"""Shared request-boundary security checks used by every query-shaped router.

Extracted from `routers/query.py` so `agent_query.py`/`agent_stream.py`
reuse the exact same, already-tested JWT-precedence and DoS-limit logic
rather than re-implementing it.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from fastapi import HTTPException

from rag.api.auth import VerifiedIdentity
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig
from rag.mcp.business.schemas import CaseApproval
from rag.retrieval.authorization import AuthorizationContext


def build_authorization_context(
    identity: VerifiedIdentity | None,
    tenant_id: str | None,
    roles: list[str] | None,
    as_of: date | None,
    require_trust_level: str | None,
) -> AuthorizationContext | None:
    """Build the `AuthorizationContext` for one request.

    When `identity` is present, tenant and role claims come strictly from
    the verified JWT identity. If the caller's own `tenant_id`/`roles`
    disagree, this logs a `forged_claim_attempt` event but does not use
    those values for authorization.

    When no verified identity is present, falls back to the caller-
    supplied `tenant_id`/`roles`/`as_of`/`require_trust_level`. That path
    is used when JWT auth is disabled, or when insecure development mode
    allows a request without a token.

    Parameters
    ----------
    identity : VerifiedIdentity | None
        The verified caller identity, or `None`.
    tenant_id, roles : str | None, list[str] | None
        Caller-supplied identity claims (only trusted when `identity` is
        `None`. See above).
    as_of : date | None
        Freshness resolution date; always honored (a query parameter,
        not an identity claim).
    require_trust_level : str | None
        Required trust level; always honored, same reasoning as `as_of`.

    Returns
    -------
    AuthorizationContext | None
        `None` means fully unrestricted retrieval.
    """
    if identity is not None:
        body_tenant_differs = tenant_id is not None and tenant_id != identity.tenant_id
        body_roles_differ = bool(roles) and set(roles or []) != set(identity.roles)
        if body_tenant_differs or body_roles_differ:
            log_audit_event(
                "forged_claim_attempt",
                subject=pseudonymous_subject(identity.subject),
                verified_tenant_id=identity.tenant_id,
                body_tenant_id=tenant_id,
            )
        return AuthorizationContext(
            tenant_id=identity.tenant_id,
            roles=identity.roles,
            as_of=as_of,
            require_trust_level=require_trust_level,
        )

    has_auth_fields = (
        tenant_id is not None or roles or as_of is not None or require_trust_level is not None
    )
    return (
        AuthorizationContext(
            tenant_id=tenant_id,
            roles=roles or [],
            as_of=as_of,
            require_trust_level=require_trust_level,
        )
        if has_auth_fields
        else None
    )


def enforce_dos_limits(
    query: str, top_k: int | None, filters: dict[str, Any] | None, config: AppConfig
) -> None:
    """Reject oversized requests with a 422, per `security.dos_limits`.

    Parameters
    ----------
    query : str
        The request's query text.
    top_k : int | None
        The request's `top_k`, if supplied.
    filters : dict[str, Any] | None
        The request's `filters`, if supplied.
    config : AppConfig
        Application configuration.

    Raises
    ------
    HTTPException
        422, when `query`/`top_k`/`filters` exceed their configured bound,
        or when `top_k` is less than 1.
    """
    limits = config.security.dos_limits
    if len(query) > limits.max_query_length:
        log_audit_event("oversized_request_rejected", field="query", limit=limits.max_query_length)
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds maximum length of {limits.max_query_length} characters",
        )
    if top_k is not None and top_k < 1:
        log_audit_event("oversized_request_rejected", field="top_k", reason="below_minimum")
        raise HTTPException(status_code=422, detail="top_k must be >= 1")
    if top_k is not None and top_k > limits.max_top_k:
        log_audit_event("oversized_request_rejected", field="top_k", limit=limits.max_top_k)
        raise HTTPException(status_code=422, detail=f"top_k exceeds maximum of {limits.max_top_k}")
    if filters is not None:
        filters_bytes = len(json.dumps(filters).encode("utf-8"))
        if filters_bytes > limits.max_filters_bytes:
            log_audit_event(
                "oversized_request_rejected", field="filters", limit=limits.max_filters_bytes
            )
            raise HTTPException(
                status_code=422,
                detail=f"filters exceeds maximum size of {limits.max_filters_bytes} bytes",
            )


def enforce_case_approval_limits(
    case_approvals: list[CaseApproval] | None, config: AppConfig
) -> None:
    """Reject a request carrying more `case_approvals` than the configured maximum.

    A runtime-config-driven check (not a Pydantic field constraint),
    matching `enforce_dos_limits`'s own reasoning: the bound must read
    from `config`, not a value baked into the request model at
    class-definition time. Bounds the number of approvals that can reach
    `mint_internal_token`'s `case_approvals` claim.

    Parameters
    ----------
    case_approvals : list[CaseApproval] | None
        The request's `case_approvals` field, if supplied.
    config : AppConfig
        Application configuration; reads
        `mcp.business_actions.max_case_approvals_per_request`.

    Raises
    ------
    HTTPException
        422, when `case_approvals` exceeds the configured maximum.
    """
    if not case_approvals:
        return
    limit = config.mcp.business_actions.max_case_approvals_per_request
    if len(case_approvals) > limit:
        log_audit_event("oversized_request_rejected", field="case_approvals", limit=limit)
        raise HTTPException(
            status_code=422, detail=f"case_approvals exceeds maximum of {limit} entries"
        )


def resolve_case_approvals(
    identity: VerifiedIdentity | None,
    case_approvals: list[CaseApproval] | None,
    config: AppConfig,
) -> list[CaseApproval]:
    """Return `case_approvals` only when the verified caller holds an approval role.

    Never trusts an unauthenticated caller or one lacking a configured
    approval role; a rejected attempt is audit-logged (reusing
    `authorization_denied`, the same event other elevated-capability
    denials use) but not treated as an error -- the request still
    proceeds, simply without the requested approval attached.

    Parameters
    ----------
    identity : VerifiedIdentity | None
        The verified caller identity, or `None`.
    case_approvals : list[CaseApproval] | None
        The request's `case_approvals` field, if supplied.
    config : AppConfig
        Application configuration; reads
        `mcp.business_actions.approval_roles`.

    Returns
    -------
    list[CaseApproval]
        `case_approvals` unchanged if the caller is authorized to grant
        it, otherwise an empty list.
    """
    if not case_approvals:
        return []
    if identity is None:
        log_audit_event(
            "authorization_denied", action="case_status_approval", reason="unauthenticated"
        )
        return []
    approval_roles = set(config.mcp.business_actions.approval_roles)
    if not set(identity.roles) & approval_roles:
        log_audit_event(
            "authorization_denied",
            action="case_status_approval",
            subject=pseudonymous_subject(identity.subject),
            reason="missing_approval_role",
        )
        return []
    return case_approvals
