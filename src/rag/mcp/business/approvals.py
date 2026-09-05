"""Resolves pre-authorized case-status approvals from an MCP call's transport.

Independent of `rag.mcp.identity`: identity resolution establishes *who*
is calling, this resolves one additional, optional claim minted only by
`rag.agent.mcp_client.mint_internal_token` (`case_approvals`), so the
general identity model never carries a business-specific concept. Never
reads a tool argument; a malformed or unverifiable token yields no
approvals rather than raising, since identity resolution on the same
call already fails the request closed for a genuinely invalid token.

A verifiable JWT alone is not enough to grant an approval: this deployment's
tokens are minted and verified in the same trust domain (see `JWTConfig`'s
docstring), so any caller able to mint themselves an ordinary token could
otherwise attach a `case_approvals` claim to it directly and bypass
`rag.api.request_auth.resolve_case_approvals`'s role gate entirely by
calling the MCP server directly instead of `POST /agent/query`. `case_
approvals` is honored only on a token that is unambiguously an internal
service token (`sub`/`token_use` match `mint_internal_token`'s fixed
markers), and only when that same token's own `roles` claim -- the
original caller's already-role-gated roles, embedded by `mint_internal_
token` -- still intersects `mcp.business_actions.approval_roles`.
"""

from __future__ import annotations

from collections.abc import Mapping

import jwt
from pydantic import ValidationError

from rag.config import AppConfig
from rag.mcp.business.schemas import MAX_CASE_APPROVALS, CaseApproval


def resolve_case_action_approvals(
    headers: Mapping[str, str] | None, config: AppConfig
) -> list[CaseApproval]:
    """Extract this call's `case_approvals` claim from its bearer token, if any.

    Parameters
    ----------
    headers : Mapping[str, str] | None
        The current MCP request's transport headers.
    config : AppConfig
        Application configuration; reads `security.auth.jwt` for the
        same signing key/algorithm/issuer/audience `verify_jwt` uses.

    Returns
    -------
    list[CaseApproval]
        The approved `(case_id, new_status)` pairs, or an empty list
        when auth is disabled, no token is present, the token doesn't
        verify, the token is not recognized as an internal service
        token, that token's own `roles` claim holds no approval role,
        or the claim is missing or malformed.
    """
    if not config.security.auth.enabled:
        return []

    header = None
    if headers is not None:
        header = headers.get("authorization") or headers.get("Authorization")
    if header is None:
        return []

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return []

    jwt_config = config.security.auth.jwt
    try:
        claims = jwt.decode(
            token,
            config.jwt_signing_key(),
            algorithms=[jwt_config.algorithm],
            issuer=jwt_config.issuer,
            audience=jwt_config.audience,
            leeway=jwt_config.leeway_seconds,
            options={"require": []},
        )
    except jwt.InvalidTokenError:
        return []

    client_cfg = config.mcp.client
    if claims.get("sub") != client_cfg.internal_token_subject:
        return []
    if claims.get("token_use") != "mcp_internal_service":
        return []

    approval_roles = set(config.mcp.business_actions.approval_roles)
    token_roles = set(claims.get("roles") or [])
    if not token_roles & approval_roles:
        return []

    raw = claims.get("case_approvals") or []
    if not isinstance(raw, list):
        return []

    approvals: list[CaseApproval] = []
    for item in raw[:MAX_CASE_APPROVALS]:
        try:
            approvals.append(CaseApproval.model_validate(item))
        except ValidationError:
            continue
    return approvals
