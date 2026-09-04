"""Agent-side MCP client for the two remote business tools (MCP Stage 2).

Every other agent tool (`search_knowledge_base`/`get_document`/
`get_latest_document`/`get_related_context`) stays a direct, in-process
call into `rag.agent.tools`. This module exists only for
`get_customer_case`/`get_case_status`, dispatched as a real MCP client
call against the Stage 1B business tools served by `rag.mcp.server`,
never a shortcut direct function call.

Security posture:

- Fails closed without an authenticated caller identity.
  `validate_startup_config` refuses to start the process at all when
  `mcp.client.enabled=True` and `security.auth.enabled=False`, since the
  business tools' own tenant/role authorization has no kill-switch.
  `dispatch_remote_tool_sync` additionally refuses any call whose
  resolved `AuthorizationContext` has no `tenant_id`.
- The caller's original inbound JWT is never forwarded. Each call mints
  a fresh, short-lived internal service token (`mint_internal_token`)
  from the already-verified `AuthorizationContext` on `AgentState`,
  signed with the same `security.auth.jwt` secret the receiving
  `rag.mcp.identity.resolve_http_identity` verifies with. Only `HS256`
  is supported for minting, since it is the only algorithm this
  codebase holds a signing (not just verifying) key for.
- One fresh `mcp.ClientSession` per remote tool call; never pooled or
  reused across calls or requests.
- The returned evidence is a synthetic `SearchResult`
  (`origin="mcp_remote"`, a fabricated `mcp://business/...`
  `source`/`chunk_id`), never fetched from `VectorStore` and never
  passed through `RetrievalPipeline.retrieve()`/
  `expand_with_relationships()`/`resolve_auth()`, so document-specific
  behaviors (freshness resolution, relationship expansion, document-level
  ACL) structurally never apply to it. Authorization for this evidence is
  already fully enforced server-side, inside `rag.mcp.business.store`,
  before this module ever sees a result.
"""

from __future__ import annotations

import time

import jwt

from rag.config import AppConfig
from rag.retrieval.authorization import AuthorizationContext


def validate_startup_config(config: AppConfig) -> None:
    """Fail fast at process startup for an invalid `mcp.client` configuration.

    Called once from `rag.api.main` at import time. A no-op when
    `mcp.client.enabled` is `False` (the default).

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Raises
    ------
    RuntimeError
        If `mcp.client.enabled=True` and any of the following hold:
        `security.auth.enabled=False` (no trustworthy identity to attach
        to a business-tool call); `security.auth.jwt.algorithm` is not
        `HS256` (internal token minting needs a symmetric signing key,
        and RS256/ES256 config only holds a public, verify-only key);
        or `mcp.client.transport="asgi"` while `mcp.enabled=False` (the
        in-process transport has no server object to bind to).
    """
    client_cfg = config.mcp.client
    if not client_cfg.enabled:
        return
    if not config.security.auth.enabled:
        raise RuntimeError(
            "mcp.client.enabled=True requires security.auth.enabled=True: the Stage "
            "1B business tools (get_customer_case/get_case_status) are tenant/role "
            "protected with no kill-switch, so there is no trustworthy identity to "
            "attach to an agent-originated call when authentication is disabled. "
            "Enable security.auth, or disable mcp.client."
        )
    if config.security.auth.jwt.algorithm != "HS256":
        raise RuntimeError(
            f"mcp.client.enabled=True with security.auth.enabled=True requires "
            f"security.auth.jwt.algorithm='HS256', got "
            f"{config.security.auth.jwt.algorithm!r}: internal service-token minting "
            "needs a symmetric signing key, and RS256/ES256 config only holds a "
            "public (verify-only) key -- see AppConfig.jwt_signing_key. Configure "
            "HS256, or disable mcp.client."
        )
    if client_cfg.transport == "asgi" and not config.mcp.enabled:
        raise RuntimeError(
            "mcp.client.transport='asgi' requires mcp.enabled=True: the in-process "
            "ASGI transport calls the same MCP server object main.py would otherwise "
            "mount, so one must actually be built. Set mcp.enabled=True, or switch "
            "mcp.client.transport to 'http' with a real server_url."
        )


def mint_internal_token(auth: AuthorizationContext, config: AppConfig) -> str:
    """Mint a fresh, short-lived internal service token for one remote MCP call.

    Never forwards the caller's original inbound JWT; always a new
    token, signed here, carrying only the already-verified
    `tenant_id`/`roles` from `auth`. Reuses `config.security.auth.jwt`'s
    configured secret, verified by the same `verify_jwt` every other
    caller of this deployment's MCP server or `/query` boundary goes
    through, with no special-cased relaxation for this token.

    `iss` is always set (to `security.auth.jwt.issuer` when configured,
    else `mcp.client.internal_token_issuer`), since an unchecked `iss`
    claim on a token is never inspected by `verify_jwt`. `aud` is
    included only when `security.auth.jwt.audience` is actually
    configured: PyJWT rejects any token carrying an `aud` claim when the
    verifier expects none, so setting an "informational" audience
    unconditionally would make every internal token fail verification. A
    `token_use` claim marks this as an internal service token, not
    itself checked by `verify_jwt`, useful for anyone inspecting a
    decoded token or log line.

    Parameters
    ----------
    auth : AuthorizationContext
        The caller's already-resolved authorization context. Must carry
        a non-`None` `tenant_id`; see `dispatch_remote_tool_sync`,
        which checks this before ever calling here.
    config : AppConfig
        Application configuration.

    Returns
    -------
    str
        The signed JWT, ready for an `Authorization: Bearer <token>`
        header.

    Raises
    ------
    RuntimeError
        If `security.auth.jwt.algorithm` is not `HS256`. Defense in
        depth: `validate_startup_config` already refuses to start the
        process in this state, so this should be unreachable in
        practice.
    """
    jwt_config = config.security.auth.jwt
    if jwt_config.algorithm != "HS256":
        raise RuntimeError(
            f"Cannot mint an internal MCP service token: "
            f"security.auth.jwt.algorithm={jwt_config.algorithm!r} is not 'HS256'."
        )
    client_cfg = config.mcp.client
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": client_cfg.internal_token_subject,
        "tenant_id": auth.tenant_id,
        "roles": list(auth.roles),
        "iat": now,
        "exp": now + client_cfg.internal_token_ttl_seconds,
        "iss": jwt_config.issuer or client_cfg.internal_token_issuer,
        "token_use": "mcp_internal_service",
    }
    # aud is deliberately omitted, not set to an "informational" fallback, when
    # jwt_config.audience is unset; see the docstring's note on PyJWT's
    # asymmetric iss/aud handling.
    if jwt_config.audience:
        claims["aud"] = jwt_config.audience
    key = config.jwt_signing_key()
    return jwt.encode(claims, key, algorithm="HS256")
