"""MCP-transport identity resolution: transport-owned, never tool-argument-owned.

Mirrors `rag.api.deps.get_current_identity`'s JWT verification rules
exactly, adapted to the two transports this project ships: Streamable
HTTP (per tool call, from the request's `Authorization` header) and
stdio (once per process, from an environment variable -- stdio has no
per-request headers at all). Neither path ever reads `tenant_id`/`roles`
from a tool call's arguments; both resolve a `VerifiedIdentity` (or
`None`, when `security.auth.enabled` is `False`) from a transport-level
credential only. Governed entirely by `security.auth` -- there is no
separate, weaker `mcp.auth.*` toggle that could drift from it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from mcp.server.mcpserver.exceptions import ToolError

from rag.api.auth import AuthenticationError, VerifiedIdentity, verify_jwt
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig


def resolve_http_identity(
    headers: Mapping[str, str] | None, config: AppConfig
) -> VerifiedIdentity | None:
    """Resolve the caller's verified identity from one MCP tool call's HTTP headers.

    Byte-identical rules to `rag.api.deps.get_current_identity`: returns
    `None` when `security.auth.enabled` is `False`; otherwise requires a
    valid `Authorization: Bearer <jwt>` header, unless
    `security.auth.insecure_dev_mode` allows a call with no header at
    all. An invalid/expired/malformed/signature-mismatched token always
    fails closed regardless of that flag. Verified on every call (no
    per-session caching), matching `get_current_identity`'s own
    no-hot-reload, no-memoization contract.

    Parameters
    ----------
    headers : Mapping[str, str] | None
        The current MCP request's transport headers (`Context.headers`).
        `None` on a transport with no headers (stdio); never trusted as
        an identity assertion by itself.
    config : AppConfig
        Application configuration.

    Returns
    -------
    VerifiedIdentity | None
        The verified caller identity, or `None` when auth is disabled.

    Raises
    ------
    ToolError
        When a token is required but missing, malformed, or fails
        verification. Carries no token contents -- matches
        `AuthenticationError`'s own audit-safe design.
    """
    if not config.security.auth.enabled:
        return None

    header = None
    if headers is not None:
        header = headers.get("authorization") or headers.get("Authorization")

    if header is None:
        if config.security.auth.insecure_dev_mode:
            return None
        log_audit_event("mcp_auth_failure", reason="missing_token")
        raise ToolError("Missing Authorization header")

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        log_audit_event("mcp_auth_failure", reason="malformed")
        raise ToolError("Malformed Authorization header")

    try:
        identity = verify_jwt(token, config)
    except AuthenticationError as exc:
        log_audit_event("mcp_auth_failure", reason=exc.reason)
        raise ToolError("Invalid or expired token") from exc

    log_audit_event(
        "mcp_auth_success",
        subject=pseudonymous_subject(identity.subject),
        tenant_id=identity.tenant_id,
    )
    return identity


def resolve_stdio_identity(config: AppConfig) -> VerifiedIdentity | None:
    """Resolve a single fixed identity for one stdio server process, at startup only.

    stdio has no per-request headers, so identity is verified once, from
    a JWT supplied via the `MCP_AUTH_TOKEN` environment variable --
    through the exact same `verify_jwt` every other transport uses, never
    a separate, weaker "trust this tenant_id" bypass. Every tool call
    within this process shares this one identity: a stdio server is a
    single-user, local-development convenience, never a shared or
    multi-tenant deployment target.

    Returns `None` (fully unrestricted) when `security.auth.enabled` is
    `False`, matching the HTTP transport exactly -- deliberately no
    stdio-specific relaxation beyond what HTTP already allows.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    VerifiedIdentity | None
        The verified process-wide identity, or `None` when auth is
        disabled.

    Raises
    ------
    RuntimeError
        `security.auth.enabled` is `True` but `MCP_AUTH_TOKEN` is unset
        or fails verification. Raised at process startup (not per tool
        call), since stdio identity never changes mid-process.
    """
    if not config.security.auth.enabled:
        return None

    token = os.environ.get("MCP_AUTH_TOKEN")
    if not token:
        raise RuntimeError(
            "security.auth.enabled is true; set MCP_AUTH_TOKEN to a valid JWT "
            "before starting the stdio MCP server."
        )
    try:
        identity = verify_jwt(token, config)
    except AuthenticationError as exc:
        raise RuntimeError(f"MCP_AUTH_TOKEN failed verification: {exc.reason}") from exc

    log_audit_event(
        "mcp_auth_success",
        subject=pseudonymous_subject(identity.subject),
        tenant_id=identity.tenant_id,
    )
    return identity
