"""Agent-side MCP client for the remote business tools (MCP Stage 2).

Every other agent tool (`search_knowledge_base`/`get_document`/
`get_latest_document`/`get_related_context`) stays a direct, in-process
call into `rag.agent.tools`. This module exists only for
`get_customer_case`/`get_case_status`/`update_case_status`, dispatched as
a real MCP client call against the business tools served by
`rag.mcp.server`, never a shortcut direct function call. The first two
are read-only; `update_case_status` is the one write action, gated by a
trusted `case_approvals` claim (see `mint_internal_token`) rather than
anything in its own tool arguments.

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
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio
import httpx2
import jwt
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from opentelemetry import propagate

from rag.agent.tool_schemas import GetCaseStatusArgs, GetCustomerCaseArgs, UpdateCaseStatusArgs
from rag.agent.tools import ToolExecutionError
from rag.config import AppConfig
from rag.mcp.business.schemas import CaseActionOutcome, CaseApproval, CaseStatusResult, CustomerCase
from rag.retrieval.authorization import AuthorizationContext
from rag.schemas import Chunk, ChunkMetadata, SearchResult

# The MCP SDK's DNS-rebinding Host-header protection (default host="127.0.0.1")
# only allows "127.0.0.1:*"/"localhost:*"/"[::1]:*". A portless host is omitted
# from the Host header entirely and gets rejected with 421 Misdirected Request,
# so a non-default port is required here even though nothing listens on it.
_ASGI_INTERNAL_BASE_URL = "http://127.0.0.1:1/"


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


def mint_internal_token(
    auth: AuthorizationContext,
    config: AppConfig,
    case_approvals: Sequence[CaseApproval] = (),
) -> str:
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

    `case_approvals`, when non-empty, is embedded as a `case_approvals`
    claim: pre-authorized `(case_id, new_status)` pairs for
    `update_case_status`, resolved server-side by `rag.mcp.business.
    approvals.resolve_case_action_approvals`, never read from a tool
    argument. Reuses this token's existing signature rather than a
    second trust channel.

    Parameters
    ----------
    auth : AuthorizationContext
        The caller's already-resolved authorization context. Must carry
        a non-`None` `tenant_id`; see `dispatch_remote_tool_sync`,
        which checks this before ever calling here.
    config : AppConfig
        Application configuration.
    case_approvals : Sequence[CaseApproval], optional
        Pre-authorized case-status transitions to attach, already
        bounded at the API boundary (see `rag.api.request_auth.
        enforce_case_approval_limits`).

    Returns
    -------
    str
        The signed JWT, ready for an `Authorization: Bearer <token>`
        header.

    Raises
    ------
    RuntimeError
        If `security.auth.jwt.algorithm` is not `HS256` (defense in
        depth: `validate_startup_config` already refuses to start the
        process in this state), or if `case_approvals` exceeds `mcp.
        business_actions.max_case_approvals_per_request` (defense in
        depth: the API boundary already enforces this before `AgentState.
        case_approvals` is ever populated).
    """
    jwt_config = config.security.auth.jwt
    if jwt_config.algorithm != "HS256":
        raise RuntimeError(
            f"Cannot mint an internal MCP service token: "
            f"security.auth.jwt.algorithm={jwt_config.algorithm!r} is not 'HS256'."
        )
    max_approvals = config.mcp.business_actions.max_case_approvals_per_request
    if len(case_approvals) > max_approvals:
        raise RuntimeError(
            f"Cannot mint an internal MCP service token: case_approvals has "
            f"{len(case_approvals)} entries, exceeding the configured maximum of "
            f"{max_approvals}."
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
    if case_approvals:
        claims["case_approvals"] = [a.model_dump(mode="json") for a in case_approvals]
    key = config.jwt_signing_key()
    return jwt.encode(claims, key, algorithm="HS256")


async def _call_tool_async(
    tool_name: str,
    args: GetCustomerCaseArgs | GetCaseStatusArgs | UpdateCaseStatusArgs,
    *,
    auth: AuthorizationContext,
    config: AppConfig,
    mcp_app: Any | None,
    case_approvals: Sequence[CaseApproval] = (),
) -> dict[str, Any] | None:
    """Open one fresh MCP session, call `tool_name`, and return its raw structured result.

    Session-per-call by design (see the module docstring). Raises a
    plain `Exception` on any timeout, connection failure, or tool-side
    error (`CallToolResult.is_error`); the caller
    (`rag.agent.graph._execute_tool`) already has a generic
    catch-and-record-as-failed-ToolCallRecord envelope around any tool
    dispatch, local or remote, so no special-case error handling is
    needed here.
    """
    client_cfg = config.mcp.client
    token = mint_internal_token(auth, config, case_approvals)
    headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
    # Injects the current OpenTelemetry span's trace context (if tracing is
    # enabled), so Jaeger shows one connected trace across the agent -> MCP
    # client -> MCP server boundary instead of two disconnected ones. A no-op,
    # cheap dict-touch when tracing is disabled (the default no-op propagator).
    propagate.inject(headers)

    if client_cfg.transport == "asgi":
        if mcp_app is None:
            raise RuntimeError(
                "mcp.client.transport='asgi' but run_agent() was not given an "
                "mcp_app (see rag.api.deps.get_mcp_asgi_app())."
            )
        transport: httpx2.AsyncBaseTransport | None = httpx2.ASGITransport(app=mcp_app)
        base_url = _ASGI_INTERNAL_BASE_URL
    else:
        transport = None
        base_url = config.mcp_client_server_url()

    async with httpx2.AsyncClient(
        transport=transport, headers=headers, timeout=client_cfg.timeout_seconds
    ) as http_client:
        async with streamable_http_client(base_url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name,
                    args.model_dump(),
                    read_timeout_seconds=client_cfg.timeout_seconds,
                )

    if result.is_error:
        first = result.content[0] if result.content else None
        message = getattr(first, "text", None) or "no error detail"
        raise RuntimeError(f"MCP tool {tool_name!r} returned an error: {message}")
    structured = result.structured_content or {}
    return structured.get("result")


def _render_case_action_outcome(outcome: CaseActionOutcome) -> str:
    """Render one `update_case_status` outcome as plain, unambiguous evidence text.

    Every wording states the true outcome explicitly, so a model
    synthesizing an answer from this text has no basis to claim a
    mutation happened when it didn't (`approval_required`/`invalid_
    transition`) or to imply an action was needed when it wasn't
    (`already_in_status`).
    """
    if outcome.outcome == "executed":
        return (
            f"Case {outcome.case_id} status was changed from {outcome.previous_status} "
            f"to {outcome.new_status}."
        )
    if outcome.outcome == "already_in_status":
        return f"Case {outcome.case_id} is already {outcome.new_status}; no change was made."
    if outcome.outcome == "invalid_transition":
        return (
            f"Requested status change for case {outcome.case_id} from "
            f"{outcome.previous_status} to {outcome.new_status} is not a valid transition. "
            f"No change was made; status remains {outcome.previous_status}."
        )
    return (
        f"Status change for case {outcome.case_id} from {outcome.previous_status} to "
        f"{outcome.new_status} requires approval before it can be applied. No change has "
        f"been made."
    )


def _business_result_to_search_result(
    tool_name: str, case_id: str, data: dict[str, Any]
) -> SearchResult:
    """Validate a raw business-tool result against its structured schema and wrap it as evidence.

    Re-validates client-side against `rag.mcp.business.schemas`
    (`CustomerCase`/`CaseStatusResult`/`CaseActionOutcome`) rather than
    trusting the raw dict as-is, keeping the structured-schema guarantee
    end-to-end across the process boundary.

    The synthetic `Chunk`/`ChunkMetadata` this builds deliberately leaves
    every document-governance field (`document_version`/`status`/
    `effective_from`/`supersedes_source`/`allowed_roles`/`classification`/
    `trust_level`) unset, so this evidence never participates in
    freshness resolution or document-level ACL. `tenant_id` is set from
    the case's own real tenant purely for citation/audit display; it is
    never re-checked against an `AuthorizationContext`, since that
    authorization already happened inside `rag.mcp.business.store`
    before this function ever runs.
    """
    if tool_name == "get_customer_case":
        case = CustomerCase.model_validate(data)
        content = (
            f"Customer support case {case.case_id} (tenant: {case.tenant_id}, "
            f"customer: {case.customer_name})\n"
            f"Subject: {case.subject}\n"
            f"Status: {case.status}, priority: {case.priority}, "
            f"assigned to: {case.assigned_team}\n"
            f"Description: {case.description}"
        )
        tenant_id = case.tenant_id
        created_at, last_modified = case.created_at, case.updated_at
    elif tool_name == "update_case_status":
        outcome = CaseActionOutcome.model_validate(data)
        content = _render_case_action_outcome(outcome)
        tenant_id = None
        created_at = last_modified = outcome.updated_at
    else:
        status = CaseStatusResult.model_validate(data)
        content = (
            f"Customer support case {status.case_id} status: {status.status}, "
            f"priority: {status.priority}, last updated: {status.updated_at.isoformat()}"
        )
        tenant_id = None
        created_at = last_modified = status.updated_at

    metadata = ChunkMetadata(
        document_id=case_id,
        chunk_id=f"mcp:{tool_name}:{case_id}",
        source=f"mcp://business/{case_id}",
        source_type="mcp_business",
        created_at=created_at,
        last_modified=last_modified,
        chunk_index=0,
        dataset_id="mcp_business",
        tenant_id=tenant_id,
    )
    chunk = Chunk(id=metadata.chunk_id, content=content, metadata=metadata)
    return SearchResult(chunk=chunk, score=1.0, origin="mcp_remote")


def dispatch_remote_tool_sync(
    tool_name: str,
    args: GetCustomerCaseArgs | GetCaseStatusArgs | UpdateCaseStatusArgs,
    *,
    auth: AuthorizationContext | None,
    config: AppConfig,
    mcp_app: Any | None,
    case_approvals: Sequence[CaseApproval] = (),
) -> list[SearchResult]:
    """Dispatch one remote business-tool call, synchronously, and return it as agent evidence.

    The synchronous entrypoint `rag.agent.graph._execute_tool` calls;
    bridges to the async MCP client via `anyio.run` since that module's
    node functions are synchronous by design. Safe to call from a plain
    worker thread with no already-running event loop; both callers of
    `run_agent()` (`POST /agent/query`'s sync route, and the SSE stream's
    `run_in_threadpool`-wrapped call) already satisfy this.

    Parameters
    ----------
    tool_name : {"get_customer_case", "get_case_status", "update_case_status"}
        Which remote business tool to call.
    args : GetCustomerCaseArgs | GetCaseStatusArgs | UpdateCaseStatusArgs
        The validated tool-argument instance.
    auth : AuthorizationContext | None
        The caller's resolved authorization context. Must be non-`None`
        with a non-`None` `tenant_id`, or this raises `ToolExecutionError`
        before attempting any call; there is no anonymous path for these
        tools (see the module docstring's security posture).
    config : AppConfig
        Application configuration; `config.mcp.client` governs transport/
        timeout/token settings.
    mcp_app : Any | None
        The in-process MCP ASGI app object (see
        `rag.api.deps.get_mcp_asgi_app`), required when
        `config.mcp.client.transport="asgi"` (the default); unused for
        `transport="http"`.
    case_approvals : Sequence[CaseApproval], optional
        Trusted, pre-authorized transitions for `update_case_status`,
        forwarded to `mint_internal_token`. Ignored (harmless) for the
        two read tools.

    Returns
    -------
    list[SearchResult]
        Empty when the case doesn't exist or the caller isn't authorized
        for it (the business store's own, deliberately indistinguishable
        "not found" result, not a tool failure); otherwise exactly one
        synthetic, `origin="mcp_remote"` result.

    Raises
    ------
    ToolExecutionError
        If `auth` is `None` or has no `tenant_id`; no authenticated
        identity to attach to the call.
    """
    if auth is None or auth.tenant_id is None:
        raise ToolExecutionError(
            f"{tool_name} requires an authenticated caller identity with a tenant_id"
        )
    case_id = args.case_id
    call = partial(
        _call_tool_async,
        tool_name,
        args,
        auth=auth,
        config=config,
        mcp_app=mcp_app,
        case_approvals=case_approvals,
    )
    raw_result = anyio.run(call)
    if raw_result is None:
        return []
    return [_business_result_to_search_result(tool_name, case_id, raw_result)]
