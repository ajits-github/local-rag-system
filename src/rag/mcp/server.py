"""Builds the MCP server exposing the four core RAG tools plus two synthetic business tools.

Each RAG tool is a thin adapter over `rag.agent.tools.*`, the same
functions the in-process agent graph calls; no retrieval or
authorization logic is reimplemented here. Identity is resolved once
per call from the transport (see `rag.mcp.identity`) via the SDK's
`Resolve` parameter-injection mechanism, which excludes resolver-filled
parameters from a tool's generated JSON schema, so a tool call can never
supply or override it. Every RAG result passes through
`RetrievalPipeline.sanitize_evidence` in `_run_tool` before being
returned, mirroring `rag.agent.graph._execute_tool`'s dispatch-then-
sanitize pattern.

`get_customer_case`/`get_case_status` are a separate tool family: thin
adapters over `rag.mcp.business.store`, a synthetic case backend with
its own tenant/role authorization (see that module for why it doesn't
reuse `AuthorizationContext`/`sanitize_evidence`). They demonstrate MCP
as an integration layer to a separate backend system, and do not route
through `_run_tool`.

Every registered tool's argument model is hardened after registration
(see `_harden_argument_schemas`) so an unknown argument is rejected
rather than silently dropped.

This module does not use ``from __future__ import annotations``: the
SDK resolves tool parameter annotations via `eval()`, which fails for
the closure-local resolver names used below under postponed evaluation.
"""

import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, Literal, TypeVar

from mcp.server.mcpserver import Context, MCPServer, Resolve
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from rag.agent import tools
from rag.agent.tool_schemas import (
    ContentTypeFilter,
    GetDocumentArgs,
    GetLatestDocumentArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
)
from rag.api.auth import VerifiedIdentity
from rag.api.request_auth import build_authorization_context
from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.mcp.business import store as business_store
from rag.mcp.business.schemas import CaseStatusResult, CustomerCase
from rag.mcp.identity import resolve_http_identity
from rag.mcp.schemas import McpChunkResult, to_mcp_result
from rag.observability import metrics as observability_metrics
from rag.observability import tracing
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

_ToolName = Literal[
    "search_knowledge_base", "get_document", "get_latest_document", "get_related_context"
]
_BusinessToolName = Literal["get_customer_case", "get_case_status"]
_T = TypeVar("_T")


class _Unset:
    """Sentinel type distinguishing "not passed" from an explicit `None` for `fixed_identity`.

    A dedicated class, not a bare `object()`, so mypy can narrow
    `fixed_identity is _UNSET` correctly: `object` would collapse the
    `VerifiedIdentity | None | object` union down to `object` and defeat
    narrowing entirely.
    """


_UNSET = _Unset()


def _harden_argument_schemas(server: MCPServer) -> None:
    """Make every registered tool's argument model reject an unknown field.

    The SDK's dynamically-built per-tool argument model defaults to
    Pydantic's `extra="ignore"` with no exposed strictness knob, so an
    unrecognized argument (e.g. an attempted `tenant_id`/`roles`/`auth`
    injection) would otherwise be silently dropped rather than rejected.
    This switches each tool's argument model to `extra="forbid"` and
    regenerates its cached JSON schema, so validation fails loudly
    before the tool function or any resolver runs. `auth` itself is
    never reachable through tool arguments regardless (it is exclusively
    resolver-injected, see `Resolve(_resolve_auth)` below); this closes
    the separate unknown-key gap.
    """
    for tool in server._tool_manager.list_tools():
        arg_model = tool.fn_metadata.arg_model
        arg_model.model_config["extra"] = "forbid"
        arg_model.model_rebuild(force=True)
        tool.parameters = arg_model.model_json_schema(by_alias=True)


def build_mcp_server(
    config: AppConfig,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    *,
    fixed_identity: VerifiedIdentity | None | _Unset = _UNSET,
) -> MCPServer:
    """Build the MCP server exposing the four RAG tools and two business-case tools.

    Parameters
    ----------
    config : AppConfig
        Application configuration; `security.auth` governs identity
        resolution, and `agent.max_tool_top_k`/
        `agent.max_chunks_per_document_fetch*` bound the same
        server-controlled limits the in-process agent tools use.
    pipeline, vectorstore, embedder
        Process-wide singletons; `embedder` is used by `get_document`/
        `get_latest_document`'s relevance-selection pass.
    fixed_identity : VerifiedIdentity | None, keyword-only
        Left unset (the default) for the Streamable-HTTP transport,
        where identity is resolved per call from that call's request
        headers. Pass an explicit value (including `None`, meaning
        auth disabled) only for the stdio transport, which resolves one
        identity at process startup and reuses it for every call.

    Returns
    -------
    MCPServer
        A server with all six tools registered. Callers mount it via
        `.streamable_http_app()` or run it via `.run_stdio_async()`.
    """
    server: MCPServer = MCPServer(
        name="local-rag-system",
        instructions=(
            "Search and fetch content from this deployment's authorized knowledge "
            "base, and look up synthetic customer-support cases from a separate "
            "backend system. Every result is already scoped to your verified "
            "identity -- you cannot and should not supply a tenant, role, or "
            "authorization value yourself; any such argument is ignored."
        ),
    )

    # Resolved once here: mypy does not preserve isinstance narrowing of a
    # variable captured by a nested closure, so the _Unset case is collapsed
    # into a plain `VerifiedIdentity | None` local before any closure reads it.
    use_fixed_identity = not isinstance(fixed_identity, _Unset)
    resolved_fixed_identity: VerifiedIdentity | None = (
        fixed_identity if isinstance(fixed_identity, VerifiedIdentity) else None
    )

    def _resolve_identity(ctx: Context) -> VerifiedIdentity | None:
        """Resolve this call's verified identity from the transport, never from tool args.

        The shared first half of `_resolve_auth` below, factored out so
        the business-case tools (which need a bare `VerifiedIdentity`,
        not an `AuthorizationContext`) can reuse the exact same
        transport-resolution rule rather than a second copy of it.
        """
        return (
            resolved_fixed_identity
            if use_fixed_identity
            else resolve_http_identity(ctx.headers, config)
        )

    async def _resolve_auth(
        ctx: Context, as_of: date | None, require_trust_level: str | None
    ) -> AuthorizationContext | None:
        """Build this call's `AuthorizationContext` from the transport, never from tool args.

        `as_of`/`require_trust_level` are ordinary, non-privileged tool
        arguments (matching `AgentQueryRequest`'s own fields), not
        identity claims, so they are read by name from the calling
        tool, the same as `build_authorization_context` treats them for
        `/query`/`/agent/query`. `tenant_id`/`roles` are always passed
        as `None`; this project's transports never source them from
        anything but a verified identity.
        """
        identity = _resolve_identity(ctx)
        return build_authorization_context(identity, None, None, as_of, require_trust_level)

    async def _resolve_identity_only(ctx: Context) -> VerifiedIdentity | None:
        """`Resolve()`-compatible async wrapper around `_resolve_identity`.

        `Resolve()` requires an async callable, which is the only reason
        this thin wrapper exists separately from `_resolve_identity`.
        """
        return _resolve_identity(ctx)

    def _dispatch(
        tool_name: _ToolName,
        args: Any,
        *,
        auth: AuthorizationContext | None,
        filters: dict[str, Any] | None,
        dataset_id: str | None,
        query: str,
    ) -> list[SearchResult]:
        """Call the matching `rag.agent.tools` function.

        The same dispatch `agent.graph._dispatch_tool` makes.
        """
        if tool_name == "search_knowledge_base":
            clamped_top_k = min(args.top_k, config.agent.max_tool_top_k)
            clamped_args = args.model_copy(update={"top_k": clamped_top_k})
            return list(tools.search_knowledge_base(clamped_args, pipeline, filters, auth))
        if tool_name == "get_document":
            chunks = tools.get_document(
                args,
                pipeline,
                vectorstore,
                dataset_id,
                query,
                embedder,
                auth,
                config.agent.max_chunks_per_document_fetch,
                config.agent.max_chunks_per_document_fetch_hard_ceiling,
            )
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        if tool_name == "get_latest_document":
            chunks = tools.get_latest_document(
                args,
                pipeline,
                vectorstore,
                dataset_id,
                query,
                embedder,
                auth,
                config.agent.max_chunks_per_document_fetch,
                config.agent.max_chunks_per_document_fetch_hard_ceiling,
            )
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        if tool_name == "get_related_context":
            chunks = tools.get_related_context(args, pipeline, vectorstore, auth, dataset_id)
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )  # unreachable: tool_name is Literal-validated

    def _run_tool(
        tool_name: _ToolName,
        args: Any,
        *,
        auth: AuthorizationContext | None,
        filters: dict[str, Any] | None,
        dataset_id: str | None,
        query: str,
    ) -> list[McpChunkResult]:
        """Dispatch, sanitize, observe, and serialize one RAG tool call.

        All four RAG tool handlers route through this function, so
        sanitization is applied centrally and no tool can bypass it.
        """
        t0 = time.perf_counter()
        try:
            with tracing.start_span(tool_name, attributes={"tool_name": tool_name}) as span:
                results = _dispatch(
                    tool_name, args, auth=auth, filters=filters, dataset_id=dataset_id, query=query
                )
                tracing.set_attributes(span, {"tool_success": True, "result_count": len(results)})
        except tools.ToolExecutionError as exc:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            raise ToolError(str(exc)) from None
        except Exception:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            logger.exception("MCP tool %s failed", tool_name)
            raise  # an unanticipated failure: let the SDK report it generically, never leak details

        effective_auth = pipeline.resolve_auth(
            auth, {"dataset_id": dataset_id} if dataset_id else None
        )
        sanitized = pipeline.sanitize_evidence(results, effective_auth)
        latency_seconds = time.perf_counter() - t0
        observability_metrics.observe_tool_call(tool_name, True, latency_seconds)
        return [to_mcp_result(r) for r in sanitized]

    def _run_business_tool(tool_name: _BusinessToolName, fn: Callable[[], _T]) -> _T:
        """Dispatch one business-case tool: observe, then delegate authorization to `fn`.

        Parallel to `_run_tool` (tracing, latency, error metrics) but
        skips `pipeline.sanitize_evidence`, which redacts chunk/field
        content a business case doesn't have; `rag.mcp.business.store`
        already returns `None` for both "not found" and "not
        authorized". `fn` is a zero-arg closure so this stays generic
        over which `store` function it wraps.
        """
        t0 = time.perf_counter()
        try:
            with tracing.start_span(tool_name, attributes={"tool_name": tool_name}) as span:
                result = fn()
                tracing.set_attributes(span, {"tool_success": True, "found": result is not None})
        except Exception:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            logger.exception("MCP tool %s failed", tool_name)
            raise  # an unanticipated failure: let the SDK report it generically, never leak details

        latency_seconds = time.perf_counter() - t0
        observability_metrics.observe_tool_call(tool_name, True, latency_seconds)
        return result

    @server.tool(
        name="search_knowledge_base",
        description=(
            "Search the authorized knowledge base for chunks relevant to a query. "
            "Results are already scoped to your verified identity, freshness-"
            "filtered, and redacted."
        ),
    )
    async def search_knowledge_base(
        query: str,
        top_k: Annotated[int, Field(ge=1, le=20)] = 5,
        content_type: ContentTypeFilter | None = None,
        dataset_id: str | None = None,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = SearchKnowledgeBaseArgs(query=query, top_k=top_k, content_type=content_type)
        filters = {"dataset_id": dataset_id} if dataset_id else None
        return _run_tool(
            "search_knowledge_base",
            args,
            auth=auth,
            filters=filters,
            dataset_id=dataset_id,
            query=query,
        )

    @server.tool(
        name="get_document",
        description=(
            "Fetch a specific authorized document by its source path, bounded and "
            "relevance-selected against your current question."
        ),
    )
    async def get_document(
        source: str,
        dataset_id: str,
        query: str,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetDocumentArgs(source=source)
        return _run_tool(
            "get_document", args, auth=auth, filters=None, dataset_id=dataset_id, query=query
        )

    @server.tool(
        name="get_latest_document",
        description=(
            "Resolve a document's source path to its currently-effective version "
            "(if superseded), then fetch it, bounded and relevance-selected "
            "against your current question."
        ),
    )
    async def get_latest_document(
        source: str,
        dataset_id: str,
        query: str,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetLatestDocumentArgs(source=source)
        return _run_tool(
            "get_latest_document", args, auth=auth, filters=None, dataset_id=dataset_id, query=query
        )

    @server.tool(
        name="get_related_context",
        description=(
            "Fetch parent/neighbor context for an already-retrieved chunk, by its "
            "chunk_id (copy the chunk_id value from a prior result verbatim)."
        ),
    )
    async def get_related_context(
        chunk_id: str,
        dataset_id: str | None = None,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetRelatedContextArgs(chunk_id=chunk_id)
        return _run_tool(
            "get_related_context", args, auth=auth, filters=None, dataset_id=dataset_id, query=""
        )

    @server.tool(
        name="get_customer_case",
        description=(
            "Fetch a synthetic customer-support case by its case_id, from a separate "
            "backend business system (not this deployment's knowledge base). Scoped "
            "to your verified tenant and role; a case you may not access is "
            "indistinguishable from one that doesn't exist."
        ),
    )
    async def get_customer_case(
        case_id: str,
        identity: Annotated[VerifiedIdentity | None, Resolve(_resolve_identity_only)] = None,
    ) -> CustomerCase | None:
        cross_tenant_support_roles = config.security.authorization.cross_tenant_support_roles
        return _run_business_tool(
            "get_customer_case",
            lambda: business_store.get_customer_case(case_id, identity, cross_tenant_support_roles),
        )

    @server.tool(
        name="get_case_status",
        description=(
            "Fetch only the status, priority, and last-updated time of a synthetic "
            "customer-support case by its case_id -- a narrower read than "
            "get_customer_case for callers that only need to check case state."
        ),
    )
    async def get_case_status(
        case_id: str,
        identity: Annotated[VerifiedIdentity | None, Resolve(_resolve_identity_only)] = None,
    ) -> CaseStatusResult | None:
        cross_tenant_support_roles = config.security.authorization.cross_tenant_support_roles
        return _run_business_tool(
            "get_case_status",
            lambda: business_store.get_case_status(case_id, identity, cross_tenant_support_roles),
        )

    _harden_argument_schemas(server)
    return server
