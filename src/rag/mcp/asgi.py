"""Builds the MCP server's Streamable-HTTP ASGI app for mounting inside the existing FastAPI app.

Not a second server or process: `rag.api.main` mounts this app's ASGI
callable directly under `config.mcp.server.mount_path`, sharing the same
process-wide singletons (`rag.api.deps`) every other route already uses.
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.mcp.server import build_mcp_server
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore


def build_mcp_asgi_app(
    config: AppConfig,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
) -> Starlette:
    """Build the mountable Streamable-HTTP ASGI app for the MCP server.

    Parameters
    ----------
    config, pipeline, vectorstore, embedder
        Passed straight through to `rag.mcp.server.build_mcp_server`.

    Returns
    -------
    Starlette
        The app `rag.api.main` mounts at `config.mcp.server.mount_path`,
        registered with `streamable_http_path="/"` since the mount
        prefix already supplies that path segment (the SDK's own `/mcp`
        default would otherwise double up as `/mcp/mcp`). Its
        `.router.lifespan_context` must be entered by the parent app's
        own lifespan, or the MCP session manager's background task never
        starts.
    """
    server = build_mcp_server(config, pipeline, vectorstore, embedder)
    return server.streamable_http_app(streamable_http_path="/")


class _BareMountPathMiddleware:
    """Rewrite the bare MCP mount path to its trailing-slash form, server-side.

    `Starlette.mount(mount_path, app)` only matches `<mount_path>/...`,
    so a request to the bare path (no trailing slash) falls through to
    Starlette's 307 redirect, which the MCP SDK's Streamable HTTP client
    does not follow during session initialization. This rewrites the
    ASGI scope's `path` (and `raw_path`) in place, before routing runs,
    whenever it exactly equals `mount_path`, so both spellings resolve
    identically with no client-visible redirect. Every other path passes
    through untouched.
    """

    def __init__(self, app: ASGIApp, mount_path: str) -> None:
        self._app = app
        self._mount_path = mount_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == self._mount_path:
            scope = dict(scope)
            scope["path"] = self._mount_path + "/"
            raw_path = scope.get("raw_path")
            if raw_path is not None:
                scope["raw_path"] = raw_path + b"/"
        await self._app(scope, receive, send)


def mount_mcp_app(app: Any, mcp_app: Starlette, mount_path: str) -> None:
    """Mount the MCP ASGI app so both `mount_path` and `mount_path/` work, with no visible redirect.

    Layers `_BareMountPathMiddleware` on top of a normal `app.mount()`
    so the bare `mount_path` (no trailing slash) resolves the same as
    `mount_path/` instead of 307-redirecting. Every other route in
    `app` is unaffected.

    Parameters
    ----------
    app
        The outer FastAPI/Starlette application to mount onto. Typed
        loosely since only `.mount()`/`.add_middleware()` are used.
    mcp_app : Starlette
        The app returned by `build_mcp_asgi_app`.
    mount_path : str
        `config.mcp.server.mount_path`, e.g. `"/mcp"`.
    """
    app.mount(mount_path, mcp_app)
    app.add_middleware(_BareMountPathMiddleware, mount_path=mount_path)
