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
        The app `rag.api.main` mounts at `config.mcp.server.mount_path`.
        Registered with `streamable_http_path="/"` (not the SDK's own
        `/mcp` default) since the mount prefix itself already supplies
        that path segment -- registering the SDK's default here would
        require every request to repeat it (`/mcp/mcp`).

        Its `.router.lifespan_context` must be entered by the parent
        app's own lifespan: Starlette does not auto-propagate a mounted
        sub-app's lifespan, so without this the MCP session manager's
        background task group would never start. See `rag.api.main`'s
        combined lifespan.
    """
    server = build_mcp_server(config, pipeline, vectorstore, embedder)
    return server.streamable_http_app(streamable_http_path="/")


class _BareMountPathMiddleware:
    """Rewrites a request for the bare MCP mount path to its trailing-slash form, server-side.

    `Starlette.mount(mount_path, app)` builds a `Mount` route whose match
    regex is `<mount_path>/{path:path}` (confirmed against the installed
    `starlette==1.3.1`), which requires a literal `/` immediately after
    `mount_path` -- so it matches `<mount_path>/` and everything under
    it, but never the bare `<mount_path>` itself. A request to the bare
    path therefore falls through to Starlette's own `Router`-level
    `redirect_slashes` handling and receives a 307 to `<mount_path>/` --
    which the MCP SDK's own Streamable HTTP client does not follow
    during session initialization (confirmed against a real client
    against a real Docker container; see ISSUES.md). This rewrites the
    ASGI scope's `path` (and `raw_path`, if present) in place, before
    routing runs, whenever it exactly equals `mount_path` -- so both
    spellings resolve to the identical request with no client-visible
    redirect at all. Every other path is passed through untouched.
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

    `app.mount(mount_path, mcp_app)` alone leaves the bare `mount_path`
    (no trailing slash) 307-redirecting to `mount_path/` -- see
    `_BareMountPathMiddleware`'s docstring for the exact Starlette
    mechanics. This does the mount and then layers that middleware on
    top, so `mount_path` is the reliable, documented canonical MCP
    endpoint; `mount_path/` keeps working exactly as before, unaffected.

    A no-op for every other route in `app`: the middleware only rewrites
    a request whose path is an exact match for `mount_path`.

    Parameters
    ----------
    app
        The outer FastAPI/Starlette application to mount onto. Typed
        loosely (rather than importing `fastapi.FastAPI`) since only
        `.mount()`/`.add_middleware()` are used, both already part of
        Starlette's own `Starlette` interface FastAPI subclasses.
    mcp_app : Starlette
        The app returned by `build_mcp_asgi_app`.
    mount_path : str
        `config.mcp.server.mount_path`, e.g. `"/mcp"`.
    """
    app.mount(mount_path, mcp_app)
    app.add_middleware(_BareMountPathMiddleware, mount_path=mount_path)
