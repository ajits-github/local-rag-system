"""Builds the MCP server's Streamable-HTTP ASGI app for mounting inside the existing FastAPI app.

Not a second server or process: `rag.api.main` mounts this app's ASGI
callable directly under `config.mcp.server.mount_path`, sharing the same
process-wide singletons (`rag.api.deps`) every other route already uses.
"""

from __future__ import annotations

from starlette.applications import Starlette

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
