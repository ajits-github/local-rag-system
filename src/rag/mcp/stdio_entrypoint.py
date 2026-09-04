"""stdio entrypoint for the MCP server: `python -m rag.mcp.stdio_entrypoint`.

Host-only, local-development convenience for a desktop MCP client (e.g.
Claude Desktop) that talks stdio rather than HTTP; never containerized,
never a shared/multi-tenant deployment target. Builds the same
`rag.mcp.server.build_mcp_server` every Streamable-HTTP request uses;
the only difference is that identity is resolved once at startup from
`MCP_AUTH_TOKEN`, instead of per call from an HTTP header.
"""

from __future__ import annotations

import anyio

from rag.api.deps import get_config, get_embedder, get_retrieval_pipeline, get_vectorstore
from rag.mcp.identity import resolve_stdio_identity
from rag.mcp.server import build_mcp_server


async def _run() -> None:
    """Resolve the process's one fixed identity, then serve over stdio until disconnected."""
    config = get_config()
    identity = resolve_stdio_identity(config)
    server = build_mcp_server(
        config,
        get_retrieval_pipeline(),
        get_vectorstore(),
        get_embedder(),
        fixed_identity=identity,
    )
    await server.run_stdio_async()


def main() -> None:
    """CLI entrypoint."""
    anyio.run(_run)


if __name__ == "__main__":
    main()
