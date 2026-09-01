"""Real-Postgres adversarial proof: MCP tool calls cannot bypass tenant isolation.

Complements `tests/integration/test_mcp_end_to_end.py` (which uses fake
doubles to prove the MCP transport/identity/dispatch wiring is correct)
with one real, SQL-backed scenario per tool, mirroring
`tests/integration/test_agent_tool_tenant_isolation.py`'s exact
ingestion/fixture pattern -- proving the real `VectorStore` authorization
predicate holds when reached through the MCP transport too, not just
through the in-process agent graph. Since `rag.mcp.server` calls the
exact same, already-adversarially-tested `rag.agent.tools.*` functions
unmodified, this file is deliberately a small, targeted spot-check (one
scenario per tool), not a re-run of that file's full matrix.

Self-skips without Postgres (`require_postgres`, same as every other
integration test); no Ollama needed, these tools never call the LLM.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx2
import jwt
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rag.factory import build_embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.mcp.asgi import build_mcp_asgi_app
from rag.retrieval.pipeline import RetrievalPipeline

_SECRET = "mcp-tenant-isolation-test-secret"


class _NoOpEmbedder:
    """Placeholder embedder for get_document/get_latest_document's required parameter."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        else:
            lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(app) -> Iterator[str]:
    import uvicorn

    port = _free_port()
    uconfig = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(uconfig)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn server did not start in time"
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _token(tenant_id: str, roles: list[str]) -> str:
    now = int(time.time())
    claims = {"sub": "alice", "tenant_id": tenant_id, "roles": roles, "iat": now, "exp": now + 3600}
    return jwt.encode(claims, _SECRET, algorithm="HS256")


async def _call_tool(base_url: str, token: str, name: str, arguments: dict):
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http_client:
        async with streamable_http_client(base_url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)


def _secure_config(config, monkeypatch):
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    secure.security.auth.enabled = True
    secure.security.auth.jwt.secret_env_var = "JWT_HS256_SECRET"
    secure.mcp.enabled = True
    return secure


@pytest.mark.asyncio
async def test_search_knowledge_base_cannot_cross_tenant_boundary(
    require_postgres, config, tmp_path: Path, monkeypatch
):
    """A tenant_beta-scoped MCP session never sees tenant_alpha's document, and vice versa."""
    ns = f"pytest-mcp-e2e-{uuid.uuid4()}"
    secure = _secure_config(config, monkeypatch)
    ingestion = IngestionPipeline(secure)

    path = _write_doc(
        tmp_path,
        "alpha-only.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "The Alpha-only rollback code is ALPHA-ROLLBACK-4471.",
    )
    result = ingestion.ingest_file(path, ns)
    vectorstore = ingestion._vectorstore
    # search_knowledge_base routes through RetrievalPipeline.retrieve(), which embeds
    # the query for a real vector search -- unlike get_document/get_related_context,
    # this needs the real, dimension-matching embedder, not a placeholder.
    embedder = build_embedder(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=embedder)
    app = build_mcp_asgi_app(secure, pipeline, vectorstore, embedder)

    try:
        with _serve(app) as base_url:
            beta_result = await _call_tool(
                base_url,
                _token("tenant_beta", ["tenant_beta_operator"]),
                "search_knowledge_base",
                {"query": "what is the rollback code", "top_k": 5, "dataset_id": ns},
            )
            assert beta_result.structured_content["result"] == []

            alpha_result = await _call_tool(
                base_url,
                _token("tenant_alpha", ["tenant_alpha_operator"]),
                "search_knowledge_base",
                {"query": "what is the rollback code", "top_k": 5, "dataset_id": ns},
            )
            contents = [r["content"] for r in alpha_result.structured_content["result"]]
            assert any("ALPHA-ROLLBACK-4471" in c for c in contents)
    finally:
        vectorstore.delete_document(result["document_id"])


@pytest.mark.asyncio
async def test_get_document_cannot_cross_tenant_boundary(
    require_postgres, config, tmp_path: Path, monkeypatch
):
    """get_document is a direct VectorStore fetch, unlike retrieve(); still enforces tenant ACL."""
    ns = f"pytest-mcp-e2e-{uuid.uuid4()}"
    secure = _secure_config(config, monkeypatch)
    ingestion = IngestionPipeline(secure)

    path = _write_doc(
        tmp_path,
        "beta-secret.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "The Beta secret rotation window is 30 days.",
    )
    result = ingestion.ingest_file(path, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=_NoOpEmbedder())
    app = build_mcp_asgi_app(secure, pipeline, vectorstore, _NoOpEmbedder())

    try:
        with _serve(app) as base_url:
            wrong_tenant = await _call_tool(
                base_url,
                _token("tenant_alpha", ["tenant_alpha_operator"]),
                "get_document",
                {"source": str(path), "dataset_id": ns, "query": "rotation window"},
            )
            assert wrong_tenant.structured_content["result"] == []

            right_tenant = await _call_tool(
                base_url,
                _token("tenant_beta", ["tenant_beta_operator"]),
                "get_document",
                {"source": str(path), "dataset_id": ns, "query": "rotation window"},
            )
            contents = [r["content"] for r in right_tenant.structured_content["result"]]
            assert any("30 days" in c for c in contents)
    finally:
        vectorstore.delete_document(result["document_id"])
