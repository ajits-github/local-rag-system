"""Real MCP client/server end-to-end tests over the actual wire protocol.

Runs a real uvicorn server (background thread, ephemeral localhost port)
hosting the Streamable-HTTP app `rag.mcp.asgi.build_mcp_asgi_app` builds,
and drives it with the official `mcp` Python SDK's own client -- not just
calling internal dispatch functions directly. Uses `_FakePipeline`/
`_FakeVectorStore`/`_FakeEmbedder` doubles (same pattern as
`tests/unit/test_agent_tools.py`) rather than a real Postgres/Ollama: this
file's job is proving the MCP transport, identity resolution, and
dispatch/sanitize wiring are correct, not re-proving `VectorStore`'s
SQL-level ACL (already covered by
`tests/integration/test_agent_tool_tenant_isolation.py`, whose
`rag.agent.tools.*` functions this milestone reuses unmodified). No
`require_postgres`/`require_ollama` needed -- self-contained, always runs.

See `tests/integration/test_mcp_tenant_isolation.py` for the one
real-Postgres adversarial cross-tenant check.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime

import httpx2
import jwt
import pytest
import uvicorn
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rag.config import load_config
from rag.mcp.asgi import build_mcp_asgi_app, mount_mcp_app
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_SECRET = "mcp-e2e-test-secret-not-real"


class _FakeEmbedder:
    """Deterministic, zero-cost embedder -- relevance-selection is never exercised here."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _FakePipeline:
    """RetrievalPipeline double recording every retrieve/resolve_auth/sanitize_evidence call."""

    def __init__(self) -> None:
        self.retrieve_results: list[SearchResult] = []
        self.retrieve_calls: list[dict] = []
        self.resolve_auth_calls: list[dict] = []
        self.sanitize_calls: list[dict] = []

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        self.retrieve_calls.append(
            {"query": query, "filters": filters, "candidate_k": candidate_k, "auth": auth}
        )
        return self.retrieve_results

    def resolve_auth(self, auth, filters=None, versions=None):
        self.resolve_auth_calls.append({"auth": auth, "filters": filters})
        return auth

    def sanitize_evidence(self, results, auth):
        self.sanitize_calls.append({"result_count": len(results), "auth": auth})
        return [
            r.model_copy(update={"chunk": r.chunk.model_copy(update={"content": "[REDACTED]"})})
            if self._should_redact(r, auth)
            else r
            for r in results
        ]

    def _should_redact(self, result: SearchResult, auth) -> bool:
        # Simulates field redaction: a chunk tagged "secret" is redacted unless the
        # caller's roles include "admin" -- enough to prove sanitize_evidence's
        # *auth argument* actually reaches the redaction decision, without
        # reimplementing the real field_policy machinery.
        if "secret" not in result.chunk.content:
            return False
        roles = auth.roles if auth is not None else []
        return "admin" not in roles

    def expand_with_relationships(self, results, auth=None):
        return results


class _FakeVectorStore:
    """VectorStore double resolving from fixed source/id -> Chunk mappings."""

    def __init__(self) -> None:
        self.chunks_by_source: dict[tuple[str, str], list[Chunk]] = {}
        self.chunks_by_id: dict[str, Chunk] = {}
        self.get_chunks_by_source_calls: list[dict] = []
        self.get_chunks_by_ids_calls: list[dict] = []

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        self.get_chunks_by_source_calls.append(
            {"source": source, "dataset_id": dataset_id, "auth": auth, "limit": limit}
        )
        return list(self.chunks_by_source.get((source, dataset_id), []))

    def get_chunks_by_ids(self, chunk_ids, auth=None):
        self.get_chunks_by_ids_calls.append({"chunk_ids": chunk_ids, "auth": auth})
        return [self.chunks_by_id[c] for c in chunk_ids if c in self.chunks_by_id]

    def list_document_versions(self, dataset_id):
        return []


def _chunk(chunk_id: str, content: str = "hello world", document_id: str = "doc-1") -> Chunk:
    now = datetime.now(UTC)
    meta = ChunkMetadata(
        document_id=document_id,
        chunk_id=chunk_id,
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds1",
    )
    return Chunk(id=chunk_id, content=content, metadata=meta)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(app) -> Iterator[str]:
    """Run `app` via a real uvicorn server on an ephemeral port, in a background thread."""
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


def _token(**claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _mcp_config(*, auth_enabled: bool):
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(
        update={"enabled": auth_enabled, "jwt": jwt_config}
    )
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


async def _call_tool(base_url: str, token: str | None, name: str, arguments: dict):
    """Open a fresh MCP session against `base_url` and call one tool."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx2.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(base_url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)


async def _list_tool_names(base_url: str) -> list[str]:
    async with httpx2.AsyncClient() as http_client:
        async with streamable_http_client(base_url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [t.name for t in result.tools]


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)


@pytest.mark.asyncio
async def test_server_exposes_exactly_the_four_core_tools():
    """Stage 1A exposes only the four core RAG tools -- no business-backend example yet."""
    config = _mcp_config(auth_enabled=False)
    app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    with _serve(app) as base_url:
        names = await _list_tool_names(base_url)
    assert set(names) == {
        "search_knowledge_base",
        "get_document",
        "get_latest_document",
        "get_related_context",
    }


@pytest.mark.asyncio
async def test_search_knowledge_base_threads_verified_identity_into_pipeline_retrieve():
    """A valid JWT's identity -- not any tool argument -- is what reaches pipeline.retrieve()."""
    config = _mcp_config(auth_enabled=True)
    pipeline = _FakePipeline()
    pipeline.retrieve_results = [SearchResult(chunk=_chunk("c1"), score=0.9)]
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url, _token(), "search_knowledge_base", {"query": "hi", "top_k": 3}
        )

    assert result.is_error is False
    assert len(pipeline.retrieve_calls) == 1
    auth = pipeline.retrieve_calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["tenant_alpha_operator"]
    assert pipeline.retrieve_calls[0]["candidate_k"] == 3
    # sanitize_evidence's own resolve_auth call reuses the same resolved context.
    assert pipeline.sanitize_calls[0]["auth"].tenant_id == "tenant_alpha"


@pytest.mark.asyncio
async def test_injected_tenant_id_roles_and_auth_arguments_are_rejected_loudly():
    """A client-supplied tenant_id/roles/auth-shaped argument is rejected, not silently dropped.

    No tool function has a tenant_id/roles parameter to bind these into,
    and `auth` is exclusively resolver-injected, so these keys were
    already structurally unable to influence authorization. But the
    SDK's default argument-model behavior (`extra="ignore"`) used to
    silently drop them rather than reject the call -- a worse failure
    mode than a loud rejection at a security boundary. `rag.mcp.server`
    hardens every tool's argument model to `extra="forbid"`
    (`_harden_argument_schemas`), so this call now fails argument
    validation before the tool function -- and therefore
    `pipeline.retrieve()` -- ever runs at all.
    """
    config = _mcp_config(auth_enabled=True)
    pipeline = _FakePipeline()
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url,
            _token(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"]),
            "search_knowledge_base",
            {
                "query": "hi",
                "tenant_id": "tenant_evil",
                "roles": ["security_admin"],
                "auth": {"tenant_id": "tenant_evil"},
            },
        )

    assert result.is_error is True
    assert pipeline.retrieve_calls == []


@pytest.mark.asyncio
async def test_arbitrary_unknown_argument_is_rejected_loudly():
    """A client-supplied field with no relation to authorization at all is still rejected.

    Proves the hardening is a general unknown-field rejection, not a
    special case hardcoded for tenant_id/roles/auth specifically.
    """
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url, None, "search_knowledge_base", {"query": "hi", "banana": "not a real field"}
        )

    assert result.is_error is True
    assert pipeline.retrieve_calls == []


@pytest.mark.asyncio
async def test_all_declared_optional_arguments_are_still_accepted_after_hardening():
    """Hardening rejects unknown fields only -- every legitimate optional field still validates.

    A regression guard for `_harden_argument_schemas`: `extra="forbid"`
    must reject keys absent from the schema, never a key the schema
    actually declares.
    """
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    pipeline.retrieve_results = [SearchResult(chunk=_chunk("c1"), score=0.9)]
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url,
            None,
            "search_knowledge_base",
            {
                "query": "hi",
                "top_k": 2,
                "content_type": "table",
                "dataset_id": "ds1",
                "as_of": "2026-01-01",
                "require_trust_level": "authoritative",
            },
        )

    assert result.is_error is False
    assert pipeline.retrieve_calls[0]["filters"] == {
        "dataset_id": "ds1",
        "content_type": "table",
    }


@pytest.mark.asyncio
async def test_missing_token_fails_closed_when_auth_enabled():
    """No Authorization header: the tool call fails closed, the pipeline is never reached."""
    config = _mcp_config(auth_enabled=True)
    pipeline = _FakePipeline()
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(base_url, None, "search_knowledge_base", {"query": "hi"})

    assert result.is_error is True
    assert "Missing Authorization header" in result.content[0].text
    assert pipeline.retrieve_calls == []


@pytest.mark.asyncio
async def test_invalid_token_fails_closed_when_auth_enabled():
    """A signature-mismatched token fails closed, the pipeline is never reached."""
    config = _mcp_config(auth_enabled=True)
    pipeline = _FakePipeline()
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())
    bad_token = jwt.encode(
        {"sub": "eve", "tenant_id": "x", "roles": []}, "wrong-secret", algorithm="HS256"
    )

    with _serve(app) as base_url:
        result = await _call_tool(base_url, bad_token, "search_knowledge_base", {"query": "hi"})

    assert result.is_error is True
    assert "Invalid or expired token" in result.content[0].text
    assert pipeline.retrieve_calls == []


@pytest.mark.asyncio
async def test_auth_disabled_yields_unrestricted_access_with_no_token_needed():
    """security.auth.enabled=False: no token required, auth=None reaches the pipeline."""
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    pipeline.retrieve_results = [SearchResult(chunk=_chunk("c1"), score=0.9)]
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(base_url, None, "search_knowledge_base", {"query": "hi"})

    assert result.is_error is False
    assert pipeline.retrieve_calls[0]["auth"] is None


@pytest.mark.asyncio
async def test_search_knowledge_base_top_k_is_clamped_to_max_tool_top_k():
    """The server-side config ceiling wins even when a client requests a larger top_k."""
    config = _mcp_config(auth_enabled=False)
    config = config.model_copy(
        update={"agent": config.agent.model_copy(update={"max_tool_top_k": 4})}
    )
    pipeline = _FakePipeline()
    app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        await _call_tool(base_url, None, "search_knowledge_base", {"query": "hi", "top_k": 20})

    assert pipeline.retrieve_calls[0]["candidate_k"] == 4


@pytest.mark.asyncio
async def test_get_document_dispatches_with_dataset_id_and_query_and_sanitizes_result():
    """get_document reaches VectorStore with the right args; its output passes through sanitize."""
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    vectorstore = _FakeVectorStore()
    vectorstore.chunks_by_source[("policy.md", "ds1")] = [
        _chunk("c1", "the secret rotation policy")
    ]
    app = build_mcp_asgi_app(config, pipeline, vectorstore, _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url,
            None,
            "get_document",
            {"source": "policy.md", "dataset_id": "ds1", "query": "what is the rotation policy"},
        )

    assert result.is_error is False
    assert vectorstore.get_chunks_by_source_calls[0]["source"] == "policy.md"
    assert vectorstore.get_chunks_by_source_calls[0]["dataset_id"] == "ds1"
    # sanitize_evidence ran: this fake's redaction rule replaces "secret" content.
    assert result.structured_content["result"][0]["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_get_document_output_is_authorized_and_not_redacted_for_an_admin_role():
    """sanitize_evidence's redaction decision depends on the real resolved auth, not a stub."""
    config = _mcp_config(auth_enabled=True)
    pipeline = _FakePipeline()
    vectorstore = _FakeVectorStore()
    vectorstore.chunks_by_source[("policy.md", "ds1")] = [
        _chunk("c1", "the secret rotation policy")
    ]
    app = build_mcp_asgi_app(config, pipeline, vectorstore, _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url,
            _token(roles=["admin"]),
            "get_document",
            {"source": "policy.md", "dataset_id": "ds1", "query": "what is the rotation policy"},
        )

    assert result.structured_content["result"][0]["content"] == "the secret rotation policy"


@pytest.mark.asyncio
async def test_get_related_context_dispatches_by_chunk_id():
    """get_related_context resolves its seed chunk via VectorStore.get_chunks_by_ids."""
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    vectorstore = _FakeVectorStore()
    vectorstore.chunks_by_id["c1"] = _chunk("c1", "parent prose")
    app = build_mcp_asgi_app(config, pipeline, vectorstore, _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(base_url, None, "get_related_context", {"chunk_id": "c1"})

    assert result.is_error is False
    assert vectorstore.get_chunks_by_ids_calls[0]["chunk_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_get_document_requires_dataset_id_argument():
    """dataset_id has no default: an omitted call is rejected before the tool even runs."""
    config = _mcp_config(auth_enabled=False)
    app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())

    with _serve(app) as base_url:
        result = await _call_tool(
            base_url, None, "get_document", {"source": "policy.md", "query": "hi"}
        )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_bare_mount_path_and_mounted_lifespan_work_through_a_wrapping_app():
    """Mounted under a real outer app (mirroring rag.api.main), both /mcp and /mcp/ work.

    Every other test in this file serves `build_mcp_asgi_app`'s return
    value directly as uvicorn's own root app: the MCP SDK's session
    manager lifespan starts automatically (uvicorn's own lifespan
    protocol reaches it directly) and there is no `Mount` prefix in the
    way at all -- neither of which is how `rag.api.main` actually uses
    this app. This test instead builds a small wrapping FastAPI app the
    same way `rag.api.main` does: `mount_mcp_app` for the mount (plus
    its bare-mount-path fix) and a composed lifespan that enters the
    mounted sub-app's own `router.lifespan_context` (mirroring
    `rag.api.main`'s own `_lifespan`). Proves two things together: the
    mounted session manager actually starts (a real tool call would hang
    or error on session init otherwise -- this is real "mounted-app
    lifespan coverage", not an inspection of the wiring), and the bare
    mount path (no trailing slash) succeeds directly rather than
    307-redirecting -- the exact failure a real MCP client hit against a
    real Docker container before `mount_mcp_app` existed (see
    ISSUES.md). If either the mount, the middleware, or the lifespan
    composition were wrong, `_call_tool`'s `session.initialize()` would
    raise or hang rather than return a result.
    """
    config = _mcp_config(auth_enabled=False)
    pipeline = _FakePipeline()
    pipeline.retrieve_results = [SearchResult(chunk=_chunk("c1"), score=0.9)]
    mcp_app = build_mcp_asgi_app(config, pipeline, _FakeVectorStore(), _FakeEmbedder())

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
            yield

    wrapper = FastAPI(lifespan=_lifespan)
    mount_mcp_app(wrapper, mcp_app, "/mcp")

    with _serve(wrapper) as base_url:
        root = base_url.rstrip("/")

        bare_result = await _call_tool(
            f"{root}/mcp", None, "search_knowledge_base", {"query": "hi"}
        )
        assert bare_result.is_error is False

        slash_result = await _call_tool(
            f"{root}/mcp/", None, "search_knowledge_base", {"query": "hi"}
        )
        assert slash_result.is_error is False
