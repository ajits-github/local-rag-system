"""Real MCP client/server end-to-end tests for the agent's Stage 2 business tools.

Runs `run_agent()` all the way through, dispatching `get_customer_case`/
`get_case_status` via a real `mcp.ClientSession` against a real (in-process,
ASGI-transport) MCP server object built the same way `rag.api.main` builds
it -- not a mocked dispatch function. Uses `_FakePipeline`/`_FakeVectorStore`/
`_FakeEmbedder` doubles (same pattern as
`tests/integration/test_mcp_end_to_end.py`) for the local-tool side, which
these tests never exercise, and the real `rag.mcp.business.store` synthetic
dataset for the business-tool side. No Postgres/Ollama needed -- self-
contained, always runs (the business store has no such dependency either).
"""

from __future__ import annotations

import asyncio
import time

import jwt
import pytest

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.mcp.asgi import build_mcp_asgi_app
from rag.retrieval.authorization import AuthorizationContext

_SECRET = "agent-mcp-stage2-e2e-secret-not-real"


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _FakePipeline:
    """RetrievalPipeline double; these tests only exercise the remote business tools."""

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        return []

    def resolve_auth(self, auth, filters=None, versions=None):
        return auth

    def sanitize_evidence(self, results, auth):
        return results

    def expand_with_relationships(self, results, auth=None):
        return results


class _FakeVectorStore:
    def health_check(self) -> bool:
        return True


class ScriptedRemoteToolLLM:
    """Classifies complex, decomposes once, calls the named remote tool, then synthesizes."""

    def __init__(self, tool_name: str, case_id: str) -> None:
        self._tool_name = tool_name
        self._case_id = case_id
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        """Record the call and return the scripted response for whichever component asked."""
        self.calls.append((system, user))
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return (
                f'{{"tool_name": "{self._tool_name}", '
                f'"tool_args": {{"case_id": "{self._case_id}"}}}}'
            )
        if "evidence-sufficiency component" in system:
            return '{"sufficient": true}'
        return f"Case {self._case_id}'s status is described in the evidence above."

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class NeverSufficientLLM:
    """Same shape as ScriptedRemoteToolLLM but never reports evidence sufficient.

    Used for the denial-path tests: since a denied/nonexistent case
    yields zero evidence every retry, this proves the run terminates
    safely (max_agent_steps/insufficient_evidence) rather than hanging
    or crashing when the business tool never returns anything.
    """

    def __init__(self, tool_name: str, case_id: str) -> None:
        self._tool_name = tool_name
        self._case_id = case_id

    def generate(self, system: str, user: str) -> str:
        """Record the call and return the scripted response for whichever component asked."""
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return (
                f'{{"tool_name": "{self._tool_name}", '
                f'"tool_args": {{"case_id": "{self._case_id}"}}}}'
            )
        if "evidence-sufficiency component" in system:
            return '{"sufficient": false, "reformulated_query": "q1 again"}'
        return "no evidence available"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def _secure_mcp_config(*, monkeypatch, ttl_seconds: int | None = None):
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = load_config().model_copy(deep=True)
    config.security.auth.enabled = True
    config.security.auth.jwt.secret_env_var = "JWT_HS256_SECRET"
    config.mcp.enabled = True
    config.mcp.client.enabled = True
    config.agent.enabled = True
    config.agent.max_retrieval_attempts = 1000
    config.agent.max_agent_steps = 12
    if ttl_seconds is not None:
        config.mcp.client.internal_token_ttl_seconds = ttl_seconds
    return config


@pytest.mark.asyncio
async def test_agent_run_synthesizes_an_answer_citing_an_authorized_business_case(monkeypatch):
    """Full run_agent() -> get_customer_case -> synthesized answer with an mcp_remote citation."""
    config = _secure_mcp_config(monkeypatch=monkeypatch)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    llm = ScriptedRemoteToolLLM("get_customer_case", "CASE-1001")
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    state = AgentState(original_query="what is case CASE-1001 about", authorization_context=auth)

    async with mcp_app.router.lifespan_context(mcp_app):
        result = await asyncio.to_thread(
            run_agent,
            state,
            pipeline=_FakePipeline(),
            vectorstore=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            llm=llm,
            config=config,
            mcp_app=mcp_app,
        )

    assert result.state.termination_reason == "synthesized"
    remote_calls = [
        r for r in result.state.tool_call_history if r.tool_name == "get_customer_case"
    ]
    assert remote_calls and remote_calls[0].success is True
    assert any(c.source == "mcp://business/CASE-1001" for c in result.state.citations)
    assert any(c.chunk_id == "mcp:get_customer_case:CASE-1001" for c in result.state.citations)


@pytest.mark.asyncio
async def test_agent_run_denies_wrong_role_same_tenant_case_access(monkeypatch):
    """CASE-1002 is admin-only; a tenant_alpha_operator gets no evidence, no crash."""
    config = _secure_mcp_config(monkeypatch=monkeypatch)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    llm = NeverSufficientLLM("get_customer_case", "CASE-1002")
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    state = AgentState(original_query="what is case CASE-1002 about", authorization_context=auth)

    async with mcp_app.router.lifespan_context(mcp_app):
        result = await asyncio.to_thread(
            run_agent,
            state,
            pipeline=_FakePipeline(),
            vectorstore=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            llm=llm,
            config=config,
            mcp_app=mcp_app,
        )

    assert result.state.retrieved_evidence == []
    assert result.state.citations == []
    assert result.state.termination_reason in ("insufficient_evidence", "max_steps")
    remote_calls = [
        r for r in result.state.tool_call_history if r.tool_name == "get_customer_case"
    ]
    assert remote_calls and all(r.success is True and r.result_count == 0 for r in remote_calls)


@pytest.mark.asyncio
async def test_agent_run_denies_cross_tenant_case_access_without_support_role(monkeypatch):
    """CASE-1001 belongs to tenant_alpha; a plain tenant_beta_operator gets no evidence."""
    config = _secure_mcp_config(monkeypatch=monkeypatch)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    llm = NeverSufficientLLM("get_customer_case", "CASE-1001")
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    state = AgentState(original_query="what is case CASE-1001 about", authorization_context=auth)

    async with mcp_app.router.lifespan_context(mcp_app):
        result = await asyncio.to_thread(
            run_agent,
            state,
            pipeline=_FakePipeline(),
            vectorstore=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            llm=llm,
            config=config,
            mcp_app=mcp_app,
        )

    assert result.state.retrieved_evidence == []
    assert result.state.citations == []


def test_mcp_tool_error_does_not_escape_run_agent(monkeypatch):
    """A malformed/missing mcp_app (an MCP-layer failure) is a recorded failure, not a crash.

    Simulates the "server unreachable" family of failures without a real
    network: transport='asgi' with no mcp_app wired is exactly the
    failure shape a genuine connection/timeout error would also produce
    (an exception raised inside the MCP client call, caught by
    _execute_tool's generic envelope) -- run_agent() must still return a
    normal AgentRunResult, never raise. Deliberately a plain sync test
    (not async/pytest.mark.asyncio): run_agent()'s own internal
    anyio.run() bridge requires no event loop already running on its
    calling thread, mirroring exactly how both real callers invoke it
    (a sync FastAPI route handler thread, or run_in_threadpool's worker
    thread) -- calling it directly from an async test function's own
    already-running loop would raise a *different*, unrelated
    "already running" error instead of exercising the failure this test
    is actually about.
    """
    config = _secure_mcp_config(monkeypatch=monkeypatch)
    llm = NeverSufficientLLM("get_case_status", "CASE-1001")
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    state = AgentState(original_query="status of case CASE-1001", authorization_context=auth)

    result = run_agent(
        state,
        pipeline=_FakePipeline(),
        vectorstore=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
        llm=llm,
        config=config,
        mcp_app=None,  # deliberately not wired -> _call_tool_async raises RuntimeError
    )

    assert result.state.final_answer is not None
    failed_calls = [r for r in result.state.tool_call_history if r.tool_name == "get_case_status"]
    assert failed_calls and all(r.success is False for r in failed_calls)
    assert all("mcp_app" in (r.error or "") for r in failed_calls)


@pytest.mark.asyncio
async def test_original_caller_credential_never_reaches_the_mcp_server(monkeypatch):
    """The internal token the MCP server actually receives is never any original end-user token.

    Builds a stand-in "original" JWT (as if it had arrived on the inbound
    HTTP request) purely to prove it never appears anywhere in the
    request `run_agent()` sends onward: AgentState/AuthorizationContext
    structurally carry no raw-token field at all (see rag.agent.state's
    docstring), and this test additionally captures the actual
    Authorization header the MCP server received and decodes it to prove
    its subject is the fixed internal-service marker, never the
    "original" caller's subject.
    """
    config = _secure_mcp_config(monkeypatch=monkeypatch)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())

    now = int(time.time())
    original_caller_token = jwt.encode(
        {
            "sub": "real-end-user-alice",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
            "iat": now,
            "exp": now + 3600,
        },
        _SECRET,
        algorithm="HS256",
    )

    captured_headers: list[dict[str, str]] = []

    async def _capturing_app(scope, receive, send):
        """Wrap `mcp_app` as a plain ASGI callable, capturing each request's headers.

        Not an instance-attribute override.

        Reassigning `mcp_app.__call__` directly would silently not work:
        Python's implicit special-method lookup for `mcp_app(...)` (which
        is exactly how `httpx2.ASGITransport` invokes it) resolves
        `__call__` on the *type*, not the instance, so an instance-level
        override is never seen. A real wrapping callable, passed as the
        `mcp_app` argument instead of the real object, is what actually
        gets invoked.
        """
        if scope.get("type") == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            captured_headers.append(headers)
        await mcp_app(scope, receive, send)

    llm = ScriptedRemoteToolLLM("get_case_status", "CASE-1001")
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    state = AgentState(original_query="status of case CASE-1001", authorization_context=auth)

    async with mcp_app.router.lifespan_context(mcp_app):
        result = await asyncio.to_thread(
            run_agent,
            state,
            pipeline=_FakePipeline(),
            vectorstore=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            llm=llm,
            config=config,
            mcp_app=_capturing_app,
        )

    assert result.state.termination_reason == "synthesized"
    auth_headers = [h["authorization"] for h in captured_headers if "authorization" in h]
    assert auth_headers, "expected at least one Authorization header to reach the MCP server"
    for header in auth_headers:
        scheme, _, token = header.partition(" ")
        assert scheme == "Bearer"
        assert token != original_caller_token
        claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
        assert claims["sub"] == config.mcp.client.internal_token_subject
        assert claims["sub"] != "real-end-user-alice"
