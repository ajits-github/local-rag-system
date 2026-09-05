"""Real MCP client/server end-to-end tests for the agent's Stage 2 business tools.

Runs `run_agent()` all the way through, dispatching `get_customer_case`/
`get_case_status`/`update_case_status` via a real `mcp.ClientSession`
against a real (in-process, ASGI-transport) MCP server object built the
same way `rag.api.main` builds it -- not a mocked dispatch function. Uses
`_FakePipeline`/`_FakeVectorStore`/`_FakeEmbedder` doubles (same pattern
as `tests/integration/test_mcp_end_to_end.py`) for the local-tool side,
which these tests never exercise, and the real `rag.mcp.business.store`
synthetic dataset for the business-tool side. No Postgres/Ollama needed --
self-contained, always runs (the business store has no such dependency
either). `_reset_case_store` restores the shared, in-memory case dataset
after every test, since the `update_case_status` tests genuinely mutate it.
"""

from __future__ import annotations

import asyncio
import copy
import time

import jwt
import pytest

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.mcp.asgi import build_mcp_asgi_app
from rag.mcp.business import store as business_store
from rag.mcp.business.schemas import CaseApproval
from rag.retrieval.authorization import AuthorizationContext

_SECRET = "agent-mcp-stage2-e2e-secret-not-real"


@pytest.fixture(autouse=True)
def _reset_case_store():
    original = copy.deepcopy(business_store._SYNTHETIC_CASES)
    yield
    business_store._SYNTHETIC_CASES.clear()
    business_store._SYNTHETIC_CASES.update(original)


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


class ScriptedUpdateCaseStatusLLM:
    """Classifies complex, decomposes once, calls update_case_status, then synthesizes."""

    def __init__(self, case_id: str, new_status: str) -> None:
        self._case_id = case_id
        self._new_status = new_status

    def generate(self, system: str, user: str) -> str:
        """Return the scripted response for whichever decision component asked."""
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return (
                '{"tool_name": "update_case_status", '
                f'"tool_args": {{"case_id": "{self._case_id}", '
                f'"new_status": "{self._new_status}"}}}}'
            )
        if "evidence-sufficiency component" in system:
            return '{"sufficient": true}'
        return f"Here is the outcome of the requested change for case {self._case_id}."

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


def _secure_mcp_config(
    *, monkeypatch, ttl_seconds: int | None = None, business_actions: bool = False
):
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = load_config().model_copy(deep=True)
    config.security.auth.enabled = True
    config.security.auth.jwt.secret_env_var = "JWT_HS256_SECRET"
    config.mcp.enabled = True
    config.mcp.client.enabled = True
    config.mcp.business_actions.enabled = business_actions
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
    remote_calls = [r for r in result.state.tool_call_history if r.tool_name == "get_customer_case"]
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
    remote_calls = [r for r in result.state.tool_call_history if r.tool_name == "get_customer_case"]
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


@pytest.mark.asyncio
async def test_agent_run_reports_approval_required_without_mutating_the_case(monkeypatch):
    """Resolved -> closed on CASE-2001, with no approval attached: no mutation, no retry."""
    config = _secure_mcp_config(monkeypatch=monkeypatch, business_actions=True)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    llm = ScriptedUpdateCaseStatusLLM("CASE-2001", "closed")
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    state = AgentState(original_query="close case CASE-2001", authorization_context=auth)

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
    write_calls = [r for r in result.state.tool_call_history if r.tool_name == "update_case_status"]
    assert len(write_calls) == 1
    assert write_calls[0].success is True
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"
    assert any(c.source == "mcp://business/CASE-2001" for c in result.state.citations)
    assert any("requires approval" in r.chunk.content for r in result.state.retrieved_evidence)


@pytest.mark.asyncio
async def test_agent_run_executes_the_mutation_when_case_approvals_supplied(monkeypatch):
    """The identical request, but with a matching approval, actually mutates the case.

    The caller's roles include case_status_approver alongside their
    ordinary case-access role: the internal token embeds the caller's
    full role set, and the MCP server re-checks that set against
    mcp.business_actions.approval_roles before honoring case_approvals
    (defense in depth, independent of the API-boundary role gate).
    """
    config = _secure_mcp_config(monkeypatch=monkeypatch, business_actions=True)
    mcp_app = build_mcp_asgi_app(config, _FakePipeline(), _FakeVectorStore(), _FakeEmbedder())
    llm = ScriptedUpdateCaseStatusLLM("CASE-2001", "closed")
    auth = AuthorizationContext(
        tenant_id="tenant_beta", roles=["tenant_beta_operator", "case_status_approver"]
    )
    state = AgentState(
        original_query="close case CASE-2001",
        authorization_context=auth,
        case_approvals=[CaseApproval(case_id="CASE-2001", new_status="closed")],
    )

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
    write_calls = [r for r in result.state.tool_call_history if r.tool_name == "update_case_status"]
    assert len(write_calls) == 1
    assert write_calls[0].success is True
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"


def test_remote_update_case_status_failure_does_not_crash_run_agent(monkeypatch):
    """A malformed/missing mcp_app for update_case_status is a recorded failure, not a crash.

    Same reasoning and sync-test shape as
    test_mcp_tool_error_does_not_escape_run_agent above.
    """
    config = _secure_mcp_config(monkeypatch=monkeypatch, business_actions=True)
    llm = ScriptedUpdateCaseStatusLLM("CASE-2001", "closed")
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    state = AgentState(original_query="close case CASE-2001", authorization_context=auth)

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
    failed_calls = [
        r for r in result.state.tool_call_history if r.tool_name == "update_case_status"
    ]
    assert failed_calls and all(r.success is False for r in failed_calls)
    assert len(failed_calls) == 1  # no retry after a write-action failure either
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"
