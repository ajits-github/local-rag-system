"""Shared fixtures for the LangGraph experiment's test suite.

Most tests here need no Postgres/Ollama at all: every test exercises the
case-store branch only (`rag.mcp.business.store` is a self-contained,
in-memory backend -- see `wiring.GraphDeps`'s docstring), so a `GraphDeps`
built without a `RetrievalPipeline`/`LLM`/`action_ledger` is all any test
needs. The production-hardening idempotency-ledger tests are the one
exception (`ActionLedger` is Postgres-backed by design -- see
`idempotency.py`); those use the `postgres_deps`/`require_postgres`
fixtures below and self-skip when Postgres isn't reachable, mirroring
this repo's own `tests/integration/conftest.py` convention.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import psycopg2
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_experiment.graph import build_graph
from langgraph_experiment.idempotency import ActionLedger
from langgraph_experiment.wiring import GraphDeps
from rag.config import load_config
from rag.mcp.business import store as case_store
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def make_search_result(
    chunk_id: str, source: str, content: str, score: float = 0.9
) -> SearchResult:
    """Build a minimal, valid `SearchResult` for a fake retrieval pipeline's return value."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        source=source,
        source_type="markdown",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    chunk = Chunk(id=chunk_id, content=content, metadata=metadata)
    return SearchResult(chunk=chunk, score=score, origin="retrieved")


class FakePipeline:
    """A narrow `RetrievalPipeline` stand-in: only `.retrieve()` is used by `nodes.retrieve`."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results if results is not None else []

    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        return self._results


class FakeLLM:
    """A narrow `LLM` stand-in: only `.generate()` is used by `nodes.synthesize_rag`."""

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer

    def health_check(self) -> bool:
        return True


class FakePromptTemplate:
    """A narrow `PromptTemplate` stand-in: only `.render()` is used by `nodes.synthesize_rag`."""

    def render(self, **kwargs):
        return ("system prompt", f"query={kwargs.get('query')} context={kwargs.get('context')}")


@pytest.fixture(autouse=True)
def _restore_synthetic_cases():
    """Snapshot/restore `rag.mcp.business.store`'s shared in-memory case dict.

    Mirrors the identical autouse fixture in `tests/unit/
    test_mcp_business_case_actions.py` (production's own test file for
    this same module): `update_case_status` mutates process-global state,
    so a test that approves a real mutation must not leak that change
    into the next test.
    """
    snapshot = copy.deepcopy(case_store._SYNTHETIC_CASES)
    yield
    case_store._SYNTHETIC_CASES.clear()
    case_store._SYNTHETIC_CASES.update(snapshot)


@pytest.fixture
def deps() -> GraphDeps:
    """A `GraphDeps` with the real case store, no retrieval pipeline."""
    config = load_config()
    return GraphDeps(
        config=config,
        pipeline=None,
        llm=None,
        rag_prompt_template=None,
        approval_roles=("case_status_approver",),
        cross_tenant_support_roles=("techfusion_support",),
    )


@pytest.fixture
def deps_with_retrieval() -> GraphDeps:
    """A `GraphDeps` with a fake retrieval pipeline/LLM, for the read-only RAG branch.

    Mocks at the narrowest point that avoids real network/model I/O
    (`FakePipeline`/`FakeLLM`/`FakePromptTemplate` in this module),
    matching this repo's `tests/unit` convention -- see CLAUDE.md's
    "Testing conventions" section.
    """
    config = load_config()
    result = make_search_result("c1", "docs/policy.md", "Passwords rotate every 90 days.")
    return GraphDeps(
        config=config,
        pipeline=FakePipeline([result]),
        llm=FakeLLM("Passwords must be rotated every 90 days per docs/policy.md."),
        rag_prompt_template=FakePromptTemplate(),
    )


@pytest.fixture
def graph(deps: GraphDeps):
    """A compiled graph against a fresh in-memory checkpointer (no SQLite file)."""
    checkpointer = InMemorySaver()
    return build_graph(deps, checkpointer)


@pytest.fixture
def graph_with_retrieval(deps_with_retrieval: GraphDeps):
    """A compiled graph wired with the fake retrieval pipeline/LLM fixtures."""
    checkpointer = InMemorySaver()
    return build_graph(deps_with_retrieval, checkpointer)


@pytest.fixture
def thread_config():
    """A fresh, unique thread config for one test's run."""
    import uuid

    return {"configurable": {"thread_id": str(uuid.uuid4())}}


@pytest.fixture
def require_postgres(deps: GraphDeps):
    """Skip a test cleanly (not fail) when the local Postgres isn't reachable.

    Mirrors `tests/integration/conftest.py`'s identical-purpose fixture
    for the main test suite -- a missing `make up` is an environment
    fact, not a test failure.
    """
    try:
        conn = psycopg2.connect(deps.config.database_url(), connect_timeout=2)
        conn.close()
    except psycopg2.OperationalError:
        pytest.skip("Postgres not reachable; run `make up` (or `docker compose up -d postgres`)")


@pytest.fixture
def postgres_deps(deps: GraphDeps, require_postgres) -> GraphDeps:
    """A `GraphDeps` like `deps`, plus a real, schema-ensured `ActionLedger`."""
    ledger = ActionLedger(deps.config.database_url())
    ledger.ensure_schema()
    return GraphDeps(
        config=deps.config,
        pipeline=None,
        llm=None,
        rag_prompt_template=None,
        approval_roles=deps.approval_roles,
        cross_tenant_support_roles=deps.cross_tenant_support_roles,
        action_ledger=ledger,
    )
