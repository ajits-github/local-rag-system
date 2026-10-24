"""Shared fixtures for the multi-agent experiment's test suite.

No Postgres/Ollama here, same as `langgraph_experiment/tests/conftest.py`:
`FakePipeline`/`FakeLLM`/`FakePromptTemplate` mock at the narrowest point
that avoids real network/model I/O (this repo's `tests/unit` convention --
see CLAUDE.md's "Testing conventions" section), and the business branch
exercises the real, self-contained, in-memory `rag.mcp.business.store`.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.graph import build_graph
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
    """A narrow `RetrievalPipeline` stand-in: only `.retrieve()` is used by `knowledge_agent`.

    `results_by_call` lets a test script a *sequence* of return values
    (e.g. empty on the first call, non-empty on a retry) -- if fewer
    entries than calls are given, the last entry repeats. `raises`, when
    set, is raised on every call instead (for the "specialist fails"
    scenarios), taking priority over `results_by_call`.
    """

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        results_by_call: list[list[SearchResult]] | None = None,
        raises: type[Exception] | None = None,
    ) -> None:
        self._results_by_call = results_by_call if results_by_call is not None else [results or []]
        self._raises = raises
        self.calls: list[dict] = []

    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        """Record the call and return the next scripted result (or raise `self._raises`)."""
        self.calls.append({"query": query, "filters": filters, "candidate_k": candidate_k})
        if self._raises is not None:
            raise self._raises("simulated retrieval failure")
        index = min(len(self.calls) - 1, len(self._results_by_call) - 1)
        return self._results_by_call[index]


class FakeLLM:
    """A narrow `LLM` stand-in: only `.generate()` is used by `orchestrator.final_synthesis`."""

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        """Record the call and return the fixed `self.answer`."""
        self.calls.append((system, user))
        return self.answer

    def health_check(self) -> bool:
        """Report a healthy status unconditionally."""
        return True


class FakePromptTemplate:
    """A narrow `PromptTemplate` stand-in: only `.render()` is used by final synthesis."""

    def render(self, **kwargs):
        """Return a fixed system prompt plus a user prompt echoing `query`/`context`."""
        return ("system prompt", f"query={kwargs.get('query')} context={kwargs.get('context')}")


@pytest.fixture(autouse=True)
def _restore_synthetic_cases():
    """Snapshot/restore `rag.mcp.business.store`'s shared in-memory case dict.

    Identical pattern to `langgraph_experiment/tests/conftest.py`'s own
    autouse fixture and production's `tests/unit/
    test_mcp_business_case_actions.py` -- `update_case_status` mutates
    process-global state, so a test that approves a real mutation must
    not leak that change into the next test.
    """
    snapshot = copy.deepcopy(case_store._SYNTHETIC_CASES)
    yield
    case_store._SYNTHETIC_CASES.clear()
    case_store._SYNTHETIC_CASES.update(snapshot)


@pytest.fixture
def deps() -> GraphDeps:
    """Build a `GraphDeps` with the real case store, no retrieval pipeline."""
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
    """Build a `GraphDeps` with a fake retrieval pipeline/LLM, one evidence chunk by default."""
    config = load_config()
    result = make_search_result(
        "c1", "docs/api-policy.md", "The API request timeout is 30 seconds."
    )
    return GraphDeps(
        config=config,
        pipeline=FakePipeline([result]),
        llm=FakeLLM("The API request timeout policy is 30 seconds, per docs/api-policy.md."),
        rag_prompt_template=FakePromptTemplate(),
        approval_roles=("case_status_approver",),
        cross_tenant_support_roles=("techfusion_support",),
    )


@pytest.fixture
def graph(deps: GraphDeps):
    """Build a compiled graph against a fresh in-memory checkpointer (no SQLite file)."""
    return build_graph(deps, InMemorySaver())


@pytest.fixture
def graph_with_retrieval(deps_with_retrieval: GraphDeps):
    """Build a compiled graph wired with the fake retrieval pipeline/LLM fixtures."""
    return build_graph(deps_with_retrieval, InMemorySaver())


@pytest.fixture
def thread_config():
    """Build a fresh, unique thread config for one test's run."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
