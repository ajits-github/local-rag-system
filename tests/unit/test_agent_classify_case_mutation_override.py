"""Proves the deterministic case-mutation routing backstop added to `_classify_query`.

The specialized-tool evaluation (see ISSUES.md) found `agent_classify_v2`
misrouting phrasings like "please close case X" as "simple", which skips
the agent loop -- and therefore MCP's `update_case_status` -- entirely.
`_looks_like_case_mutation_request` (rag.agent.graph) forces such queries
onto the agent path regardless of what the LLM classified them as. These
tests use a `ScriptedLLM` that deliberately classifies every query
"simple" (reproducing the exact misclassification bug) so a passing
`route == "agent"` assertion proves the override, not the model's own
judgment, decided the route.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.agent.graph import _looks_like_case_mutation_request, run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk() -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="doc-1_0",
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id="doc-1_0", content="some evidence", metadata=metadata)


class ScriptedLLM:
    """Returns one queued response per call, in order; records every call."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses this double's generate() will pop from."""
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        """Record `system`/`user` and return the next queued response."""
        self.calls.append((system, user))
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double recording every answer()/retrieve()/sanitize_evidence() call."""

    def __init__(self) -> None:
        """Start with empty call logs and a fixed classic-RAG answer."""
        self.answer_result = {
            "answer": "classic answer",
            "sources": [],
            "retrieval_ms": 1.0,
            "generation_ms": 2.0,
            "total_ms": 3.0,
        }
        self.answer_calls: list[dict] = []

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return the fixed classic-RAG answer."""
        self.answer_calls.append({"query": query, "filters": filters, "auth": auth})
        return self.answer_result

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Return one fixed evidence result, enough to reach synthesis."""
        return [SearchResult(chunk=_chunk(), score=0.9)]

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization parity is tested elsewhere."""
        return auth

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results

    def expand_with_relationships(self, results, auth=None):
        """Return results unchanged; not exercised by these tests."""
        return results


class FakeVectorStore:
    """Minimal VectorStore double that only answers health checks."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


def _agent_config(**overrides):
    """Return `load_config()` with the agent enabled and `overrides` applied."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True, **overrides})
    return config.model_copy(update={"agent": agent})


def _run_forced_complex_flow(query: str) -> tuple[str, ScriptedLLM]:
    """Run `run_agent` with a classifier that always says "simple".

    The classify call always returns `simple`; if the run still reaches
    the agent loop, only the deterministic override could have caused
    it. The remaining scripted calls drive a minimal, one-tool
    convergence so the run can reach synthesis if it does take the agent
    path.
    """
    llm = ScriptedLLM(
        [
            '{"query_type": "simple", "reasoning": "misclassified by the model"}',
            '{"subquestions": ["find the requested case information"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "case lookup"}}',
            '{"sufficient": true}',
            "final synthesized answer",
        ]
    )
    pipeline = FakePipeline()
    state = AgentState(original_query=query)

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )
    return result.route, llm


@pytest.mark.parametrize(
    "query",
    [
        "As a Tenant Beta operator, please close case CASE-2001.",
        "As a Tenant Alpha operator, please resolve case CASE-1001.",
        "As a Tenant Alpha operator, please change case CASE-1001's status to in_progress.",
        "Please reopen case CASE-1003 by setting its status back to open.",
        "Update case CASE-1001's status to resolved.",
    ],
)
def test_case_mutation_phrasing_is_detected(query):
    """Every phrasing the task names as a required action request is flagged."""
    assert _looks_like_case_mutation_request(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the current status and priority of customer support case CASE-1001?",
        "As a Tenant Alpha operator, what is the current status of case CASE-2001?",
        "As a Tenant Alpha operator, what is case CASE-1002 about?",
        "What is the maximum file size allowed for a single upload?",
        "How many days after cancellation is a customer's account data deleted?",
    ],
)
def test_read_only_and_ordinary_questions_are_not_flagged(query):
    """Read-only case questions and ordinary knowledge questions never trigger the override."""
    assert _looks_like_case_mutation_request(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "As a Tenant Beta operator, please close case CASE-2001.",
        "As a Tenant Alpha operator, please resolve case CASE-1001.",
        "As a Tenant Alpha operator, please change case CASE-1001's status to in_progress.",
    ],
)
def test_action_requests_route_to_agent_even_when_the_model_says_simple(query):
    """The deterministic override forces the agent route despite a "simple" classification."""
    route, llm = _run_forced_complex_flow(query)

    assert route == "agent"
    # classify + decompose + tool_select + evidence_sufficiency + synthesize.
    assert len(llm.calls) == 5


def test_read_only_case_status_question_still_routes_classic_rag_when_model_says_simple():
    """A read-only case question is unaffected: the model's own "simple" call still decides."""
    route, llm = _run_forced_complex_flow(
        "As a Tenant Alpha operator, what is the current status of case CASE-2001?"
    )

    assert route == "classic_rag"
    assert len(llm.calls) == 1  # only the classify call ran.


def test_normal_simple_rag_question_still_routes_classic_rag():
    """An ordinary simple RAG question with no case-mutation phrasing is unaffected."""
    route, llm = _run_forced_complex_flow(
        "What is the maximum number of retries allowed for a failed API call?"
    )

    assert route == "classic_rag"
    assert len(llm.calls) == 1
