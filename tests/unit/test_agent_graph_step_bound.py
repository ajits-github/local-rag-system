"""Proves max_agent_steps and max_tool_calls each independently bound the run.

Both terminate safely (best-effort synthesis, or insufficient-evidence) --
no infinite loop, ever.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
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


class InfiniteInsufficientLLM:
    """An agent that would loop forever without a bound.

    Always classifies complex, decomposes to one subquestion, picks
    search_knowledge_base, and always reports evidence insufficient with a
    reformulated query. The very last call (whatever it is) is treated as
    synthesis text if reached.
    """

    def __init__(self) -> None:
        """Start with a zero call count."""
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        """Record the call and return the scripted response for whichever component asked."""
        self.calls += 1
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}'
        if "evidence-sufficiency component" in system:
            return '{"sufficient": false, "reformulated_query": "q1 again"}'
        return "final best-effort answer"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double returning one fixed evidence result on every retrieve() call."""

    def __init__(self) -> None:
        """Start with a zero retrieve() call count."""
        self.retrieve_call_count = 0

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return one fixed evidence result."""
        self.retrieve_call_count += 1
        return [SearchResult(chunk=_chunk(), score=0.5)]

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results


class FakeVectorStore:
    """Minimal VectorStore double that only answers health checks."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text):
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


def _agent_config(**overrides):
    """Return `load_config()` with the agent enabled, generous attempt/call caps, and overrides."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(
        update={
            "enabled": True,
            "max_retrieval_attempts": 1000,  # not the bound under test
            "max_tool_calls": 1000,
            **overrides,
        }
    )
    return config.model_copy(update={"agent": agent})


def test_max_agent_steps_terminates_the_run_and_still_answers():
    """An agent that would loop forever ('insufficient' always) stops at max_agent_steps."""
    llm = InfiniteInsufficientLLM()
    pipeline = FakePipeline()
    state = AgentState(original_query="an unanswerable-in-practice question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_agent_steps=6),
    )

    assert result.state.termination_reason == "max_steps"
    assert result.state.step_count >= 6
    # It stopped -- bounded, not infinite.
    assert llm.calls < 1000
    assert result.state.final_answer is not None


def test_max_tool_calls_terminates_independently_of_max_agent_steps():
    """max_tool_calls bounds total tool dispatches even when steps/attempts are generous."""
    llm = InfiniteInsufficientLLM()
    pipeline = FakePipeline()
    state = AgentState(original_query="another hard question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_agent_steps=1000, max_tool_calls=3),
    )

    assert result.state.termination_reason == "max_tool_calls"
    assert pipeline.retrieve_call_count == 3
    assert result.state.final_answer is not None
