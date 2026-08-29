"""Proves the reformulate-and-retry loop is bounded by max_retrieval_attempts.

This bound holds independent of max_agent_steps/max_tool_calls.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk() -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata and weak evidence text."""
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
    return Chunk(id="doc-1_0", content="weak evidence", metadata=metadata)


class ScriptedLLM:
    """LLM double that returns each response in order, recording every call."""

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
    """RetrievalPipeline double returning one fixed weak-evidence result per call."""

    def __init__(self) -> None:
        """Start with no recorded retrieve() calls."""
        self.retrieve_calls: list[dict] = []

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the query and return one fixed weak-evidence result."""
        self.retrieve_calls.append({"query": query})
        return [SearchResult(chunk=_chunk(), score=0.5)]

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization parity is tested elsewhere."""
        return auth

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Fail the test if classic_rag routing is reached for this complex question."""
        raise AssertionError("classic_rag should not run for a complex question")


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
    """Return `load_config()` with the agent enabled, a generous step cap, and `overrides` set."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True, "max_agent_steps": 50, **overrides})
    return config.model_copy(update={"agent": agent})


def test_retrieval_attempts_bounded_and_terminates_with_evidence_gathered():
    """The retry loop stops at max_retrieval_attempts and still synthesizes gathered evidence."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            # attempt 1
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}',
            '{"sufficient": false, "reformulated_query": "q1 retry"}',
            # attempt 2 (max_retrieval_attempts=2 -> this is the last allowed attempt)
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1 retry"}}',
            '{"sufficient": false, "reformulated_query": "q1 retry again"}',
            # finalize
            "best-effort answer",
        ]
    )
    pipeline = FakePipeline()
    state = AgentState(original_query="a hard question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_retrieval_attempts=2, max_tool_calls=50),
    )

    assert result.state.retrieval_attempts == 2
    assert result.state.termination_reason == "max_retrieval_attempts"
    assert len(pipeline.retrieve_calls) == 2
    # Never a third retrieval attempt, even though evidence stayed insufficient.
    assert pipeline.retrieve_calls[1]["query"] == "q1 retry"
    assert result.state.final_answer == "best-effort answer"
