"""Proves the agent can return a genuine insufficient-evidence response.

This covers the case where nothing useful was ever gathered. no LLM
synthesis call, and never a fabricated answer.
"""

from __future__ import annotations

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config


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


class EmptyResultsPipeline:
    """RetrievalPipeline double whose retrieve() always returns no results."""

    def __init__(self) -> None:
        """Start with a zero retrieve() call count."""
        self.retrieve_calls = 0

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return an empty result list, simulating a total miss."""
        self.retrieve_calls += 1
        return []  # every search comes back empty

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
    """Return `load_config()` with the agent enabled and `overrides` applied."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True, **overrides})
    return config.model_copy(update={"agent": agent})


def test_zero_evidence_gathered_returns_insufficient_evidence_without_a_synthesis_call():
    """When every retrieval attempt comes back empty, the agent refuses without a synthesis call."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}',
            '{"sufficient": false, "reformulated_query": "q1 again"}',
        ]
    )
    pipeline = EmptyResultsPipeline()
    state = AgentState(original_query="a question with no answer in the corpus")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_retrieval_attempts=1, max_tool_calls=5),
    )

    assert result.state.termination_reason == "max_retrieval_attempts"
    assert result.state.retrieved_evidence == []
    assert result.state.final_answer is not None
    assert "don't have enough" in result.state.final_answer.lower()
    assert result.state.citations == []
    # No synthesis LLM call was made. exactly the 4 decision calls, nothing more.
    assert len(llm.calls) == 4
