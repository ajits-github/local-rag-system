"""Proves best-effort token accounting.

AgentState.prompt_tokens/completion_tokens accumulate across every LLM call
in a run, read via the same getattr(...) pattern RetrievalPipeline.answer()
already uses. 0 for an LLM that doesn't track them.
"""

from __future__ import annotations

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config


class TokenTrackingLLM:
    """Reports a fixed token count per call, mimicking OllamaLLM's instance-attribute tracking."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses this double's generate() will pop from."""
        self._responses = list(responses)
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def generate(self, system: str, user: str) -> str:
        """Record a fixed token count and return the next queued response."""
        self.last_prompt_tokens = 100
        self.last_completion_tokens = 20
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class NoTokenTrackingLLM:
    """No last_prompt_tokens/last_completion_tokens attributes at all. must never raise."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses this double's generate() will pop from."""
        self._responses = list(responses)

    def generate(self, system: str, user: str) -> str:
        """Return the next queued response."""
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double whose answer() returns a fixed classic-RAG result with tokens."""

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Return a fixed classic-RAG answer with fixed prompt/completion token counts."""
        return {
            "answer": "classic answer",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
            "prompt_tokens": 50,
            "completion_tokens": 10,
        }


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


def _agent_config(enabled: bool):
    """Return `load_config()` with `config.agent.enabled` set to `enabled`."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": enabled})
    return config.model_copy(update={"agent": agent})


def test_classic_route_accumulates_tokens_from_pipeline_answer():
    """When routed classic (agent disabled), state tokens come from pipeline.answer()'s counts."""
    llm = NoTokenTrackingLLM([])
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(enabled=False),
    )

    assert result.state.prompt_tokens == 50
    assert result.state.completion_tokens == 10


def test_agent_route_accumulates_tokens_across_every_decision_call():
    """When routed to the agent, state tokens sum every LLM decision call plus answer()'s own."""
    llm = TokenTrackingLLM(['{"query_type": "simple"}'])
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(enabled=True),
    )

    # Only the classify call happened (routed to classic_rag). one LLM call's worth
    # of tokens (100/20) plus the classic pipeline.answer() call's own (50/10).
    assert result.state.prompt_tokens == 150
    assert result.state.completion_tokens == 30
