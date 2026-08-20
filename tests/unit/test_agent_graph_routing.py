"""Proves the two headline routing invariants.

Simple questions stay on the fast classic-RAG path with zero tool calls, and
complex questions reach the bounded agent loop -- plus config.agent.enabled=False
is a true, zero-extra-call kill-switch.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str = "doc-1_0", content: str = "content") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


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

    def __init__(self, retrieve_results=None, answer_result=None) -> None:
        """Store the fixed results this double's retrieve()/answer() will return."""
        self.retrieve_results = retrieve_results or []
        self.answer_result = answer_result or {
            "answer": "classic answer",
            "sources": [],
            "retrieval_ms": 1.0,
            "generation_ms": 2.0,
            "total_ms": 3.0,
        }
        self.answer_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self.sanitize_calls: list[dict] = []

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return the fixed classic-RAG answer."""
        self.answer_calls.append({"query": query, "filters": filters, "auth": auth})
        return self.answer_result

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return the fixed retrieve results."""
        self.retrieve_calls.append({"query": query, "filters": filters, "auth": auth})
        return self.retrieve_results

    def sanitize_evidence(self, results, auth):
        """Record the call and return results unchanged."""
        self.sanitize_calls.append({"results": results, "auth": auth})
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


def test_simple_question_routes_to_classic_rag_with_no_tool_calls():
    """A `query_type: simple` classification routes to classic RAG with no tool/decompose calls."""
    llm = ScriptedLLM(['{"query_type": "simple", "reasoning": "single fact"}'])
    pipeline = FakePipeline()
    state = AgentState(original_query="What is the retention period?")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert result.route == "classic_rag"
    assert result.state.termination_reason == "synthesized"
    assert result.state.tool_call_history == []
    assert len(pipeline.answer_calls) == 1
    assert len(llm.calls) == 1  # only the classify call -- no decompose/tool-select/etc.


def test_complex_question_reaches_the_bounded_agent_loop_and_synthesizes():
    """A `query_type: complex` classification runs the tool-using agent loop to synthesis."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["find the deployment", "find the rollback procedure"]}',
            '{"tool_name": "search_knowledge_base", '
            '"tool_args": {"query": "deployment", "top_k": 5}}',
            '{"sufficient": true, "reasoning": "enough evidence"}',
            "The rollback procedure is documented in Source 1.",
        ]
    )
    pipeline = FakePipeline(
        retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)],
    )
    state = AgentState(original_query="Which service caused the backlog and what is the rollback?")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert result.route == "agent"
    assert result.state.termination_reason == "synthesized"
    assert result.state.tool_call_history[0].tool_name == "search_knowledge_base"
    assert result.state.tool_call_history[0].success is True
    assert result.state.final_answer == "The rollback procedure is documented in Source 1."
    # Every tool result passed through the universal sanitization path.
    assert len(pipeline.sanitize_calls) == 1


def test_disabled_agent_always_routes_classic_with_zero_llm_calls():
    """config.agent.enabled=False is a true kill-switch -- not even the classify call happens."""
    llm = ScriptedLLM([])  # any call at all would raise IndexError (queue empty)
    pipeline = FakePipeline()
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(enabled=False),
    )

    assert result.route == "classic_rag"
    assert llm.calls == []
    assert len(pipeline.answer_calls) == 1


def test_classify_parse_failure_defaults_to_simple_and_still_routes_classic():
    """A malformed classify response fails safely onto the fast, well-tested classic path."""
    llm = ScriptedLLM(["not valid json at all"])
    pipeline = FakePipeline()
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_json_parse_retries=0),
    )

    assert result.route == "classic_rag"
    assert result.state.query_type == "simple"
