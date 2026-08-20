"""Proves tool failures are handled safely.

Both invalid LLM-supplied arguments and an exception raised by the tool
itself are recorded, never propagated, never crash the request.
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
    return Chunk(id="doc-1_0", content="content", metadata=metadata)


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
    """RetrievalPipeline double returning one fixed result on every retrieve() call."""

    def __init__(self) -> None:
        """Start with a zero retrieve() call count."""
        self.retrieve_calls = 0

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return one fixed result."""
        self.retrieve_calls += 1
        return [SearchResult(chunk=_chunk(), score=0.5)]

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results


class FakeVectorStore:
    """VectorStore double whose get_chunks_by_source() must never be reached in these tests."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        """Fail the test if reached -- these tests never supply a dataset_id."""
        raise AssertionError("should not be reached: dataset_id is missing")


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


def test_invalid_tool_arguments_are_rejected_and_the_run_recovers():
    """An invalid tool call fails safely, and a subsequent valid tool call still succeeds."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            # First attempt: smuggled "roles" key -> SearchKnowledgeBaseArgs rejects it.
            '{"tool_name": "search_knowledge_base", '
            '"tool_args": {"query": "q1", "roles": ["security_admin"]}}',
            '{"sufficient": false, "reformulated_query": "q1 retry"}',
            # Second attempt: valid.
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1 retry"}}',
            '{"sufficient": true}',
            "final answer",
        ]
    )
    pipeline = FakePipeline()
    state = AgentState(original_query="a question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_retrieval_attempts=5, max_tool_calls=5),
    )

    first_record, second_record = result.state.tool_call_history
    assert first_record.success is False
    assert first_record.error == "invalid_arguments"
    assert second_record.success is True
    assert result.state.final_answer == "final answer"
    assert pipeline.retrieve_calls == 1  # only the valid call actually reached the pipeline


def test_tool_execution_error_is_recorded_and_does_not_crash_the_run():
    """A ToolExecutionError (missing dataset_id) is a recorded failure, not an unhandled raise."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "get_document", "tool_args": {"source": "policy.md"}}',
            # No dataset_id was ever set on state.filters, so get_document raises --
            # evaluate_evidence still runs once on the (empty) evidence gathered so far.
            '{"sufficient": false}',
        ]
    )
    pipeline = FakePipeline()
    state = AgentState(original_query="a question", filters=None)

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_tool_calls=1),
    )

    assert result.state.tool_call_history[0].success is False
    assert result.state.tool_call_history[0].tool_name == "get_document"
    assert result.state.termination_reason == "max_tool_calls"
    # No evidence was gathered (the only tool call failed) -> insufficient-evidence response.
    assert result.state.final_answer is not None
