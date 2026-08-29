"""Node-level timing: individual-invocation capture, aggregation, and the LLM-vs-overhead split.

Uses the same ScriptedLLM/FakePipeline/FakeVectorStore/FakeEmbedder
doubles as `test_agent_graph_routing.py`, plus a `SleepyLLM` that adds a
deliberate delay inside `generate()` so `llm_ms` is distinguishable from
`overhead_ms` in a test, not just in production timing noise.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str = "doc-1_0") -> Chunk:
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
    return Chunk(id=chunk_id, content="content", metadata=metadata)


class ScriptedLLM:
    """Returns one queued response per call, in order."""

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


class SleepyLLM(ScriptedLLM):
    """Like ScriptedLLM, but each generate() call sleeps for a fixed duration first."""

    def __init__(self, responses: list[str], sleep_seconds: float) -> None:
        """Store the sleep duration alongside the queued responses."""
        super().__init__(responses)
        self._sleep_seconds = sleep_seconds

    def generate(self, system: str, user: str) -> str:
        """Sleep for the configured duration, then behave like ScriptedLLM."""
        time.sleep(self._sleep_seconds)
        return super().generate(system, user)


class FakePipeline:
    """RetrievalPipeline double; sanitize_evidence is a pass-through."""

    def __init__(self, retrieve_results=None) -> None:
        """Store the fixed results this double's retrieve() will return."""
        self.retrieve_results = retrieve_results or []

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Return a fixed classic-RAG answer."""
        return {
            "answer": "classic answer",
            "sources": [],
            "retrieval_ms": 1.0,
            "generation_ms": 2.0,
            "total_ms": 3.0,
        }

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Return the fixed retrieve results."""
        return self.retrieve_results

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization parity is tested elsewhere."""
        return auth

    def sanitize_evidence(self, results, auth):
        """Return results unchanged."""
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


def test_node_timings_recorded_and_aggregated_across_a_retry_loop():
    """A retry loop (insufficient then sufficient) proves both invocation and aggregate timing."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1", "q2"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            '{"sufficient": false, "reformulated_query": "q1 retry", "reasoning": "not enough"}',
            '{"tool_name": "search_knowledge_base", '
            '"tool_args": {"query": "q1 retry", "top_k": 5}}',
            '{"sufficient": true, "reasoning": "enough now"}',
            "final synthesized answer",
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    state = AgentState(original_query="a multi-hop question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_retrieval_attempts=2, max_agent_steps=20),
    )

    assert result.route == "agent"
    assert result.state.termination_reason == "synthesized"

    # Per-invocation timing on state: tool_select/tool_execute/evidence_sufficiency ran twice.
    raw = result.state.node_timings_ms
    assert len(raw["classify"]) == 1
    assert len(raw["decompose"]) == 1
    assert len(raw["tool_select"]) == 2
    assert len(raw["tool_execute"]) == 2
    assert len(raw["evidence_sufficiency"]) == 2
    assert len(raw["synthesize"]) == 1

    # Aggregate timing on the result: count matches, total_ms sums the invocations.
    agg = result.node_timings_ms
    assert agg["tool_select"].count == 2
    assert agg["tool_select"].total_ms == sum(t.total_ms for t in raw["tool_select"])
    assert agg["tool_select"].mean_ms == agg["tool_select"].total_ms / 2


def test_llm_ms_and_overhead_ms_split_for_an_llm_calling_node():
    """A deliberately slow generate() call shows up as llm_ms, not overhead_ms."""
    sleep_seconds = 0.05
    llm = SleepyLLM(['{"query_type": "simple"}'], sleep_seconds=sleep_seconds)
    pipeline = FakePipeline()
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    classify_timing = result.state.node_timings_ms["classify"][0]
    assert classify_timing.llm_ms is not None
    assert classify_timing.overhead_ms is not None
    # llm_ms should be close to the sleep duration and dominate total_ms;
    # overhead_ms (JSON parsing/validation) should be small in comparison.
    assert classify_timing.llm_ms >= sleep_seconds * 1000 * 0.8
    assert classify_timing.overhead_ms < classify_timing.llm_ms
    reconstructed = classify_timing.llm_ms + classify_timing.overhead_ms
    assert abs(classify_timing.total_ms - reconstructed) < 1.0


def test_execute_tool_has_no_llm_ms_not_a_misleading_zero():
    """execute_tool makes no direct LLM call, so llm_ms/overhead_ms stay None, not 0.0."""
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            '{"sufficient": true}',
            "answer",
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    state = AgentState(original_query="a question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    tool_execute_timing = result.state.node_timings_ms["tool_execute"][0]
    assert tool_execute_timing.llm_ms is None
    assert tool_execute_timing.overhead_ms is None
    assert tool_execute_timing.total_ms >= 0.0

    agg = result.node_timings_ms["tool_execute"]
    assert agg.llm_ms_mean is None
    assert agg.overhead_ms_mean is None


def test_node_token_usage_tracks_per_node_totals():
    """Per-node token usage sums alongside the existing run-wide prompt_tokens/completion_tokens."""

    class TokenTrackingLLM(ScriptedLLM):
        """Reports a fixed token count per call, mimicking OllamaLLM's own tracking attributes."""

        def generate(self, system, user):
            """Record a fixed token count and return the next queued response."""
            self.last_prompt_tokens = 10
            self.last_completion_tokens = 2
            return super().generate(system, user)

    llm = TokenTrackingLLM(['{"query_type": "simple"}'])
    pipeline = FakePipeline()
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert result.node_token_usage["classify"] == {"prompt": 10, "completion": 2}
    assert result.state.prompt_tokens >= 10


def test_agent_run_result_backward_compatible_fields_unchanged():
    """retrieval_ms/generation_ms/total_ms keep their existing agent-route formulas."""
    llm = ScriptedLLM(['{"query_type": "simple"}'])
    pipeline = FakePipeline()
    state = AgentState(original_query="hello")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert result.retrieval_ms >= 0.0
    assert result.generation_ms >= 0.0
    assert result.total_ms >= 0.0
    assert result.llm_call_count >= 1


def test_average_run_latency_with_instrumentation_stays_small(capsys):
    """Rough overhead sanity check, not a strict benchmark (see docs/architecture.md)."""
    durations = []
    for _ in range(20):
        llm = ScriptedLLM(['{"query_type": "simple"}'])
        result = run_agent(
            AgentState(original_query="hello"),
            pipeline=FakePipeline(),
            vectorstore=FakeVectorStore(),
            embedder=FakeEmbedder(),
            llm=llm,
            config=_agent_config(),
        )
        durations.append(result.total_ms)

    mean_ms = sum(durations) / len(durations)
    print(f"mean run latency with instrumentation over {len(durations)} runs: {mean_ms:.3f}ms")
    assert mean_ms < 50.0
