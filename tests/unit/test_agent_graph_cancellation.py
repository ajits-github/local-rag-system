"""Proves `run_agent`'s `cancel_event` cooperatively stops the bounded tool-call loop early.

Before this fix, an SSE client disconnect (`POST /agent/query/stream`)
stopped further streaming but had no way to actually cancel the
already-running agent turn: `run_agent` had no cancellation hook of any
kind, so an abandoned stream still burned a full run's worth of LLM/DB/
retrieval work. `run_agent`'s bounded tool-call loop now checks a
`threading.Event` once per iteration and, once set, stops immediately
(labeling `termination_reason="cancelled"`) and skips the final synthesis
LLM call entirely, unlike every other termination reason.

Uses the same fake-pipeline/scripted-LLM style as
`tests/unit/test_agent_graph_step_bound.py`.
"""

from __future__ import annotations

import threading
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


class FakePipeline:
    """RetrievalPipeline double returning one fixed evidence result on every retrieve() call."""

    def __init__(self) -> None:
        """Start with a zero retrieve() call count."""
        self.retrieve_call_count = 0

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return one fixed evidence result."""
        self.retrieve_call_count += 1
        return [SearchResult(chunk=_chunk(), score=0.5)]

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization parity is tested elsewhere."""
        return auth

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


class InfiniteInsufficientLLM:
    """An agent that would loop forever without a bound.

    Always classifies complex, decomposes to one subquestion, picks
    search_knowledge_base, and always reports evidence insufficient with a
    reformulated query -- identical shape to
    `test_agent_graph_step_bound.py`'s own fixture, reused here to prove
    cancellation (not a step/tool-call/retrieval-attempt bound) is what
    actually stopped this run.
    """

    def __init__(self, cancel_event: threading.Event | None = None, cancel_after: int = 0) -> None:
        """Start with a zero call count; optionally set `cancel_event` after N calls."""
        self.calls = 0
        self._cancel_event = cancel_event
        self._cancel_after = cancel_after

    def generate(self, system: str, user: str) -> str:
        """Record the call, optionally trip the cancel event, and answer whichever node asked."""
        self.calls += 1
        if (
            self._cancel_event is not None
            and self._cancel_after
            and self.calls >= self._cancel_after
        ):
            self._cancel_event.set()
        if "routing component" in system:
            return '{"query_type": "complex"}'
        if "decomposition component" in system:
            return '{"subquestions": ["q1"]}'
        if "tool-selection component" in system:
            return '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}'
        if "evidence-sufficiency component" in system:
            return '{"sufficient": false, "reformulated_query": "q1 again"}'
        return "final best-effort answer"  # pragma: no cover -- synthesis is skipped once cancelled

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def _agent_config(**overrides):
    """Return `load_config()` with the agent enabled and generous bounds, plus overrides."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(
        update={
            "enabled": True,
            "max_agent_steps": 1000,
            "max_retrieval_attempts": 1000,
            "max_tool_calls": 1000,
            **overrides,
        }
    )
    return config.model_copy(update={"agent": agent})


def test_cancel_event_set_before_the_run_stops_the_tool_loop_before_it_starts():
    """A cancel_event already set when run_agent starts never dispatches a single tool call.

    classify/decompose still run once each (cancellation is only checked at
    the top of the bounded tool-call loop), but the loop itself never
    executes an iteration, and the final synthesis LLM call is skipped
    entirely -- proving this run did not execute anywhere near its full
    node sequence.
    """
    cancel_event = threading.Event()
    cancel_event.set()
    llm = InfiniteInsufficientLLM()
    pipeline = FakePipeline()
    state = AgentState(original_query="a question nobody is waiting for the answer to")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
        cancel_event=cancel_event,
    )

    assert result.state.termination_reason == "cancelled"
    assert result.state.final_answer is None
    assert result.state.citations == []
    assert pipeline.retrieve_call_count == 0
    # Only classify + decompose ran; no tool_select/evidence_sufficiency/synthesize call.
    assert llm.calls == 2


def test_cancel_event_set_mid_run_stops_before_the_next_loop_iteration():
    """A cancel_event tripped partway through the tool loop stops at the next checkpoint.

    Simulates a disconnect detected while the first tool_select/
    tool_execute/evidence_sufficiency iteration is already in flight: that
    iteration is allowed to finish (cancellation never interrupts a node
    call already in progress), but the second iteration never starts, and
    the loop would otherwise run forever (`InfiniteInsufficientLLM` always
    reports insufficient evidence).
    """
    cancel_event = threading.Event()
    # classify(1) + decompose(2) + tool_select(3) + evidence_sufficiency(4):
    # trip the event once the first full iteration's LLM calls are done, so
    # the *second* iteration's tool_select (call 5) never happens.
    llm = InfiniteInsufficientLLM(cancel_event=cancel_event, cancel_after=4)
    pipeline = FakePipeline()
    state = AgentState(original_query="a question whose caller disconnects mid-run")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
        cancel_event=cancel_event,
    )

    assert result.state.termination_reason == "cancelled"
    assert result.state.final_answer is None
    # Exactly one search_knowledge_base dispatch (the in-flight iteration), never a second.
    assert pipeline.retrieve_call_count == 1
    assert llm.calls == 4


def test_no_cancel_event_runs_the_full_loop_unaffected():
    """Omitting cancel_event (the default) never cancels; bounded-run behavior is unchanged."""
    llm = InfiniteInsufficientLLM()
    pipeline = FakePipeline()
    state = AgentState(original_query="a normal question with no cancellation in play")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(max_agent_steps=6),
    )

    assert result.state.termination_reason == "max_steps"
    assert result.state.final_answer is not None
