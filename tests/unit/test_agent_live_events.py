"""Live-event emission: safe payload shape, correct ordering, and no sensitive content.

`AgentEvent` has no free-text field at all (see `rag.agent.events`), so
"no chain-of-thought/reasoning/evidence-content leaks" is checked both by
construction (the model has nowhere to put it) and by asserting the
emitted events' *values* never contain the LLM's raw `reasoning` text or
retrieved chunk content, which the scripted LLM below deliberately
includes in its JSON responses so a leak would be detectable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.events import AgentEvent
from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_SECRET_REASONING = "SECRET-INTERNAL-REASONING-do-not-leak-this-string"
_SECRET_CHUNK_CONTENT = "SECRET-CHUNK-CONTENT-do-not-leak-this-either"


def _chunk() -> Chunk:
    """Build a single Chunk whose content embeds the secret marker text."""
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
    return Chunk(id="doc-1_0", content=_SECRET_CHUNK_CONTENT, metadata=metadata)


class ScriptedLLM:
    """Returns one queued response per call, in order."""

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


def test_event_sequence_for_a_full_agent_run_matches_expected_transitions():
    """A full agent run emits the documented event sequence with correct metadata."""
    llm = ScriptedLLM(
        [
            f'{{"query_type": "complex", "reasoning": "{_SECRET_REASONING}"}}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            f'{{"sufficient": true, "reasoning": "{_SECRET_REASONING}"}}',
            "final answer",
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    events: list[AgentEvent] = []

    result = run_agent(
        AgentState(original_query="a question"),
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
        on_event=events.append,
    )

    assert result.state.termination_reason == "synthesized"
    event_types = [e.event_type for e in events]
    assert event_types == [
        "query_received",
        "route_selected",
        "decomposition_started",
        "decomposition_completed",
        "tool_selected",
        "tool_started",
        "tool_completed",
        "evidence_evaluated",
        "synthesis_started",
        "completed",
    ]
    assert events[1].route == "agent"
    assert events[4].tool_name == "search_knowledge_base"
    assert events[6].retrieved_chunk_count == 1
    assert events[7].evidence_sufficient is True


def test_events_never_contain_reasoning_text_or_retrieved_content():
    """No emitted event's JSON payload contains the LLM's reasoning text or chunk content."""
    llm = ScriptedLLM(
        [
            f'{{"query_type": "complex", "reasoning": "{_SECRET_REASONING}"}}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1", "top_k": 5}}',
            f'{{"sufficient": true, "reasoning": "{_SECRET_REASONING}"}}',
            "final answer",
        ]
    )
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk(), score=0.9)])
    events: list[AgentEvent] = []

    run_agent(
        AgentState(original_query="a question"),
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
        on_event=events.append,
    )

    for event in events:
        payload = event.model_dump_json()
        assert _SECRET_REASONING not in payload
        assert _SECRET_CHUNK_CONTENT not in payload
        # No free-text field exists on the model at all. assert the field set
        # itself never grows a text-shaped surface beyond the documented ones.
        assert set(type(event).model_fields) == {
            "event_type",
            "step",
            "tool_name",
            "elapsed_ms",
            "retrieved_chunk_count",
            "evidence_sufficient",
            "termination_reason",
            "route",
        }


def test_broken_event_sink_never_crashes_the_run():
    """A consumer callback that raises never propagates out of `run_agent`."""

    def _raising_sink(event: AgentEvent) -> None:
        raise RuntimeError("consumer callback failure")

    llm = ScriptedLLM(['{"query_type": "simple"}'])
    result = run_agent(
        AgentState(original_query="hello"),
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
        on_event=_raising_sink,
    )

    assert result.route == "classic_rag"


def test_classic_route_still_emits_query_received_route_selected_and_completed():
    """The classic_rag (agent-disabled) route emits exactly the 3 top-level lifecycle events."""
    llm = ScriptedLLM([])
    events: list[AgentEvent] = []

    run_agent(
        AgentState(original_query="hello"),
        pipeline=FakePipeline(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(enabled=False),
        on_event=events.append,
    )

    assert [e.event_type for e in events] == ["query_received", "route_selected", "completed"]
    assert events[1].route == "classic_rag"
