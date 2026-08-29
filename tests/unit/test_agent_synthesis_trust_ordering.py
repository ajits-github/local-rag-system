"""Proves synthesis presents authoritative evidence ahead of conflicting untrusted evidence.

Regression test for the Q17 finding in experiment_029 (see
experiments/reports/agentic_rag_baseline_v1.md section 3): the agent
retrieved both an authoritative and an untrusted, contradicting source,
but synthesized the untrusted value as the primary answer. The fix is
`rag.agent.graph._order_evidence_for_synthesis`, which the graph's
`_synthesize` node now applies before rendering context. this test
proves it via the observable prompt text (matching this codebase's
existing convention of asserting through `run_agent`'s public entrypoint
plus a recording `ScriptedLLM`, never by importing graph.py's private
functions directly).
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str, source: str, content: str, trust_level: str | None) -> Chunk:
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=chunk_id,
        chunk_id=chunk_id,
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
        trust_level=trust_level,
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


class ScriptedLLM:
    """Returns one queued response per call, in order; records every call."""

    def __init__(self, responses: list[str]) -> None:
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
    """RetrievalPipeline double returning fixed, deliberately untrusted-first evidence."""

    def __init__(self, retrieve_results: list[SearchResult]) -> None:
        """Store the fixed evidence list this double's retrieve() will return."""
        self.retrieve_results = retrieve_results

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Return the fixed evidence list, unchanged."""
        return self.retrieve_results

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

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


def _agent_config(**overrides):
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True, **overrides})
    return config.model_copy(update={"agent": agent})


def test_untrusted_evidence_gathered_first_is_still_presented_after_authoritative_evidence():
    """Retrieval order (untrusted first) must not become synthesis/citation order."""
    untrusted_chunk = _chunk(
        "untrusted_0",
        "knowledge_base/security_evaluation/internal_techfusion/untrusted-operations-notes.md",
        "UNTRUSTED_CLAIM: retention is 7 days.",
        trust_level="untrusted",
    )
    authoritative_chunk = _chunk(
        "auth_0",
        "knowledge_base/security_evaluation/tenant_alpha/retention-policy-v2.md",
        "AUTHORITATIVE_VALUE: retention is 90 days.",
        trust_level="authoritative",
    )
    # Deliberately returned untrusted-first, mirroring how a broad hybrid
    # search could rank an untrusted-but-lexically-similar page above the
    # authoritative one.
    pipeline = FakePipeline(
        [
            SearchResult(chunk=untrusted_chunk, score=0.95),
            SearchResult(chunk=authoritative_chunk, score=0.80),
        ]
    )
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["current retention period"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "retention period"}}',
            '{"sufficient": true}',
            "The current authoritative retention period is 90 days (Source 1).",
        ]
    )
    state = AgentState(original_query="What is the current approved retention period?")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    synthesis_prompt = llm.calls[-1][1]
    authoritative_pos = synthesis_prompt.index("AUTHORITATIVE_VALUE")
    untrusted_pos = synthesis_prompt.index("UNTRUSTED_CLAIM")
    assert authoritative_pos < untrusted_pos, "authoritative evidence must be numbered/read first"
    assert synthesis_prompt.index("[Source 1:") < synthesis_prompt.index("[Source 2:")
    # Citations follow the same reordering, so Source-N numbering used by
    # eval tooling's citation parsing stays consistent with the context.
    assert result.state.citations[0].source == authoritative_chunk.metadata.source
    assert result.state.citations[1].source == untrusted_chunk.metadata.source


def test_evidence_ordering_is_unaffected_when_no_evidence_is_untrusted():
    """Two authoritative/untagged sources keep their original relative (stable-sort) order."""
    first = _chunk("a_0", "a.md", "first content", trust_level=None)
    second = _chunk("b_0", "b.md", "second content", trust_level="authoritative")
    pipeline = FakePipeline(
        [SearchResult(chunk=first, score=0.9), SearchResult(chunk=second, score=0.8)]
    )
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}',
            '{"sufficient": true}',
            "answer",
        ]
    )
    state = AgentState(original_query="a question")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert [c.source for c in result.state.citations] == ["a.md", "b.md"]
