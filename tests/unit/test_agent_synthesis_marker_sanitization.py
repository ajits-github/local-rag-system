"""Proves the agentic path sanitizes echoed redaction markers."""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str, source: str, content: str) -> Chunk:
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
    """RetrievalPipeline double returning a fixed, already-redacted evidence chunk."""

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
        """Return results unchanged; not exercised by this test."""
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


def test_synthesize_sanitizes_an_echoed_redaction_marker_in_the_final_answer():
    """A synthesized answer that echoes the literal marker is sanitized before storage."""
    redacted_chunk = _chunk("c1", "integration-runbook.md", "Test key: [REDACTED:SENSITIVE_FIELD].")
    pipeline = FakePipeline([SearchResult(chunk=redacted_chunk, score=0.9)])
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["what is the test key"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "test key"}}',
            '{"sufficient": true}',
            "The synthetic test key is `[REDACTED:SENSITIVE_FIELD]` (Source 1).",
        ]
    )
    state = AgentState(original_query="What is the test key?")

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=_agent_config(),
    )

    assert "[REDACTED:SENSITIVE_FIELD]" not in result.state.final_answer
    assert "this value is unavailable at your access level" in result.state.final_answer
