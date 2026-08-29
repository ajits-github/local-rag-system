"""Tests that get_related_context can use real chunk ids from evidence.

get_related_context requires a real chunk_id, and the tool-selection model
can only supply one it was actually shown. These tests prove the evidence
summary exposes each item's real chunk_id, and that a full search then
get_related_context agent run can dispatch get_related_context using the
exact id shown in that summary.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import _summarize_evidence, run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str, content: str, source: str = "a.md") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


def test_summarize_evidence_exposes_the_real_chunk_id():
    """Each evidence line names its real chunk_id, not just a bracketed index.

    The tool-selection prompt depends on this value to call
    get_related_context with a valid seed chunk.
    """
    evidence = [SearchResult(chunk=_chunk("doc-1_7", "alpha content"), score=0.9)]

    summary = _summarize_evidence(evidence)

    assert "chunk_id=doc-1_7" in summary
    assert "source=a.md" in summary


def test_summarize_evidence_does_not_expose_unnecessary_internal_metadata():
    """The summary stays limited to chunk_id/source/text; no other internal fields leak in."""
    chunk = _chunk("doc-1_7", "alpha content")
    chunk.metadata.tenant_id = "tenant_alpha"
    chunk.metadata.sensitive_field_ids = ["synthetic_admin_credential"]
    evidence = [SearchResult(chunk=chunk, score=0.9)]

    summary = _summarize_evidence(evidence)

    assert "tenant_alpha" not in summary
    assert "sensitive_field_ids" not in summary
    assert "synthetic_admin_credential" not in summary


class ScriptedLLM:
    """LLM double that returns each response in order, recording every (system, user) call."""

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
    """retrieve() returns one fixed seed chunk; expand_with_relationships() returns one related."""

    def __init__(self) -> None:
        """Start with no recorded expand_with_relationships() calls."""
        self.expand_calls: list[dict] = []

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Return one fixed seed result, ignoring the query."""
        return [SearchResult(chunk=_chunk("doc-1_0", "alpha seed content"), score=0.9)]

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results

    def resolve_auth(self, auth, filters=None):
        """Return `auth` unchanged; authorization resolution is outside this file's scope."""
        return auth

    def expand_with_relationships(self, results, auth=None):
        """Record the call and append one fixed related chunk."""
        self.expand_calls.append({"results": results, "auth": auth})
        related = _chunk("doc-1_1", "related neighbor content")
        return list(results) + [
            SearchResult(chunk=related, score=0.9, origin="expanded", expanded_from="doc-1_0")
        ]


class FakeVectorStore:
    """get_chunks_by_ids() resolves only the known seed chunk_id "doc-1_0"."""

    def get_chunks_by_ids(self, chunk_ids, auth=None):
        """Return the seed chunk when asked for it, else nothing."""
        return [_chunk("doc-1_0", "alpha seed content")] if "doc-1_0" in chunk_ids else []


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


def test_full_search_then_get_related_context_flow_using_the_exposed_chunk_id():
    """search_knowledge_base gathers a chunk, then get_related_context uses its exposed id.

    End-to-end: get_related_context is dispatched with the exact chunk_id
    the evidence summary exposed to the model, and it actually returns
    related content (result_count > 0).
    """
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "q1"}}',
            '{"sufficient": false, "reformulated_query": "need related context"}',
            '{"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc-1_0"}}',
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

    # The second select_tool prompt must show "doc-1_0"; the next tool call
    # then proves the model used an exposed id rather than an invented one.
    second_select_tool_prompt = llm.calls[4][1]
    assert "chunk_id=doc-1_0" in second_select_tool_prompt

    search_record, related_record = result.state.tool_call_history
    assert search_record.tool_name == "search_knowledge_base"
    assert related_record.tool_name == "get_related_context"
    assert related_record.success is True
    assert related_record.result_count > 0
    assert pipeline.expand_calls  # expand_with_relationships was actually reached
