"""Unit tests for `rag.context.normalize`."""

from __future__ import annotations

from datetime import UTC, datetime

from rag.context.normalize import (
    content_hash,
    normalize_conversation_turn,
    normalize_search_result,
)
from rag.context.types import ConversationTurn
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id="c1", content="hello", **kwargs) -> Chunk:
    now = datetime.now(UTC)
    meta = ChunkMetadata(
        document_id=f"doc_{chunk_id}",
        chunk_id=chunk_id,
        source=f"docs/{chunk_id}.md",
        source_type=kwargs.pop("source_type", "markdown"),
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test",
        **kwargs,
    )
    return Chunk(id=chunk_id, content=content, metadata=meta)


def test_document_result_normalizes_to_document_source_type():
    """Document result normalizes to document source type."""
    result = SearchResult(chunk=_chunk(), score=0.9, origin="retrieved")
    item = normalize_search_result(result)
    assert item.source_type == "document"
    assert item.chunk_id == "c1"
    assert item.relevance_score == 0.9
    assert item.content == "hello"


def test_tool_fetched_result_normalizes_to_tool_fetched_source_type():
    """Tool fetched result normalizes to tool fetched source type."""
    result = SearchResult(chunk=_chunk(), score=1.0, origin="tool_fetched")
    item = normalize_search_result(result)
    assert item.source_type == "tool_fetched"


def test_graph_result_normalizes_to_graph_path_source_type():
    """Graph result normalizes to graph path source type."""
    result = SearchResult(
        chunk=_chunk(content="Alpha API DEPENDS_ON Alpha Database", source_type="graph"),
        score=1.0,
        origin="graph",
    )
    item = normalize_search_result(result)
    assert item.source_type == "graph_path"
    assert len(item.graph_path) == 1
    assert item.graph_path[0].relation == "DEPENDS_ON"


def test_mcp_remote_result_normalizes_to_mcp_tool_and_extracts_tool_name():
    """Mcp remote result normalizes to mcp tool and extracts tool name."""
    chunk = _chunk(chunk_id="mcp:get_customer_case:CASE-1001", content="Case detail")
    result = SearchResult(chunk=chunk, score=1.0, origin="mcp_remote")
    item = normalize_search_result(result)
    assert item.source_type == "mcp_tool"
    assert item.tool_name == "get_customer_case"


def test_authority_score_reflects_trust_level():
    """Authority score reflects trust level."""
    authoritative = normalize_search_result(
        SearchResult(chunk=_chunk(trust_level="authoritative"), score=0.5)
    )
    untrusted = normalize_search_result(
        SearchResult(chunk=_chunk(chunk_id="c2", trust_level="untrusted"), score=0.5)
    )
    unset = normalize_search_result(SearchResult(chunk=_chunk(chunk_id="c3"), score=0.5))
    assert authoritative.authority_score == 1.0
    assert untrusted.authority_score == 0.0
    assert unset.authority_score == 0.5


def test_redacted_field_ids_and_injection_flag_are_carried_through():
    """Redacted field ids and injection flag are carried through."""
    result = SearchResult(chunk=_chunk(), score=0.5, injection_suspected=True)
    result.redacted_field_ids = ["synthetic_admin_credential"]
    item = normalize_search_result(result)
    assert item.injection_suspected is True
    assert item.sensitive_field_ids == ["synthetic_admin_credential"]


def test_conversation_turn_normalizes_with_role_prefixed_content():
    """Conversation turn normalizes with role prefixed content."""
    turn = ConversationTurn(role="user", content="What is the timeout?")
    item = normalize_conversation_turn(turn, 0)
    assert item.source_type == "conversation_history"
    assert item.content == "User: What is the timeout?"
    assert item.relevance_score is None


def test_conversation_turn_benign_content_is_not_flagged():
    """An ordinary history turn is not flagged."""
    turn = ConversationTurn(role="assistant", content="The timeout is 30 seconds.")
    item = normalize_conversation_turn(turn, 0)
    assert item.injection_suspected is False


def test_content_hash_is_stable_and_order_insensitive_to_object_identity():
    """Content hash is stable and order insensitive to object identity."""
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
