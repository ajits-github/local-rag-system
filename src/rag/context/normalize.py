"""Converts existing evidence/history types into normalized `ContextItem`s.

Every function here assumes its input has already passed through
`RetrievalPipeline.sanitize_evidence` (field redaction + injection
flagging) or an equivalent already-authorized source; this module adds
no authorization or redaction logic of its own, it only repackages
already-safe data. See section 17 of the design: "No evidence should
enter the context before authorization/redaction" is upheld by never
calling these functions earlier than the existing pipelines already
call `sanitize_evidence`/`retrieve()`.
"""

from __future__ import annotations

import hashlib

from rag.context.graph_packing import parse_graph_fact_text
from rag.context.prioritize import authority_score_for_trust_level
from rag.context.types import ContextItem, ContextSourceType, ConversationTurn
from rag.retrieval.injection_detection import detect_injection
from rag.schemas import SearchResult


def _source_type_for_result(result: SearchResult) -> ContextSourceType:
    """Map a `SearchResult`'s origin/metadata onto a `ContextSourceType`."""
    if result.chunk.metadata.source_type == "graph" or result.origin == "graph":
        return "graph_path"
    if result.origin == "mcp_remote":
        return "mcp_tool"
    if result.origin == "tool_fetched":
        return "tool_fetched"
    # "retrieved" and "expanded" are both ordinary corpus evidence.
    return "document"


def _is_current_from_status(status: str | None) -> bool | None:
    """Map `status` onto `is_current`: `True` for `"active"`, `False` otherwise, `None` if unset."""
    if status is None:
        return None
    return status == "active"


def normalize_search_result(result: SearchResult) -> ContextItem:
    """Convert one `SearchResult` into a `ContextItem`.

    Parameters
    ----------
    result : SearchResult
        Already-sanitized evidence (document, graph, MCP, or
        tool-fetched).

    Returns
    -------
    ContextItem
        `content` is `result.chunk.content` unchanged; `citation` reuses
        `rag.retrieval.pipeline.source_label` so wording matches the
        legacy `build_context` path exactly.
    """
    # Deferred import: `rag.retrieval.pipeline` imports `rag.context.engine`
    # (which imports this module) to build its own prompt context, so a
    # module-level import here would be circular. By call time, both
    # modules are already fully loaded.
    from rag.retrieval.pipeline import source_label

    meta = result.chunk.metadata
    source_type = _source_type_for_result(result)
    graph_path = parse_graph_fact_text(result.chunk.content) if source_type == "graph_path" else []
    tool_name: str | None = None
    if source_type == "mcp_tool" and meta.chunk_id.startswith("mcp:"):
        # `mcp_client._business_result_to_search_result`'s chunk_id shape:
        # "mcp:<tool_name>:<case_id>".
        tool_name = meta.chunk_id.split(":")[1] if meta.chunk_id.count(":") >= 2 else None
    return ContextItem(
        content=result.chunk.content,
        source_type=source_type,
        source_id=meta.chunk_id,
        citation=source_label(result),
        relevance_score=result.score,
        reranker_score=None,
        authority_score=authority_score_for_trust_level(meta.trust_level),
        freshness_status=meta.status,
        is_current=_is_current_from_status(meta.status),
        tenant_id=meta.tenant_id,
        dataset_id=meta.dataset_id,
        sensitive_field_ids=list(result.redacted_field_ids),
        injection_suspected=result.injection_suspected,
        document_id=meta.document_id,
        chunk_id=meta.chunk_id,
        content_type=meta.content_type,
        graph_path=graph_path,
        tool_name=tool_name,
        origin=result.origin,
    )


def normalize_search_results(results: list[SearchResult]) -> list[ContextItem]:
    """Convert a list of `SearchResult`s, preserving order. See `normalize_search_result`."""
    return [normalize_search_result(r) for r in results]


def normalize_conversation_turn(turn: ConversationTurn, index: int) -> ContextItem:
    """Convert one `ConversationTurn` into a `ContextItem`.

    Parameters
    ----------
    turn : ConversationTurn
    index : int
        Position in the supplied history list (oldest first), used to
        build a stable `source_id` when `turn.turn_id` is absent.

    Returns
    -------
    ContextItem
        `relevance_score`/`authority_score`/`freshness_status` are all
        `None`: history turns have no retrieval-ranking or governance
        concept. `injection_suspected` is set via the same
        `detect_injection` heuristic every document/graph/tool evidence
        item already carries (telemetry/reinforcement only, exactly as
        for any other evidence: the trust boundary is that history
        renders as data inside the user turn, never as an instruction,
        regardless of this flag).
    """
    source_id = turn.turn_id or f"turn_{index}"
    label = "User" if turn.role == "user" else "Assistant"
    return ContextItem(
        content=f"{label}: {turn.content}",
        source_type="conversation_history",
        source_id=source_id,
        citation=f"conversation turn {index} ({turn.role})",
        injection_suspected=detect_injection(turn.content),
    )


def normalize_conversation_turns(turns: list[ConversationTurn]) -> list[ContextItem]:
    """Convert a list of turns, preserving order. See `normalize_conversation_turn`."""
    return [normalize_conversation_turn(t, i) for i, t in enumerate(turns)]


def content_hash(text: str) -> str:
    """Stable sha256 hex digest of `text`, for exact-duplicate detection in `rag.context.dedup`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
