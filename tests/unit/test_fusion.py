from __future__ import annotations

from datetime import UTC, datetime

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _make_result(chunk_id: str, score: float, content: str = "content") -> SearchResult:
    """Build a SearchResult with minimal-but-valid chunk metadata."""
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
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=score)


def test_rrf_sums_reciprocal_ranks_for_chunk_in_both_lists():
    """A chunk ranked 1st in both lists gets 2 * 1/(k+1)."""
    a = _make_result("a", score=0.9)
    dense = [a]
    keyword = [a]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    assert len(fused) == 1
    assert fused[0].score == 2 * (1.0 / 61)


def test_rrf_includes_chunk_present_in_only_one_list():
    """A chunk in only one list contributes only that list's reciprocal-rank term."""
    a = _make_result("a", score=0.9)
    b = _make_result("b", score=0.5)
    dense = [a, b]
    keyword = [a]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    fused_by_id = {r.chunk.id: r.score for r in fused}
    assert fused_by_id["a"] == (1.0 / 61) + (1.0 / 61)
    assert fused_by_id["b"] == 1.0 / 62


def test_rrf_dedups_by_chunk_id_keeping_first_seen_content():
    """A chunk in both lists produces exactly one fused entry, keeping the first-seen content."""
    a_dense = _make_result("a", score=0.9, content="dense version")
    a_keyword = _make_result("a", score=12.3, content="keyword version")

    fused = reciprocal_rank_fusion([[a_dense], [a_keyword]], k=60)

    assert len(fused) == 1
    assert fused[0].chunk.content == "dense version"
    assert fused[0].score not in (0.9, 12.3)  # replaced by the fused RRF score


def test_rrf_sorts_descending_by_fused_score():
    """Fused results are sorted best (highest fused score) first."""
    a = _make_result("a", score=0.1)
    b = _make_result("b", score=0.9)
    dense = [b, a]  # b ranked 1st, a ranked 2nd
    keyword = [b, a]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    assert [r.chunk.id for r in fused] == ["b", "a"]


def test_rrf_empty_lists_returns_empty():
    """No candidates in, none out."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_relative_order():
    """A single ranked list fuses to the same relative order (degenerate case)."""
    a = _make_result("a", score=0.9)
    b = _make_result("b", score=0.5)

    fused = reciprocal_rank_fusion([[a, b]], k=60)

    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_rrf_custom_k_changes_fused_scores_but_not_correct_ordering():
    """A different k changes the fused score values while preserving correct relative ordering."""
    a = _make_result("a", score=0.9)
    b = _make_result("b", score=0.5)
    dense = [a, b]
    keyword = [a]

    fused_k60 = reciprocal_rank_fusion([dense, keyword], k=60)
    fused_k1 = reciprocal_rank_fusion([dense, keyword], k=1)

    assert [r.chunk.id for r in fused_k60] == ["a", "b"]
    assert [r.chunk.id for r in fused_k1] == ["a", "b"]
    a_score_k60 = next(r.score for r in fused_k60 if r.chunk.id == "a")
    a_score_k1 = next(r.score for r in fused_k1 if r.chunk.id == "a")
    assert a_score_k60 != a_score_k1
