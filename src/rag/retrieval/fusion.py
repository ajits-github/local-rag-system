"""Reciprocal Rank Fusion: merges multiple ranked SearchResult lists into one."""

from __future__ import annotations

from rag.schemas import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Fuse multiple ranked SearchResult lists into one via Reciprocal Rank Fusion.

    Deduplicates by `SearchResult.chunk.id` (stable across a dense-search
    row and a keyword-search row for the same underlying chunk, since
    both are read off the same `chunk_id` column). Each list contributes
    `1 / (k + rank)` to a chunk's fused score, where `rank` starts at 1;
    scores from multiple lists for the same chunk are summed. The first
    occurrence of a chunk (in `ranked_lists` order, then within-list
    rank order) supplies the returned `SearchResult`'s `chunk`
    content/metadata; only `.score` is replaced.

    Parameters
    ----------
    ranked_lists : list[list[SearchResult]]
        Two or more ranked result lists to fuse (e.g. dense-search
        results and keyword-search results), each already sorted
        best-first.
    k : int, optional
        RRF's rank-damping constant, by default 60 (the standard value
        from the original RRF paper, also used as Elasticsearch's and
        Weaviate's default).

    Returns
    -------
    list[SearchResult]
        Deduplicated, fused results sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = result

    fused = [
        first_seen[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused
