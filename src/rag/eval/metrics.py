"""Basic retrieval metrics: recall@k, MRR, and hit rate, over generic id
lists so the caller decides whether ids are document_ids (stable across
chunking experiments), chunk_ids, or normalized source paths.

Every function accepts an optional `match_fn(retrieved_id, relevant_id) ->
bool` for callers whose notion of a "match" isn't exact string equality
(e.g. gold data that references documents by relative path, which must be
matched against a possibly differently-rooted stored `source` value by
suffix rather than equality). Omitting it preserves plain exact-match
semantics.
"""

from __future__ import annotations

from typing import Callable, Iterable

MatchFn = Callable[[str, str], bool]


def _is_hit(retrieved: Iterable[str], relevant: str, match_fn: MatchFn | None) -> bool:
    if match_fn is None:
        return relevant in retrieved
    return any(match_fn(r, relevant) for r in retrieved)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    relevant_ids = list(relevant_ids)
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rel in relevant_ids if _is_hit(top_k, rel, match_fn))
    return hits / len(relevant_ids)


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    """1.0 if ANY relevant id appears in the top-k, else 0.0 — unlike
    recall_at_k, doesn't average over multiple relevant ids per query."""
    relevant_ids = list(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(_is_hit(top_k, rel, match_fn) for rel in relevant_ids) else 0.0


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    match_fn: MatchFn | None = None,
) -> float:
    relevant_ids = list(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if match_fn is None:
            if item in relevant_ids:
                return 1.0 / rank
        elif any(match_fn(item, rel) for rel in relevant_ids):
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(
    all_retrieved: list[list[str]],
    all_relevant: list[Iterable[str]],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    if not all_retrieved:
        return 0.0
    scores = [recall_at_k(r, rel, k, match_fn) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)


def mean_hit_rate_at_k(
    all_retrieved: list[list[str]],
    all_relevant: list[Iterable[str]],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    if not all_retrieved:
        return 0.0
    scores = [hit_rate_at_k(r, rel, k, match_fn) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(
    all_retrieved: list[list[str]],
    all_relevant: list[Iterable[str]],
    match_fn: MatchFn | None = None,
) -> float:
    if not all_retrieved:
        return 0.0
    scores = [reciprocal_rank(r, rel, match_fn) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)
