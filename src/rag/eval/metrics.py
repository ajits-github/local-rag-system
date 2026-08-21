"""Basic retrieval metrics: recall@k, MRR, and hit rate, over generic id lists.

Ids are generic so the caller decides whether they're document_ids (stable
across chunking experiments), chunk_ids, or normalized source paths.

Every function accepts an optional `match_fn(retrieved_id, relevant_id) ->
bool` for callers whose notion of a "match" isn't exact string equality
(e.g. gold data that references documents by relative path, which must be
matched against a possibly differently-rooted stored `source` value by
suffix rather than equality). Omitting it preserves plain exact-match
semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

MatchFn = Callable[[str, str], bool]


def _is_hit(retrieved: Iterable[str], relevant: str, match_fn: MatchFn | None) -> bool:
    """Return whether `relevant` matches any id in `retrieved`."""
    if match_fn is None:
        return relevant in retrieved
    return any(match_fn(r, relevant) for r in retrieved)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    """Fraction of relevant ids that appear in the top-k retrieved ids.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ids returned by retrieval, in ranked order.
    relevant_ids : Iterable[str]
        Ids considered relevant (gold-labeled) for this query.
    k : int
        Cutoff rank; only ``retrieved_ids[:k]`` is considered.
    match_fn : MatchFn | None, optional
        Custom equality function ``(retrieved_id, relevant_id) -> bool``.
        Defaults to exact string equality.

    Returns
    -------
    float
        ``hits / len(relevant_ids)``, or ``0.0`` if `relevant_ids` is empty.
    """
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
    """Return 1.0 if any relevant id appears in the top-k, else 0.0.

    Unlike `recall_at_k`, this doesn't average over multiple relevant ids
    per query. It only asks whether retrieval found *at least one*.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ids returned by retrieval, in ranked order.
    relevant_ids : Iterable[str]
        Ids considered relevant (gold-labeled) for this query.
    k : int
        Cutoff rank; only ``retrieved_ids[:k]`` is considered.
    match_fn : MatchFn | None, optional
        Custom equality function ``(retrieved_id, relevant_id) -> bool``.
        Defaults to exact string equality.

    Returns
    -------
    float
        ``1.0`` or ``0.0``.
    """
    relevant_ids = list(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(_is_hit(top_k, rel, match_fn) for rel in relevant_ids) else 0.0


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    match_fn: MatchFn | None = None,
) -> float:
    """Return ``1 / rank`` of the first relevant id found, else 0.0.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ids returned by retrieval, in ranked order (rank 1 = first item).
    relevant_ids : Iterable[str]
        Ids considered relevant (gold-labeled) for this query.
    match_fn : MatchFn | None, optional
        Custom equality function ``(retrieved_id, relevant_id) -> bool``.
        Defaults to exact string equality.

    Returns
    -------
    float
        ``1.0 / rank`` of the first hit, or ``0.0`` if there is no hit.
    """
    relevant_ids = list(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if match_fn is None:
            if item in relevant_ids:
                return 1.0 / rank
        elif any(match_fn(item, rel) for rel in relevant_ids):
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(
    all_retrieved: Sequence[list[str]],
    all_relevant: Sequence[Iterable[str]],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    """Average `recall_at_k` across a batch of queries.

    Parameters
    ----------
    all_retrieved : Sequence[list[str]]
        One retrieved-id list per query.
    all_relevant : Sequence[Iterable[str]]
        One relevant-id collection per query, paired by index with
        `all_retrieved`.
    k : int
        Cutoff rank passed through to `recall_at_k`.
    match_fn : MatchFn | None, optional
        Custom equality function passed through to `recall_at_k`.

    Returns
    -------
    float
        Mean recall@k over all queries, or ``0.0`` if `all_retrieved` is
        empty.
    """
    if not all_retrieved:
        return 0.0
    scores = [
        recall_at_k(r, rel, k, match_fn) for r, rel in zip(all_retrieved, all_relevant, strict=True)
    ]
    return sum(scores) / len(scores)


def mean_hit_rate_at_k(
    all_retrieved: Sequence[list[str]],
    all_relevant: Sequence[Iterable[str]],
    k: int,
    match_fn: MatchFn | None = None,
) -> float:
    """Average `hit_rate_at_k` across a batch of queries.

    Parameters
    ----------
    all_retrieved : Sequence[list[str]]
        One retrieved-id list per query.
    all_relevant : Sequence[Iterable[str]]
        One relevant-id collection per query, paired by index with
        `all_retrieved`.
    k : int
        Cutoff rank passed through to `hit_rate_at_k`.
    match_fn : MatchFn | None, optional
        Custom equality function passed through to `hit_rate_at_k`.

    Returns
    -------
    float
        Mean hit-rate@k over all queries, or ``0.0`` if `all_retrieved` is
        empty.
    """
    if not all_retrieved:
        return 0.0
    scores = [
        hit_rate_at_k(r, rel, k, match_fn)
        for r, rel in zip(all_retrieved, all_relevant, strict=True)
    ]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(
    all_retrieved: Sequence[list[str]],
    all_relevant: Sequence[Iterable[str]],
    match_fn: MatchFn | None = None,
) -> float:
    """Average `reciprocal_rank` across a batch of queries.

    Parameters
    ----------
    all_retrieved : Sequence[list[str]]
        One retrieved-id list per query.
    all_relevant : Sequence[Iterable[str]]
        One relevant-id collection per query, paired by index with
        `all_retrieved`.
    match_fn : MatchFn | None, optional
        Custom equality function passed through to `reciprocal_rank`.

    Returns
    -------
    float
        Mean reciprocal rank over all queries, or ``0.0`` if `all_retrieved`
        is empty.
    """
    if not all_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(r, rel, match_fn)
        for r, rel in zip(all_retrieved, all_relevant, strict=True)
    ]
    return sum(scores) / len(scores)
