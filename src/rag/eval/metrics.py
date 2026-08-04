"""Basic retrieval metrics: recall@k and MRR, over generic id lists so the
caller decides whether ids are document_ids (stable across chunking
experiments) or chunk_ids."""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(
    all_retrieved: list[list[str]], all_relevant: list[set[str]], k: int
) -> float:
    if not all_retrieved:
        return 0.0
    scores = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)


def mean_reciprocal_rank(
    all_retrieved: list[list[str]], all_relevant: list[set[str]]
) -> float:
    if not all_retrieved:
        return 0.0
    scores = [reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)
