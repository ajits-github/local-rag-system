from __future__ import annotations

from rag.eval.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_hits_within_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 1 / 3  # only "b" is in top-3


def test_recall_at_k_full_recall():
    retrieved = ["a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_hit():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean_recall_at_k_averages_across_queries():
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"z"}]
    assert mean_recall_at_k(all_retrieved, all_relevant, k=2) == 0.5


def test_mean_reciprocal_rank_averages_across_queries():
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"y"}]
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == (1.0 + 0.5) / 2
