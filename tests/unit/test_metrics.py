from __future__ import annotations

from rag.eval.metrics import (
    hit_rate_at_k,
    mean_hit_rate_at_k,
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


def test_hit_rate_at_k_is_binary_unlike_recall():
    # Two relevant ids, only one found -- recall is fractional, hit rate is 1.0.
    retrieved = ["a", "b", "c"]
    relevant = {"a", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_zero_when_no_hit_in_window():
    assert hit_rate_at_k(["a", "b", "c"], {"z"}, k=2) == 0.0


def test_mean_hit_rate_at_k_averages_across_queries():
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"z"}]
    assert mean_hit_rate_at_k(all_retrieved, all_relevant, k=2) == 0.5


def test_recall_at_k_with_custom_match_fn():
    # match_fn treats "knowledge_base/x.md" as matching any retrieved id
    # ending with that suffix -- used for path-based gold data.
    match_fn = lambda retrieved, relevant: retrieved.endswith(relevant)
    retrieved = ["data/knowledge_base/x.md", "data/knowledge_base/y.md"]
    relevant = ["knowledge_base/x.md"]
    assert recall_at_k(retrieved, relevant, k=2, match_fn=match_fn) == 1.0


def test_reciprocal_rank_with_custom_match_fn():
    match_fn = lambda retrieved, relevant: retrieved.endswith(relevant)
    retrieved = ["data/a.md", "data/knowledge_base/x.md"]
    relevant = ["knowledge_base/x.md"]
    assert reciprocal_rank(retrieved, relevant, match_fn=match_fn) == 0.5
