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
    """Only relevant ids within the top-k window count toward recall."""
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 1 / 3  # only "b" is in top-3


def test_recall_at_k_full_recall():
    """recall_at_k is 1.0 when every relevant id is retrieved within k."""
    retrieved = ["a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_recall_at_k_no_relevant_returns_zero():
    """recall_at_k is 0.0 when there are no relevant ids at all."""
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_first_hit():
    """reciprocal_rank is 1.0 when the first result is relevant."""
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_hit():
    """reciprocal_rank is 1/rank of the first relevant result."""
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    """reciprocal_rank is 0.0 when no result is relevant."""
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean_recall_at_k_averages_across_queries():
    """mean_recall_at_k averages per-query recall_at_k scores."""
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"z"}]
    assert mean_recall_at_k(all_retrieved, all_relevant, k=2) == 0.5


def test_mean_reciprocal_rank_averages_across_queries():
    """mean_reciprocal_rank averages per-query reciprocal_rank scores."""
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"y"}]
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == (1.0 + 0.5) / 2


def test_hit_rate_at_k_is_binary_unlike_recall():
    """hit_rate_at_k is 1.0 as soon as any relevant id is found, unlike recall."""
    # Two relevant ids, only one found -- recall is fractional, hit rate is 1.0.
    retrieved = ["a", "b", "c"]
    relevant = {"a", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_zero_when_no_hit_in_window():
    """hit_rate_at_k is 0.0 when no relevant id falls within the top-k window."""
    assert hit_rate_at_k(["a", "b", "c"], {"z"}, k=2) == 0.0


def test_mean_hit_rate_at_k_averages_across_queries():
    """mean_hit_rate_at_k averages per-query hit_rate_at_k scores."""
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"z"}]
    assert mean_hit_rate_at_k(all_retrieved, all_relevant, k=2) == 0.5


def _suffix_match(retrieved: str, relevant: str) -> bool:
    """Match path-based gold data by suffix, e.g. "knowledge_base/x.md"."""
    return retrieved.endswith(relevant)


def test_recall_at_k_with_custom_match_fn():
    """match_fn treats a relative path as matching any retrieved suffix."""
    retrieved = ["data/knowledge_base/x.md", "data/knowledge_base/y.md"]
    relevant = ["knowledge_base/x.md"]
    assert recall_at_k(retrieved, relevant, k=2, match_fn=_suffix_match) == 1.0


def test_reciprocal_rank_with_custom_match_fn():
    """reciprocal_rank respects a custom suffix-based match_fn."""
    retrieved = ["data/a.md", "data/knowledge_base/x.md"]
    relevant = ["knowledge_base/x.md"]
    assert reciprocal_rank(retrieved, relevant, match_fn=_suffix_match) == 0.5
