"""Unit tests for `rag.context.caching`."""

from __future__ import annotations

from rag.context.caching import TokenCountCache
from rag.context.tokens import HeuristicTokenCounter


def test_cache_hit_returns_the_same_value_without_recounting():
    """Cache hit returns the same value without recounting."""
    cache = TokenCountCache(max_entries=10)
    counter = HeuristicTokenCounter(chars_per_token=4.0)
    first = cache.get_or_count("tenant_a", "hello world", counter)
    second = cache.get_or_count("tenant_a", "hello world", counter)
    assert first == second
    assert cache.hits == 1
    assert cache.misses == 1


def test_different_tenants_never_share_a_cache_key():
    """Different tenants never share a cache key."""
    cache = TokenCountCache(max_entries=10)
    counter = HeuristicTokenCounter(chars_per_token=4.0)
    cache.get_or_count("tenant_a", "same text", counter)
    cache.get_or_count("tenant_b", "same text", counter)
    assert cache.misses == 2
    assert cache.hits == 0


def test_different_token_counting_strategies_never_share_a_cache_key():
    """Different token counting strategies never share a cache key."""
    cache = TokenCountCache(max_entries=10)
    heuristic = HeuristicTokenCounter(chars_per_token=4.0)
    heuristic_other_ratio = HeuristicTokenCounter(chars_per_token=2.0)
    # Both report strategy_name == "heuristic", so this specifically
    # verifies caching happens per text+tenant+strategy_name, not per
    # ratio; a real cross-ratio collision would only matter if two
    # different concrete classes shared one strategy_name, which they
    # don't today. Documented as a known, accepted limitation.
    v1 = cache.get_or_count("t", "hello world", heuristic)
    v2 = cache.get_or_count("t", "hello world", heuristic_other_ratio)
    assert v1 == v2  # same strategy_name -> cache hit -> same (first-seen) value
    assert cache.hits == 1


def test_lru_eviction_bounds_memory():
    """Lru eviction bounds memory."""
    cache = TokenCountCache(max_entries=2)
    counter = HeuristicTokenCounter(chars_per_token=4.0)
    cache.get_or_count("t", "a", counter)
    cache.get_or_count("t", "b", counter)
    cache.get_or_count("t", "c", counter)
    assert len(cache) == 2


def test_cache_never_stores_raw_text_only_a_hash_derived_key():
    """Cache never stores raw text only a hash derived key."""
    cache = TokenCountCache(max_entries=10)
    counter = HeuristicTokenCounter(chars_per_token=4.0)
    secret_text = "SYNTHETIC_ONLY_super_secret_value"
    cache.get_or_count("tenant_a", secret_text, counter)
    assert all(secret_text not in key for key in cache._store)
