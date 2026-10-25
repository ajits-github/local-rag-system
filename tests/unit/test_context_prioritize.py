"""Unit tests for `rag.context.prioritize`."""

from __future__ import annotations

from rag.context.prioritize import authority_score_for_trust_level, prioritize_items
from rag.context.types import ContextItem


def _item(source_id, *, authority_score=None, is_current=None) -> ContextItem:
    return ContextItem(
        content="x",
        source_type="document",
        source_id=source_id,
        authority_score=authority_score,
        is_current=is_current,
    )


def test_authority_score_mapping_is_fixed_and_documented():
    """Authority score mapping is fixed and documented."""
    assert authority_score_for_trust_level("authoritative") == 1.0
    assert authority_score_for_trust_level("AUTHORITATIVE") == 1.0  # case-insensitive
    assert authority_score_for_trust_level("untrusted") == 0.0
    assert authority_score_for_trust_level(None) == 0.5
    assert authority_score_for_trust_level("internal") == 0.5  # unknown label -> neutral


def test_current_items_rank_before_superseded_items_regardless_of_input_order():
    """Current items rank before superseded items regardless of input order."""
    stale = _item("stale", is_current=False)
    current = _item("current", is_current=True)
    ranked = prioritize_items([stale, current])
    assert [i.source_id for i in ranked] == ["current", "stale"]
    assert ranked[0].priority_rank == 0
    assert ranked[1].priority_rank == 1


def test_authoritative_items_rank_before_untrusted_items_within_the_same_freshness_tier():
    """Authoritative items rank before untrusted items within the same freshness tier."""
    untrusted = _item("untrusted", authority_score=0.0)
    authoritative = _item("authoritative", authority_score=1.0)
    neutral = _item("neutral", authority_score=None)
    ranked = prioritize_items([untrusted, neutral, authoritative])
    assert [i.source_id for i in ranked] == ["authoritative", "neutral", "untrusted"]


def test_freshness_outranks_authority():
    """Freshness outranks authority."""
    # A stale-but-authoritative item still ranks below a current,
    # authority-neutral item: freshness tier is checked first.
    stale_authoritative = _item("stale_authoritative", authority_score=1.0, is_current=False)
    current_neutral = _item("current_neutral", authority_score=None, is_current=True)
    ranked = prioritize_items([stale_authoritative, current_neutral])
    assert [i.source_id for i in ranked] == ["current_neutral", "stale_authoritative"]


def test_ties_break_deterministically_on_original_order():
    """Ties break deterministically on original order."""
    a = _item("a")
    b = _item("b")
    c = _item("c")
    ranked_once = [i.source_id for i in prioritize_items([a, b, c])]
    ranked_again = [i.source_id for i in prioritize_items([a, b, c])]
    assert ranked_once == ranked_again == ["a", "b", "c"]


def test_prioritize_never_drops_or_duplicates_items():
    """Prioritize never drops or duplicates items."""
    items = [_item(str(i)) for i in range(10)]
    ranked = prioritize_items(items)
    assert {i.source_id for i in ranked} == {str(i) for i in range(10)}
    assert len(ranked) == 10
