"""Unit tests for `rag.context.ordering` (the "lost in the middle" render-order strategies)."""

from __future__ import annotations

from rag.context.ordering import (
    order_authority_first,
    order_begin_end_sandwich,
    order_for_rendering,
    order_grouped_by_source,
    order_priority_first,
)
from rag.context.types import ContextItem


def _item(
    source_id, priority_rank=None, authority_score=None, source_type="document"
) -> ContextItem:
    return ContextItem(
        content="x",
        source_type=source_type,
        source_id=source_id,
        priority_rank=priority_rank,
        authority_score=authority_score,
    )


def test_priority_first_orders_by_priority_rank():
    """Priority first orders by priority rank."""
    items = [_item("c", priority_rank=2), _item("a", priority_rank=0), _item("b", priority_rank=1)]
    ordered = order_priority_first(items)
    assert [i.source_id for i in ordered] == ["a", "b", "c"]


def test_authority_first_orders_by_authority_score_descending():
    """Authority first orders by authority score descending."""
    items = [
        _item("low", authority_score=0.0, priority_rank=0),
        _item("high", authority_score=1.0, priority_rank=1),
    ]
    ordered = order_authority_first(items)
    assert [i.source_id for i in ordered] == ["high", "low"]


def test_begin_end_sandwich_places_the_best_items_at_both_ends():
    """Begin end sandwich places the best items at both ends."""
    items = [_item(str(i), priority_rank=i) for i in range(5)]
    ordered = order_begin_end_sandwich(items)
    # Highest priority (rank 0) always leads.
    assert ordered[0].source_id == "0"
    # The second-best (rank 1) trails.
    assert ordered[-1].source_id == "1"
    assert {i.source_id for i in ordered} == {str(i) for i in range(5)}
    assert len(ordered) == 5


def test_begin_end_sandwich_never_drops_or_duplicates_items():
    """Begin end sandwich never drops or duplicates items."""
    items = [_item(str(i), priority_rank=i) for i in range(7)]
    ordered = order_begin_end_sandwich(items)
    assert sorted(i.source_id for i in ordered) == sorted(str(i) for i in range(7))


def test_grouped_by_source_groups_documents_before_graph_before_mcp_before_history():
    """Grouped by source groups documents before graph before mcp before history."""
    items = [
        _item("h1", priority_rank=0, source_type="conversation_history"),
        _item("m1", priority_rank=1, source_type="mcp_tool"),
        _item("g1", priority_rank=2, source_type="graph_path"),
        _item("d1", priority_rank=3, source_type="document"),
    ]
    ordered = order_grouped_by_source(items)
    assert [i.source_id for i in ordered] == ["d1", "g1", "m1", "h1"]


def test_order_for_rendering_dispatches_to_the_right_strategy():
    """Order for rendering dispatches to the right strategy."""
    items = [_item("a", priority_rank=1), _item("b", priority_rank=0)]
    assert [i.source_id for i in order_for_rendering(items, "priority_first")] == ["b", "a"]
