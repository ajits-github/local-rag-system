"""Unit tests for `rag.context.packing`.

Includes the overflow-regression invariants from section 26 of the
design: the packed result never exceeds its budget regardless of
strategy, and an oversized single item is handled explicitly (compressed
or excluded), never silently truncated at an arbitrary offset.
"""

from __future__ import annotations

from rag.context.packing import pack_naive_token_first, pack_priority_aware, pack_top_n
from rag.context.tokens import HeuristicTokenCounter
from rag.context.types import ContextBudget, ContextItem, SourceBudget

_COUNTER = HeuristicTokenCounter(chars_per_token=1.0)  # 1 char = 1 token, for exact arithmetic


def _count(text: str) -> int:
    return _COUNTER.count(text)


def _item(source_id, token_count, source_type="document") -> ContextItem:
    item = ContextItem(content="x" * token_count, source_type=source_type, source_id=source_id)
    item.token_count = token_count
    return item


def _budget(available: int, **bucket_tokens: int) -> ContextBudget:
    buckets = {
        "document": 0,
        "graph_path": 0,
        "mcp_tool": 0,
        "conversation_history": 0,
        **bucket_tokens,
    }
    return ContextBudget(
        model_context_window=100000,
        reserved_output_tokens=0,
        system_prompt_tokens=0,
        user_query_tokens=0,
        framework_overhead_tokens=0,
        safety_margin_tokens=0,
        available_context_tokens=available,
        source_budgets={
            k: SourceBudget(source_type=k, allocated_tokens=v) for k, v in buckets.items()
        },
    )


def test_top_n_takes_a_fixed_count_per_bucket_ignoring_token_size():
    """Top n takes a fixed count per bucket ignoring token size."""
    items = [_item("a", 1000), _item("b", 5), _item("c", 5)]
    kept, trace = pack_top_n(items, n_per_bucket={"document": 2})
    assert [i.source_id for i in kept] == ["a", "b"]
    assert trace.strategy == "top_n"


def test_naive_token_first_never_exceeds_the_global_budget():
    """Naive token first never exceeds the global budget."""
    items = [_item(str(i), 10) for i in range(20)]
    kept, trace = pack_naive_token_first(items, global_budget_tokens=55)
    assert sum(i.token_count for i in kept) <= 55
    assert len(kept) == 5


def test_naive_token_first_skips_an_oversized_item_and_continues():
    """Naive token first skips an oversized item and continues."""
    items = [_item("huge", 1000), _item("small", 5)]
    kept, trace = pack_naive_token_first(items, global_budget_tokens=10)
    assert [i.source_id for i in kept] == ["small"]


def test_priority_aware_never_exceeds_the_global_budget():
    """Priority aware never exceeds the global budget."""
    items = [_item(str(i), 30) for i in range(10)]
    budget = _budget(available=100, document=100)
    kept, trace = pack_priority_aware(
        items,
        budget=budget,
        count_tokens=_count,
        query="x",
        compression_enabled=False,
        compression_threshold_tokens=1000,
        compression_strategy="extractive",
        compression_max_sentences=8,
    )
    assert sum(i.token_count for i in kept) <= 100
    assert sum(i.token_count for i in kept) <= budget.available_context_tokens


def test_priority_aware_never_exceeds_a_per_bucket_budget_even_with_global_room():
    """Priority aware never exceeds a per bucket budget even with global room."""
    doc_items = [_item(f"d{i}", 30) for i in range(5)]
    graph_items = [_item(f"g{i}", 30, source_type="graph_path") for i in range(5)]
    budget = _budget(available=1000, document=40, graph_path=1000)
    kept, trace = pack_priority_aware(
        doc_items + graph_items,
        budget=budget,
        count_tokens=_count,
        query="x",
        compression_enabled=False,
        compression_threshold_tokens=1000,
        compression_strategy="extractive",
        compression_max_sentences=8,
    )
    doc_tokens = sum(i.token_count for i in kept if i.source_type == "document")
    assert doc_tokens <= 40


def test_priority_aware_compresses_an_oversized_item_to_fit_rather_than_dropping_it():
    """Priority aware compresses an oversized item to fit rather than dropping it."""
    text = "This is one sentence. " * 40  # many short, splittable sentences, well over budget
    item = ContextItem(content=text, source_type="document", source_id="big", content_type="prose")
    item.token_count = len(text)  # exact char count with our 1.0 ratio counter
    budget = _budget(available=50, document=50)
    kept, trace = pack_priority_aware(
        [item],
        budget=budget,
        count_tokens=_count,
        query="word",
        compression_enabled=True,
        compression_threshold_tokens=10,
        compression_strategy="sentence_trim",
        compression_max_sentences=8,
    )
    assert len(kept) == 1
    assert kept[0].compressed is True
    assert kept[0].token_count <= 50
    assert trace.truncation_count == 1


def test_priority_aware_excludes_an_item_that_doesnt_fit_even_after_compression():
    """Priority aware excludes an item that doesnt fit even after compression."""
    # A single word with no sentence boundary at all cannot be shrunk.
    item = ContextItem(content="x" * 500, source_type="document", source_id="big")
    item.token_count = 500
    budget = _budget(available=10, document=10)
    kept, trace = pack_priority_aware(
        [item],
        budget=budget,
        count_tokens=_count,
        query="x",
        compression_enabled=True,
        compression_threshold_tokens=10,
        compression_strategy="sentence_trim",
        compression_max_sentences=8,
    )
    assert kept == []
    assert trace.dropped_item_count == 1
    assert trace.decisions[0].included is False


def test_priority_aware_prefers_higher_priority_items_when_budget_is_tight():
    """Priority aware prefers higher priority items when budget is tight."""
    stale = _item("stale", 20)
    stale.is_current = False
    current = _item("current", 20)
    current.is_current = True
    budget = _budget(available=20, document=20)
    kept, trace = pack_priority_aware(
        [stale, current],
        budget=budget,
        count_tokens=_count,
        query="x",
        compression_enabled=False,
        compression_threshold_tokens=1000,
        compression_strategy="extractive",
        compression_max_sentences=8,
    )
    assert [i.source_id for i in kept] == ["current"]


def test_priority_aware_continues_past_a_skipped_item_to_a_smaller_one():
    """Priority aware continues past a skipped item to a smaller one."""
    big = _item("big", 100)
    small = _item("small", 5)
    budget = _budget(available=10, document=10)
    kept, trace = pack_priority_aware(
        [big, small],
        budget=budget,
        count_tokens=_count,
        query="x",
        compression_enabled=False,
        compression_threshold_tokens=1000,
        compression_strategy="extractive",
        compression_max_sentences=8,
    )
    assert [i.source_id for i in kept] == ["small"]
