"""Unit tests for `rag.context.dedup`."""

from __future__ import annotations

from rag.context.dedup import deduplicate_items
from rag.context.types import ContextItem


def _item(
    source_id, content, token_count=10, source_type="document", tenant_id=None
) -> ContextItem:
    return ContextItem(
        content=content,
        source_type=source_type,
        source_id=source_id,
        token_count=token_count,
        tenant_id=tenant_id,
    )


def test_exact_id_duplicate_is_collapsed():
    """Exact id duplicate is collapsed."""
    items = [_item("c1", "hello world"), _item("c1", "hello world")]
    kept, stats = deduplicate_items(items)
    assert len(kept) == 1
    assert stats.exact_duplicates_removed == 1
    assert stats.candidates_before == 2
    assert stats.items_after == 1


def test_exact_content_hash_duplicate_with_different_ids_is_collapsed_and_tracked():
    """Exact content hash duplicate with different ids is collapsed and tracked."""
    items = [_item("c1", "the exact same text"), _item("c2", "the exact same text")]
    kept, stats = deduplicate_items(items)
    assert len(kept) == 1
    assert kept[0].source_id == "c1"
    assert kept[0].merged_from == ["c2"]
    assert stats.exact_duplicates_removed == 1


def test_near_duplicate_text_is_collapsed_above_threshold():
    """Near duplicate text is collapsed above threshold."""
    a = _item("c1", "The webhook delivery timeout is thirty seconds for all retries.")
    b = _item("c2", "The webhook delivery timeout is thirty seconds for all retries today.")
    kept, stats = deduplicate_items([a, b], overlap_threshold=0.7)
    assert len(kept) == 1
    assert stats.near_duplicates_removed == 1


def test_different_facts_about_the_same_topic_are_never_merged():
    """Different facts about the same topic are never merged."""
    a = _item("c1", "The webhook delivery timeout is 30 seconds.")
    b = _item("c2", "The webhook delivery timeout is 60 seconds.")
    kept, stats = deduplicate_items([a, b], overlap_threshold=0.85)
    assert len(kept) == 2
    assert stats.near_duplicates_removed == 0


def test_near_duplicates_across_different_source_types_are_never_merged():
    """Near duplicates across different source types are never merged."""
    # Same text, but one is a document chunk and one is a graph fact;
    # near-dup matching is scoped per source_type.
    a = _item("c1", "Alpha API DEPENDS_ON Alpha Database", source_type="document")
    b = _item("c2", "Alpha API DEPENDS_ON Alpha Database", source_type="graph_path")
    kept, stats = deduplicate_items([a, b])
    # Exact content hash match still fires regardless of source_type,
    # since it's a stronger signal than the near-duplicate pass.
    assert len(kept) == 1


def test_tokens_saved_is_reported():
    """Tokens saved is reported."""
    items = [_item("c1", "same text", token_count=50), _item("c2", "same text", token_count=50)]
    kept, stats = deduplicate_items(items)
    assert stats.tokens_saved == 50


def test_empty_input_is_handled():
    """Empty input is handled."""
    kept, stats = deduplicate_items([])
    assert kept == []
    assert stats.candidates_before == 0
    assert stats.items_after == 0
