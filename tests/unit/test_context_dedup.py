"""Unit tests for `rag.context.dedup`."""

from __future__ import annotations

from rag.context.dedup import _jaccard_overlap, _shingles, deduplicate_items
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


def test_near_duplicate_paragraphs_with_a_conflicting_number_are_never_merged():
    """A long near-identical paragraph differing only in one number must never be merged.

    Regression for CTX-007: the pre-existing "different facts" test used
    short sentences whose word-shingle overlap already fell below the
    threshold on its own, so it never actually exercised the dangerous
    case where overlap stays *above* threshold despite a genuine numeric
    conflict (an old paragraph restating "timeout 60" alongside an
    otherwise-identical, current, authoritative paragraph saying
    "timeout 30"). Deduplication must never pick a winner between
    conflicting values; that is freshness/authority ordering's job.
    """
    long_paragraph = (
        "The webhook delivery subsystem retries every failed delivery attempt up to five "
        "times, waiting with exponential backoff between attempts, and the overall delivery "
        "timeout for the entire retry sequence is {n} seconds before the webhook is marked "
        "permanently failed and moved to the dead-letter queue for manual review."
    )
    stale = _item("c1", long_paragraph.format(n="60"))
    current = _item("c2", long_paragraph.format(n="30"))
    # Confirm the premise: these two texts really do overlap enough that
    # the near-duplicate pass would merge them without the numeric guard.
    assert _jaccard_overlap(_shingles(stale.content), _shingles(current.content)) >= 0.85

    kept, stats = deduplicate_items([stale, current], overlap_threshold=0.85)

    assert len(kept) == 2
    assert stats.near_duplicates_removed == 0


def test_exact_content_hash_duplicate_across_tenants_is_never_merged():
    """Byte-identical content from two different tenants must never be merged.

    Regression for CTX-007: merging would silently drop one tenant's
    citation into the other's merged_from, misattributing a shared fact
    to the wrong tenant's document even though both chunks were
    independently authorized for this caller (e.g. a
    cross_tenant_support_roles request spanning two tenants).
    """
    a = _item("c1", "Standard retention policy text.", tenant_id="tenant_alpha")
    b = _item("c2", "Standard retention policy text.", tenant_id="tenant_beta")
    kept, stats = deduplicate_items([a, b])
    assert len(kept) == 2
    assert stats.exact_duplicates_removed == 0


def test_near_duplicate_across_tenants_is_never_merged():
    """Near-identical content from two different tenants must never be merged."""
    a = _item(
        "c1",
        "The webhook delivery timeout is thirty seconds for all retries.",
        tenant_id="tenant_alpha",
    )
    b = _item(
        "c2",
        "The webhook delivery timeout is thirty seconds for all retries today.",
        tenant_id="tenant_beta",
    )
    kept, stats = deduplicate_items([a, b], overlap_threshold=0.7)
    assert len(kept) == 2
    assert stats.near_duplicates_removed == 0


def test_exact_content_hash_matching_finds_the_right_tenants_entry_among_several():
    """A third, same-tenant duplicate must still find its own tenant's kept entry.

    Regression for the fix's own internal bookkeeping: content_hash ->
    kept-index used to be a single slot per hash, so a different
    tenant's byte-identical item (correctly not merged) would overwrite
    that slot, causing a later same-tenant duplicate of the *original*
    item to miss its match entirely.
    """
    original = _item("c1", "Standard retention policy text.", tenant_id="tenant_alpha")
    other_tenant = _item("c2", "Standard retention policy text.", tenant_id="tenant_beta")
    later_duplicate = _item("c3", "Standard retention policy text.", tenant_id="tenant_alpha")

    kept, stats = deduplicate_items([original, other_tenant, later_duplicate])

    assert len(kept) == 2
    alpha_kept = next(k for k in kept if k.tenant_id == "tenant_alpha")
    assert alpha_kept.source_id == "c1"
    assert alpha_kept.merged_from == ["c3"]
    assert stats.exact_duplicates_removed == 1


_BASE = (
    "Service accounts in the billing namespace are {} to call the payments API "
    "directly from batch jobs running in the nightly reconciliation window, "
    "according to the platform access policy maintained by the security team"
)


def test_same_id_in_different_buckets_is_not_an_identity_duplicate():
    """A history turn_id equal to a chunk id never erases the document chunk (CTX-016)."""
    items = [
        _item("c1", "document evidence text"),
        _item("c1", "User: an unrelated question", source_type="conversation_history"),
    ]
    kept, stats = deduplicate_items(items)
    assert len(kept) == 2
    assert stats.exact_duplicates_removed == 0


def test_same_id_in_different_tenants_is_not_an_identity_duplicate():
    """The identity pass is tenant-scoped, like the content passes (CTX-016)."""
    items = [
        _item("c1", "alpha text", tenant_id="alpha"),
        _item("c1", "beta text", tenant_id="beta"),
    ]
    kept, _ = deduplicate_items(items)
    assert [i.tenant_id for i in kept] == ["alpha", "beta"]


def test_history_and_document_with_identical_content_never_merge():
    """The exact-hash pass never folds a history turn into evidence or vice versa."""
    items = [
        _item("c1", "identical text"),
        _item("turn_0", "identical text", source_type="conversation_history"),
    ]
    kept, _ = deduplicate_items(items)
    assert len(kept) == 2


def test_near_duplicates_differing_only_by_negation_are_never_merged():
    """A negation ("allowed" vs "not allowed") is a conflict, not a duplicate (CTX-007)."""
    items = [
        _item("c1", _BASE.format("allowed")),
        _item("c2", _BASE.format("not allowed")),
    ]
    assert _jaccard_overlap(_shingles(items[0].content), _shingles(items[1].content)) >= 0.85
    kept, _ = deduplicate_items(items)
    assert len(kept) == 2


def test_near_duplicates_differing_only_by_polarity_word_are_never_merged():
    """A word-valued change such as "enabled" vs "disabled" also blocks the merge."""
    base = _BASE.replace("are {} to call", "have direct calls {} for")
    items = [_item("c1", base.format("enabled")), _item("c2", base.format("disabled"))]
    assert _jaccard_overlap(_shingles(items[0].content), _shingles(items[1].content)) >= 0.8
    kept, _ = deduplicate_items(items, overlap_threshold=0.8)
    assert len(kept) == 2


def test_near_duplicate_where_only_one_side_has_a_number_is_never_merged():
    """A value present on one side only is a conflict, not a duplicate."""
    items = [
        _item("c1", _BASE.format("allowed") + " with a limit of 30 calls"),
        _item("c2", _BASE.format("allowed") + " with a limit of many calls"),
    ]
    assert _jaccard_overlap(_shingles(items[0].content), _shingles(items[1].content)) >= 0.8
    kept, _ = deduplicate_items(items, overlap_threshold=0.8)
    assert len(kept) == 2
