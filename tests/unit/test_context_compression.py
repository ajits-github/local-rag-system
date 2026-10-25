"""Unit tests for `rag.context.compression` and `rag.context.graph_packing`."""

from __future__ import annotations

from rag.context.compression import compress_item
from rag.context.graph_packing import parse_graph_fact_text, shrink_graph_path
from rag.context.tokens import HeuristicTokenCounter
from rag.context.types import ContextItem

_COUNTER = HeuristicTokenCounter(chars_per_token=4.0)


def _count(text: str) -> int:
    return _COUNTER.count(text)


def test_item_within_budget_is_returned_unchanged():
    """Item within budget is returned unchanged."""
    item = ContextItem(content="short text", source_type="document", source_id="c1")
    item.token_count = _count(item.content)
    result = compress_item(
        item,
        target_tokens=1000,
        count_tokens=_count,
        query="x",
        strategy="extractive",
        max_sentences=8,
    )
    assert result.content == item.content
    assert result.compressed is False


def test_extractive_compression_keeps_query_relevant_sentences_in_original_order():
    """Extractive compression keeps query relevant sentences in original order."""
    text = (
        "The onboarding process takes three weeks. "
        "The webhook delivery timeout is thirty seconds. "
        "New hires receive a laptop on day one. "
        "The webhook retry policy allows three attempts."
    )
    item = ContextItem(content=text, source_type="document", source_id="c1", content_type="prose")
    item.token_count = _count(text)
    compressed = compress_item(
        item,
        target_tokens=20,
        count_tokens=_count,
        query="What is the webhook timeout?",
        strategy="extractive",
        max_sentences=2,
    )
    assert compressed.compressed is True
    assert "webhook" in compressed.content.lower()
    assert compressed.original_token_count == item.token_count
    # Original sentence order is preserved among the kept sentences.
    idx_timeout = compressed.content.find("timeout")
    idx_retry = compressed.content.find("retry")
    if idx_retry != -1:
        assert idx_timeout < idx_retry


def test_compression_never_reorders_or_invents_text_only_selects_existing_sentences():
    """Compression never reorders or invents text only selects existing sentences."""
    text = "Fact one is true. Fact two is also true. Fact three as well."
    item = ContextItem(content=text, source_type="document", source_id="c1")
    item.token_count = _count(text)
    compressed = compress_item(
        item,
        target_tokens=5,
        count_tokens=_count,
        query="fact",
        strategy="sentence_trim",
        max_sentences=8,
    )
    for sentence in compressed.content.split(". "):
        assert sentence.strip(".") + "." in text or sentence.strip() in text


def test_table_content_type_uses_line_based_trimming_and_keeps_the_header():
    """Table content type uses line based trimming and keeps the header."""
    table = "| Name | Value |\n| a | 1 |\n| b | 2 |\n| c | 3 |\n| d | 4 |"
    item = ContextItem(content=table, source_type="document", source_id="c1", content_type="table")
    item.token_count = _count(table)
    compressed = compress_item(
        item,
        target_tokens=10,
        count_tokens=_count,
        query="value",
        strategy="extractive",
        max_sentences=2,
    )
    assert compressed.content.splitlines()[0] == "| Name | Value |"


def test_redaction_marker_is_never_reassembled_by_compression():
    """Redaction marker is never reassembled by compression."""
    text = (
        "The admin credential for this tenant is [REDACTED:SENSITIVE_FIELD]. "
        "This value must never be disclosed to unauthorized readers. "
        "Contact your tenant administrator for access requests."
    )
    item = ContextItem(content=text, source_type="document", source_id="c1")
    item.token_count = _count(text)
    compressed = compress_item(
        item,
        target_tokens=15,
        count_tokens=_count,
        query="admin credential",
        strategy="extractive",
        max_sentences=1,
    )
    # The marker may be kept or dropped whole, but never split into a
    # fragment that could be confused with a real, partially-redacted value.
    assert (
        "[REDACTED:SENSITIVE_FIELD]" in compressed.content or "REDACTED" not in compressed.content
    )


def test_parse_dependency_path_fact_text():
    """Parse dependency path fact text."""
    hops = parse_graph_fact_text("Dependency path: Alpha API -> Alpha Database -> PostgreSQL")
    assert [h.subject for h in hops] == ["Alpha API", "Alpha Database"]
    assert [h.object for h in hops] == ["Alpha Database", "PostgreSQL"]


def test_shrink_graph_path_drops_whole_trailing_hops_not_mid_hop_text():
    """Shrink graph path drops whole trailing hops not mid hop text."""
    hops = parse_graph_fact_text(
        "Dependency path: Alpha API -> Alpha Database -> PostgreSQL -> Shared Storage"
    )
    item = ContextItem(
        content="Dependency path: Alpha API -> Alpha Database -> PostgreSQL -> Shared Storage",
        source_type="graph_path",
        source_id="g1",
        graph_path=hops,
    )
    shrunk = shrink_graph_path(item, target_tokens=8, token_counter_fn=_count)
    assert shrunk.startswith("Dependency path: Alpha API")
    assert "Shared Storage" not in shrunk or shrunk == item.content
    # Never truncated mid-name.
    assert "Alpha Data" not in shrunk.replace("Alpha Database", "")
