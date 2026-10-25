"""Unit tests for `rag.context.history`."""

from __future__ import annotations

from rag.context.history import select_history_items
from rag.context.normalize import normalize_conversation_turn
from rag.context.tokens import HeuristicTokenCounter
from rag.context.types import ConversationTurn

_COUNTER = HeuristicTokenCounter(chars_per_token=4.0)


def _count(text: str) -> int:
    return _COUNTER.count(text)


def _turns(n: int) -> list[ConversationTurn]:
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(ConversationTurn(role=role, content=f"Turn {i} content about topic {i}."))
    return turns


def test_full_strategy_keeps_every_turn():
    """Full strategy keeps every turn."""
    turns = _turns(10)
    items = select_history_items(
        turns,
        strategy="full",
        max_turns=3,
        summarize_min_turns_to_trigger=4,
        query="x",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    assert len(items) == 10


def test_sliding_window_keeps_only_the_most_recent_turns():
    """Sliding window keeps only the most recent turns."""
    turns = _turns(10)
    items = select_history_items(
        turns,
        strategy="sliding_window",
        max_turns=3,
        summarize_min_turns_to_trigger=4,
        query="x",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    assert len(items) == 3
    assert "Turn 9" in items[-1].content
    assert "Turn 7" in items[0].content


def test_summarized_strategy_produces_one_summary_item_plus_recent_raw_turns():
    """Summarized strategy produces one summary item plus recent raw turns."""
    turns = _turns(10)
    items = select_history_items(
        turns,
        strategy="summarized",
        max_turns=3,
        summarize_min_turns_to_trigger=2,
        query="topic 2",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    # 1 summary + 3 recent raw turns.
    assert len(items) == 4
    assert items[0].source_id == "history_summary"
    assert "summary of 7 earlier conversation turns" in items[0].citation


def test_summarized_strategy_falls_back_to_sliding_window_below_the_trigger():
    """Summarized strategy falls back to sliding window below the trigger."""
    turns = _turns(4)
    items = select_history_items(
        turns,
        strategy="summarized",
        max_turns=3,
        summarize_min_turns_to_trigger=4,
        query="x",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    # Only 1 older turn (4 - 3 = 1), at or below the trigger of 4 -> no summary.
    assert len(items) == 4
    assert all(i.source_id != "history_summary" for i in items)


def test_summary_never_fabricates_a_fact_only_selects_existing_sentences():
    """Summary never fabricates a fact only selects existing sentences."""
    turns = [
        ConversationTurn(role="user", content="The deployment window is Tuesday at noon."),
        ConversationTurn(role="assistant", content="Understood, I will note that."),
        ConversationTurn(role="user", content="Also the rollback plan needs sign-off."),
    ]
    items = select_history_items(
        turns,
        strategy="summarized",
        max_turns=0,
        summarize_min_turns_to_trigger=0,
        query="deployment window",
        count_tokens=_count,
        summary_target_tokens=20,
        normalize_fn=normalize_conversation_turn,
    )
    assert len(items) == 1
    summary_text = items[0].content
    # Every word in the summary must trace back to an original turn.
    original_text = " ".join(t.content for t in turns)
    for word in summary_text.replace("(summary of 3 earlier turns)", "").split():
        assert word.strip(".") in original_text or word in original_text


def test_retrieval_strategy_selects_the_most_query_relevant_turns():
    """Retrieval strategy selects the most query relevant turns."""
    turns = [
        ConversationTurn(role="user", content="What is the weather like today?"),
        ConversationTurn(role="assistant", content="It is sunny."),
        ConversationTurn(role="user", content="What is the webhook delivery timeout?"),
        ConversationTurn(role="assistant", content="It is thirty seconds."),
    ]
    items = select_history_items(
        turns,
        strategy="retrieval",
        max_turns=2,
        summarize_min_turns_to_trigger=4,
        query="webhook timeout",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    contents = " ".join(i.content for i in items)
    assert "webhook" in contents.lower() or "thirty" in contents.lower()


def test_empty_history_returns_no_items():
    """Empty history returns no items."""
    items = select_history_items(
        [],
        strategy="sliding_window",
        max_turns=3,
        summarize_min_turns_to_trigger=4,
        query="x",
        count_tokens=_count,
        summary_target_tokens=50,
        normalize_fn=normalize_conversation_turn,
    )
    assert items == []


def test_history_turn_role_is_never_system():
    """History turn role is never system."""
    # ConversationTurn's own Literal type already enforces this at
    # construction time; this test documents the invariant explicitly.
    import pydantic

    try:
        ConversationTurn(role="system", content="x")
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("ConversationTurn must reject role='system'")
