"""Conversation-history management: sliding window, deterministic summarization, retrieval.

This codebase has no server-side session store (see "Known deviations"
below) and no prior milestone added one: `POST /query`/`POST
/agent/query` are, and remain, fully stateless per request. A caller
that wants multi-turn behavior re-sends prior turns as
`ConversationTurn`s on each request, the same "caller asserts, server
only bounds/uses it" pattern this codebase already applies to
`tenant_id`/`roles`/`case_approvals`. This module's job is purely how
much of that caller-supplied history actually reaches the prompt, never
where it is stored.

**Trust boundary.** A `ConversationTurn.role` is only ever `"user"` or
`"assistant"` (see `rag.context.types`); nothing here ever writes a
turn into the fixed system-instruction string a `PromptTemplate`
renders. History turns, once converted to `ContextItem`s by
`rag.context.normalize.normalize_conversation_turn`, flow through the
same budgeting/dedup/packing/prioritization pipeline as any other
evidence and are rendered inside the *user* turn's context block
alongside retrieved evidence, never the system turn. A malicious prior
turn (e.g. containing "ignore previous instructions") is therefore
data, not an instruction channel, exactly like a retrieved document
(see section 16, "Context poisoning").

**Known deviations.** `"summarized"`'s summary is produced by a
deterministic, extractive sentence-selection pass (see
`_summarize_turns`), never an LLM call: an LLM-generated summary would
itself need a token budget, latency, and failure-handling story, and
would not be "deterministic enough to test" (section 14's own
requirement). The summary intentionally never asserts a fact the
original turns didn't contain; it only ever *selects* a subset of
already-said sentences, the same "selection, not regeneration" principle
`rag.context.compression` uses for oversized evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from rag.context.types import ContextItem, ConversationTurn

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _query_terms(query: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(query)}


def _sliding_window(turns: list[ConversationTurn], max_turns: int) -> list[ConversationTurn]:
    """Keep only the most recent `max_turns` turns."""
    if max_turns <= 0:
        return []
    return turns[-max_turns:]


def _summarize_turns(
    older_turns: list[ConversationTurn],
    query: str,
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> str:
    """Deterministically extract the most query-relevant sentences from `older_turns`.

    Retains user goals, established facts, explicit constraints, and
    unresolved questions by scoring each sentence on query-term overlap
    (the same signal `rag.context.compression`'s extractive strategy
    uses) plus a small recency bonus favoring later turns, since the
    most recently stated goal/constraint is more likely still relevant
    than an earlier, possibly-superseded one. Selected sentences are
    rendered back in their original chronological order, never
    reordered, so causal references ("as I said before, ...") still
    read coherently.

    Parameters
    ----------
    older_turns : list[ConversationTurn]
        Turns older than the sliding window, oldest first.
    query : str
        Current query, for relevance scoring.
    target_tokens : int
        Token budget for the produced summary.
    count_tokens : Callable[[str], int]

    Returns
    -------
    str
        The summary text, or `""` if `older_turns` is empty.
    """
    if not older_turns:
        return ""
    terms = _query_terms(query)
    sentences: list[tuple[int, str, str]] = []  # (turn_index, role, sentence)
    for turn_index, turn in enumerate(older_turns):
        for sentence in _SENTENCE_SPLIT_RE.split(turn.content.strip()):
            if sentence:
                sentences.append((turn_index, turn.role, sentence))
    if not sentences:
        return ""

    def score(entry: tuple[int, str, str]) -> float:
        turn_index, _role, sentence = entry
        words = {w.lower() for w in _WORD_RE.findall(sentence)}
        overlap = len(words & terms) if terms else 0
        recency_bonus = turn_index / max(len(older_turns) - 1, 1) * 0.5
        return overlap + recency_bonus

    ranked_indices = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    kept_indices: set[int] = set()
    running_tokens = 0
    for idx in ranked_indices:
        sentence_tokens = count_tokens(sentences[idx][2])
        if kept_indices and running_tokens + sentence_tokens > target_tokens:
            continue
        kept_indices.add(idx)
        running_tokens += sentence_tokens
    if not kept_indices:
        kept_indices = {ranked_indices[0]}
    ordered = sorted(kept_indices)
    return " ".join(sentences[i][2] for i in ordered)


def _retrieval_select(
    turns: list[ConversationTurn], query: str, max_turns: int
) -> list[ConversationTurn]:
    """Keep the `max_turns` whole turns with the highest query-term overlap, oldest first.

    Falls back to a plain sliding window when no turn has any overlap
    at all (an uninformative query, or history genuinely unrelated to
    it), so "retrieval" never does *worse* than recency in that case.
    """
    terms = _query_terms(query)
    if not terms:
        return _sliding_window(turns, max_turns)
    scored = [
        (i, len({w.lower() for w in _WORD_RE.findall(t.content)} & terms))
        for i, t in enumerate(turns)
    ]
    if not any(score > 0 for _, score in scored):
        return _sliding_window(turns, max_turns)
    best = sorted(scored, key=lambda p: p[1], reverse=True)[:max_turns]
    top_indices = sorted(best, key=lambda p: p[0])
    return [turns[i] for i, _ in top_indices]


def select_history_items(
    turns: list[ConversationTurn],
    *,
    strategy: str,
    max_turns: int,
    summarize_min_turns_to_trigger: int,
    query: str,
    count_tokens: Callable[[str], int],
    summary_target_tokens: int,
    normalize_fn: Callable[[ConversationTurn, int], ContextItem],
) -> list[ContextItem]:
    """Apply `strategy` to `turns` and return the resulting `ContextItem`s.

    Parameters
    ----------
    turns : list[ConversationTurn]
        Caller-supplied history, oldest first.
    strategy : {"full", "sliding_window", "summarized", "retrieval"}
    max_turns : int
        Recent turns kept raw (ignored for `"full"`).
    summarize_min_turns_to_trigger : int
        `"summarized"` only summarizes once there are more older turns
        than this; otherwise behaves like `"sliding_window"`.
    query : str
        Current query text.
    count_tokens : Callable[[str], int]
    summary_target_tokens : int
        Token budget for a produced summary item.
    normalize_fn : Callable[[ConversationTurn, int], ContextItem]
        Typically `rag.context.normalize.normalize_conversation_turn`;
        injected rather than imported directly to keep this module
        free of a hard dependency on that one, narrower, concern.

    Returns
    -------
    list[ContextItem]
        Oldest first; a produced summary (if any) is placed before the
        raw recent turns it summarizes, preserving chronological order.
    """
    if not turns:
        return []

    if strategy == "full":
        return [normalize_fn(t, i) for i, t in enumerate(turns)]

    if strategy == "retrieval":
        selected = _retrieval_select(turns, query, max_turns)
        return [normalize_fn(t, i) for i, t in enumerate(selected)]

    if strategy == "summarized":
        recent = _sliding_window(turns, max_turns)
        older = turns[: len(turns) - len(recent)]
        if len(older) <= summarize_min_turns_to_trigger:
            return [normalize_fn(t, i) for i, t in enumerate(turns)]
        summary_text = _summarize_turns(older, query, summary_target_tokens, count_tokens)
        items = [
            ContextItem(
                content=f"(summary of {len(older)} earlier turns) {summary_text}",
                source_type="conversation_history",
                source_id="history_summary",
                citation=f"summary of {len(older)} earlier conversation turns",
            )
        ]
        items.extend(normalize_fn(t, i) for i, t in enumerate(recent))
        return items

    # "sliding_window" (the default).
    recent = _sliding_window(turns, max_turns)
    return [normalize_fn(t, i) for i, t in enumerate(recent)]
