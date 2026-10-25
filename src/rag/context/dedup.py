"""Context deduplication: exact, identity-based, and near-duplicate collapsing.

Three passes, in order, each strictly narrowing what survives:

1. **Exact identity** -- the same `chunk_id`/`source_id` appearing twice
   (e.g. a chunk surfaced by both direct retrieval and relationship
   expansion, or by both dense and BM25 before fusion already
   deduplicated it, kept here as a second safety net for callers that
   assemble candidates from more than one pipeline call).
2. **Exact content hash** -- different ids, byte-identical text (e.g.
   the same document fact surfaced once as a retrieved chunk and again
   as a graph fact derived from that same sentence).
3. **Near-duplicate overlap** -- different ids, different-but-highly-
   overlapping text, via word-shingle Jaccard similarity
   (`config.context.deduplication.overlap_threshold`). Deliberately a
   cheap, dependency-free, explainable heuristic, not an embedding-
   similarity model: a second embedding call per candidate pair would
   add real latency and a second "confidence value" nothing else in
   this budget calls for, and the two coarser passes above already
   catch the common real-world cases (relationship expansion, hybrid
   fusion, graph-vs-chunk restatement).

Each pass keeps the *first* occurrence in input order (callers are
expected to hand candidates in priority order, so "first" already means
"best" for ties) and records the discarded item's `source_id` onto the
kept item's `merged_from`, never dropping provenance silently.

Never merges two items just because they are about the same *topic*;
only byte-identical or near-byte-identical text triggers a merge, so two
genuinely different facts about the same entity are never accidentally
collapsed into one.
"""

from __future__ import annotations

import re

from rag.context.normalize import content_hash
from rag.context.types import ContextItem, DeduplicationStats

_WORD_RE = re.compile(r"\w+")


def _shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Word-level shingles (overlapping n-grams) of `text`, lowercased, for Jaccard overlap."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def _jaccard_overlap(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    """Jaccard similarity of two shingle sets; `0.0` if both are empty."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def deduplicate_items(
    items: list[ContextItem], *, overlap_threshold: float = 0.85
) -> tuple[list[ContextItem], DeduplicationStats]:
    """Collapse exact-id, exact-content, and near-duplicate items.

    Parameters
    ----------
    items : list[ContextItem]
        Candidates, in priority order (first occurrence wins a merge).
        Conversation-history and system/query items are passed through
        this function like any other item; in practice duplicate
        history turns are rare but not impossible (a caller re-sending
        an identical turn), and the same exact-match passes still apply
        correctly to them.
    overlap_threshold : float, optional
        Jaccard-overlap threshold for the near-duplicate pass, by
        default 0.85 (mirrors `DeduplicationConfig.overlap_threshold`'s
        default).

    Returns
    -------
    tuple[list[ContextItem], DeduplicationStats]
        The deduplicated items (same relative order as their first
        occurrence), and counts for observability.
    """
    tokens_before = sum(i.token_count for i in items)
    seen_ids: set[str] = set()
    seen_hashes: dict[str, int] = {}
    kept: list[ContextItem] = []
    exact_removed = 0
    near_removed = 0
    shingle_cache: list[set[tuple[str, ...]]] = []

    for item in items:
        if item.source_id in seen_ids:
            exact_removed += 1
            continue
        chash = content_hash(item.content)
        if chash in seen_hashes:
            idx = seen_hashes[chash]
            kept[idx] = kept[idx].model_copy(
                update={"merged_from": [*kept[idx].merged_from, item.source_id]}
            )
            seen_ids.add(item.source_id)
            exact_removed += 1
            continue

        item_shingles = _shingles(item.content)
        merged_into_existing = False
        for idx, existing in enumerate(kept):
            if existing.source_type != item.source_type:
                continue
            if _jaccard_overlap(shingle_cache[idx], item_shingles) < overlap_threshold:
                continue
            kept[idx] = existing.model_copy(
                update={"merged_from": [*existing.merged_from, item.source_id]}
            )
            seen_ids.add(item.source_id)
            near_removed += 1
            merged_into_existing = True
            break
        if merged_into_existing:
            continue

        seen_ids.add(item.source_id)
        seen_hashes[chash] = len(kept)
        kept.append(item)
        shingle_cache.append(item_shingles)

    tokens_after = sum(i.token_count for i in kept)
    stats = DeduplicationStats(
        candidates_before=len(items),
        items_after=len(kept),
        tokens_saved=max(tokens_before - tokens_after, 0),
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
    )
    return kept, stats
