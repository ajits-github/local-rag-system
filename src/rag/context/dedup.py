"""Context deduplication: exact, identity-based, and near-duplicate collapsing.

Three passes, in order, each strictly narrowing what survives:

1. **Exact identity** -- the same `source_id` appearing twice within one
   budget bucket, tenant and dataset (e.g. a chunk surfaced by both
   direct retrieval and relationship expansion, or by both dense and BM25
   before fusion already deduplicated it, kept here as a second safety
   net for callers that assemble candidates from more than one pipeline
   call). The id alone is never trusted across buckets: a caller-chosen
   history `turn_id` must not erase a document chunk with the same id.
2. **Exact content hash** -- different ids, byte-identical text (e.g.
   the same document fact surfaced once as a retrieved chunk and again
   as a graph fact derived from that same sentence). Conversation
   history only ever merges with history, never with evidence.
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

Never merges two items just because they are about the same *topic*. High
overlap alone proves nothing: "retention is 30 days" and "retention is 30
hours" share almost every word of a long paragraph, and merging keeps whichever
came first regardless of which is current. So the near-duplicate pass drops an
item only when it adds nothing: its words must appear, in order, within the
kept item's, and both must state the same numbers and negation or polarity
words. That also refuses swapped values, changed names and moved negation, at
the cost of keeping some harmless rewordings, which packing can still drop.

Both the exact-content-hash and near-duplicate passes additionally
require matching `tenant_id` before merging (CTX-007): a single
request's candidate set can legitimately span two tenants (a
`cross_tenant_support_roles` caller), and merging byte-identical or
near-identical boilerplate text across that boundary would silently
drop one tenant's citation into the other's `merged_from`, misattributing
a shared fact to the wrong tenant's document even though both chunks
were independently authorized for this caller.
"""

from __future__ import annotations

import re

from rag.context.normalize import content_hash
from rag.context.planner import bucket_for_item
from rag.context.types import ContextItem, DeduplicationStats

_WORD_RE = re.compile(r"\w+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_POLARITY_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "allowed",
        "denied",
        "forbidden",
        "prohibited",
        "permitted",
        "enabled",
        "disabled",
        "required",
        "optional",
        "true",
        "false",
        "deprecated",
        "supported",
        "unsupported",
    }
)


def _numbers(text: str) -> set[str]:
    """Distinct numeric tokens in `text` (e.g. `{"30", "60"}`), for the conflict guard."""
    return set(_NUMBER_RE.findall(text))


def _polarity(text: str) -> set[str]:
    """Negation/polarity words in `text`, with any `n't` contraction read as `not`."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    lowered = text.lower().replace("\u2019", "'")
    if "n't" in lowered:
        words.add("not")
    return words & _POLARITY_WORDS


def _words(text: str) -> list[str]:
    """Lowercased word tokens of `text`, in order."""
    return _WORD_RE.findall(text.lower())


def _adds_nothing(candidate: list[str], kept: list[str]) -> bool:
    """Whether `candidate` is an in-order subsequence of `kept`, so dropping it loses no words."""
    remaining = iter(kept)
    return all(word in remaining for word in candidate)


def _identity_key(item: ContextItem) -> tuple[str, str | None, str | None, str]:
    """Key for the exact-identity pass: bucket, tenant and dataset scope the id."""
    return (bucket_for_item(item), item.tenant_id, item.dataset_id, item.source_id)


def _is_history(item: ContextItem) -> bool:
    """Whether `item` is a conversation-history turn."""
    return item.source_type == "conversation_history"


def _tenant_matches(a: ContextItem, b: ContextItem) -> bool:
    """Whether two items may be merged under the tenant-scoping rule.

    `None` (untenanted, pre-governance content) is treated as its own
    single bucket: two untenanted items may still merge with each other,
    but never with a real tenant's content, and two different real
    tenants' items never merge with each other.
    """
    return a.tenant_id == b.tenant_id


def _shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Word-level shingles (overlapping n-grams) of `text`, lowercased, for Jaccard overlap."""
    words = _words(text)
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
    seen_ids: set[tuple[str, str | None, str | None, str]] = set()
    # content_hash -> every kept index with that hash, not just one: two
    # different tenants' byte-identical content (CTX-007) must each get
    # their own kept entry under the same hash, so a later same-tenant
    # duplicate of either still finds its own matching one rather than
    # whichever tenant happened to occupy the (overwritten) single slot.
    seen_hashes: dict[str, list[int]] = {}
    kept: list[ContextItem] = []
    exact_removed = 0
    near_removed = 0
    shingle_cache: list[set[tuple[str, ...]]] = []

    for item in items:
        identity = _identity_key(item)
        if identity in seen_ids:
            exact_removed += 1
            continue
        chash = content_hash(item.content)
        matched_hash_idx = next(
            (
                idx
                for idx in seen_hashes.get(chash, [])
                if _tenant_matches(kept[idx], item) and _is_history(kept[idx]) == _is_history(item)
            ),
            None,
        )
        if matched_hash_idx is not None:
            kept[matched_hash_idx] = kept[matched_hash_idx].model_copy(
                update={"merged_from": [*kept[matched_hash_idx].merged_from, item.source_id]}
            )
            seen_ids.add(identity)
            exact_removed += 1
            continue
        # Either no hash match at all, or only a different tenant's
        # byte-identical content (CTX-007): falls through to be kept as
        # its own item below, exactly as if no hash match had occurred.

        item_shingles = _shingles(item.content)
        item_numbers = _numbers(item.content)
        item_polarity = _polarity(item.content)
        item_words = _words(item.content)
        merged_into_existing = False
        for idx, existing in enumerate(kept):
            if existing.source_type != item.source_type:
                continue
            if not _tenant_matches(existing, item):
                continue
            if _jaccard_overlap(shingle_cache[idx], item_shingles) < overlap_threshold:
                continue
            if (
                _numbers(existing.content) != item_numbers
                or _polarity(existing.content) != item_polarity
                or not _adds_nothing(item_words, _words(existing.content))
            ):
                # High text overlap but possibly a conflicting or updated fact.
                # Never merge; let freshness/authority ordering decide which
                # one to prioritize.
                continue
            kept[idx] = existing.model_copy(
                update={"merged_from": [*existing.merged_from, item.source_id]}
            )
            seen_ids.add(identity)
            near_removed += 1
            merged_into_existing = True
            break
        if merged_into_existing:
            continue

        seen_ids.add(identity)
        seen_hashes.setdefault(chash, []).append(len(kept))
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
