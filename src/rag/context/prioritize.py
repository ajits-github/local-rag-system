"""Relevance vs. authority vs. freshness prioritization, with deterministic tie-breaking.

Relevance, authority, and freshness answer three different questions,
and this module never collapses them into one fabricated confidence
score:

- **Relevance** (`ContextItem.relevance_score`): how well an item
  matches *this query*, as the originating retriever/reranker already
  scored it. Kept for observability but **not** used as a cross-source
  sort key here, because dense cosine similarity, BM25/RRF scores, and
  the fixed `1.0` constants assigned to graph/tool/tool-fetched evidence
  are not on comparable scales (`RetrievalPipeline`'s own docstring
  already documents the same dense-vs-BM25 incomparability). Within one
  source, the originating retriever's own rank order already *is*
  relevance ordering, so this module preserves it.
- **Authority** (`ContextItem.authority_score`): how trustworthy the
  *source* is, independent of query match, derived only from the
  existing `trust_level` governance field. A document with no
  `trust_level` at all is treated as neutral (`0.5`), never guessed
  finer than the three bins this codebase's own data actually encodes
  (`"authoritative"` / unset / `"untrusted"`).
- **Freshness** (`ContextItem.is_current`): whether this item's
  document is the currently-effective member of its version family
  (from the `status` field alone; see `normalize.py`'s documented
  limitation on why a true freshness-family re-resolution doesn't
  happen at this layer).

Final ordering is a stable sort: freshness tier, then authority tier,
then each item's original position (its source's own pre-existing rank
order). Ties are therefore always broken the same way for the same
input, a requirement of section 7 of the design.
"""

from __future__ import annotations

from rag.context.types import ContextItem


def authority_score_for_trust_level(trust_level: str | None) -> float:
    """Map `ChunkMetadata.trust_level` onto a fixed, documented authority score.

    Parameters
    ----------
    trust_level : str | None
        `None`, `"authoritative"`, `"untrusted"`, or any other label
        this corpus happens to use.

    Returns
    -------
    float
        `1.0` for `"authoritative"`, `0.0` for `"untrusted"`, `0.5`
        (neutral) for `None` or anything else. Never a finer-grained
        value: the governance data itself only distinguishes these
        three cases.
    """
    if trust_level is None:
        return 0.5
    normalized = trust_level.lower()
    if normalized == "authoritative":
        return 1.0
    if normalized == "untrusted":
        return 0.0
    return 0.5


def _freshness_tier(item: ContextItem) -> int:
    """`1` for a known-superseded item, `0` for current or freshness-unknown."""
    return 1 if item.is_current is False else 0


def _authority_tier(item: ContextItem) -> int:
    """`0` authoritative, `1` neutral/unknown, `2` untrusted."""
    if item.authority_score is None:
        return 1
    if item.authority_score >= 0.9:
        return 0
    if item.authority_score <= 0.1:
        return 2
    return 1


def prioritize_items(items: list[ContextItem]) -> list[ContextItem]:
    """Stable-sort `items` by freshness tier, then authority tier, then original order.

    Never reorders within a tier beyond what the input order already
    gave (each source's own rank order); never drops or adds an item.

    Parameters
    ----------
    items : list[ContextItem]
        Candidate items, in their pre-existing (per-source) rank order.

    Returns
    -------
    list[ContextItem]
        A new list, same items (as copies with `priority_rank` set),
        reordered. `priority_rank` is `0` for the highest-priority item.
    """
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (_freshness_tier(pair[1]), _authority_tier(pair[1]), pair[0]))
    return [
        item.model_copy(update={"priority_rank": rank}) for rank, (_, item) in enumerate(indexed)
    ]
