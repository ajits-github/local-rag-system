"""Final render-order strategies for packed context items ("lost in the middle").

Packing (`rag.context.packing`) decides *which* items survive the
budget; this module decides what *order* the survivors are rendered in,
a separate concern. Long-context literature on decoder-only LLMs
(frequently summarized as "lost in the middle") reports that models
attend most reliably to the beginning and end of a long context and
least reliably to the middle, which motivates the `"begin_end_sandwich"`
strategy below; `rag.eval.run_context_eval`'s lost-in-the-middle
experiment (see `docs/context_engineering.md`) measures whether that
holds for this project's own local model and prompt rather than
asserting it as fact up front.
"""

from __future__ import annotations

from typing import Literal

from rag.context.types import ContextItem

OrderingStrategy = Literal[
    "priority_first", "authority_first", "begin_end_sandwich", "grouped_by_source"
]


def order_priority_first(items: list[ContextItem]) -> list[ContextItem]:
    """Highest-priority item first, in `priority_rank` order (ties: input order)."""

    def _rank(item: ContextItem) -> int:
        return item.priority_rank if item.priority_rank is not None else 1_000_000

    return sorted(items, key=_rank)


def order_authority_first(items: list[ContextItem]) -> list[ContextItem]:
    """Highest `authority_score` first (ties: `priority_rank`, then input order)."""
    indexed = list(enumerate(items))
    indexed.sort(
        key=lambda pair: (
            -(pair[1].authority_score if pair[1].authority_score is not None else 0.5),
            pair[1].priority_rank if pair[1].priority_rank is not None else 1_000_000,
            pair[0],
        )
    )
    return [item for _, item in indexed]


def order_begin_end_sandwich(items: list[ContextItem]) -> list[ContextItem]:
    """Place the highest-priority items at both the beginning and the end.

    Ranks by `priority_rank` (falling back to input order), then
    alternates placing the next-best remaining item at the front or the
    back of the output, so the single best item leads, the second-best
    trails, the third-best leads (second position), and so on; the
    lowest-priority items end up adjacent to each other in the middle.

    Parameters
    ----------
    items : list[ContextItem]

    Returns
    -------
    list[ContextItem]
        Same items, reordered; never drops or duplicates one.
    """
    ranked = order_priority_first(items)
    front: list[ContextItem] = []
    back: list[ContextItem] = []
    for i, item in enumerate(ranked):
        if i % 2 == 0:
            front.append(item)
        else:
            back.insert(0, item)
    return front + back


def order_grouped_by_source(items: list[ContextItem]) -> list[ContextItem]:
    """Group by `source_type` (documents, then graph paths, then tool evidence, then history).

    Within each group, items keep their `priority_rank` order.
    """
    group_order = ["document", "tool_fetched", "graph_path", "mcp_tool", "conversation_history"]
    ranked = order_priority_first(items)
    buckets: dict[str, list[ContextItem]] = {g: [] for g in group_order}
    for item in ranked:
        buckets.setdefault(item.source_type, []).append(item)
    ordered: list[ContextItem] = []
    for group in group_order:
        ordered.extend(buckets.get(group, []))
    # Any source_type not in group_order (forward-compatibility) goes last.
    for key, bucket_items in buckets.items():
        if key not in group_order:
            ordered.extend(bucket_items)
    return ordered


_STRATEGIES = {
    "priority_first": order_priority_first,
    "authority_first": order_authority_first,
    "begin_end_sandwich": order_begin_end_sandwich,
    "grouped_by_source": order_grouped_by_source,
}


def order_for_rendering(items: list[ContextItem], strategy: OrderingStrategy) -> list[ContextItem]:
    """Dispatch to one of the ordering strategies above.

    Parameters
    ----------
    items : list[ContextItem]
        Packed (included) items.
    strategy : OrderingStrategy

    Returns
    -------
    list[ContextItem]
    """
    return _STRATEGIES[strategy](items)
