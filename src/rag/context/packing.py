"""Token-aware context packing: three comparable strategies, never a blind truncate.

`pack_items` never takes "top-N and concatenate" as its real algorithm;
that mode (`"top_n"`) exists only so `rag.eval.run_context_eval`/
`scripts/benchmark_context_engineering.py` can measure it as baseline
**A** against baseline **B** (`"naive_token_first"`) and the real
algorithm **C** (`"priority_aware"`), per section 9/25 of the design.

System and query text are never `ContextItem`s at all and never enter
this module: `rag.context.planner.plan_context_budget` subtracts their
measured token cost from `available_context_tokens` *before* any bucket
budget is computed, so mandatory, trusted text can never be dropped by
a packing decision here (section 30: "System instructions and security
constraints must never be dropped to make room for retrieved
evidence.").
"""

from __future__ import annotations

from collections.abc import Callable

from rag.context.compression import compress_item
from rag.context.planner import bucket_for_item
from rag.context.prioritize import prioritize_items
from rag.context.types import ContextBudget, ContextItem, PackingDecision, PackingTrace


def pack_top_n(
    items: list[ContextItem], *, n_per_bucket: dict[str, int]
) -> tuple[list[ContextItem], PackingTrace]:
    """Baseline **A**: take the first `n_per_bucket[bucket]` items per bucket, no budgeting at all.

    Reproduces the pre-Context-Engineering behavior
    (`RetrievalPipeline.retrieve`'s fixed `generation_context_top_n`
    cutoff) for comparison. Ignores token size entirely; a single huge
    item is included whole, uncompressed, even if it alone would
    overflow the model's context window downstream.

    Parameters
    ----------
    items : list[ContextItem]
        Candidates, in their pre-existing per-bucket rank order.
    n_per_bucket : dict[str, int]
        Maximum items to keep per bucket (e.g.
        `{"document": 3, "graph_path": 3, "mcp_tool": 3,
        "conversation_history": 3}`).

    Returns
    -------
    tuple[list[ContextItem], PackingTrace]
    """
    counts: dict[str, int] = {}
    decisions: list[PackingDecision] = []
    kept: list[ContextItem] = []
    for item in items:
        bucket = bucket_for_item(item)
        counts.setdefault(bucket, 0)
        limit = n_per_bucket.get(bucket, 0)
        included = counts[bucket] < limit
        if included:
            counts[bucket] += 1
            kept.append(item.model_copy(update={"included": True}))
        decisions.append(
            PackingDecision(
                source_id=item.source_id,
                source_type=item.source_type,
                token_count=item.token_count,
                included=included,
                reason="included" if included else "top_n_limit_reached",
            )
        )
    trace = PackingTrace(
        strategy="top_n",
        decisions=decisions,
        dropped_item_count=sum(1 for d in decisions if not d.included),
    )
    return kept, trace


def pack_naive_token_first(
    items: list[ContextItem], *, global_budget_tokens: int
) -> tuple[list[ContextItem], PackingTrace]:
    """Baseline **B**: greedily include items in rank order until the global budget is spent.

    Token-aware (never overflows the budget), but ignores authority/
    freshness/source-budget allocation entirely: one source can still
    consume the whole budget if it happens to rank first. No
    compression; an oversized single item is simply skipped whole.

    Parameters
    ----------
    items : list[ContextItem]
        Candidates, in their pre-existing per-bucket rank order
        (buckets are interleaved in whatever order the caller handed
        them in; this strategy deliberately does not reorder by bucket).
    global_budget_tokens : int
        Total token budget across every bucket combined.

    Returns
    -------
    tuple[list[ContextItem], PackingTrace]
    """
    remaining = global_budget_tokens
    decisions: list[PackingDecision] = []
    kept: list[ContextItem] = []
    for item in items:
        fits = item.token_count <= remaining
        if fits:
            remaining -= item.token_count
            kept.append(item.model_copy(update={"included": True}))
        decisions.append(
            PackingDecision(
                source_id=item.source_id,
                source_type=item.source_type,
                token_count=item.token_count,
                included=fits,
                reason="included" if fits else "global_budget_exhausted",
            )
        )
    trace = PackingTrace(
        strategy="naive_token_first",
        decisions=decisions,
        dropped_item_count=sum(1 for d in decisions if not d.included),
    )
    return kept, trace


def pack_priority_aware(
    items: list[ContextItem],
    *,
    budget: ContextBudget,
    count_tokens: Callable[[str], int],
    query: str,
    compression_enabled: bool,
    compression_threshold_tokens: int,
    compression_strategy: str,
    compression_max_sentences: int,
) -> tuple[list[ContextItem], PackingTrace]:
    """Strategy **C**: the real algorithm (prioritize, respect per-bucket budgets, compress, pack).

    Algorithm (section 9 of the design):

    1. Rank eligible items via `rag.context.prioritize.prioritize_items`
       (freshness tier, then authority tier, then original rank).
    2. Walk the ranked list once. For each item, compute its token
       cost against *both* its bucket's remaining budget and the
       global remaining budget (a bucket budget can never be exceeded
       even if the global budget has room, and vice versa).
    3. If it fits both as-is, include it whole (preserving the complete
       evidence unit, never splitting it to "almost fit").
    4. If it doesn't fit, but the remaining headroom is still at least
       `compression_threshold_tokens` (a floor on how small a target is
       worth compressing into at all; compressing a 300-token item into
       a 2-token sliver would be meaningless), attempt compression
       toward that headroom. Include the compressed result if it then
       fits; this is the one case an item is partially included rather
       than whole-or-nothing. Deliberately *not* gated on the item's own
       original size: a 150-token item that doesn't fit a 60-token
       headroom is just as valid a compression candidate as a 2000-token
       one, see `ISSUES.md`'s context-engineering entry for the
       regression test that caught the original, wrong "only compress
       items already over a fixed global size" gating.
    5. Otherwise, exclude it and record why (`source_budget_exhausted`
       or `global_budget_exhausted`), continuing to smaller candidates
       rather than stopping the whole walk (a later, smaller item may
       still fit after a larger one was skipped).
    6. Never spend past `budget.available_context_tokens`, which already
       has the reserved-output/system/query/overhead/safety-margin
       tokens subtracted, so generation's own budget is never touched.

    Parameters
    ----------
    items : list[ContextItem]
        Deduplicated candidates (dedup must run before this).
    budget : ContextBudget
        The resolved budget, with per-bucket allocations.
    count_tokens : Callable[[str], int]
        Token counter, for sizing compressed candidates.
    query : str
        Current query text, for extractive compression scoring.
    compression_enabled : bool
        When `False`, step 4 never runs; an oversized item is excluded
        outright instead (same as `pack_naive_token_first`'s behavior
        for that one item).
    compression_threshold_tokens, compression_strategy, compression_max_sentences
        See `rag.context.compression.compress_item`.

    Returns
    -------
    tuple[list[ContextItem], PackingTrace]
    """
    ranked = prioritize_items(items)
    bucket_remaining = {name: sb.allocated_tokens for name, sb in budget.source_budgets.items()}
    global_remaining = budget.available_context_tokens
    decisions: list[PackingDecision] = []
    kept: list[ContextItem] = []
    truncation_count = 0

    for item in ranked:
        bucket = bucket_for_item(item)
        bucket_budget = bucket_remaining.get(bucket, 0)
        headroom = min(bucket_budget, global_remaining)

        if item.token_count <= headroom:
            kept.append(item.model_copy(update={"included": True}))
            bucket_remaining[bucket] = bucket_budget - item.token_count
            global_remaining -= item.token_count
            decisions.append(
                PackingDecision(
                    source_id=item.source_id,
                    source_type=item.source_type,
                    token_count=item.token_count,
                    included=True,
                    reason="included",
                )
            )
            continue

        if compression_enabled and headroom >= compression_threshold_tokens:
            compressed = compress_item(
                item,
                target_tokens=headroom,
                count_tokens=count_tokens,
                query=query,
                strategy=compression_strategy,
                max_sentences=compression_max_sentences,
            )
            if compressed.compressed and compressed.token_count <= headroom:
                kept.append(compressed.model_copy(update={"included": True}))
                bucket_remaining[bucket] = bucket_budget - compressed.token_count
                global_remaining -= compressed.token_count
                truncation_count += 1
                decisions.append(
                    PackingDecision(
                        source_id=item.source_id,
                        source_type=item.source_type,
                        token_count=compressed.token_count,
                        included=True,
                        reason="included_compressed",
                    )
                )
                continue

        reason = (
            "source_budget_exhausted"
            if item.token_count > bucket_budget
            else (
                "global_budget_exhausted"
                if item.token_count > global_remaining
                else "oversized_after_compression"
            )
        )
        decisions.append(
            PackingDecision(
                source_id=item.source_id,
                source_type=item.source_type,
                token_count=item.token_count,
                included=False,
                reason=reason,
            )
        )

    trace = PackingTrace(
        strategy="priority_aware",
        decisions=decisions,
        truncation_count=truncation_count,
        dropped_item_count=sum(1 for d in decisions if not d.included),
    )
    return kept, trace
