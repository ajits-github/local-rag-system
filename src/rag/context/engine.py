"""`ContextEngine`: the single, explicit context-assembly abstraction.

Replaces ad-hoc, per-caller context construction
(`rag.retrieval.pipeline.build_context`, called directly by
`RetrievalPipeline.answer`, `GraphRAGPipeline`, and `agent/graph.py`'s
`_synthesize` node) with one pipeline every caller shares:

    normalize -> dedup -> plan budget -> pack -> order -> render

`build()` is the only public entry point. It never raises for an
ordinary request: a `TokenCounter`/compression/summarization failure is
caught, logged, and degrades to the safest available fallback (see
`_safe_count`), since a context-assembly bug must never turn into a
500 the way a broken Prometheus metric must not (the same
defensiveness precedent `rag.observability.metrics`/`tracing` already
establish). The one thing that is never silently swallowed is dropping
*evidence* to fit the budget; that is reported through `PackedContext.
truncated`/`packing_trace`, not hidden.
"""

from __future__ import annotations

import logging
import time

from rag.config import AppConfig
from rag.context.caching import TokenCountCache
from rag.context.dedup import deduplicate_items
from rag.context.history import select_history_items
from rag.context.normalize import normalize_conversation_turn, normalize_search_results
from rag.context.ordering import order_for_rendering
from rag.context.packing import pack_naive_token_first, pack_priority_aware, pack_top_n
from rag.context.planner import plan_context_budget
from rag.context.tokens import TokenCounter, build_token_counter
from rag.context.types import (
    CompressionStats,
    ContextItem,
    ConversationTurn,
    DeduplicationStats,
    PackedContext,
)
from rag.observability import metrics as observability_metrics
from rag.observability import tracing
from rag.schemas import SearchResult

logger = logging.getLogger(__name__)


def render_context(items: list[ContextItem]) -> str:
    """Render ordered, packed items as a `"[Source N: ...]"`-labeled text block.

    Byte-for-byte compatible labeling convention with
    `rag.retrieval.pipeline.build_context`, so a downstream
    `PromptTemplate.render(context=...)` call needs no change at all to
    accept either function's output.

    Parameters
    ----------
    items : list[ContextItem]
        Already ordered for rendering (see `rag.context.ordering`).

    Returns
    -------
    str
    """
    labeled = [f"[Source {i}: {item.citation}]\n{item.content}" for i, item in enumerate(items, 1)]
    return "\n\n---\n\n".join(labeled)


def context_engineering_summary(packed: PackedContext) -> dict:
    """Render a small, API-response-safe summary of one `PackedContext`.

    Deliberately omits `included_items`' raw content (already present,
    redacted, in the caller's own `sources` list) and the full
    per-decision `packing_trace` (available via tracing/metrics for
    debugging, not meant for every API response payload).

    Parameters
    ----------
    packed : PackedContext

    Returns
    -------
    dict
        JSON-serializable summary: candidate/packed token counts,
        utilization, reduction, dedup/compression counts, truncation
        flag, and the packing strategy used.
    """
    return {
        "packing_strategy": packed.packing_trace.strategy,
        "ordering_strategy": None,  # set by the caller if it wants to echo config
        "candidate_token_count": packed.candidate_token_count,
        "packed_token_count": packed.packed_token_count,
        "available_context_tokens": packed.budget.available_context_tokens,
        "context_utilization": round(packed.context_utilization, 4),
        "context_reduction": round(packed.context_reduction, 4),
        "items_before_dedup": packed.dedup_stats.candidates_before,
        "items_after_dedup": packed.dedup_stats.items_after,
        "items_compressed": packed.compression_stats.items_compressed,
        "compression_ratio": round(packed.compression_stats.compression_ratio, 4),
        "dropped_item_count": packed.packing_trace.dropped_item_count,
        "truncated": packed.truncated,
        "build_latency_ms": round(packed.build_latency_ms, 2),
    }


class ContextEngine:
    """Assembles a token-budget-aware, deduplicated, prioritized prompt context.

    One instance is cheap to construct (no I/O); `RetrievalPipeline`/
    `GraphRAGPipeline`/`agent/graph.py` each hold one, built once from
    `config.context`, exactly like every other injected singleton in
    this codebase's DI pattern.
    """

    def __init__(self, config: AppConfig, token_counter: TokenCounter | None = None) -> None:
        """Construct the engine.

        Parameters
        ----------
        config : AppConfig
            Application configuration; `config.context` supplies every
            tunable.
        token_counter : TokenCounter | None, optional
            Built via `build_token_counter(config)` if omitted.
        """
        self._config = config
        self._token_counter = token_counter or build_token_counter(config)
        self._cache = (
            TokenCountCache(config.context.cache.max_entries)
            if (config.context.cache.enabled)
            else None
        )

    def _count(self, tenant_id: str | None, text: str) -> int:
        """Count tokens in `text`, via the cache when enabled.

        Defensive: a counting failure (e.g. a broken optional
        `tiktoken` install mid-process) falls back to a plain
        `len(text) // 4` estimate rather than raising, since a token
        count is advisory input to a budgeting decision, not something
        worth crashing a request over.
        """
        try:
            if self._cache is not None:
                return self._cache.get_or_count(tenant_id, text, self._token_counter)
            return self._token_counter.count(text)
        except Exception:
            logger.warning(
                "Token counting failed; falling back to a char/4 estimate", exc_info=True
            )
            return max(len(text) // 4, 0)

    def build(
        self,
        *,
        query: str,
        system_prompt_text: str,
        search_results: list[SearchResult],
        history_turns: list[ConversationTurn] | None = None,
        tenant_id: str | None = None,
    ) -> PackedContext:
        """Run the full context-assembly pipeline for one request.

        Parameters
        ----------
        query : str
            The current query/question text.
        system_prompt_text : str
            The already-rendered system instruction text. Never mixed
            into any `ContextItem`; measured only to reserve its token
            cost in the budget.
        search_results : list[SearchResult]
            Already-authorized, already-redacted evidence (document,
            graph, MCP, or tool-fetched), in whatever rank order the
            caller gathered it. Never re-authorized or re-redacted here
            (see the module/package docstrings: this subsystem consumes
            already-safe evidence, it does not produce safety itself).
        history_turns : list[ConversationTurn] | None, optional
            Caller-supplied prior conversation turns, oldest first.
        tenant_id : str | None, optional
            For the token-count cache key only (see
            `rag.context.caching`); never used for authorization here.

        Returns
        -------
        PackedContext
        """
        t0 = time.perf_counter()
        cfg = self._config.context

        with tracing.start_span("context_normalize"):
            evidence_items = normalize_search_results(search_results)
            for item in evidence_items:
                item.token_count = self._count(tenant_id, item.content)

            summary_target = max(cfg.safety_margin_tokens, 64)
            history_items = select_history_items(
                history_turns or [],
                strategy=cfg.history.strategy,
                max_turns=cfg.history.max_turns,
                summarize_min_turns_to_trigger=cfg.history.summarize_min_turns_to_trigger,
                query=query,
                count_tokens=lambda t: self._count(tenant_id, t),
                summary_target_tokens=summary_target,
                normalize_fn=normalize_conversation_turn,
            )
            for item in history_items:
                if item.token_count == 0:
                    item.token_count = self._count(tenant_id, item.content)

            all_candidates = evidence_items + history_items
            candidate_token_count = sum(i.token_count for i in all_candidates)

        with tracing.start_span("context_deduplication") as dedup_span:
            if cfg.deduplication.enabled and all_candidates:
                deduped, dedup_stats = deduplicate_items(
                    all_candidates, overlap_threshold=cfg.deduplication.overlap_threshold
                )
            else:
                deduped = all_candidates
                dedup_stats = DeduplicationStats(
                    candidates_before=len(all_candidates),
                    items_after=len(all_candidates),
                    tokens_saved=0,
                )
            tracing.set_attributes(
                dedup_span,
                {
                    "candidates_before": dedup_stats.candidates_before,
                    "items_after": dedup_stats.items_after,
                },
            )

        budget = plan_context_budget(
            self._config,
            self._token_counter,
            system_prompt_text=system_prompt_text,
            user_query_text=query,
            candidate_items=deduped,
            has_history=bool(history_turns),
        )

        with tracing.start_span("context_packing") as pack_span:
            strategy = cfg.packing_strategy
            if strategy == "top_n":
                n_per_bucket = {
                    "document": self._config.retrieval.generation_context_top_n,
                    "graph_path": 3,
                    "mcp_tool": 3,
                    "conversation_history": cfg.history.max_turns,
                }
                kept, trace = pack_top_n(deduped, n_per_bucket=n_per_bucket)
            elif strategy == "naive_token_first":
                kept, trace = pack_naive_token_first(
                    deduped, global_budget_tokens=budget.available_context_tokens
                )
            else:
                kept, trace = pack_priority_aware(
                    deduped,
                    budget=budget,
                    count_tokens=lambda t: self._count(tenant_id, t),
                    query=query,
                    compression_enabled=cfg.compression.enabled,
                    compression_threshold_tokens=cfg.compression.threshold_tokens,
                    compression_strategy=cfg.compression.strategy,
                    compression_max_sentences=cfg.compression.max_sentences,
                )
            tracing.set_attributes(
                pack_span,
                {
                    "dropped_item_count": trace.dropped_item_count,
                    "truncation_count": trace.truncation_count,
                },
            )

        ordered = order_for_rendering(kept, strategy=cfg.ordering_strategy)
        rendered = render_context(ordered)
        packed_token_count = sum(i.token_count for i in ordered)

        tokens_before_compression = sum(
            (i.original_token_count if i.original_token_count is not None else i.token_count)
            for i in kept
        )
        compressed_count = sum(1 for i in kept if i.compressed)
        compression_stats = CompressionStats(
            items_compressed=compressed_count,
            tokens_before=tokens_before_compression,
            tokens_after=packed_token_count,
        )

        available = budget.available_context_tokens
        utilization = (packed_token_count / available) if available > 0 else 0.0
        reduction = (
            1.0 - (packed_token_count / candidate_token_count) if candidate_token_count > 0 else 0.0
        )
        truncated = trace.dropped_item_count > 0

        build_latency_ms = (time.perf_counter() - t0) * 1000

        dropped_by_reason: dict[str, int] = {}
        for decision in trace.decisions:
            if not decision.included:
                dropped_by_reason[decision.reason] = dropped_by_reason.get(decision.reason, 0) + 1
        observability_metrics.observe_context_build(
            candidate_tokens=candidate_token_count,
            packed_tokens=packed_token_count,
            build_seconds=build_latency_ms / 1000,
            strategy=strategy,
            utilization=min(utilization, 1.0),
            dropped_by_reason=dropped_by_reason,
            dedup_by_kind={
                "exact": dedup_stats.exact_duplicates_removed,
                "near": dedup_stats.near_duplicates_removed,
            },
            compressed_item_count=compressed_count,
            overflow_prevented=truncated,
        )

        return PackedContext(
            rendered_context=rendered,
            included_items=ordered,
            budget=budget,
            dedup_stats=dedup_stats,
            compression_stats=compression_stats,
            packing_trace=trace,
            context_utilization=utilization,
            context_reduction=max(reduction, 0.0),
            candidate_token_count=candidate_token_count,
            packed_token_count=packed_token_count,
            build_latency_ms=build_latency_ms,
            truncated=truncated,
        )
