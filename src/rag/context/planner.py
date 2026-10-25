"""Context budget planner: resolves `ContextBudget` for one request.

Computes the top-level `available_context_tokens` (section 4/5 of the
design) and allocates it across evidence-source buckets, either via the
fixed `config.context.source_budgets` shares or, when
`config.context.adaptive.enabled`, via a query-type-dependent override
chosen from the evidence actually gathered this request (never a second
LLM call: see `classify_context_query_type`).
"""

from __future__ import annotations

from rag.config import AppConfig, SourceBudgetConfig
from rag.context.model_limits import resolve_model_context_limits
from rag.context.tokens import TokenCounter
from rag.context.types import ContextBudget, ContextItem, ContextSourceType, SourceBudget

#: query-type labels `classify_context_query_type` can return.
ContextQueryType = str

#: Evidence-bearing source types a budget bucket actually covers.
#: `"tool_fetched"` shares the `"document"` bucket (see
#: `SourceBudgetConfig`'s docstring); `"system"`/`"query"` items are
#: mandatory and never budgeted against a source bucket at all (see
#: `ContextBudget.available_context_tokens`'s own docstring).
_BUCKET_BY_SOURCE_TYPE: dict[ContextSourceType, str] = {
    "document": "document",
    "tool_fetched": "document",
    "graph_path": "graph_path",
    "mcp_tool": "mcp_tool",
    "conversation_history": "conversation_history",
}


def classify_context_query_type(items: list[ContextItem], has_history: bool) -> str:
    """Classify this request's dominant evidence composition, deterministically.

    Never an LLM call: a plain, explainable rule over which buckets the
    candidate evidence actually populated, computed *before* budgeting
    so the allocation can favor the bucket that matters. Ties favor
    `"knowledge_only"` (the common case) over a narrower, more specific
    label.

    Parameters
    ----------
    items : list[ContextItem]
        Every candidate item gathered for this request (pre-dedup,
        pre-packing), excluding conversation history and the
        system/query items.
    has_history : bool
        Whether any conversation-history turns were supplied at all.

    Returns
    -------
    str
        One of `"relationship"`, `"business"`, `"conversational"`, or
        `"knowledge_only"`. Matches `AdaptiveBudgetConfig.overrides`'
        keys; a caller with `adaptive.enabled=False` never calls this.
    """
    graph_count = sum(1 for i in items if i.source_type == "graph_path")
    tool_count = sum(1 for i in items if i.source_type == "mcp_tool")
    document_count = sum(1 for i in items if i.source_type in ("document", "tool_fetched"))

    if tool_count > 0 and tool_count >= document_count and tool_count >= graph_count:
        return "business"
    if graph_count > 0 and graph_count >= document_count:
        return "relationship"
    if has_history and document_count == 0 and graph_count == 0 and tool_count == 0:
        return "conversational"
    return "knowledge_only"


def _resolve_source_budget_config(
    config: AppConfig, items: list[ContextItem], has_history: bool
) -> SourceBudgetConfig:
    """Pick the fixed or adaptive `SourceBudgetConfig` to allocate with."""
    cfg = config.context
    if not cfg.adaptive.enabled:
        return cfg.source_budgets
    query_type = classify_context_query_type(items, has_history)
    return cfg.adaptive.overrides.get(query_type, cfg.source_budgets)


def plan_context_budget(
    config: AppConfig,
    token_counter: TokenCounter,
    *,
    system_prompt_text: str,
    user_query_text: str,
    candidate_items: list[ContextItem],
    has_history: bool,
) -> ContextBudget:
    """Compute the resolved `ContextBudget` for one request.

    Parameters
    ----------
    config : AppConfig
        Application configuration; `config.context` supplies every bound.
    token_counter : TokenCounter
        Used to measure `system_prompt_text`/`user_query_text`.
    system_prompt_text : str
        The rendered system prompt (never includes retrieved content).
    user_query_text : str
        The user's query text alone, not the full rendered user turn.
    candidate_items : list[ContextItem]
        Every candidate evidence item gathered this request (before
        dedup/packing), used only to classify the query type for
        adaptive allocation; never mutated.
    has_history : bool
        Whether any conversation-history turns were supplied.

    Returns
    -------
    ContextBudget
        `available_context_tokens` floored at 0, with `source_budgets`
        populated from whichever `SourceBudgetConfig` was selected.
    """
    limits = resolve_model_context_limits(config)
    reserved_output = (
        config.context.reserved_output_tokens
        if config.context.reserved_output_tokens is not None
        else limits.default_reserved_output_tokens
    )
    system_tokens = token_counter.count(system_prompt_text)
    query_tokens = token_counter.count(user_query_text)
    overhead_tokens = config.context.framework_overhead_tokens
    safety_margin = config.context.safety_margin_tokens

    available = (
        limits.context_window_tokens
        - reserved_output
        - system_tokens
        - query_tokens
        - overhead_tokens
        - safety_margin
    )
    available = max(available, 0)

    source_budget_cfg = _resolve_source_budget_config(config, candidate_items, has_history)
    caps = {
        "document": source_budget_cfg.max_document_tokens,
        "graph_path": source_budget_cfg.max_graph_tokens,
        "mcp_tool": source_budget_cfg.max_tool_tokens,
        "conversation_history": source_budget_cfg.max_history_tokens,
    }
    shares = {
        "document": source_budget_cfg.document_share,
        "graph_path": source_budget_cfg.graph_path_share,
        "mcp_tool": source_budget_cfg.mcp_tool_share,
        "conversation_history": source_budget_cfg.conversation_history_share,
    }
    source_budgets: dict[str, SourceBudget] = {}
    for bucket, share in shares.items():
        allocated = int(available * share)
        cap = caps[bucket]
        if cap is not None:
            allocated = min(allocated, cap)
        source_budgets[bucket] = SourceBudget(source_type=bucket, allocated_tokens=allocated)

    return ContextBudget(
        model_context_window=limits.context_window_tokens,
        reserved_output_tokens=reserved_output,
        system_prompt_tokens=system_tokens,
        user_query_tokens=query_tokens,
        framework_overhead_tokens=overhead_tokens,
        safety_margin_tokens=safety_margin,
        available_context_tokens=available,
        source_budgets=source_budgets,
    )


def bucket_for_item(item: ContextItem) -> str:
    """Return the budget-bucket name for `item`'s `source_type`.

    Parameters
    ----------
    item : ContextItem

    Returns
    -------
    str
        One of `"document"`, `"graph_path"`, `"mcp_tool"`,
        `"conversation_history"`. `"system"`/`"query"` items never pass
        through packing and have no bucket.
    """
    return _BUCKET_BY_SOURCE_TYPE[item.source_type]
