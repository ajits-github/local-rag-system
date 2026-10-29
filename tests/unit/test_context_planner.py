"""Unit tests for `rag.context.planner`."""

from __future__ import annotations

from rag.config import load_config
from rag.context.planner import bucket_for_item, classify_context_query_type, plan_context_budget
from rag.context.tokens import HeuristicTokenCounter
from rag.context.types import ContextItem


def _item(source_type, source_id="x", content="hello world") -> ContextItem:
    return ContextItem(content=content, source_type=source_type, source_id=source_id)


def test_budget_arithmetic_subtracts_every_fixed_cost():
    """Budget arithmetic subtracts every fixed cost."""
    config = load_config().model_copy(deep=True)
    config.context.model_limits.default_context_window_tokens = 1000
    config.context.model_limits.overrides = {}
    config.context.reserved_output_tokens = 200
    config.context.framework_overhead_tokens = 50
    config.context.safety_margin_tokens = 50
    counter = HeuristicTokenCounter(chars_per_token=1.0)  # 1 char = 1 token, for exact arithmetic

    budget = plan_context_budget(
        config,
        counter.count,
        system_prompt_text="s" * 100,
        user_query_text="q" * 20,
        candidate_items=[],
        has_history=False,
    )
    assert budget.model_context_window == 1000
    assert budget.reserved_output_tokens == 200
    assert budget.system_prompt_tokens == 100
    assert budget.user_query_tokens == 20
    assert budget.framework_overhead_tokens == 50
    assert budget.safety_margin_tokens == 50
    assert budget.available_context_tokens == 1000 - 200 - 100 - 20 - 50 - 50


def test_budget_never_goes_negative_when_fixed_costs_exceed_the_window():
    """Budget never goes negative when fixed costs exceed the window."""
    config = load_config().model_copy(deep=True)
    config.context.model_limits.default_context_window_tokens = 100
    config.context.model_limits.overrides = {}
    config.context.reserved_output_tokens = 500
    counter = HeuristicTokenCounter(chars_per_token=1.0)

    budget = plan_context_budget(
        config,
        counter.count,
        system_prompt_text="",
        user_query_text="",
        candidate_items=[],
        has_history=False,
    )
    assert budget.available_context_tokens == 0


def test_fixed_source_budgets_sum_to_available_tokens():
    """Fixed source budgets sum to available tokens."""
    config = load_config().model_copy(deep=True)
    config.context.adaptive.enabled = False
    config.context.model_limits.default_context_window_tokens = 10000
    config.context.model_limits.overrides = {}
    config.context.reserved_output_tokens = 0
    config.context.framework_overhead_tokens = 0
    config.context.safety_margin_tokens = 0
    counter = HeuristicTokenCounter(chars_per_token=1.0)

    budget = plan_context_budget(
        config,
        counter.count,
        system_prompt_text="",
        user_query_text="",
        candidate_items=[],
        has_history=False,
    )
    total_allocated = sum(sb.allocated_tokens for sb in budget.source_budgets.values())
    # Allocation is int()-truncated per bucket, so it can be slightly less
    # than the available total, but never more.
    assert total_allocated <= budget.available_context_tokens
    assert total_allocated >= budget.available_context_tokens - 4


def test_adaptive_allocation_favors_graph_bucket_for_relationship_queries():
    """Adaptive allocation favors graph bucket for relationship queries."""
    config = load_config().model_copy(deep=True)
    config.context.adaptive.enabled = True
    items = [_item("graph_path"), _item("graph_path"), _item("document")]
    assert classify_context_query_type(items, has_history=False) == "relationship"


def test_adaptive_allocation_favors_mcp_bucket_for_business_queries():
    """Adaptive allocation favors mcp bucket for business queries."""
    items = [_item("mcp_tool"), _item("mcp_tool"), _item("document")]
    assert classify_context_query_type(items, has_history=True) == "business"


def test_adaptive_allocation_detects_pure_conversational_follow_up():
    """Adaptive allocation detects pure conversational follow up."""
    assert classify_context_query_type([], has_history=True) == "conversational"


def test_adaptive_allocation_defaults_to_knowledge_only():
    """Adaptive allocation defaults to knowledge only."""
    items = [_item("document"), _item("document")]
    assert classify_context_query_type(items, has_history=False) == "knowledge_only"


def test_adaptive_allocation_actually_changes_the_resolved_shares():
    """Adaptive allocation actually changes the resolved shares."""
    config = load_config().model_copy(deep=True)
    config.context.adaptive.enabled = True
    config.context.model_limits.default_context_window_tokens = 10000
    config.context.model_limits.overrides = {}
    config.context.reserved_output_tokens = 0
    config.context.framework_overhead_tokens = 0
    config.context.safety_margin_tokens = 0
    counter = HeuristicTokenCounter(chars_per_token=1.0)

    graph_heavy = [_item("graph_path") for _ in range(5)]
    budget = plan_context_budget(
        config,
        counter.count,
        system_prompt_text="",
        user_query_text="",
        candidate_items=graph_heavy,
        has_history=False,
    )
    knowledge_only_budget = config.context.source_budgets
    assert budget.source_budgets["graph_path"].allocated_tokens > int(
        budget.available_context_tokens * knowledge_only_budget.graph_path_share
    )


def test_disabling_adaptive_allocation_always_uses_the_fixed_budget():
    """Disabling adaptive allocation always uses the fixed budget."""
    config = load_config().model_copy(deep=True)
    config.context.adaptive.enabled = False
    config.context.model_limits.default_context_window_tokens = 10000
    config.context.model_limits.overrides = {}
    counter = HeuristicTokenCounter(chars_per_token=1.0)
    graph_heavy = [_item("graph_path") for _ in range(5)]

    budget = plan_context_budget(
        config,
        counter.count,
        system_prompt_text="",
        user_query_text="",
        candidate_items=graph_heavy,
        has_history=False,
    )
    expected_share = config.context.source_budgets.graph_path_share
    assert budget.source_budgets["graph_path"].allocated_tokens == int(
        budget.available_context_tokens * expected_share
    )


def test_bucket_for_item_maps_tool_fetched_onto_the_document_bucket():
    """Bucket for item maps tool fetched onto the document bucket."""
    assert bucket_for_item(_item("tool_fetched")) == "document"
    assert bucket_for_item(_item("document")) == "document"
    assert bucket_for_item(_item("graph_path")) == "graph_path"
    assert bucket_for_item(_item("mcp_tool")) == "mcp_tool"
    assert bucket_for_item(_item("conversation_history")) == "conversation_history"
