"""Unit tests for `rag.eval.run_context_eval` (the synthetic Context Engineering benchmark)."""

from __future__ import annotations

from rag.config import load_config
from rag.eval.run_context_eval import build_scenarios, evaluate_context_engineering, run_scenario


def test_build_scenarios_covers_every_required_category():
    """Build scenarios covers every required category."""
    scenarios = build_scenarios(candidate_token_target=2000)
    categories = {s.category for s in scenarios}
    expected = {
        "simple_short_context",
        "long_document",
        "many_relevant_chunks",
        "many_irrelevant_chunks",
        "duplicate_chunks",
        "graph_plus_document",
        "mcp_plus_document",
        "conflicting_evidence",
        "stale_vs_current",
        "long_conversation_history",
        "prompt_injection_evidence",
        "context_larger_than_model_window",
        "one_extremely_large_chunk",
        "multi_agent_context_case",
        "unanswerable_query",
    }
    assert expected <= categories


def test_run_scenario_never_raises_for_any_strategy():
    """Run scenario never raises for any strategy."""
    config = load_config().model_copy(deep=True)
    config.context.model_limits.overrides = {}
    config.context.model_limits.default_context_window_tokens = 2048
    config.context.reserved_output_tokens = 256
    scenarios = build_scenarios(candidate_token_target=2000)
    for scenario in scenarios:
        for strategy in ["top_n", "naive_token_first", "dedup_plus_budget", "full_pipeline"]:
            run_config = config.model_copy(deep=True)
            result = run_scenario(run_config, scenario, strategy, 2000)
            assert result.packed_token_count <= result.available_context_tokens


def test_evaluate_context_engineering_runs_the_full_sweep_without_error():
    """Evaluate context engineering runs the full sweep without error."""
    report = evaluate_context_engineering(
        model_context_window_tokens=1024,
        reserved_output_tokens=128,
        candidate_token_targets=[2000],
        strategies=["top_n", "full_pipeline"],
    )
    assert len(report.results) == 15 * 2  # 15 scenarios x 2 strategies x 1 size
    summary = report.by_strategy_summary()
    assert "top_n" in summary
    assert "full_pipeline" in summary


def test_full_pipeline_never_exceeds_budget_even_under_extreme_candidate_volume():
    """Full pipeline never exceeds budget even under extreme candidate volume."""
    report = evaluate_context_engineering(
        model_context_window_tokens=512,
        reserved_output_tokens=128,
        candidate_token_targets=[32000],
        strategies=["full_pipeline"],
    )
    for row in report.results:
        assert row.packed_token_count <= row.available_context_tokens
        assert row.context_overflow is False
        assert row.context_excess_tokens == 0


def test_top_n_baseline_genuinely_overflows_and_reports_it_honestly():
    """The top_n baseline can exceed its budget by design; the report must say so, unclamped.

    Regression for CTX-011: `context_utilization` used to be
    clamped to 1.0 in this report, making a run that overflowed its
    budget by 600% read identically to one that landed exactly at
    capacity. Under the same extreme-candidate-volume condition that
    `full_pipeline` is proven clean against above, `top_n` (which
    ignores token size entirely, by design) must show real, unclamped
    overflow.
    """
    # A larger window than the other extreme-volume tests in this file:
    # enough headroom that available_context_tokens is meaningfully
    # positive (not the degenerate all-fixed-costs-exhaust-the-window
    # case), while still far smaller than the 32000-token candidate
    # volume, so top_n (which ignores token size entirely by design)
    # has real room to overflow by a meaningful, non-zero ratio.
    report = evaluate_context_engineering(
        model_context_window_tokens=8192,
        reserved_output_tokens=512,
        candidate_token_targets=[32000],
        strategies=["top_n"],
    )
    overflowing = [
        row for row in report.results if row.context_overflow and row.available_context_tokens > 0
    ]
    assert overflowing, "expected at least one genuinely overflowing top_n row"
    for row in overflowing:
        assert row.context_excess_tokens > 0
        assert row.context_utilization > 1.0
        assert row.packed_token_count > row.available_context_tokens

    summary = report.by_strategy_summary()["top_n"]
    assert summary["overflow_run_count"] > 0
