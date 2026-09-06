"""Unit tests for scripts/record_agent_experiment.py's record-flattening logic.

scripts/ isn't part of the rag package (it inserts src/ onto sys.path
itself, at import time, so its own modules can import `rag.*`); this test
file mirrors that same pattern for its own import, inserting scripts/
onto sys.path so it can import record_agent_experiment directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from record_agent_experiment import build_agent_experiment_record  # noqa: E402

from rag.config import load_config  # noqa: E402


def _agent_report(**overrides: Any) -> dict[str, Any]:
    """Build a minimal rag.eval.run_agent_eval --verbose-shaped report."""
    base: dict[str, Any] = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "corpus_lineage": {"dataset_id": "techfusion", "corpus_version": "v1"},
        "per_example": [{"steps": 3}, {"steps": 5}],
        "num_examples": 2,
        "routing_accuracy": {"rate": 1.0},
        "node_latency_breakdown_ms": {
            "classify": {"mean_ms_per_invocation": 120.0},
            "synthesize": {"mean_ms_per_invocation": 450.0},
        },
        "termination_reason_breakdown": {
            "by_reason": {
                "synthesized": {"count": 1, "rate": 0.5},
                "max_steps": {"count": 1, "rate": 0.5},
            },
            "guardrail_termination_count": 1,
            "guardrail_termination_rate": 0.5,
        },
        "tool_usage_breakdown": {
            "by_tool": {
                "search_knowledge_base": {"count": 2, "rate": 1.0, "success_rate": 1.0},
            },
        },
    }
    base.update(overrides)
    return base


def test_build_agent_experiment_record_flattens_node_latency_breakdown():
    """Per-node latency means are flattened into agent_node_<name>_latency_ms_mean fields."""
    config = load_config()

    record = build_agent_experiment_record(
        _agent_report(), None, config, "experiment_999", "unit test"
    )

    assert record["agent_node_classify_latency_ms_mean"] == 120.0
    assert record["agent_node_synthesize_latency_ms_mean"] == 450.0
    # A node absent from this report's breakdown (e.g. no decompose step
    # ran on any scored example) is None, not a KeyError.
    assert record["agent_node_decompose_latency_ms_mean"] is None
    assert record["agent_node_tool_select_latency_ms_mean"] is None
    assert record["agent_node_tool_execute_latency_ms_mean"] is None
    assert record["agent_node_evidence_sufficiency_latency_ms_mean"] is None


def test_build_agent_experiment_record_flattens_termination_reason_breakdown():
    """Per-termination-reason counts/rates and the guardrail bucket are both flattened."""
    config = load_config()

    record = build_agent_experiment_record(
        _agent_report(), None, config, "experiment_999", "unit test"
    )

    assert record["agent_termination_synthesized_count"] == 1
    assert record["agent_termination_synthesized_rate"] == 0.5
    assert record["agent_termination_max_steps_count"] == 1
    assert record["agent_termination_max_steps_rate"] == 0.5
    # A reason absent from this report (no example ever hit it) is None.
    assert record["agent_termination_insufficient_evidence_count"] is None
    assert record["agent_termination_max_retrieval_attempts_count"] is None
    assert record["agent_termination_max_tool_calls_count"] is None
    assert record["agent_guardrail_termination_count"] == 1
    assert record["agent_guardrail_termination_rate"] == 0.5


def test_build_agent_experiment_record_flattens_tool_usage_breakdown():
    """Per-tool-name counts/rates are flattened into agent_tool_usage_<name>_* fields."""
    config = load_config()

    record = build_agent_experiment_record(
        _agent_report(), None, config, "experiment_999", "unit test"
    )

    assert record["agent_tool_usage_search_knowledge_base_count"] == 2
    assert record["agent_tool_usage_search_knowledge_base_rate"] == 1.0
    # A tool never dispatched in this report is None, not a KeyError.
    assert record["agent_tool_usage_get_document_count"] is None
    assert record["agent_tool_usage_update_case_status_count"] is None


def test_build_agent_experiment_record_handles_missing_breakdown_sections_gracefully():
    """An older report predating this milestone (no breakdown sections at all) never raises.

    Every new field simply reports None -- the same graceful-degradation
    pattern every other additive field in this record already follows.
    """
    config = load_config()
    older_report: dict[str, Any] = {
        "generated_at": "2025-01-01T00:00:00+00:00",
        "corpus_lineage": {},
        "per_example": [],
        "num_examples": 0,
    }

    record = build_agent_experiment_record(
        older_report, None, config, "experiment_998", "older report"
    )

    assert record["agent_node_classify_latency_ms_mean"] is None
    assert record["agent_node_synthesize_latency_ms_mean"] is None
    assert record["agent_termination_synthesized_rate"] is None
    assert record["agent_guardrail_termination_count"] is None
    assert record["agent_guardrail_termination_rate"] is None
    assert record["agent_tool_usage_search_knowledge_base_count"] is None
