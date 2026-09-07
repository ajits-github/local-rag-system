"""Unit tests for scripts/check_eval_gates.py.

scripts/ is a standalone CLI directory, not part of the rag package (no
__init__.py, not on pytest's configured pythonpath), so the module under
test is loaded dynamically via importlib rather than a normal import --
this keeps the fix local to this file instead of widening pytest's
pythonpath for the whole suite.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_eval_gates", REPO_ROOT / "scripts" / "check_eval_gates.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check_eval_gates = importlib.util.module_from_spec(_SPEC)
# dataclasses' @dataclass processing resolves its owning module via
# sys.modules[cls.__module__], which only exists once registered here --
# exec_module alone (without this line) raises AttributeError on the
# GateResult dataclass at import time.
sys.modules["check_eval_gates"] = check_eval_gates
_SPEC.loader.exec_module(check_eval_gates)


def _gate(id_="recall_at_5", direction="higher_is_better", baseline=0.6, blocking=True, **extra):
    gate = {
        "id": id_,
        "metric_path": "retrieval.recall@5",
        "direction": direction,
        "baseline": baseline,
    }
    gate["floor" if direction == "higher_is_better" else "ceiling"] = extra.pop("limit", 0.5)
    gate["blocking"] = blocking
    gate.update(extra)
    return gate


def _report(recall_at_5=0.625, **extra):
    report = {"retrieval": {"recall@5": recall_at_5}}
    report.update(extra)
    return report


class TestEvaluateGate:
    def test_higher_is_better_gate_passes_at_or_above_floor(self):
        """A value at or above its floor passes."""
        result = check_eval_gates.evaluate_gate(_gate(limit=0.5), _report(recall_at_5=0.625))
        assert result.status == "PASS"
        assert result.value == 0.625

    def test_higher_is_better_gate_fails_below_floor(self):
        """A value below its floor fails -- the core quality-regression case."""
        result = check_eval_gates.evaluate_gate(_gate(limit=0.5), _report(recall_at_5=0.25))
        assert result.status == "FAIL"

    def test_lower_is_better_gate_fails_above_ceiling(self):
        """A lower_is_better metric above its ceiling fails."""
        gate = {
            "id": "leakage",
            "metric_path": "safety.leak.rate",
            "direction": "lower_is_better",
            "baseline": 0.0,
            "ceiling": 0.0,
        }
        report = {"safety": {"leak": {"rate": 0.1}}}
        result = check_eval_gates.evaluate_gate(gate, report)
        assert result.status == "FAIL"

    def test_zero_tolerance_security_metric_passes_at_exactly_zero(self):
        """A zero-tolerance security gate (ceiling=0.0) passes when the metric is exactly 0."""
        gate = {
            "id": "cross_tenant_leakage",
            "metric_path": "safety.cross_tenant_leakage_rate.rate",
            "direction": "lower_is_better",
            "baseline": 0.0,
            "ceiling": 0.0,
        }
        report = {"safety": {"cross_tenant_leakage_rate": {"rate": 0.0}}}
        result = check_eval_gates.evaluate_gate(gate, report)
        assert result.status == "PASS"

    def test_missing_metric_reports_missing_not_a_crash(self):
        """A metric_path absent from the report is reported as MISSING, never raises."""
        result = check_eval_gates.evaluate_gate(_gate(), _report())
        result2 = check_eval_gates.evaluate_gate(
            {**_gate(), "metric_path": "retrieval.does_not_exist"}, _report()
        )
        assert result.status == "PASS"  # sanity: the "does exist" case is unaffected
        assert result2.status == "MISSING"
        assert result2.value is None

    def test_malformed_metric_value_reports_malformed_not_a_crash(self):
        """A non-numeric metric value is reported as MALFORMED, never raises."""
        gate = _gate()
        report = {"retrieval": {"recall@5": "not-a-number"}}
        result = check_eval_gates.evaluate_gate(gate, report)
        assert result.status == "MALFORMED"

    def test_bool_metric_value_is_malformed_not_silently_scored(self):
        """Bool is a subclass of int in Python; True/False must never pass as 1.0/0.0."""
        gate = _gate()
        report = {"retrieval": {"recall@5": True}}
        result = check_eval_gates.evaluate_gate(gate, report)
        assert result.status == "MALFORMED"

    def test_tolerance_absorbs_tiny_floating_point_noise_at_the_boundary(self):
        """A value a hair below the floor, within epsilon, still passes."""
        gate = _gate(limit=0.5)
        result = check_eval_gates.evaluate_gate(gate, _report(recall_at_5=0.5 - 1e-12))
        assert result.status == "PASS"

    def test_value_meaningfully_below_floor_still_fails_despite_tolerance(self):
        """Epsilon tolerance never masks a real regression."""
        gate = _gate(limit=0.5)
        result = check_eval_gates.evaluate_gate(gate, _report(recall_at_5=0.5 - 1e-3))
        assert result.status == "FAIL"

    def test_non_blocking_gate_can_fail_without_being_reported_as_blocking(self):
        """A non-blocking gate still reports FAIL; its `blocking` flag carries through."""
        gate = _gate(limit=0.9, blocking=False)
        result = check_eval_gates.evaluate_gate(gate, _report(recall_at_5=0.1))
        assert result.status == "FAIL"
        assert result.blocking is False


class TestCheckReport:
    def test_all_gates_passing_does_not_fail_ci(self):
        """The all-pass case: check_report reports success and no failure."""
        gate_groups = {"retrieval_gates": {"gates": [_gate(limit=0.5)]}}
        results, should_fail = check_eval_gates.check_report(
            _report(recall_at_5=0.625), gate_groups
        )
        assert should_fail is False
        assert all(r.passed for r in results)

    def test_one_blocking_quality_gate_failing_fails_ci(self):
        """One failing blocking quality gate is enough to fail the whole report."""
        gate_groups = {"retrieval_gates": {"gates": [_gate(limit=0.9)]}}
        _, should_fail = check_eval_gates.check_report(_report(recall_at_5=0.625), gate_groups)
        assert should_fail is True

    def test_zero_tolerance_security_gate_failing_fails_ci_even_if_quality_gates_pass(self):
        """A failing zero-tolerance security gate fails CI regardless of quality gates."""
        quality_gate = _gate(limit=0.1)  # trivially passes
        security_gate = {
            "id": "leakage",
            "metric_path": "safety.leak.rate",
            "direction": "lower_is_better",
            "baseline": 0.0,
            "ceiling": 0.0,
        }
        gate_groups = {
            "retrieval_gates": {"gates": [quality_gate]},
            "security_gates": {"gates": [security_gate]},
        }
        report = _report(recall_at_5=0.9, safety={"leak": {"rate": 0.05}})
        _, should_fail = check_eval_gates.check_report(report, gate_groups)
        assert should_fail is True

    def test_missing_required_metric_fails_ci(self):
        """A blocking gate whose metric is absent from the report fails CI (fail-closed)."""
        gate_groups = {
            "retrieval_gates": {"gates": [{**_gate(), "metric_path": "retrieval.missing_metric"}]}
        }
        _, should_fail = check_eval_gates.check_report(_report(), gate_groups)
        assert should_fail is True

    def test_non_blocking_failure_never_fails_ci_alone(self):
        """A lone non-blocking failure is reported but never fails the overall check."""
        gate_groups = {"retrieval_gates": {"gates": [_gate(limit=0.99, blocking=False)]}}
        _, should_fail = check_eval_gates.check_report(_report(recall_at_5=0.625), gate_groups)
        assert should_fail is False


class TestLoadGateDefinitions:
    def test_loads_a_well_formed_gates_file(self, tmp_path: Path):
        """A well-formed gates YAML file loads into the expected group/gate structure."""
        gates_path = tmp_path / "gates.yaml"
        gates_path.write_text(
            "retrieval_gates:\n"
            "  gates:\n"
            "    - id: recall_at_5\n"
            "      metric_path: retrieval.recall@5\n"
            "      direction: higher_is_better\n"
            "      baseline: 0.6\n"
            "      floor: 0.5\n"
        )
        groups = check_eval_gates.load_gate_definitions(gates_path)
        assert groups["retrieval_gates"]["gates"][0]["id"] == "recall_at_5"

    def test_malformed_gates_file_missing_required_field_raises(self, tmp_path: Path):
        """A gate entry missing a required field (metric_path/baseline) raises ValueError."""
        gates_path = tmp_path / "gates.yaml"
        gates_path.write_text(
            "retrieval_gates:\n  gates:\n    - id: recall_at_5\n      direction: higher_is_better\n"
        )
        with pytest.raises(ValueError, match="missing required field"):
            check_eval_gates.load_gate_definitions(gates_path)

    def test_gates_file_missing_floor_for_higher_is_better_raises(self, tmp_path: Path):
        """A higher_is_better gate with no `floor` value raises ValueError."""
        gates_path = tmp_path / "gates.yaml"
        gates_path.write_text(
            "retrieval_gates:\n"
            "  gates:\n"
            "    - id: recall_at_5\n"
            "      metric_path: retrieval.recall@5\n"
            "      direction: higher_is_better\n"
            "      baseline: 0.6\n"
        )
        with pytest.raises(ValueError, match="floor"):
            check_eval_gates.load_gate_definitions(gates_path)

    def test_gates_file_with_no_gate_groups_raises(self, tmp_path: Path):
        """A YAML file with no `*_gates` key at all raises ValueError."""
        gates_path = tmp_path / "gates.yaml"
        gates_path.write_text("version: 1\n")
        with pytest.raises(ValueError, match="no '\\*_gates' group"):
            check_eval_gates.load_gate_definitions(gates_path)


class TestFailingExampleQuestions:
    def test_reports_up_to_max_examples_for_a_recall_gate(self):
        """Failing questions for a recall gate are recomputed and bounded by max_examples."""
        report = {
            "per_example": [
                {"question": "Q1", "relevant_documents": ["a.md"], "retrieved_sources": ["b.md"]},
                {"question": "Q2", "relevant_documents": ["c.md"], "retrieved_sources": ["c.md"]},
                {"question": "Q3", "relevant_documents": ["d.md"], "retrieved_sources": ["e.md"]},
            ]
        }
        failing = check_eval_gates.failing_example_questions(report, "recall_at_5", max_examples=1)
        assert failing == ["Q1"]

    def test_unanswerable_examples_with_no_relevant_documents_are_never_listed(self):
        """An unanswerable question (no relevant_documents) is excluded, matching run_eval.py."""
        report = {
            "per_example": [
                {"question": "Q1", "relevant_documents": [], "retrieved_sources": ["a.md"]},
            ]
        }
        failing = check_eval_gates.failing_example_questions(report, "recall_at_5", max_examples=5)
        assert failing == []

    def test_non_example_level_gate_returns_empty(self):
        """A gate id with no per-example recomputation rule (e.g. mrr) returns no questions."""
        report = {
            "per_example": [
                {"question": "Q1", "relevant_documents": ["a.md"], "retrieved_sources": []}
            ]
        }
        assert check_eval_gates.failing_example_questions(report, "mrr", max_examples=5) == []

    def test_report_without_per_example_returns_empty(self):
        """A non-verbose report (no `per_example` key) degrades gracefully to no drill-down."""
        assert check_eval_gates.failing_example_questions({}, "recall_at_5", max_examples=5) == []


class TestMainCli:
    def _write(self, tmp_path: Path, name: str, data: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    def _write_gates_yaml(self, tmp_path: Path, floor: float) -> Path:
        path = tmp_path / "gates.yaml"
        path.write_text(
            "retrieval_gates:\n"
            "  gates:\n"
            "    - id: recall_at_5\n"
            "      metric_path: retrieval.recall@5\n"
            "      direction: higher_is_better\n"
            f"      baseline: 0.625\n      floor: {floor}\n"
        )
        return path

    def test_cli_exits_zero_when_all_gates_pass(self, tmp_path: Path):
        """The CLI exits 0 and prints RESULT: PASS when every gate passes."""
        report_path = self._write(tmp_path, "report.json", _report(recall_at_5=0.625))
        gates_path = self._write_gates_yaml(tmp_path, floor=0.5)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_eval_gates.py"),
                "--report",
                str(report_path),
                "--gates",
                str(gates_path),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "RESULT: PASS" in result.stdout

    def test_cli_exits_non_zero_when_a_blocking_gate_fails(self, tmp_path: Path):
        """The CLI exits non-zero and prints RESULT: FAIL when a blocking gate fails."""
        report_path = self._write(tmp_path, "report.json", _report(recall_at_5=0.1))
        gates_path = self._write_gates_yaml(tmp_path, floor=0.5)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_eval_gates.py"),
                "--report",
                str(report_path),
                "--gates",
                str(gates_path),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 1
        assert "RESULT: FAIL" in result.stdout
        assert "FAIL" in result.stdout
