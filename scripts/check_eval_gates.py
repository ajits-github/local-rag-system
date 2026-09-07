"""Fast CI evaluation gate: PASS/FAIL a deterministic run_eval.py report.

Compares metrics in a `rag.eval.run_eval` JSON report against the explicit
thresholds in `config/eval_gates.yaml` and exits non-zero the moment any
*blocking* gate fails. Every threshold's baseline/floor/rationale lives in
that YAML file, never as a magic number in CI YAML or in this script -- see
docs/ci_eval_gates.md for the full design and how to update a baseline.

Usage
-----
    python scripts/check_eval_gates.py --report path/to/report.json

To reproduce the report this reads, see docs/ci_eval_gates.md's "Run the
gate locally" section.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.path_matching import source_matches_relevant

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES_PATH = REPO_ROOT / "config" / "eval_gates.yaml"

# Absorbs float accumulation noise only -- never widens an intentional floor/ceiling.
_EPSILON = 1e-9


@dataclass
class GateResult:
    """The outcome of checking one gate from config/eval_gates.yaml against a report."""

    id: str
    direction: str
    blocking: bool
    baseline: float
    limit: float | None
    failure_message: str
    value: float | None = None
    status: str = "FAIL"  # PASS | FAIL | MISSING | MALFORMED

    @property
    def passed(self) -> bool:
        """Whether this gate did not fail (PASS is the only passing status)."""
        return self.status == "PASS"


def _get_path(data: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted path such as ``safety.duplicate_sensitive_field_miss_rate.rate``."""
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_gate_definitions(gates_path: Path) -> dict[str, Any]:
    """Load and lightly validate the gate-threshold YAML file.

    Parameters
    ----------
    gates_path : Path
        Path to a gate-definitions YAML file (see config/eval_gates.yaml).

    Returns
    -------
    dict[str, Any]
        The parsed gate-group mapping (e.g. ``{"retrieval_gates": ..., "security_gates": ...}``).

    Raises
    ------
    ValueError
        If the file has no recognizable ``*_gates`` group, or a gate entry
        is missing a required field.
    """
    with gates_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    gate_groups = {k: v for k, v in data.items() if k.endswith("_gates")}
    if not gate_groups:
        raise ValueError(f"{gates_path} defines no '*_gates' group")

    required_fields = {"id", "metric_path", "direction", "baseline"}
    for group_name, group in gate_groups.items():
        for gate in group.get("gates", []):
            missing = required_fields - gate.keys()
            if missing:
                raise ValueError(
                    f"{gates_path}: gate {gate.get('id', '<unnamed>')!r} in "
                    f"{group_name!r} is missing required field(s): {sorted(missing)}"
                )
            if gate["direction"] not in ("higher_is_better", "lower_is_better"):
                raise ValueError(
                    f"{gates_path}: gate {gate['id']!r} has an unknown direction "
                    f"{gate['direction']!r} (expected 'higher_is_better' or 'lower_is_better')"
                )
            limit_field = "floor" if gate["direction"] == "higher_is_better" else "ceiling"
            if limit_field not in gate:
                raise ValueError(
                    f"{gates_path}: gate {gate['id']!r} is '{gate['direction']}' "
                    f"but has no '{limit_field}' value"
                )

    return gate_groups


def load_report(report_path: Path) -> dict[str, Any]:
    """Load a `rag.eval.run_eval` JSON report from disk."""
    with report_path.open(encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(gate: dict[str, Any], report: dict[str, Any]) -> GateResult:
    """Check one gate definition against one report.

    Never raises on a missing/malformed metric -- both are reported as
    their own status (`MISSING` / `MALFORMED`) so one bad report produces a
    readable failure line instead of a traceback.
    """
    direction = gate["direction"]
    blocking = gate.get("blocking", True)
    baseline = gate["baseline"]
    limit = gate.get("floor") if direction == "higher_is_better" else gate.get("ceiling")
    failure_message = gate.get("failure_message", f"{gate['id']} failed its threshold")
    base = {
        "id": gate["id"],
        "direction": direction,
        "blocking": blocking,
        "baseline": baseline,
        "limit": limit,
        "failure_message": failure_message,
    }

    raw_value = _get_path(report, gate["metric_path"])
    if raw_value is None:
        return GateResult(**base, status="MISSING")

    # bool is a subclass of int; reject it explicitly rather than silently
    # scoring True/False as 1.0/0.0.
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return GateResult(**base, status="MALFORMED")

    value = float(raw_value)
    if direction == "higher_is_better":
        passed = value >= limit - _EPSILON
    else:
        passed = value <= limit + _EPSILON

    return GateResult(**base, value=value, status="PASS" if passed else "FAIL")


def check_report(
    report: dict[str, Any], gate_groups: dict[str, Any]
) -> tuple[list[GateResult], bool]:
    """Evaluate every gate in every group against one report.

    Returns
    -------
    tuple[list[GateResult], bool]
        Every gate's result, and whether CI should fail (True if any
        *blocking* gate did not PASS).
    """
    results = [
        evaluate_gate(gate, report)
        for group in gate_groups.values()
        for gate in group.get("gates", [])
    ]
    should_fail = any(not r.passed and r.blocking for r in results)
    return results, should_fail


def format_result(result: GateResult) -> str:
    """Render one gate result as a single, human-scannable report line."""
    if result.status in ("MISSING", "MALFORMED"):
        suffix = "" if result.blocking else " (non-blocking)"
        return f"{result.status:9s} {result.id}{suffix}"

    op = ">=" if result.direction == "higher_is_better" else "<="
    suffix = "" if result.blocking else " (non-blocking)"
    return (
        f"{result.status:9s} {result.id} = {result.value:.4f} {op} "
        f"{result.limit:.4f} (baseline {result.baseline:.4f}){suffix}"
    )


def _recall_hit_at_k(example: dict[str, Any], k: int, require_all: bool) -> bool | None:
    """Recompute one gold example's own recall@k (or hit-rate@k) from `per_example` data.

    Returns None when the example has no `relevant_documents` to score
    against (an unanswerable/out-of-scope question), matching how
    `rag.eval.run_eval` itself excludes those from recall/hit-rate.
    """
    relevant = example.get("relevant_documents") or []
    if not relevant:
        return None
    retrieved_top_k = (example.get("retrieved_sources") or [])[:k]
    matched = [
        rel for rel in relevant if any(source_matches_relevant(src, rel) for src in retrieved_top_k)
    ]
    return len(matched) == len(relevant) if require_all else len(matched) > 0


# Which failing gate ids trigger a per-example drill-down, and how to recompute
# that one gate's pass/fail per example from the report's `per_example` list.
_EXAMPLE_LEVEL_GATES: dict[str, tuple[int, bool]] = {
    "recall_at_5": (5, True),
    "recall_at_10": (10, True),
    "hit_rate_at_5": (5, False),
    "hit_rate_at_10": (10, False),
}


def failing_example_questions(report: dict[str, Any], gate_id: str, max_examples: int) -> list[str]:
    """Return up to `max_examples` gold questions responsible for a failing recall/hit-rate gate.

    Empty when the report has no `per_example` detail (i.e. run_eval.py
    wasn't invoked with --verbose) or the gate isn't one of the recall/
    hit-rate family this can recompute per-example.
    """
    spec = _EXAMPLE_LEVEL_GATES.get(gate_id)
    per_example = report.get("per_example")
    if spec is None or not per_example:
        return []
    k, require_all = spec
    failing = [ex for ex in per_example if _recall_hit_at_k(ex, k, require_all) is False]
    return [ex["question"] for ex in failing[:max_examples]]


def main() -> None:
    """CLI entrypoint: check a report against gate thresholds, print a summary, exit 0/1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", required=True, type=Path, help="Path to a rag.eval.run_eval JSON report"
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=DEFAULT_GATES_PATH,
        help=f"Path to the gate thresholds YAML (default: {DEFAULT_GATES_PATH})",
    )
    parser.add_argument(
        "--max-example-failures",
        type=int,
        default=5,
        help="Bound on how many failing question strings are printed per failing gate",
    )
    args = parser.parse_args()

    gate_groups = load_gate_definitions(args.gates)
    report = load_report(args.report)
    results, should_fail = check_report(report, gate_groups)

    print("CI evaluation gate report")
    print(f"  report: {args.report}")
    print(f"  gates:  {args.gates}")
    print("-" * 72)
    for result in results:
        print(format_result(result))
        if result.status == "FAIL":
            questions = failing_example_questions(report, result.id, args.max_example_failures)
            for q in questions:
                print(f"           - {q}")
    print("-" * 72)
    print("RESULT: FAIL" if should_fail else "RESULT: PASS")
    sys.exit(1 if should_fail else 0)


if __name__ == "__main__":
    main()
