"""Compare RAGAS judge scores against manually-labeled rows.

Joins by `question_index` (not question text). Computes, per metric/label
pair, an "agreement rate": thresholding the continuous [0,1] RAGAS score
at 0.5 and comparing the resulting boolean against the human label, over
only the rows a reviewer has actually filled in. This is a simple, honest
first signal, not a validated statistical benchmark (n is small, the
threshold is uncalibrated). The generated report says so explicitly.

Metric <-> human-label pairing:
    faithfulness       <-> human_faithful
    answer_relevancy    <-> human_relevant
    answer_correctness  <-> human_correct
`human_correct_refusal` has no RAGAS counterpart (RAGAS has no refusal
metric) and is reported separately, human-label-only, over unanswerable
rows.

Usage:
    python scripts/compare_ragas_manual.py --ragas-output /tmp/ragas_report.json \
        --manual-review data/eval/manual_review/sample_docs_manual_review.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AGREEMENT_THRESHOLD = 0.5
MIN_RECOMMENDED_LABELED = 10
METRIC_LABEL_PAIRS = [
    ("faithfulness", "human_faithful"),
    ("answer_relevancy", "human_relevant"),
    ("answer_correctness", "human_correct"),
]
CAVEAT = (
    "**Caveat:** this comparison is based on a small, manually-labeled sample and a "
    "fixed, uncalibrated 0.5 agreement threshold. It is NOT sufficient evidence that "
    "the RAGAS judge is reliable -- treat agreement numbers below as a first signal "
    "only, not a validated judge-quality benchmark. Do not use this report alone to "
    "justify trusting RAGAS scores for unreviewed questions."
)


def load_ragas_scores(report_path: Path) -> dict[int, dict[str, float | None]]:
    """Load `question_index -> {metric: score}` from a ragas report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        entry["question_index"]: entry["scores"]
        for entry in report.get("ragas", {}).get("per_question", [])
    }


def load_manual_labels(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load manual-review rows that have at least one non-null human_* label."""
    human_fields = ["human_faithful", "human_correct", "human_relevant", "human_correct_refusal"]
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if any(row.get(f) is not None for f in human_fields):
            rows.append(row)
    return rows


def compute_agreement(
    ragas_scores: dict[int, dict[str, float | None]], manual_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute per-metric-pair agreement rates plus a refusal-correctness summary.

    Parameters
    ----------
    ragas_scores : dict[int, dict[str, float | None]]
        `question_index -> {metric: score}`, from `load_ragas_scores`.
    manual_rows : list[dict[str, Any]]
        Labeled rows, from `load_manual_labels`.

    Returns
    -------
    dict[str, Any]
        ``{"pairs": {metric: {"labeled_n", "agreement_rate"}},
        "refusal": {"labeled_n", "correct_rate"}}``.

    Raises
    ------
    ValueError
        If `manual_rows` is empty (nothing to compare).
    """
    if not manual_rows:
        raise ValueError("No manually-labeled rows found -- nothing to compare.")

    pairs: dict[str, Any] = {}
    for metric, human_field in METRIC_LABEL_PAIRS:
        labeled = [row for row in manual_rows if row.get(human_field) is not None]
        matches = 0
        for row in labeled:
            scores = ragas_scores.get(row["question_index"], {})
            ragas_value = scores.get(metric)
            if ragas_value is None:
                continue
            ragas_bool = ragas_value >= AGREEMENT_THRESHOLD
            if ragas_bool == bool(row[human_field]):
                matches += 1
        pairs[metric] = {
            "labeled_n": len(labeled),
            "agreement_rate": (matches / len(labeled)) if labeled else None,
        }

    refusal_rows = [
        row
        for row in manual_rows
        if row.get("unanswerable") and row.get("human_correct_refusal") is not None
    ]
    correct_refusals = sum(1 for row in refusal_rows if row["human_correct_refusal"])
    refusal = {
        "labeled_n": len(refusal_rows),
        "correct_rate": (correct_refusals / len(refusal_rows)) if refusal_rows else None,
    }

    return {"pairs": pairs, "refusal": refusal}


def render_report(agreement: dict[str, Any], num_labeled: int) -> str:
    """Render the agreement comparison as Markdown, caveat prepended.

    Parameters
    ----------
    agreement : dict[str, Any]
        Output of `compute_agreement`.
    num_labeled : int
        Total number of manually-labeled rows the comparison drew from.

    Returns
    -------
    str
        Markdown report.
    """
    lines = [CAVEAT, ""]
    if num_labeled < MIN_RECOMMENDED_LABELED:
        lines.append(
            f"**Note:** only {num_labeled} row(s) labeled, below the recommended "
            f"minimum of {MIN_RECOMMENDED_LABELED} -- agreement numbers below are "
            "especially unreliable."
        )
        lines.append("")

    lines.append("| Metric | Human label | Threshold | Labeled N | Agreement % |")
    lines.append("|---|---|---|---|---|")
    for metric, human_field in METRIC_LABEL_PAIRS:
        row = agreement["pairs"][metric]
        rate = "-" if row["agreement_rate"] is None else f"{row['agreement_rate']:.0%}"
        lines.append(
            f"| {metric} | {human_field} | >= {AGREEMENT_THRESHOLD} | {row['labeled_n']} | {rate} |"
        )

    refusal = agreement["refusal"]
    rate = "-" if refusal["correct_rate"] is None else f"{refusal['correct_rate']:.0%}"
    lines.append("")
    lines.append(
        f"Correct-refusal rate on unanswerable questions (human-labeled only, no "
        f"RAGAS counterpart): {rate} (n={refusal['labeled_n']})"
    )
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint: load both inputs, compute agreement, print + write the Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragas-output", required=True, help="Path to a ragas eval report")
    parser.add_argument(
        "--manual-review", required=True, help="Path to a filled-in manual-review JSONL"
    )
    parser.add_argument("--output", default=None, help="Output Markdown path")
    args = parser.parse_args()

    ragas_scores = load_ragas_scores(Path(args.ragas_output))
    manual_rows = load_manual_labels(Path(args.manual_review))
    agreement = compute_agreement(ragas_scores, manual_rows)
    report = render_report(agreement, len(manual_rows))

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.manual_review).with_name(
            Path(args.manual_review).stem + "_ragas_vs_manual.md"
        )
    )
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote report -> {output_path}")


if __name__ == "__main__":
    main()
