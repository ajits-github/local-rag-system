"""Generate a manual-review JSONL scaffold from a RAGAS eval report.

Selects a deterministic, answerable/unanswerable-stratified sample of
questions from a `rag.eval.run_ragas_eval --verbose` report's
`ragas.per_question` section (joined against the base `per_example`
section by `question_index` for the generated answer), and writes one
JSONL row per selected question with empty human-label fields for a
reviewer to fill in by hand.

Usage:
    python -m rag.eval.run_ragas_eval --gold data/eval/techfusion_gold.jsonl \
        --dataset-id techfusion --verbose > /tmp/ragas_report.json
    python scripts/generate_manual_review.py --eval-output /tmp/ragas_report.json --num-rows 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MIN_ROWS = 10
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "manual_review"


def select_rows(report: dict[str, Any], num_rows: int) -> list[dict[str, Any]]:
    """Deterministically stratify-sample `num_rows` questions from `report`.

    First-N per answerable/unanswerable bucket (in original question
    order, no randomness), proportional to each bucket's share of the
    scored pool -- guarantees at least one unanswerable row when any
    exist and `num_rows` allows it.

    Parameters
    ----------
    report : dict[str, Any]
        A `rag.eval.run_ragas_eval --verbose` report.
    num_rows : int
        Number of rows to select; must be >= `MIN_ROWS`.

    Returns
    -------
    list[dict[str, Any]]
        Selected entries from `report["ragas"]["per_question"]`, sorted by
        `question_index`.

    Raises
    ------
    ValueError
        If `num_rows` is below `MIN_ROWS`, or the report has fewer scored
        questions than requested.
    """
    if num_rows < MIN_ROWS:
        raise ValueError(f"--num-rows must be >= {MIN_ROWS}, got {num_rows}")

    per_question = report.get("ragas", {}).get("per_question", [])
    if len(per_question) < num_rows:
        raise ValueError(
            f"Report has only {len(per_question)} scored question(s), fewer than the "
            f"{num_rows} requested. Re-run rag.eval.run_ragas_eval with a larger "
            "--sample-size."
        )

    unanswerable_pool = [q for q in per_question if q["unanswerable"]]
    answerable_pool = [q for q in per_question if not q["unanswerable"]]

    num_unanswerable = round(num_rows * len(unanswerable_pool) / len(per_question))
    if unanswerable_pool and num_unanswerable == 0 and num_rows > len(answerable_pool):
        num_unanswerable = 1
    num_unanswerable = min(num_unanswerable, len(unanswerable_pool), num_rows)
    num_answerable = min(num_rows - num_unanswerable, len(answerable_pool))

    selected = answerable_pool[:num_answerable] + unanswerable_pool[:num_unanswerable]
    selected.sort(key=lambda q: q["question_index"])
    return selected


def build_manual_review_row(question_entry: dict[str, Any], answer: str) -> dict[str, Any]:
    """Build one scaffold JSONL row with empty human-label fields.

    Parameters
    ----------
    question_entry : dict[str, Any]
        One entry from `report["ragas"]["per_question"]`.
    answer : str
        The generated answer for this question, from the base report's
        `per_example` section.

    Returns
    -------
    dict[str, Any]
        A row with the question/answer/ragas-scores plus null
        `human_faithful`/`human_correct`/`human_relevant`/
        `human_correct_refusal` fields for a reviewer to fill in.
    """
    return {
        "question_index": question_entry["question_index"],
        "question": question_entry["question"],
        "unanswerable": question_entry["unanswerable"],
        "generated_answer": answer,
        "ragas_scores": question_entry["scores"],
        "human_faithful": None,
        "human_correct": None,
        "human_relevant": None,
        "human_correct_refusal": None,
        "human_notes": "",
    }


def main() -> None:
    """CLI entrypoint: select rows from a ragas report and write the scaffold JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-output", required=True, help="Path to a rag.eval.run_ragas_eval --verbose report"
    )
    parser.add_argument(
        "--num-rows", type=int, default=MIN_ROWS, help=f"Rows to select, by default {MIN_ROWS}"
    )
    parser.add_argument("--output", default=None, help="Output JSONL path")
    args = parser.parse_args()

    report = json.loads(Path(args.eval_output).read_text(encoding="utf-8"))
    rows = select_rows(report, args.num_rows)
    # per_example has no explicit question_index field; its position matches
    # the index run_ragas_eval assigned when it built ragas scoring rows.
    answers_by_index = {i: entry["answer"] for i, entry in enumerate(report.get("per_example", []))}

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_DIR / f"{report.get('dataset_id', 'unknown')}_manual_review.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in rows:
            answer = answers_by_index.get(entry["question_index"], "")
            f.write(json.dumps(build_manual_review_row(entry, answer)) + "\n")

    print(f"Wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
