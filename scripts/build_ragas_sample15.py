"""Build a stratified 15-question RAGAS sample from techfusion_gold.jsonl.

Deterministic, file-order-based selection (no randomness):
  - 1 question from each of the 9 authored specialty `content_type` buckets
    (architecture_diagram, chart, table_image, image_only,
    caption_answerable, relationship_aware, text_only, text_plus_image,
    unanswerable_visual). The first occurrence of each in file order.
  - 6 questions from the plain (content_type unset) bucket, covering the
    axes that bucket varies on: question_type (multi_hop/single_document),
    difficulty (easy/medium/hard), and unanswerable (True/False). So the
    sample isn't only edge-case multimodal questions but also represents
    the ordinary-retrieval majority of the gold set.

All fields are passed through untouched from the source row. Re-run this
to regenerate data/eval/techfusion_gold_v2_ragas_sample15.jsonl if
techfusion_gold.jsonl changes; the selection is fully deterministic given
the same source file.

Usage:
    python scripts/build_ragas_sample15.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "eval" / "techfusion_gold.jsonl"
DST = ROOT / "data" / "eval" / "techfusion_gold_v2_ragas_sample15.jsonl"

SPECIALTY_BUCKETS = [
    "architecture_diagram",
    "chart",
    "table_image",
    "image_only",
    "caption_answerable",
    "relationship_aware",
    "text_only",
    "text_plus_image",
    "unanswerable_visual",
]


def build_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 15 rows from `rows`: one per specialty content_type bucket, plus 6 plain rows.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Parsed gold examples, in file order.

    Returns
    -------
    list[dict[str, Any]]
        The 15 selected rows, in selection order.
    """
    selected: list[dict[str, Any]] = []
    selected_questions: set[str] = set()

    for bucket in SPECIALTY_BUCKETS:
        for row in rows:
            if row.get("content_type") == bucket and row["question"] not in selected_questions:
                selected.append(row)
                selected_questions.add(row["question"])
                break

    none_rows = [r for r in rows if not r.get("content_type")]

    def pick(question_type: str | None, difficulty: str | None, unanswerable: bool | None) -> None:
        for row in none_rows:
            if row["question"] in selected_questions:
                continue
            if question_type is not None and row.get("question_type") != question_type:
                continue
            if difficulty is not None and row.get("difficulty") != difficulty:
                continue
            if unanswerable is not None and row.get("unanswerable") != unanswerable:
                continue
            selected.append(row)
            selected_questions.add(row["question"])
            return

    pick("single_document", "easy", False)
    pick("single_document", "medium", False)
    pick("single_document", "hard", False)
    pick("multi_hop", "medium", False)
    pick("multi_hop", "hard", False)
    pick(None, None, True)  # at least one unanswerable=True plain question

    return selected


def main() -> None:
    """CLI entrypoint: build the sample from `SRC` and write it to `DST`."""
    rows = [
        json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    selected = build_sample(rows)
    assert len(selected) == 15, f"expected 15 rows, got {len(selected)}"

    with DST.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows to {DST}")
    for i, row in enumerate(selected, 1):
        print(
            f"{i:2}. [{row.get('content_type') or 'plain'}"
            f"/{row.get('question_type')}/{row.get('difficulty')}"
            f"/unanswerable={row.get('unanswerable')}] {row['question']}"
        )


if __name__ == "__main__":
    main()
