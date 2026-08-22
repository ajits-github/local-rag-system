"""Build a targeted RAGAS sample for the rag_answer v3-vs-v5 prompt A/B.

Deterministic, file-order-based selection (no randomness), mirroring
build_ragas_sample15.py's approach but targeted at this specific A/B rather
than stratified across the whole gold set:

  - Every row with a non-empty `forbidden_documents` (19 rows on the current
    techfusion_gold.jsonl) -- the security/authorization set. This set
    already overlaps heavily with injection_present, sensitive_data_present,
    and requires_current_document (freshness), so it representatively
    covers authorization/redaction/injection/freshness without a second,
    separately-tuned selection per category.
  - 6 plain benign rows (no forbidden_documents/injection_present/
    sensitive_data_present/requires_current_document/safety_category),
    covering the same question_type/difficulty/unanswerable axes as
    build_ragas_sample15.py's plain-bucket picks, so the sample isn't only
    security edge cases.

All fields are passed through untouched from the source row. Re-run this to
regenerate data/eval/techfusion_prompt_ab_ragas_sample.jsonl if
techfusion_gold.jsonl changes; the selection is fully deterministic given
the same source file.

Usage:
    python scripts/build_prompt_ab_ragas_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "eval" / "techfusion_gold.jsonl"
DST = ROOT / "data" / "eval" / "techfusion_prompt_ab_ragas_sample.jsonl"


def _is_benign(row: dict[str, Any]) -> bool:
    """Whether `row` has none of the security/safety flags this sample targets."""
    return not (
        row.get("forbidden_documents")
        or row.get("injection_present")
        or row.get("sensitive_data_present")
        or row.get("requires_current_document")
        or row.get("safety_category")
    )


def build_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the security set plus 6 benign control rows from `rows`.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Parsed gold examples, in file order.

    Returns
    -------
    list[dict[str, Any]]
        The selected rows, in selection order (security rows first, then
        the benign control sample).
    """
    security = [row for row in rows if row.get("forbidden_documents")]
    selected = list(security)
    selected_questions = {row["question"] for row in selected}

    benign_pool = [
        row for row in rows if _is_benign(row) and row["question"] not in selected_questions
    ]

    def pick(question_type: str | None, difficulty: str | None, unanswerable: bool | None) -> None:
        for row in benign_pool:
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
    pick(None, None, True)  # at least one unanswerable=True benign question

    return selected


def main() -> None:
    """CLI entrypoint: build the sample from `SRC` and write it to `DST`."""
    rows = [
        json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    selected = build_sample(rows)

    with DST.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows to {DST}")
    for i, row in enumerate(selected, 1):
        tag = "security" if row.get("forbidden_documents") else "benign"
        print(
            f"{i:2}. [{tag}/{row.get('safety_category') or row.get('question_type')}"
            f"/{row.get('difficulty')}/unanswerable={row.get('unanswerable')}] {row['question']}"
        )


if __name__ == "__main__":
    main()
