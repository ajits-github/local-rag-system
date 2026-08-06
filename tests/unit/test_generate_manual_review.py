from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_manual_review = _load_script("generate_manual_review")


def _report(num_answerable: int, num_unanswerable: int) -> dict:
    """Build a fake ragas report with the given answerable/unanswerable question mix."""
    per_question = [
        {
            "question_index": i,
            "question": f"Answerable question {i}",
            "unanswerable": False,
            "scores": {"faithfulness": 0.8},
        }
        for i in range(num_answerable)
    ] + [
        {
            "question_index": num_answerable + i,
            "question": f"Unanswerable question {i}",
            "unanswerable": True,
            "scores": {"faithfulness": 0.5},
        }
        for i in range(num_unanswerable)
    ]
    return {"dataset_id": "test-dataset", "ragas": {"per_question": per_question}}


def test_select_rows_rejects_num_rows_below_minimum():
    """select_rows() raises ValueError when num_rows is below the minimum of 10."""
    with pytest.raises(ValueError, match="10"):
        generate_manual_review.select_rows(_report(15, 0), num_rows=5)


def test_select_rows_rejects_insufficient_scored_questions():
    """select_rows() raises ValueError when the report has fewer scored questions than requested."""
    with pytest.raises(ValueError, match="sample-size"):
        generate_manual_review.select_rows(_report(5, 2), num_rows=10)


def test_select_rows_stratifies_answerable_and_unanswerable():
    """select_rows() includes at least one unanswerable row when the pool has any."""
    selected = generate_manual_review.select_rows(_report(12, 3), num_rows=10)

    assert len(selected) == 10
    assert any(row["unanswerable"] for row in selected)
    assert selected == sorted(selected, key=lambda r: r["question_index"])


def test_select_rows_all_answerable_when_pool_has_no_unanswerable():
    """select_rows() draws only from the answerable pool when no unanswerable rows exist."""
    selected = generate_manual_review.select_rows(_report(15, 0), num_rows=10)

    assert len(selected) == 10
    assert all(not row["unanswerable"] for row in selected)


def test_build_manual_review_row_has_null_human_label_fields():
    """build_manual_review_row() scaffolds empty human_* fields for a reviewer to fill in."""
    entry = {"question_index": 0, "question": "Q?", "unanswerable": False, "scores": {}}
    row = generate_manual_review.build_manual_review_row(entry, answer="A.")

    assert row["question"] == "Q?"
    assert row["generated_answer"] == "A."
    assert row["human_faithful"] is None
    assert row["human_correct"] is None
    assert row["human_relevant"] is None
    assert row["human_correct_refusal"] is None
    assert row["human_notes"] == ""
