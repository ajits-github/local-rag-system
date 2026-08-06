from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_ragas_manual = _load_script("compare_ragas_manual")


def _ragas_report_path(tmp_path: Path) -> Path:
    """Write a fake ragas report with two scored questions."""
    report = {
        "ragas": {
            "per_question": [
                {
                    "question_index": 0,
                    "question": "Q0",
                    "unanswerable": False,
                    "scores": {
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.9,
                        "answer_correctness": 0.2,
                    },
                },
                {
                    "question_index": 1,
                    "question": "Q1",
                    "unanswerable": True,
                    "scores": {
                        "faithfulness": 0.3,
                        "answer_relevancy": 0.3,
                        "answer_correctness": 0.1,
                    },
                },
            ]
        }
    }
    path = tmp_path / "ragas_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _manual_review_path(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a manual-review JSONL file from `rows`."""
    path = tmp_path / "manual_review.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_manual_labels_skips_rows_with_no_human_fields_set(tmp_path):
    """load_manual_labels() only keeps rows with at least one non-null human_* field."""
    rows = [
        {
            "question_index": 0,
            "human_faithful": True,
            "human_correct": None,
            "human_relevant": None,
            "human_correct_refusal": None,
        },
        {
            "question_index": 1,
            "human_faithful": None,
            "human_correct": None,
            "human_relevant": None,
            "human_correct_refusal": None,
        },
    ]
    path = _manual_review_path(tmp_path, rows)

    labeled = compare_ragas_manual.load_manual_labels(path)

    assert len(labeled) == 1
    assert labeled[0]["question_index"] == 0


def test_compute_agreement_raises_when_no_rows_labeled():
    """compute_agreement() raises ValueError when nothing has been labeled."""
    with pytest.raises(ValueError, match="No manually-labeled rows"):
        compare_ragas_manual.compute_agreement({}, [])


def test_compute_agreement_thresholds_ragas_score_against_human_bool():
    """A ragas score >= 0.5 agreeing with human_faithful=True counts as a match."""
    ragas_scores = {0: {"faithfulness": 0.9}, 1: {"faithfulness": 0.2}}
    manual_rows = [
        {
            "question_index": 0,
            "unanswerable": False,
            "human_faithful": True,
            "human_correct": None,
            "human_relevant": None,
            "human_correct_refusal": None,
        },
        {
            "question_index": 1,
            "unanswerable": False,
            "human_faithful": False,
            "human_correct": None,
            "human_relevant": None,
            "human_correct_refusal": None,
        },
    ]

    agreement = compare_ragas_manual.compute_agreement(ragas_scores, manual_rows)

    assert agreement["pairs"]["faithfulness"]["labeled_n"] == 2
    assert agreement["pairs"]["faithfulness"]["agreement_rate"] == 1.0


def test_compute_agreement_refusal_is_human_only():
    """Refusal correctness is computed from human labels alone, no RAGAS counterpart."""
    ragas_scores = {0: {}}
    manual_rows = [
        {
            "question_index": 0,
            "unanswerable": True,
            "human_faithful": None,
            "human_correct": None,
            "human_relevant": None,
            "human_correct_refusal": True,
        },
    ]

    agreement = compare_ragas_manual.compute_agreement(ragas_scores, manual_rows)

    assert agreement["refusal"]["labeled_n"] == 1
    assert agreement["refusal"]["correct_rate"] == 1.0


def test_render_report_includes_caveat_string():
    """render_report() prepends the fixed reliability caveat."""
    agreement = {
        "pairs": {
            "faithfulness": {"labeled_n": 2, "agreement_rate": 1.0},
            "answer_relevancy": {"labeled_n": 0, "agreement_rate": None},
            "answer_correctness": {"labeled_n": 0, "agreement_rate": None},
        },
        "refusal": {"labeled_n": 0, "correct_rate": None},
    }
    report = compare_ragas_manual.render_report(agreement, num_labeled=2)

    assert compare_ragas_manual.CAVEAT in report


def test_render_report_flags_small_sample_below_ten():
    """render_report() adds an explicit warning when fewer than 10 rows are labeled."""
    agreement = {
        "pairs": {
            "faithfulness": {"labeled_n": 2, "agreement_rate": 1.0},
            "answer_relevancy": {"labeled_n": 0, "agreement_rate": None},
            "answer_correctness": {"labeled_n": 0, "agreement_rate": None},
        },
        "refusal": {"labeled_n": 0, "correct_rate": None},
    }
    report = compare_ragas_manual.render_report(agreement, num_labeled=2)

    assert "below the recommended minimum" in report
