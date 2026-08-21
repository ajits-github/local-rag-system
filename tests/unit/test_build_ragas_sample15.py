from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_ragas_sample15 = _load_script("build_ragas_sample15")


def _row(question: str, **overrides: Any) -> dict[str, Any]:
    """Build a minimal gold-row dict, overriding any fields given."""
    base: dict[str, Any] = {
        "question": question,
        "content_type": None,
        "question_type": "single_document",
        "difficulty": "medium",
        "unanswerable": False,
    }
    base.update(overrides)
    return base


def _synthetic_rows() -> list[dict[str, Any]]:
    """Build a minimal source set covering every bucket build_sample selects from."""
    rows = [
        _row(f"specialty {bucket}", content_type=bucket)
        for bucket in build_ragas_sample15.SPECIALTY_BUCKETS
    ]
    rows += [
        _row("plain easy", difficulty="easy"),
        _row("plain medium", difficulty="medium"),
        _row("plain hard", difficulty="hard"),
        _row("plain multi_hop medium", question_type="multi_hop", difficulty="medium"),
        _row("plain multi_hop hard", question_type="multi_hop", difficulty="hard"),
        _row("plain unanswerable", unanswerable=True),
        _row(
            "plain extra unused", difficulty="easy"
        ),  # not picked. exact axes already covered above
    ]
    return rows


def test_build_sample_returns_exactly_fifteen_rows():
    """build_sample always selects exactly 15 rows given a well-formed source set."""
    selected = build_ragas_sample15.build_sample(_synthetic_rows())
    assert len(selected) == 15


def test_build_sample_covers_every_specialty_bucket():
    """Every one of the 9 authored specialty content_type buckets appears exactly once."""
    selected = build_ragas_sample15.build_sample(_synthetic_rows())
    selected_buckets = [r["content_type"] for r in selected if r.get("content_type")]
    assert sorted(selected_buckets) == sorted(build_ragas_sample15.SPECIALTY_BUCKETS)


def test_build_sample_includes_plain_bucket_axes():
    """The 6 plain-bucket picks cover multi_hop, unanswerable, and all three difficulties."""
    selected = build_ragas_sample15.build_sample(_synthetic_rows())
    plain = [r for r in selected if not r.get("content_type")]
    assert len(plain) == 6
    assert any(r["question_type"] == "multi_hop" for r in plain)
    assert any(r["unanswerable"] is True for r in plain)
    assert {r["difficulty"] for r in plain} == {"easy", "medium", "hard"}


def test_build_sample_is_deterministic():
    """Running build_sample twice on the same input yields the same selection, in the same order."""
    rows = _synthetic_rows()
    first = build_ragas_sample15.build_sample(rows)
    second = build_ragas_sample15.build_sample(rows)
    assert [r["question"] for r in first] == [r["question"] for r in second]


def test_build_sample_never_selects_the_same_question_twice():
    """No question appears more than once in the selection."""
    selected = build_ragas_sample15.build_sample(_synthetic_rows())
    questions = [r["question"] for r in selected]
    assert len(questions) == len(set(questions))
