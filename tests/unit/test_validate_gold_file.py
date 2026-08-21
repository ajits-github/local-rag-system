from __future__ import annotations

import importlib.util
from pathlib import Path

from rag.config import load_config
from rag.eval.gold_schema import GoldExample

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_gold_file = _load_script("validate_gold_file")


def _write_kb(tmp_path: Path) -> Path:
    """Write a synthetic kb/ with a table doc, a code doc, and a chart doc under tmp_path/data."""
    kb_root = tmp_path / "data" / "kb"
    kb_root.mkdir(parents=True)
    (kb_root / "table.md").write_text(
        "# Report\n\n| Name | Value |\n|---|---|\n| a | 1 |\n", encoding="utf-8"
    )
    (kb_root / "code.md").write_text(
        "# Runbook\n\n```python\ndef f():\n    return 1\n```\n", encoding="utf-8"
    )
    (kb_root / "chart.md").write_text(
        "# Capacity\n\n```text\nQ1 |###\n```\n\n*Chart caption: grew.*\n", encoding="utf-8"
    )
    return kb_root


def test_missing_relevant_document_is_a_hard_error(tmp_path: Path):
    """A relevant_documents path that doesn't resolve to a file is an error, not a warning."""
    kb_root = _write_kb(tmp_path)
    examples = [
        GoldExample(question="Q1", relevant_documents=["kb/table.md"]),
        GoldExample(question="Q2", relevant_documents=["kb/does-not-exist.md"]),
    ]

    errors, _warnings = validate_gold_file.validate(
        examples, kb_root.parent, load_config(), expect_count=None
    )

    assert any("does-not-exist.md" in e for e in errors)


def test_expect_count_mismatch_is_a_warning_not_an_error(tmp_path: Path):
    """A record-count mismatch is reported as a warning and never fails the run."""
    kb_root = _write_kb(tmp_path)
    examples = [GoldExample(question="Q1", relevant_documents=["kb/table.md"])]

    errors, warnings = validate_gold_file.validate(
        examples, kb_root.parent, load_config(), expect_count=5
    )

    assert any("Expected 5" in w for w in warnings)
    assert not any("Expected 5" in e for e in errors)


def test_duplicate_questions_are_a_hard_error(tmp_path: Path):
    """Two examples with the same question (case-folded) are flagged as duplicates."""
    kb_root = _write_kb(tmp_path)
    examples = [
        GoldExample(question="What is X?", relevant_documents=["kb/table.md"]),
        GoldExample(question="what is x?", relevant_documents=["kb/code.md"]),
    ]

    errors, _warnings = validate_gold_file.validate(
        examples, kb_root.parent, load_config(), expect_count=None
    )

    assert any("Duplicate question" in e for e in errors)


def test_empty_structured_bucket_is_a_hard_error(tmp_path: Path):
    """If no example classifies into a structured bucket (e.g. 'chart'), that's an error."""
    kb_root = _write_kb(tmp_path)
    examples = [
        GoldExample(question="Q1", relevant_documents=["kb/table.md"]),
        GoldExample(question="Q2", relevant_documents=["kb/code.md"]),
        # No example referencing kb/chart.md. the 'chart' bucket stays empty.
    ]

    errors, _warnings = validate_gold_file.validate(
        examples, kb_root.parent, load_config(), expect_count=None
    )

    assert any("'chart' bucket" in e for e in errors)


def test_all_buckets_present_and_no_duplicates_passes_clean(tmp_path: Path):
    """A gold file covering every structured bucket with unique questions produces no errors."""
    kb_root = _write_kb(tmp_path)
    examples = [
        GoldExample(question="What is the value in the table?", relevant_documents=["kb/table.md"]),
        GoldExample(question="What does f() return?", relevant_documents=["kb/code.md"]),
        GoldExample(
            question="What does the chart caption say?", relevant_documents=["kb/chart.md"]
        ),
    ]

    errors, _warnings = validate_gold_file.validate(
        examples, kb_root.parent, load_config(), expect_count=None
    )

    assert errors == []
