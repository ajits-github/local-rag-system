from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.eval.report_content_type import build_breakdown


def _write_kb(tmp_path: Path) -> Path:
    """Write a synthetic kb/ with a table doc and a prose doc under tmp_path/data."""
    kb_root = tmp_path / "data" / "kb"
    kb_root.mkdir(parents=True)
    (kb_root / "table.md").write_text(
        "# Report\n\n| Name | Value |\n|---|---|\n| a | 1 |\n", encoding="utf-8"
    )
    (kb_root / "prose.md").write_text("# Notes\n\nJust some prose here.\n", encoding="utf-8")
    return kb_root


def _per_example_entry(
    question: str,
    relevant_documents: list[str],
    retrieved_sources: list[str],
    *,
    question_type: str | None = None,
    unanswerable: bool = False,
    include_latency: bool = True,
) -> dict:
    """Build one rag.eval.run_eval per_example row, matching evaluate()'s real shape."""
    entry = {
        "question": question,
        "question_type": question_type,
        "difficulty": "easy",
        "unanswerable": unanswerable,
        "relevant_documents": relevant_documents,
        "retrieved_sources": retrieved_sources,
    }
    if include_latency:
        entry.update(retrieval_ms=10.0, generation_ms=20.0, total_ms=30.0)
    return entry


def test_build_breakdown_groups_by_bucket_and_reaggregates_metrics(tmp_path: Path):
    """per_example rows are grouped into buckets, each with its own recall/hit_rate/mrr/latency."""
    kb_root = _write_kb(tmp_path)
    report = {
        "per_example": [
            _per_example_entry("What is the table value?", ["kb/table.md"], ["kb/table.md"]),
            _per_example_entry("What do the notes say?", ["kb/prose.md"], ["kb/prose.md"]),
        ]
    }

    breakdown = build_breakdown(report, kb_root.parent, load_config())

    assert set(breakdown) == {"table", "prose"}
    assert breakdown["table"]["num_examples"] == 1
    assert breakdown["table"]["retrieval"]["recall@5"] == 1.0
    assert breakdown["table"]["hit_rate"]["hit_rate@5"] == 1.0
    assert breakdown["table"]["mrr"] == 1.0
    assert breakdown["table"]["latency_ms"]["retrieval_mean"] == 10.0


def test_build_breakdown_omits_latency_when_generation_was_skipped(tmp_path: Path):
    """per_example rows without retrieval_ms/generation_ms/total_ms produce no latency_ms key."""
    kb_root = _write_kb(tmp_path)
    report = {
        "per_example": [
            _per_example_entry(
                "What is the table value?",
                ["kb/table.md"],
                ["kb/table.md"],
                include_latency=False,
            ),
        ]
    }

    breakdown = build_breakdown(report, kb_root.parent, load_config())

    assert "latency_ms" not in breakdown["table"]


def test_build_breakdown_separates_unanswerable_and_multi_hop_buckets(tmp_path: Path):
    """Unanswerable and multi_hop rows land in their own buckets, not a content-type bucket."""
    kb_root = _write_kb(tmp_path)
    report = {
        "per_example": [
            _per_example_entry("Unanswerable question", ["kb/table.md"], [], unanswerable=True),
            _per_example_entry(
                "Multi hop question",
                ["kb/table.md", "kb/prose.md"],
                ["kb/table.md"],
                question_type="multi_hop",
            ),
        ]
    }

    breakdown = build_breakdown(report, kb_root.parent, load_config())

    assert set(breakdown) == {"unanswerable", "multi_hop"}
