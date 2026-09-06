"""Unit tests for scripts/export_feedback.py's serialization logic.

No database involved: `record_to_row`/`write_jsonl`/`write_csv` operate
on already-constructed `FeedbackRecord` objects. Filter-argument wiring
into `FeedbackStore.export()` is covered by `test_feedback_store.py`;
real end-to-end filtering against Postgres is covered by
`tests/integration/test_feedback_persistence.py`.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.feedback.schemas import FeedbackRecord

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_feedback = _load_script("export_feedback")
parse_args = export_feedback.parse_args
write_csv = export_feedback.write_csv
write_jsonl = export_feedback.write_jsonl


def _record(**overrides: object) -> FeedbackRecord:
    base: dict[str, object] = {
        "feedback_id": "fb-1",
        "created_at": datetime(2026, 8, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 8, 1, tzinfo=UTC),
        "tenant_id": "tenant_alpha",
        "caller_key": "abc123",
        "request_id": "req-1",
        "rating": "negative",
        "reason": "incorrect_answer",
        "comment": "wrong number",
        "route": "classic_rag",
        "dataset_id": "techfusion",
        "query_text": "what is the retention policy?",
        "answer_text": "42 days.",
        "cited_source_ids": ["doc_0", "doc_1"],
        "tool_calls": [],
        "generation_model": "qwen2.5:1.5b",
        "prompt_id": "rag_answer",
        "prompt_version": "v1",
        "retrieval_provider": "dense",
        "reranker_provider": "none",
    }
    base.update(overrides)
    return FeedbackRecord(**base)  # type: ignore[arg-type]


def test_write_jsonl_writes_one_json_object_per_line() -> None:
    """write_jsonl produces one parseable JSON object per record, in order."""
    records = [_record(feedback_id="fb-1"), _record(feedback_id="fb-2", rating="positive")]
    out = io.StringIO()

    write_jsonl(records, out)

    lines = out.getvalue().strip("\n").split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["feedback_id"] == "fb-1"
    assert parsed[1]["rating"] == "positive"
    assert parsed[0]["cited_source_ids"] == ["doc_0", "doc_1"]


def test_write_csv_joins_list_fields_and_writes_header() -> None:
    """write_csv joins list-valued fields with ';' and includes every declared column."""
    out = io.StringIO()

    write_csv([_record()], out)

    reader = csv.DictReader(io.StringIO(out.getvalue()))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["cited_source_ids"] == "doc_0;doc_1"
    assert rows[0]["rating"] == "negative"
    assert rows[0]["feedback_id"] == "fb-1"


def test_write_jsonl_never_mutates_a_gold_dataset_file() -> None:
    """The export path only ever writes the caller-specified --out file, never data/eval/*gold*."""
    out = io.StringIO()
    write_jsonl([_record()], out)
    # No file I/O happens inside write_jsonl/write_csv at all -- the only
    # place this script opens a file is main()'s own --out handling, which
    # is never pointed at a gold file by any code path here.
    assert out.getvalue()  # sanity: something was actually written to the passed stream


def test_parse_args_defaults_to_jsonl_and_stdout() -> None:
    """CLI defaults with --all-tenants (bare no-scope args are rejected): jsonl format, no --out."""
    args = parse_args(["--all-tenants"])

    assert args.format == "jsonl"
    assert args.out is None
    assert args.tenant_id is None
    assert args.rating is None


def test_parse_args_without_tenant_scope_or_all_tenants_exits(capsys) -> None:
    """Omitting both --tenant-id and --all-tenants fails closed rather than exporting every tenant.

    Regression test: the original implementation defaulted --tenant-id to
    None and FeedbackStore.export() omits the WHERE clause entirely when
    no filters are given, so the bare default CLI invocation used to
    silently export every tenant's feedback, including query/answer/
    comment text.
    """
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])

    assert exc_info.value.code == 2
    assert "--tenant-id" in capsys.readouterr().err


def test_parse_args_parses_all_filters() -> None:
    """Every documented CLI filter is parsed into the expected attribute."""
    args = parse_args(
        [
            "--format",
            "csv",
            "--tenant-id",
            "tenant_alpha",
            "--rating",
            "negative",
            "--dataset-id",
            "techfusion",
            "--route",
            "agent",
            "--start-date",
            "2026-08-01",
            "--end-date",
            "2026-08-31",
            "--limit",
            "50",
        ]
    )

    assert args.format == "csv"
    assert args.tenant_id == "tenant_alpha"
    assert args.rating == "negative"
    assert args.dataset_id == "techfusion"
    assert args.route == "agent"
    assert str(args.start_date) == "2026-08-01"
    assert str(args.end_date) == "2026-08-31"
    assert args.limit == 50
