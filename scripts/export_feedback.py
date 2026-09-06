"""Export feedback rows for human review and eval-curation, as JSONL or CSV.

Not a public API endpoint by design (this milestone's own instruction is
explicit: no broadly-accessible `GET /feedback`). This is the smallest
practical internal/admin mechanism to inspect feedback, following this
repo's existing `scripts/` convention (compare `record_experiment.py`,
`compare_experiments.py`).

Never writes to a gold dataset, and never decides that a row is "correct."
Feedback rows are a starting point for human review, not a substitute for
it -- promoting a specific example into a gold eval file (e.g.
`data/eval/techfusion_gold.jsonl`) remains a separate, deliberate,
hand-authored step. See `docs/architecture.md`'s "Feedback loop" section.

`--tenant-id` is required unless `--all-tenants` is passed explicitly: the
default CLI invocation must not silently export every tenant's query/
answer/comment text just because no scope was given.

Usage:
    python scripts/export_feedback.py --format jsonl --rating negative \
        --dataset-id techfusion --start-date 2026-08-01 --out feedback.jsonl
    python scripts/export_feedback.py --format csv --tenant-id tenant_alpha --out feedback.csv
    python scripts/export_feedback.py --all-tenants --format jsonl --out all_feedback.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.feedback.schemas import FeedbackRecord  # noqa: E402
from rag.feedback.store import FeedbackStore  # noqa: E402

CSV_FIELDS: list[str] = [
    "feedback_id",
    "created_at",
    "updated_at",
    "tenant_id",
    "caller_key",
    "request_id",
    "rating",
    "reason",
    "comment",
    "route",
    "dataset_id",
    "query_text",
    "answer_text",
    "cited_source_ids",
    "tool_calls",
    "generation_model",
    "prompt_id",
    "prompt_version",
    "retrieval_provider",
    "reranker_provider",
]


def record_to_row(record: FeedbackRecord) -> dict[str, Any]:
    """Flatten one `FeedbackRecord` to a JSON-serializable dict (ISO timestamps, plain lists)."""
    return record.model_dump(mode="json")


def write_jsonl(records: list[FeedbackRecord], out: TextIO) -> None:
    """Write one JSON object per line, one per record."""
    for record in records:
        out.write(json.dumps(record_to_row(record)) + "\n")


def write_csv(records: list[FeedbackRecord], out: TextIO) -> None:
    """Write records as CSV, joining list-valued fields with `;` for human readability."""
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        row = record_to_row(record)
        row["cited_source_ids"] = ";".join(row["cited_source_ids"])
        row["tool_calls"] = ";".join(row["tool_calls"])
        writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the export."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--out", default=None, help="Output file path; defaults to stdout")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument(
        "--all-tenants",
        action="store_true",
        help=(
            "Explicitly opt into exporting feedback across every tenant. "
            "Required when --tenant-id is omitted."
        ),
    )
    parser.add_argument("--rating", choices=["positive", "negative"], default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--route", choices=["classic_rag", "agent"], default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.tenant_id is None and not args.all_tenants:
        parser.error(
            "--tenant-id is required unless --all-tenants is passed explicitly "
            "(omitting both would export every tenant's feedback, including "
            "query/answer/comment text)"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: query `FeedbackStore.export()` and write JSONL/CSV."""
    args = parse_args(argv)
    config = load_config(args.config)
    store = FeedbackStore(config.database_url(), table=config.feedback.table_name)
    records = store.export(
        tenant_id=args.tenant_id,
        rating=args.rating,
        dataset_id=args.dataset_id,
        route=args.route,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    writer = write_jsonl if args.format == "jsonl" else write_csv
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer(records, f)
        print(f"Wrote {len(records)} feedback record(s) to {args.out}", file=sys.stderr)
    else:
        writer(records, sys.stdout)


if __name__ == "__main__":
    main()
