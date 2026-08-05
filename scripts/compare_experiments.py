"""Render a Markdown comparison table from every experiment record.

Reads every experiments/results/*.json record (written by
scripts/record_experiment.py), saves the table to
experiments/reports/comparison.md, and splices it into README.md between
the EXPERIMENTS_TABLE markers so the README never has to be hand-edited
after a new experiment is recorded.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results"
REPORTS_DIR = ROOT / "experiments" / "reports"
README_PATH = ROOT / "README.md"

TABLE_START = "<!-- EXPERIMENTS_TABLE_START -->"
TABLE_END = "<!-- EXPERIMENTS_TABLE_END -->"


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    """Load every experiment record JSON file, sorted by `experiment_id`.

    Parameters
    ----------
    results_dir : Path
        Directory containing `*.json` records written by `record_experiment.py`.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records, ordered by `experiment_id` (not filename, so
        "experiment_2" doesn't sort after "experiment_10").
    """
    records = []
    for path in sorted(results_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    # Order by experiment_id so "experiment_2" doesn't sort before
    # "experiment_10" the way plain filename sorting would.
    records.sort(key=lambda r: r.get("experiment_id", ""))
    return records


def _fmt(value: float | None, spec: str = ".3f") -> str:
    """Format a metric value, or "-" if it's missing."""
    return "-" if value is None else format(value, spec)


def _short_model_name(model_name: str | None) -> str:
    """Drop a 'namespace/' prefix for table compactness.

    E.g. 'sentence-transformers/all-MiniLM-L6-v2' -> 'all-MiniLM-L6-v2'.
    """
    if not model_name:
        return "?"
    return model_name.rsplit("/", 1)[-1]


def render_table(records: list[dict[str, Any]]) -> str:
    """Render a Markdown comparison table, one row per experiment record.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Experiment records as loaded by `load_records`.

    Returns
    -------
    str
        A Markdown table, or a placeholder message if `records` is empty.
    """
    if not records:
        return "No experiments recorded yet -- see scripts/record_experiment.py."

    header = (
        "| # | Label | Generation model | Embedder | Reranker | Recall@5 | Recall@10 "
        "| Hit Rate@10 | MRR | Answer quality | Total latency | Dataset | Date |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for i, r in enumerate(records, start=1):
        reranker = r.get("reranker_provider") or "none"
        if r.get("reranker_model"):
            reranker = f"{reranker} ({_short_model_name(r['reranker_model'])})"
        total_latency_ms = r.get("total_latency_ms")
        total_latency = f"{total_latency_ms / 1000:.1f}s" if total_latency_ms is not None else "-"
        date = (r.get("timestamp") or "")[:10] or "-"
        rows.append(
            f"| {i} | {r.get('label') or r.get('experiment_id', '?')} "
            f"| {r.get('generation_model', '?')} | {_short_model_name(r.get('embedding_model'))} "
            f"| {reranker} "
            f"| {_fmt(r.get('recall_at_5'))} | {_fmt(r.get('recall_at_10'))} "
            f"| {_fmt(r.get('hit_rate_at_10'))} | {_fmt(r.get('mrr'))} "
            f"| {_fmt(r.get('answer_quality'))} | {total_latency} "
            f"| {r.get('dataset_id', '?')} | {date} |"
        )
    return "\n".join(rows)


def update_readme(table: str) -> bool:
    """Splice `table` into README.md between the EXPERIMENTS_TABLE markers.

    Parameters
    ----------
    table : str
        Markdown table to insert.

    Returns
    -------
    bool
        True if README.md exists and both markers were found and updated.
    """
    if not README_PATH.exists():
        return False
    text = README_PATH.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        return False
    before, _, rest = text.partition(TABLE_START)
    _, _, after = rest.partition(TABLE_END)
    README_PATH.write_text(f"{before}{TABLE_START}\n{table}\n{TABLE_END}{after}", encoding="utf-8")
    return True


def main() -> None:
    """CLI entrypoint: render the table, write it to disk, and update README.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    records = load_records(Path(args.results_dir))
    table = render_table(records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")

    updated = update_readme(table)
    print(table)
    print()
    print("README.md updated." if updated else "README.md not updated (markers not found).")


if __name__ == "__main__":
    main()
