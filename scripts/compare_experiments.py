"""Render a Markdown comparison table from every experiment record.

Reads every experiments/results/*.json record (written by
scripts/record_experiment.py), saves the *full* table to
experiments/reports/comparison.md, and splices only the most recent
`README_HIGHLIGHT_COUNT` experiments into README.md between the
EXPERIMENTS_TABLE markers -- the full table has grown too long to serve
as a readable front door, so the README is deliberately kept to a
rolling recent-history window instead of every record ever recorded.

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

# The README shows only the most recent experiments (see module docstring);
# the full history always lives in experiments/reports/comparison.md.
README_HIGHLIGHT_COUNT = 7


def load_records(results_dir: Path, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """Load every experiment record JSON file, sorted by `experiment_id`.

    Parameters
    ----------
    results_dir : Path
        Directory containing `*.json` records written by `record_experiment.py`.
    exclude : set[str] | None, optional
        `experiment_id`s to omit. For records that exist (e.g. a small
        pilot subset) but aren't comparable to the rest of the table, by
        default None.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records, ordered by `experiment_id` (not filename, so
        "experiment_2" doesn't sort after "experiment_10").
    """
    exclude = exclude or set()
    records = []
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("experiment_id") not in exclude:
            records.append(record)
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
        "| Experiment | Label | Retrieval | Generation model | Embedder | Reranker "
        "| Prompt | Rel.Exp | Recall@5 | Recall@10 "
        "| Hit Rate@10 | MRR | Answer quality | Supp.Ctx Hit | Img Hit "
        "| RAGAS Faithful | RAGAS Correct "
        "| Total latency | Dataset | Date |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for r in records:
        reranker = r.get("reranker_provider") or "none"
        if r.get("reranker_model"):
            reranker = f"{reranker} ({_short_model_name(r['reranker_model'])})"
        retrieval_provider = r.get("retrieval_provider") or "dense"
        prompt_version = r.get("prompt_version") or "-"
        relationship_expansion = r.get("relationship_expansion_enabled")
        if relationship_expansion is None:
            rel_exp = "-"
        else:
            rel_exp = "on" if relationship_expansion else "off"
        total_latency_ms = r.get("total_latency_ms")
        total_latency = f"{total_latency_ms / 1000:.1f}s" if total_latency_ms is not None else "-"
        date = (r.get("timestamp") or "")[:10] or "-"
        rows.append(
            f"| {r.get('experiment_id', '?')} | {r.get('label') or r.get('experiment_id', '?')} "
            f"| {retrieval_provider} "
            f"| {r.get('generation_model', '?')} | {_short_model_name(r.get('embedding_model'))} "
            f"| {reranker} | {prompt_version} | {rel_exp} "
            f"| {_fmt(r.get('recall_at_5'))} | {_fmt(r.get('recall_at_10'))} "
            f"| {_fmt(r.get('hit_rate_at_10'))} | {_fmt(r.get('mrr'))} "
            f"| {_fmt(r.get('answer_quality'))} "
            f"| {_fmt(r.get('supporting_context_hit_rate'))} "
            f"| {_fmt(r.get('relevant_image_hit_rate'))} "
            f"| {_fmt(r.get('ragas_faithfulness'))} | {_fmt(r.get('ragas_answer_correctness'))} "
            f"| {total_latency} "
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
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated experiment_ids to omit (e.g. a non-comparable pilot subset)",
    )
    args = parser.parse_args()

    exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}
    records = load_records(Path(args.results_dir), exclude=exclude)
    table = render_table(records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")

    readme_table = render_table(records[-README_HIGHLIGHT_COUNT:])
    updated = update_readme(readme_table)
    print(table)
    print()
    if updated:
        print(
            f"README.md updated (most recent {min(len(records), README_HIGHLIGHT_COUNT)} "
            f"of {len(records)} experiments; full table in experiments/reports/comparison.md)."
        )
    else:
        print("README.md not updated (markers not found).")


if __name__ == "__main__":
    main()
