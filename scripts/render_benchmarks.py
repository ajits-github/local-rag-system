"""Render a Markdown comparison table from every saved eval baseline.

Baseline files (data/eval/baselines/*.json, produced by
`rag.eval.run_eval --verbose`) are git-ignored since they contain
dataset-derived questions/answers, but the table this script prints is
just aggregate numbers. Safe to paste into README.md.

Usage:
    python scripts/render_benchmarks.py
    python scripts/render_benchmarks.py --baselines-dir data/eval/baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_BASELINES_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "baselines"


def _load_rows(baselines_dir: Path) -> list[dict[str, Any]]:
    """Load each baseline JSON file into a flat row of table columns."""
    rows = []
    for path in sorted(baselines_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        rows.append(
            {
                "file": path.name,
                "generated_at": data.get("generated_at", "?")[:10],
                "dataset_id": data.get("dataset_id", "?"),
                "generation_model": config.get("generation_model", "?"),
                "reranker": config.get("reranker_provider", "?"),
                "recall@5": data.get("retrieval", {}).get("recall@5"),
                "recall@10": data.get("retrieval", {}).get("recall@10"),
                "mrr": data.get("mrr"),
                "total_latency_s": (data.get("latency_ms", {}).get("total_mean") or 0) / 1000,
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a Markdown comparison table, one row per baseline file.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows as loaded by `_load_rows`.

    Returns
    -------
    str
        A Markdown table, or a placeholder message if `rows` is empty.
    """
    if not rows:
        return "No baseline files found."

    header = (
        "| Generation model | Reranker | Recall@5 | Recall@10 | MRR "
        "| Avg latency | Dataset | Date |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['generation_model']} | {r['reranker']} "
            f"| {r['recall@5']:.3f} | {r['recall@10']:.3f} | {r['mrr']:.3f} "
            f"| {r['total_latency_s']:.1f}s | {r['dataset_id']} | {r['generated_at']} |"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint: load baseline files and print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-dir", default=str(DEFAULT_BASELINES_DIR))
    args = parser.parse_args()

    rows = _load_rows(Path(args.baselines_dir))
    print(render_markdown_table(rows))


if __name__ == "__main__":
    main()
