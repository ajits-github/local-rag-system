"""Run the Context Engineering benchmark and write a comparison report.

Thin CLI wrapper around `rag.eval.run_context_eval`, matching this
repo's `scripts/record_experiment.py`/`scripts/compare_experiments.py`
convention: the eval logic lives in `rag.eval`, this script just drives
it and renders a human-readable Markdown report alongside the raw JSON.

Usage:
    python scripts/benchmark_context_engineering.py
    python scripts/benchmark_context_engineering.py --model-context-window 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.eval.run_context_eval import (  # noqa: E402
    STRATEGIES,
    evaluate_context_engineering,
)

REPORT_JSON = REPO_ROOT / "experiments" / "results" / "context_engineering" / "report.json"
REPORT_MD = REPO_ROOT / "experiments" / "reports" / "context_engineering_benchmark.md"


def _render_markdown(report) -> str:
    """Render `report.by_strategy_summary()` as a Markdown comparison table."""
    summary = report.by_strategy_summary()
    lines = [
        "# Context Engineering benchmark",
        "",
        f"Generated at: {report.generated_at}",
        f"Total scenario/strategy/size combinations: {len(report.results)}",
        "",
        "| Strategy | Mean utilization | Overflowing runs | Mean excess tokens | "
        "Mean reduction | Mean build (ms) | Dropped items | Compressed items | Runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "top_n": "A: top-N (baseline)",
        "naive_token_first": "B: naive token-first",
        "dedup_plus_budget": "C: token-budget + dedup",
        "full_pipeline": "D: full pipeline",
    }
    for strategy in STRATEGIES:
        row = summary.get(strategy)
        if row is None:
            continue
        lines.append(
            f"| {labels.get(strategy, strategy)} | {row['mean_context_utilization']:.3f} | "
            f"{row['overflow_run_count']:.0f}/{row['run_count']:.0f} | "
            f"{row['mean_context_excess_tokens']:.1f} | "
            f"{row['mean_context_reduction']:.3f} | {row['mean_build_latency_ms']:.3f} | "
            f"{row['total_dropped_items']:.0f} | {row['total_items_compressed']:.0f} | "
            f"{row['run_count']:.0f} |"
        )
    sizes = sorted({r.candidate_token_target for r in report.results})
    lines.append("")
    lines.append("Candidate-size sweep: " + ", ".join(str(size) for size in sizes) + " tokens.")
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-context-window", type=int, default=4096)
    parser.add_argument("--reserved-output-tokens", type=int, default=512)
    args = parser.parse_args()

    report = evaluate_context_engineering(
        model_context_window_tokens=args.model_context_window,
        reserved_output_tokens=args.reserved_output_tokens,
    )

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
