"""Run RAGAS generation-quality scoring against agent-produced answers/evidence.

Sibling to `run_ragas_eval.py`, not a replacement for it: that module
scores the classic `RetrievalPipeline.answer()` path (via
`run_eval.evaluate()`); this module scores the agent graph's actual
output (`rag.agent.graph.run_agent()`, via `run_agent_eval.evaluate_agent`)
for `data/eval/agentic_extension_gold.jsonl` specifically. RAGAS never
drives routing/tool-selection/retry scoring -- those stay the
deterministic `run_agent_eval.py` metrics; this module only judges final-
answer/evidence quality for whatever route the agent actually took.

Every scored row's `contexts` comes from `AgentState.retrieved_evidence`
already gathered by one `run_agent_eval.evaluate_agent(..., include_evidence=True)`
pass -- not a second agent run -- so the judged context is exactly what
the agent's own synthesize step actually saw, and each source still
passes through `egress_policy.apply_egress_policy` before it may reach
the hosted judge, identical in spirit to `run_ragas_eval.py`'s
`_build_rows`.

`expected_tool_sequence`/`expected_subquestions`/`expected_reformulation`
are gold-authoring/scoring-only fields already never read by
`rag.agent.*` (the agent only ever sees `GoldExample.question` via
`AgentState.original_query`); this module doesn't change that boundary.
`expected_answer`/`reference_contexts` are read here, in the eval/judge
path only, exactly like `run_ragas_eval.py` already does.

Usage:
    python -m rag.eval.run_agent_ragas_eval \
        --gold data/eval/agentic_extension_gold.jsonl --dataset-id techfusion \
        --config config/experiments/agentic-rag-baseline-v1.yaml --verbose
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.audit import log_audit_event
from rag.config import AppConfig, load_config
from rag.eval import ragas_cache, ragas_scorer
from rag.eval.egress_policy import apply_egress_policy
from rag.eval.gold_schema import load_gold_jsonl
from rag.eval.run_agent_eval import evaluate_agent
from rag.eval.run_ragas_eval import _cache_summary, _judge_summary
from rag.factory import build_embedder, build_judge_llm, build_llm, build_vectorstore
from rag.retrieval.pipeline import RetrievalPipeline


def _build_rows(
    per_example: list[dict[str, Any]], config: AppConfig
) -> tuple[list[dict[str, Any]], int]:
    """Turn `evaluate_agent(..., include_evidence=True)`'s records into RAGAS scoring rows.

    Mirrors `run_ragas_eval._build_rows` exactly, except `contexts` comes
    from each record's `evidence_sources` (the agent's own gathered
    evidence) rather than a fresh `RetrievalPipeline.answer()` call, and
    each row additionally carries `agentic_category` for the by-category
    breakdown `run_ragas` in this module reports.

    Parameters
    ----------
    per_example : list[dict[str, Any]]
        `evaluate_agent(..., verbose=True, include_evidence=True)`'s
        `per_example` output.
    config : AppConfig
        Application configuration; reads `security.egress_policy`.

    Returns
    -------
    tuple[list[dict[str, Any]], int]
        ``(rows, num_skipped)``.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    blocked_count = 0
    for i, entry in enumerate(per_example):
        if not entry["expected_answer"]:
            skipped += 1
            continue
        contexts = []
        for source in entry.get("evidence_sources", []):
            decision = apply_egress_policy(source, config)
            if decision.allowed:
                contexts.append(decision.redacted_context)
            else:
                blocked_count += 1
        rows.append(
            {
                "question_index": i,
                "question": entry["question"],
                "unanswerable": entry["unanswerable"],
                "answer": entry["final_answer"] or "",
                "contexts": contexts,
                "expected_answer": entry["expected_answer"],
                "agentic_category": entry["agentic_category"],
                "route": entry["route"],
                "termination_reason": entry["termination_reason"],
            }
        )
    if blocked_count:
        log_audit_event("egress_policy_blocked", blocked_source_count=blocked_count)
    return rows, skipped


def _by_agentic_category(
    rows: list[dict[str, Any]], per_question: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mean RAGAS score per metric, grouped by `agentic_category`, sample size permitting."""
    categories = sorted({r["agentic_category"] for r in rows if r["agentic_category"]})
    breakdown: dict[str, Any] = {}
    for category in categories:
        indices = [i for i, r in enumerate(rows) if r["agentic_category"] == category]
        subset_scores = [per_question[i]["scores"] for i in indices]
        metric_names = subset_scores[0].keys() if subset_scores else []
        means = {}
        for metric in metric_names:
            values = [s[metric] for s in subset_scores if s.get(metric) is not None]
            means[metric] = sum(values) / len(values) if values else None
        breakdown[category] = {"count": len(indices), "mean_scores": means}
    return breakdown


def run_agent_ragas(
    gold_path: Path,
    config_path: str | None,
    dataset_id: str,
) -> dict[str, Any]:
    """Load config/gold data, run every example through the agent, then score with RAGAS.

    Parameters
    ----------
    gold_path : Path
        Path to `data/eval/agentic_extension_gold.jsonl` (or an equivalent
        agentic-schema gold file).
    config_path : str | None
        Override config path, or None to use `config/default.yaml`.
    dataset_id : str
        Namespace to restrict every retrieval/tool call to.

    Returns
    -------
    dict[str, Any]
        `{"generated_at", "num_examples", "agent_summary", "ragas"}` --
        `agent_summary` is the same deterministic report
        `run_agent_eval.evaluate_agent` produces (routing/tool/evidence
        metrics), included so this report is self-sufficient without
        cross-referencing the separate deterministic run.
    """
    config = load_config(config_path) if config_path else load_config()
    vectorstore = build_vectorstore(config)
    embedder = build_embedder(config)
    llm = build_llm(config)
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=embedder, llm=llm)
    examples = load_gold_jsonl(gold_path)

    agent_report = evaluate_agent(
        pipeline,
        vectorstore,
        embedder,
        llm,
        examples,
        dataset_id,
        config,
        verbose=True,
        include_evidence=True,
    )
    per_example = agent_report.pop("per_example")

    judge_llm = build_judge_llm(config)
    judge_embedder = build_embedder(config)
    cache = ragas_cache.build_judge_cache(config) if config.judge.cache_enabled else None
    rows, num_skipped = _build_rows(per_example, config)
    ragas_result = ragas_scorer.score(rows, judge_llm, judge_embedder, cache=cache)
    ragas_result.pop("cache_stats", None)
    ragas_result.update(
        {
            "prompt_id": "agent_synthesize",
            "prompt_version": "v1",
            "note": "Scores the agent's actual final_answer/evidence for whichever route "
            "(classic_rag or agent) each question took -- see per-row 'route'/"
            "'termination_reason' in per_question for context. Not comparable 1:1 to "
            "run_ragas_eval.py's classic-pipeline-only scoring.",
            "judge": _judge_summary(config, judge_llm),
            "cache": _cache_summary(config, judge_llm, cache),
            "sample_size": len(examples),
            "num_scored": len(rows),
            "num_skipped_missing_expected_answer": num_skipped,
            "by_agentic_category": _by_agentic_category(rows, ragas_result["per_question"]),
        }
    )
    # Attach route/termination_reason onto each per_question row for readability.
    for pq, row in zip(ragas_result["per_question"], rows, strict=True):
        pq["agentic_category"] = row["agentic_category"]
        pq["route"] = row["route"]
        pq["termination_reason"] = row["termination_reason"]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "num_examples": len(examples),
        "agent_summary": agent_report,
        "ragas": ragas_result,
    }


def main() -> None:
    """CLI entrypoint: parse args, run `run_agent_ragas`, and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to an agentic-schema gold JSONL file")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace to restrict retrieval to. Mandatory -- applied as a filter "
        "on every retrieval/tool call this run makes.",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--verbose", action="store_true", help="Include per-question detail in the printed report"
    )
    args = parser.parse_args()

    report = run_agent_ragas(Path(args.gold), args.config, args.dataset_id)
    if not args.verbose:
        report["ragas"] = {k: v for k, v in report["ragas"].items() if k != "per_question"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
