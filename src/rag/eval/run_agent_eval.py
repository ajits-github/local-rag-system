"""Run agentic-RAG evaluation against a gold JSONL dataset.

Deterministic/local only (no RAGAS, no hosted judge) -- reports the
agent-specific metrics the Agentic RAG milestone adds on top of the
classic Recall@k/MRR/hit-rate metrics `rag.eval.run_eval` already
computes. Run both against the same corpus for a full picture; this
module intentionally does not duplicate retrieval-quality scoring.

`--dataset-id` is mandatory, applied as a `filters={"dataset_id": ...}`
constraint on every call, exactly like `run_eval.py`'s existing rule.

Usage:
    python -m rag.eval.run_agent_eval --gold data/eval/agentic_extension_gold.jsonl \
        --dataset-id techfusion
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.eval.answer_quality import KeywordOverlapScorer
from rag.eval.corpus_lineage import compute_corpus_lineage
from rag.eval.gold_schema import GoldExample, load_gold_jsonl, source_matches_relevant
from rag.eval.run_eval import _build_authorization_context
from rag.factory import build_embedder, build_llm, build_vectorstore
from rag.generation.base import LLM
from rag.retrieval.pipeline import RetrievalPipeline, source_dict
from rag.vectorstore.base import VectorStore

_BOUND_TERMINATIONS = {"max_steps", "max_retrieval_attempts", "max_tool_calls"}
_answer_quality_scorer = KeywordOverlapScorer()


def _expected_route(example: GoldExample) -> str | None:
    """Infer the expected route from a gold example's agentic-milestone flags.

    Returns `None` (not scored by `routing_accuracy`) when a question
    carries no agentic signal either way -- e.g. a plain classic-RAG gold
    row from `techfusion_gold.jsonl` reused for `tool_not_needed`
    evidence is exactly the `tool_not_needed=True` case, which this
    function does classify.
    """
    if example.tool_not_needed:
        return "classic_rag"
    if (
        example.requires_query_decomposition
        or example.requires_multiple_retrieval_calls
        or example.requires_latest_document_tool
        or example.expects_insufficient_evidence_retry
        or example.adversarial_tool_instruction
    ):
        return "agent"
    return None


def _record_for_example(
    example: GoldExample,
    *,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    llm: LLM,
    config: AppConfig,
    dataset_id: str,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Run one gold example through the agent graph and capture a scoring-ready record.

    `include_evidence`, off by default, additionally attaches an
    `"evidence_sources"` key (`retrieval.pipeline.source_dict`-shaped
    dicts for every `AgentState.retrieved_evidence` entry the run
    accumulated) -- used by `run_agent_ragas_eval.py` to build RAGAS
    contexts and run egress-policy checks over agent-gathered evidence
    without a second `run_agent` call. Every other caller/existing test
    is unaffected since the field is simply absent when this stays False.
    """
    auth = _build_authorization_context(example)
    state = AgentState(
        original_query=example.question,
        authorization_context=auth,
        filters={"dataset_id": dataset_id},
    )
    result = run_agent(
        state, pipeline=pipeline, vectorstore=vectorstore, embedder=embedder, llm=llm, config=config
    )
    final_state = result.state
    tool_names = [record.tool_name for record in final_state.tool_call_history]
    record = {
        "question": example.question,
        "agentic_category": example.agentic_category,
        "expected_route": _expected_route(example),
        "route": result.route,
        "termination_reason": final_state.termination_reason,
        "steps": final_state.step_count,
        "tool_calls": tool_names,
        "tool_call_records": [
            {"tool_name": r.tool_name, "success": r.success, "result_count": r.result_count}
            for r in final_state.tool_call_history
        ],
        "retrieval_attempts": final_state.retrieval_attempts,
        "final_answer": final_state.final_answer,
        "citations": [c.source for c in final_state.citations],
        "prompt_tokens": final_state.prompt_tokens,
        "completion_tokens": final_state.completion_tokens,
        "retrieval_ms": result.retrieval_ms,
        "generation_ms": result.generation_ms,
        "total_ms": result.total_ms,
        "expected_tool_sequence": example.expected_tool_sequence,
        "expects_insufficient_evidence_retry": example.expects_insufficient_evidence_retry,
        "expects_max_step_termination": example.expects_max_step_termination,
        "tool_not_needed": example.tool_not_needed,
        "relevant_documents": example.relevant_documents,
        "expected_answer": example.expected_answer,
        "unanswerable": example.unanswerable,
    }
    if include_evidence:
        record["evidence_sources"] = [source_dict(r) for r in final_state.retrieved_evidence]
    return record


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _routing_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r["expected_route"] is not None]
    correct = [r for r in scored if r["route"] == r["expected_route"]]
    tool_not_needed = [r for r in records if r["tool_not_needed"]]
    unnecessary = [r for r in tool_not_needed if r["route"] == "agent"]
    return {
        "routing_accuracy": {
            "count": len(scored),
            "rate": len(correct) / len(scored) if scored else None,
        },
        "unnecessary_agent_rate": {
            "count": len(tool_not_needed),
            "rate": len(unnecessary) / len(tool_not_needed) if tool_not_needed else None,
        },
    }


def _tool_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r["expected_tool_sequence"]]
    hit = 0
    for r in scored:
        expected = set(r["expected_tool_sequence"])
        actual = set(r["tool_calls"])
        if expected <= actual:
            hit += 1
    all_records = [rec for r in records for rec in r["tool_call_records"]]
    successful = [rec for rec in all_records if rec["success"]]
    agent_routed = [r for r in records if r["route"] == "agent"]
    return {
        "tool_selection_accuracy": {
            "count": len(scored),
            "rate": len(scored) and hit / len(scored),
            "note": "Simplified proxy: expected tool set is a subset of the actual tool "
            "set used, not exact-sequence match -- LLM tool ordering is not perfectly "
            "reproducible.",
        },
        "tool_success_rate": {
            "count": len(all_records),
            "rate": len(successful) / len(all_records) if all_records else None,
        },
        "average_tool_calls": {
            "count": len(agent_routed),
            "value": _mean([len(r["tool_calls"]) for r in agent_routed]),
        },
    }


def _evidence_and_retry_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    retry_expected = [r for r in records if r["expects_insufficient_evidence_retry"]]
    retried = [r for r in retry_expected if r["retrieval_attempts"] >= 2]
    retry_then_succeeded = [r for r in retried if r["termination_reason"] == "synthesized"]
    max_step_expected = [r for r in records if r["expects_max_step_termination"]]
    bound_terminated = [
        r for r in max_step_expected if r["termination_reason"] in _BOUND_TERMINATIONS
    ]
    return {
        "evidence_sufficiency_accuracy": {
            "count": len(retry_expected),
            "rate": len(retried) / len(retry_expected) if retry_expected else None,
            "note": "Proxy: retrieval_attempts >= 2 stands in for 'evidence was judged "
            "insufficient at least once', since AgentState only retains the final "
            "evidence_sufficient decision, not the full history.",
        },
        "retry_success_rate": {
            "count": len(retried),
            "rate": len(retry_then_succeeded) / len(retried) if retried else None,
        },
        "max_step_termination_rate": {
            "count": len(max_step_expected),
            "rate": len(bound_terminated) / len(max_step_expected) if max_step_expected else None,
        },
    }


def _citation_support_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r["relevant_documents"] and r["citations"]]
    supported = [
        r
        for r in scored
        if all(
            any(source_matches_relevant(c, rd) for rd in r["relevant_documents"])
            for c in r["citations"]
        )
    ]
    return {
        "count": len(scored),
        "rate": len(supported) / len(scored) if scored else None,
        "note": "Fraction of examples where every citation path-suffix-matches a "
        "relevant_documents entry -- grounding, not answer correctness.",
    }


def _answer_correctness(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        r for r in records if r["expected_answer"] and not r["unanswerable"] and r["final_answer"]
    ]
    scores = [
        _answer_quality_scorer.score(r["question"], r["final_answer"], r["expected_answer"])
        for r in scored
    ]
    return {"count": len(scored), "mean_score": _mean(scores)}


def _latency_and_tokens(records: list[dict[str, Any]]) -> dict[str, Any]:
    agent_routed = [r for r in records if r["route"] == "agent"]
    classic_routed = [r for r in records if r["route"] == "classic_rag"]
    return {
        "agent_latency_ms": {
            "overall_mean": _mean([r["total_ms"] for r in records]),
            "agent_route_mean": _mean([r["total_ms"] for r in agent_routed]),
            "classic_route_mean": _mean([r["total_ms"] for r in classic_routed]),
        },
        "agent_token_usage": {
            "mean_prompt_tokens": _mean([r["prompt_tokens"] for r in records]),
            "mean_completion_tokens": _mean([r["completion_tokens"] for r in records]),
        },
    }


def _by_agentic_category(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-category routing accuracy/average tool calls, mirroring content_type_breakdown."""
    categories = sorted({r["agentic_category"] for r in records if r["agentic_category"]})
    breakdown: dict[str, Any] = {}
    for category in categories:
        subset = [r for r in records if r["agentic_category"] == category]
        scored = [r for r in subset if r["expected_route"] is not None]
        correct = [r for r in scored if r["route"] == r["expected_route"]]
        breakdown[category] = {
            "count": len(subset),
            "routing_accuracy": len(correct) / len(scored) if scored else None,
            "average_tool_calls": _mean([len(r["tool_calls"]) for r in subset]),
        }
    return breakdown


def evaluate_agent(
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    llm: LLM,
    examples: list[GoldExample],
    dataset_id: str,
    config: AppConfig,
    verbose: bool = False,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Run every example through the agent graph and compute agent-specific metrics.

    Parameters
    ----------
    pipeline, vectorstore, embedder, llm : injected components
        Shared with the classic pipeline; `config.agent` governs routing.
    examples : list[GoldExample]
        Gold rows to evaluate -- typically `agentic_extension_gold.jsonl`,
        optionally combined with the `tool_not_needed`-eligible subset of
        `techfusion_gold.jsonl` (see module docstring).
    dataset_id : str
        Mandatory dataset namespace filter.
    config : AppConfig
        Application configuration.
    verbose : bool, optional
        Include per-example records in the report.
    include_evidence : bool, optional
        Attach each record's raw evidence source dicts (see
        `_record_for_example`) -- off by default; `run_agent_ragas_eval.py`
        turns this on to build RAGAS contexts from the same agent run
        instead of re-running the graph a second time.

    Returns
    -------
    dict[str, Any]
        Report with `num_examples` plus every agent-specific metric
        section; `per_example` only when `verbose`.
    """
    records = [
        _record_for_example(
            example,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            llm=llm,
            config=config,
            dataset_id=dataset_id,
            include_evidence=include_evidence,
        )
        for example in examples
    ]
    report: dict[str, Any] = {
        "num_examples": len(examples),
        "agent_config": {
            "enabled": config.agent.enabled,
            "max_agent_steps": config.agent.max_agent_steps,
            "max_retrieval_attempts": config.agent.max_retrieval_attempts,
            "max_tool_calls": config.agent.max_tool_calls,
        },
        **_routing_metrics(records),
        **_tool_metrics(records),
        **_evidence_and_retry_metrics(records),
        "citation_support_rate": _citation_support_rate(records),
        "agent_answer_correctness": _answer_correctness(records),
        **_latency_and_tokens(records),
        "by_agentic_category": _by_agentic_category(records),
    }
    if verbose:
        report["per_example"] = records
    return report


def run(
    gold_path: Path, config_path: str | None, dataset_id: str, corpus_version: str | None = None
) -> dict[str, Any]:
    """Load config/gold data, run `evaluate_agent`, and attach a report header."""
    config = load_config(config_path) if config_path else load_config()
    vectorstore = build_vectorstore(config)
    embedder = build_embedder(config)
    llm = build_llm(config)
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=embedder, llm=llm)
    examples = load_gold_jsonl(gold_path)
    result = evaluate_agent(
        pipeline, vectorstore, embedder, llm, examples, dataset_id, config, verbose=True
    )
    lineage = compute_corpus_lineage(
        vectorstore, dataset_id, corpus_version or "unspecified", gold_path
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_lineage": lineage,
        **result,
    }


def main() -> None:
    """CLI entrypoint: parse args, run `evaluate_agent`, and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold JSONL file")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace to restrict retrieval to. Mandatory -- applied as a filter "
        "on every retrieval/tool call this run makes.",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--corpus-version", default=None, help="Free-form corpus version label for corpus_lineage"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Include per-example detail in the printed report"
    )
    args = parser.parse_args()

    report = run(Path(args.gold), args.config, args.dataset_id, corpus_version=args.corpus_version)
    if not args.verbose:
        report = {k: v for k, v in report.items() if k != "per_example"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
