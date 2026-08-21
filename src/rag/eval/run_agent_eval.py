"""Run agentic-RAG evaluation against a gold JSONL dataset.

Deterministic/local only (no RAGAS, no hosted judge). Reports the
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
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.agent.graph import AgentRunResult, run_agent
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

# Matches "Source 2", "Sources 1 and 3", "(Source 4)", etc.; the citation
# style `agent_synthesize_v1`/`v2` and `rag_answer_v3` both instruct the
# model to use (rule 5: '... Reference its source number, e.g. "(Source 2)"').
_SOURCE_CITATION_RE = re.compile(r"[Ss]ources?\s*[:#]?\s*((?:\d+\s*(?:,|and|&|\s)\s*)*\d+)")


def _extract_cited_source_numbers(answer_text: str) -> set[int]:
    """Parse every 1-indexed `[Source N]` number the answer text actually references."""
    numbers: set[int] = set()
    for match in _SOURCE_CITATION_RE.finditer(answer_text or ""):
        numbers.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    return numbers


def _cited_sources(final_answer: str | None, citations: list[str]) -> list[str] | None:
    """Return the citation sources the final answer text explicitly referenced by number.

    `None` (not just an empty list) when the answer contains zero
    "Source N" mentions, or every parsed number was out of range. See
    `_resolve_citation_attribution`, which falls back to
    `_infer_cited_sources` in that case rather than treating it as
    trivially well-grounded. `citations` must be in the same order the
    synthesis prompt numbered `[Source N]`. This is true for both routes:
    `rag.agent.graph._synthesize` builds `state.citations` from the same
    (trust-ordered) list it renders as context, and
    `RetrievalPipeline.answer()` does the same for `sources`.
    """
    numbers = _extract_cited_source_numbers(final_answer or "")
    if not numbers:
        return None
    resolved = [citations[n - 1] for n in sorted(numbers) if 0 < n <= len(citations)]
    return resolved or None


# qwen2.5:3b was found (see ISSUES.md's citation-compliance entry) to often
# skip the requested "(Source N)" citation format entirely on both
# routes, even while using the retrieved evidence correctly. Mirrors
# `run_eval.py`'s `_expansion_utilization` heuristic exactly (len>3-word
# overlap, same threshold): a triage-grade grounding proxy for when
# explicit citation parsing finds nothing, not a citation-compliance claim.
_MIN_INFERRED_CITATION_OVERLAP = 3


def _content_by_source(result: AgentRunResult, final_state: AgentState) -> dict[str, str]:
    """Map each distinct citation source path to its gathered chunk content, concatenated.

    Built from whichever route actually carries content: `result.
    classic_sources` (`classic_rag`, added for the RAGAS classic-route
    context gap; see `rag.agent.graph.AgentRunResult`) or `final_state.
    retrieved_evidence` (`agent`). This is the same underlying data
    `record["citations"]` itself is derived from, so every citation
    resolves to *some* content unless the source was never actually
    gathered (shouldn't happen, defensively tolerated via `.get(...)`` in
    `_infer_cited_sources`).
    """
    by_source: dict[str, list[str]] = {}
    if result.route == "classic_rag":
        for s in result.classic_sources:
            by_source.setdefault(s["source"], []).append(s.get("content") or "")
    else:
        for r in final_state.retrieved_evidence:
            by_source.setdefault(r.chunk.metadata.source, []).append(r.chunk.content)
    return {source: " ".join(texts) for source, texts in by_source.items()}


def _infer_cited_sources(final_answer: str, content_by_source: dict[str, str]) -> list[str]:
    """Keyword-overlap fallback: which gathered sources the answer likely drew from.

    Only consulted when explicit `(Source N)` parsing finds nothing (see
    `_resolve_citation_attribution`). A source counts as "inferred cited"
    when at least `_MIN_INFERRED_CITATION_OVERLAP` of its content's own
    words (length > 3) also appear in the answer text. This is the same
    style/threshold `run_eval.py`'s `_expansion_utilization` already uses
    for an analogous "did the answer draw on this content" question.
    Incidental vocabulary overlap can produce a false positive, and a miss
    doesn't prove a source was unused. It is a triage aid, not a grounding
    proof, exactly like that sibling heuristic.
    """
    answer_words = {w.lower() for w in final_answer.split() if len(w) > 3}
    if not answer_words:
        return []
    inferred = []
    for source, content in content_by_source.items():
        content_words = {w.lower() for w in content.split() if len(w) > 3}
        if len(content_words & answer_words) >= _MIN_INFERRED_CITATION_OVERLAP:
            inferred.append(source)
    return inferred


def _resolve_citation_attribution(
    final_answer: str | None,
    citations: list[str],
    result: AgentRunResult,
    final_state: AgentState,
) -> tuple[list[str] | None, str]:
    """Resolve which sources an answer is attributed to, and how confidently.

    Tries explicit `(Source N)` parsing first (`_cited_sources`); falls
    back to keyword-overlap inference (`_infer_cited_sources`) only when
    that finds nothing, since qwen2.5:3b frequently omits the citation
    format even when it did use the evidence correctly (see ISSUES.md).

    Returns
    -------
    tuple[list[str] | None, str]
        ``(sources, attribution)`` where `attribution` is one of
        `"explicit"` (parsed from the answer text), `"inferred"`
        (keyword-overlap fallback), or `"none"` (neither found anything;
        `sources` is `None` in that case).
    """
    explicit = _cited_sources(final_answer, citations)
    if explicit is not None:
        return explicit, "explicit"
    inferred = _infer_cited_sources(final_answer or "", _content_by_source(result, final_state))
    if inferred:
        return inferred, "inferred"
    return None, "none"


def _expected_route(example: GoldExample) -> str | None:
    """Infer the expected route from a gold example's agentic-milestone flags.

    Returns `None` (not scored by `routing_accuracy`) when a question
    carries no agentic signal either way, e.g. a plain classic-RAG gold
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
    accumulated), used by `run_agent_ragas_eval.py` to build RAGAS
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
    citation_sources = [c.source for c in final_state.citations]
    cited_sources, citation_attribution = _resolve_citation_attribution(
        final_state.final_answer, citation_sources, result, final_state
    )
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
        "citations": citation_sources,
        "cited_sources": cited_sources,
        "citation_attribution": citation_attribution,
        "prompt_tokens": final_state.prompt_tokens,
        "completion_tokens": final_state.completion_tokens,
        "retrieval_ms": result.retrieval_ms,
        "generation_ms": result.generation_ms,
        "total_ms": result.total_ms,
        "node_timings_ms": {k: v.model_dump() for k, v in result.node_timings_ms.items()},
        "llm_call_count": result.llm_call_count,
        "node_token_usage": result.node_token_usage,
        "expected_tool_sequence": example.expected_tool_sequence,
        "expects_insufficient_evidence_retry": example.expects_insufficient_evidence_retry,
        "expects_max_step_termination": example.expects_max_step_termination,
        "tool_not_needed": example.tool_not_needed,
        "relevant_documents": example.relevant_documents,
        "expected_answer": example.expected_answer,
        "unanswerable": example.unanswerable,
    }
    if include_evidence:
        record["evidence_sources"] = (
            result.classic_sources
            if result.route == "classic_rag"
            else [source_dict(r) for r in final_state.retrieved_evidence]
        )
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
            "note": "Strict metric: 1.0 only when the ENTIRE gold expected_tool_sequence "
            "set is a subset of the actual tool set used, 0.0 otherwise -- not "
            "exact-sequence match, but still an all-or-nothing gate per example. Reads "
            "as 0.0 whenever the agent's bounded tool budget can't cover a 3-4-tool "
            "expected set even if every tool it did call was correct; see "
            "'tool_selection_coverage' for graded precision/recall metrics that don't "
            "collapse to 0 on a partial-but-sensible match.",
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


def _tool_selection_coverage_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Graded tool-selection metrics that supplement `tool_selection_accuracy`'s strict gate.

    `expected_tool_sequence` is treated as an unordered set here. No row
    in the current gold file marks a strict-ordering requirement, so an
    exact-sequence-match metric would report a meaningless 0/0 today; add
    one alongside this function if a future gold row ever needs it.
    Computed per example, then macro-averaged (each example weighted
    equally regardless of its expected-tool-set size).
    """
    scored = [r for r in records if r["expected_tool_sequence"]]
    precisions: list[float] = []
    recalls: list[float] = []
    unexpected_rates: list[float] = []
    for r in scored:
        expected = set(r["expected_tool_sequence"])
        actual = r["tool_calls"]
        recalls.append(len(expected & set(actual)) / len(expected) if expected else 1.0)
        if actual:
            hit = sum(1 for t in actual if t in expected)
            precisions.append(hit / len(actual))
            unexpected_rates.append(1 - hit / len(actual))
    return {
        "tool_selection_coverage": {
            "expected_tool_precision": {
                "count": len(precisions),
                "mean": _mean(precisions),
                "note": "Mean fraction of each example's actual tool calls that were in "
                "the gold expected_tool_sequence set.",
            },
            "required_tool_coverage": {
                "count": len(recalls),
                "mean": _mean(recalls),
                "note": "Mean fraction of each example's gold expected_tool_sequence set "
                "that was actually called at least once (recall).",
            },
            "unexpected_tool_rate": {
                "count": len(unexpected_rates),
                "mean": _mean(unexpected_rates),
                "note": "Mean fraction of each example's actual tool calls that were NOT "
                "in the gold expected_tool_sequence set.",
            },
        }
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
    """Fraction of examples whose attributed sources all path-match a gold relevant document.

    Scores `record["cited_sources"]`, the sources the final answer text
    was actually *attributed* to, not every chunk a run happened to
    gather across every tool call. See the "citation-list scope
    mismatch" finding in experiments/reports/agentic_rag_baseline_v1.md
    section 7 and ISSUES.md. `state.citations` (all gathered evidence) is
    a superset of what a 2-tool-call agent run's synthesis prompt
    actually drew from, so scoring against it made an otherwise
    well-grounded answer fail the moment any single tangential chunk was
    ever retrieved.

    Attribution itself is two-tier (`record["citation_attribution"]`,
    `_resolve_citation_attribution`): "explicit" when the answer text
    parses at least one "Source N" mention, else "inferred" via a
    keyword-overlap fallback against gathered-source content (qwen2.5:3b
    was found to often skip the citation format even when it used the
    evidence correctly; see ISSUES.md), else "none". An example with
    "none" attribution is excluded from the denominator (reported
    separately as `uncited_answer_count`), not counted as a pass or a
    fail. We still cannot determine what such an answer grounded on,
    and silently treating it as "0 citations, trivially supported" would
    inflate the metric.

    Not comparable to `experiment_029`'s `citation_support_rate` (the
    pre-fix all-gathered-evidence definition) or `experiment_032`'s
    (explicit-only, no inference fallback). See this function's own
    change note in the returned dict.
    """
    eligible = [r for r in records if r["relevant_documents"]]
    scored = [r for r in eligible if r["cited_sources"]]
    uncited = [r for r in eligible if r["citation_attribution"] == "none"]
    explicit = [r for r in scored if r["citation_attribution"] == "explicit"]
    inferred = [r for r in scored if r["citation_attribution"] == "inferred"]
    supported = [
        r
        for r in scored
        if all(
            any(source_matches_relevant(c, rd) for rd in r["relevant_documents"])
            for c in r["cited_sources"]
        )
    ]
    return {
        "count": len(scored),
        "rate": len(supported) / len(scored) if scored else None,
        "explicit_count": len(explicit),
        "inferred_count": len(inferred),
        "uncited_answer_count": len(uncited),
        "note": "Fraction of examples where every attributed source (explicit 'Source N' "
        "citation, or -- only when the answer has none -- a keyword-overlap-inferred "
        "source) path-suffix-matches a relevant_documents entry -- grounding of the "
        "attributed evidence, not answer correctness. Definition changed again from "
        "experiment_032 (explicit-only) by adding the inferred fallback; not directly "
        "comparable to either that run's or experiment_029's value. uncited_answer_count "
        "tracks answers where even the fallback found nothing, excluded from this rate's "
        "denominator.",
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


def _node_latency_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-node-type timing across every example, splitting LLM vs. Overhead time.

    Reads the additive `AgentRunResult.node_timings_ms`/`node_token_usage`
    instrumentation from the observability milestone. This is purely a reporting
    aggregation, no agent decision logic involved. Empty for a run where
    every question took the `classic_rag` fast path (no per-node structure
    on that route; see `AgentRunResult`'s own docstring).
    """
    node_names = sorted({name for r in records for name in r["node_timings_ms"]})
    breakdown: dict[str, Any] = {}
    for name in node_names:
        stats_list = [r["node_timings_ms"][name] for r in records if name in r["node_timings_ms"]]
        total_count = sum(s["count"] for s in stats_list)
        total_ms = sum(s["total_ms"] for s in stats_list)
        llm_ms_total = sum(
            s["llm_ms_mean"] * s["count"] for s in stats_list if s.get("llm_ms_mean") is not None
        )
        overhead_ms_total = sum(
            s["overhead_ms_mean"] * s["count"]
            for s in stats_list
            if s.get("overhead_ms_mean") is not None
        )
        llm_count = sum(s["count"] for s in stats_list if s.get("llm_ms_mean") is not None)
        breakdown[name] = {
            "questions_using_node": len(stats_list),
            "invocation_count": total_count,
            "total_ms": total_ms,
            "mean_ms_per_invocation": total_ms / total_count if total_count else None,
            "llm_ms_mean_per_invocation": llm_ms_total / llm_count if llm_count else None,
            "overhead_ms_mean_per_invocation": overhead_ms_total / llm_count if llm_count else None,
        }
    return breakdown


def _node_token_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-node-type prompt/completion token totals across every example."""
    node_names = sorted({name for r in records for name in r["node_token_usage"]})
    return {
        name: {
            "prompt_tokens_total": sum(
                r["node_token_usage"].get(name, {}).get("prompt", 0) for r in records
            ),
            "completion_tokens_total": sum(
                r["node_token_usage"].get(name, {}).get("completion", 0) for r in records
            ),
        }
        for name in node_names
    }


def _llm_call_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Total and per-route-mean `LLM.generate()` call counts across every example."""
    agent_routed = [r for r in records if r["route"] == "agent"]
    return {
        "total_llm_calls": sum(r["llm_call_count"] for r in records),
        "mean_llm_calls_agent_route": _mean([r["llm_call_count"] for r in agent_routed]),
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
        Gold rows to evaluate, typically `agentic_extension_gold.jsonl`,
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
        `_record_for_example`). Off by default; `run_agent_ragas_eval.py`
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
        **_tool_selection_coverage_metrics(records),
        **_evidence_and_retry_metrics(records),
        "citation_support_rate": _citation_support_rate(records),
        "agent_answer_correctness": _answer_correctness(records),
        **_latency_and_tokens(records),
        "node_latency_breakdown_ms": _node_latency_breakdown(records),
        "node_token_breakdown": _node_token_breakdown(records),
        **_llm_call_summary(records),
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
