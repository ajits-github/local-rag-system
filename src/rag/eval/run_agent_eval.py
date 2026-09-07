"""Run agentic-RAG evaluation against a gold JSONL dataset.

Deterministic/local only (no RAGAS, no hosted judge). Reports
agent-specific metrics (routing, tool selection, evidence sufficiency,
citation support) on top of the classic Recall@k/MRR/hit-rate metrics
`rag.eval.run_eval` already computes; run both against the same corpus
for a full picture.

`--dataset-id` is mandatory, applied as a `filters={"dataset_id": ...}`
constraint on every call, matching `run_eval.py`'s rule.

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

import anyio

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

# AgentState.termination_reason's fixed Literal vocabulary (rag.agent.state).
# Hardcoded, not introspected, matching this module's existing
# _AGENT_PROMPT_FIELDS-style convention of a small fixed list kept in sync
# by hand; a genuinely new termination reason would need a code change
# here regardless of how it's discovered.
_TERMINATION_REASONS = [
    "synthesized",
    "max_steps",
    "max_retrieval_attempts",
    "max_tool_calls",
    "insufficient_evidence",
]

# ToolCallRecord.tool_name's fixed Literal vocabulary (rag.agent.state).
# Same fixed-list convention as _TERMINATION_REASONS above: bounded and
# small, so flattening a per-tool-name breakdown into named fields never
# risks the "high-cardinality metric name" problem a per-query or
# per-document-id breakdown would.
_TOOL_NAMES = [
    "search_knowledge_base",
    "get_document",
    "get_latest_document",
    "get_related_context",
    "get_customer_case",
    "get_case_status",
    "update_case_status",
]

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


# qwen2.5:3b frequently omits the requested "(Source N)" citation format
# even while using the retrieved evidence correctly, so explicit citation
# parsing alone under-counts grounding. Mirrors run_eval.py's
# `_expansion_utilization` heuristic (same word-length/overlap
# threshold): a triage-grade proxy, not a citation-compliance claim.
_MIN_INFERRED_CITATION_OVERLAP = 3


def _content_by_source(result: AgentRunResult, final_state: AgentState) -> dict[str, str]:
    """Map each distinct citation source path to its gathered chunk content, concatenated.

    Built from whichever route actually carries content:
    `result.classic_sources` for `classic_rag`, or
    `final_state.retrieved_evidence` for `agent`. Every citation should
    resolve to some content unless the source was never actually gathered
    (shouldn't happen, defensively tolerated via `.get(...)` in
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
    words (length > 3) also appear in the answer text, the same style/
    threshold `run_eval.py`'s `_expansion_utilization` uses for an
    analogous question. Incidental vocabulary overlap can produce a false
    positive, and a miss doesn't prove a source was unused: a triage aid,
    not a grounding proof.
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
    format even when it did use the evidence correctly.

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
    """Infer the expected route from a gold example's agentic-signal flags.

    Returns `None` (excluded from `routing_accuracy` scoring) when a
    question carries no agentic signal either way, e.g. a plain
    classic-RAG-only gold row with no `tool_not_needed`/decomposition/
    multi-retrieval/latest-document/retry/adversarial flag set.
    """
    if example.tool_not_needed:
        return "classic_rag"
    if (
        example.requires_query_decomposition
        or example.requires_multiple_retrieval_calls
        or example.requires_latest_document_tool
        or example.expects_insufficient_evidence_retry
        or example.adversarial_tool_instruction
        or example.requires_specialized_tool
        or example.requires_mcp_business_tool
    ):
        return "agent"
    return None


def _render_case_action_outcome(evidence_text: str) -> str | None:
    """Classify an update_case_status outcome from its own deterministic evidence wording.

    Matches the exact, fixed phrasings `rag.agent.mcp_client.
    _render_case_action_outcome` produces for each `CaseActionOutcome.
    outcome` value -- never a semantic/LLM judgment, since the tool's own
    output text is itself a deterministic function of the outcome type.
    Returns `None` when no known phrasing is found (e.g. the tool was
    never called, or was denied outright with no evidence at all).
    """
    if "requires approval before it can be applied" in evidence_text:
        return "approval_required"
    if "is not a valid transition" in evidence_text:
        return "invalid_transition"
    if "no change was made" in evidence_text and "already" in evidence_text:
        return "already_in_status"
    if "status was changed from" in evidence_text:
        return "executed"
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
    mcp_app: Any | None = None,
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
        state,
        pipeline=pipeline,
        vectorstore=vectorstore,
        embedder=embedder,
        llm=llm,
        config=config,
        mcp_app=mcp_app,
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

    Scores `record["cited_sources"]`: the sources the final answer text
    was actually *attributed* to, not every chunk a run happened to
    gather across every tool call. `state.citations` (all gathered
    evidence) is a superset of what a run's synthesis prompt actually
    drew from, so scoring against it would fail an otherwise well-grounded
    answer the moment any single tangential chunk was ever retrieved.

    Attribution is two-tier (`record["citation_attribution"]`,
    `_resolve_citation_attribution`): "explicit" when the answer text
    parses at least one "Source N" mention, else "inferred" via a
    keyword-overlap fallback against gathered-source content, else
    "none". An example with "none" attribution is excluded from the
    denominator (reported separately as `uncited_answer_count`) rather
    than counted as a pass or a fail, since there is no way to determine
    what such an answer was grounded on.
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
    """Aggregate per-node-type timing across every example, splitting LLM vs. overhead time.

    Purely a reporting aggregation over `AgentRunResult.node_timings_ms`/
    `node_token_usage`; no agent decision logic involved. Empty for a run
    where every question took the `classic_rag` fast path (no per-node
    structure on that route).
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


def _termination_reason_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-termination-reason counts/rates across every agent-routed example.

    Scoped to `route == "agent"` records only, matching
    `_latency_and_tokens`'s `agent_routed` scoping: `_run_classic_rag`
    also stamps `termination_reason = "synthesized"` on its single fast
    path, so including classic-routed rows would just dilute this
    agent-loop-specific breakdown by whatever fraction of the gold set
    never entered the bounded loop at all -- `_routing_metrics` already
    reports that split. `guardrail_termination_count`/`_rate` reuses the
    same `_BOUND_TERMINATIONS` set `_evidence_and_retry_metrics` scores
    `max_step_termination_rate` against, generalized here to every
    agent-routed example rather than only the gold-flagged subset.
    """
    agent_routed = [r for r in records if r["route"] == "agent"]
    total = len(agent_routed)
    counts = dict.fromkeys(_TERMINATION_REASONS, 0)
    for r in agent_routed:
        reason = r["termination_reason"]
        if reason in counts:
            counts[reason] += 1
    guardrail_count = sum(counts[reason] for reason in _BOUND_TERMINATIONS)
    return {
        "termination_reason_breakdown": {
            "count": total,
            "by_reason": {
                reason: {"count": n, "rate": (n / total if total else None)}
                for reason, n in counts.items()
            },
            "guardrail_termination_count": guardrail_count,
            "guardrail_termination_rate": guardrail_count / total if total else None,
        }
    }


def _tool_usage_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tool-name dispatch counts/rates across every agent-routed example's tool calls.

    Scoped to `route == "agent"` records (`classic_rag` never dispatches a
    tool). Counts every dispatch, successful or not, toward `count`;
    `success_rate` is scoped to that tool's own dispatches. Tool names are
    `_TOOL_NAMES`'s fixed, bounded set, so this never grows with the
    number of examples an eval run happens to score.
    """
    all_records = [rec for r in records if r["route"] == "agent" for rec in r["tool_call_records"]]
    total = len(all_records)
    by_tool: dict[str, list[dict[str, Any]]] = {name: [] for name in _TOOL_NAMES}
    for rec in all_records:
        by_tool.setdefault(rec["tool_name"], []).append(rec)
    return {
        "tool_usage_breakdown": {
            "count": total,
            "by_tool": {
                name: {
                    "count": len(recs),
                    "rate": (len(recs) / total if total else None),
                    "success_rate": (
                        sum(1 for r in recs if r["success"]) / len(recs) if recs else None
                    ),
                }
                for name, recs in by_tool.items()
            },
        }
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
    mcp_app: Any | None = None,
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
    mcp_app : Any | None, optional
        The in-process MCP ASGI app object (see
        `rag.mcp.asgi.build_mcp_asgi_app`), forwarded to `run_agent()` for
        every example so the remote business tools
        (`get_customer_case`/`get_case_status`/`update_case_status`) are
        actually reachable when `config.mcp.client.enabled=True`. Its
        lifespan must already be entered by the caller (see `run()`);
        `None` when `mcp.client.enabled=False`, matching every other
        caller of `run_agent()`.

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
            mcp_app=mcp_app,
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
        **_termination_reason_breakdown(records),
        **_tool_usage_breakdown(records),
        **_llm_call_summary(records),
        "by_agentic_category": _by_agentic_category(records),
    }
    if verbose:
        report["per_example"] = records
    return report


async def _evaluate_agent_with_mcp_lifespan(mcp_app: Any, run_eval: Any) -> dict[str, Any]:
    """Enter `mcp_app`'s lifespan once for the whole eval run, then call `run_eval` in a thread.

    Mirrors how `rag.api.main` enters the same lifespan once for the
    life of the whole process, rather than once per request/tool call.
    `run_eval` (a zero-arg closure over `evaluate_agent`) runs in a
    worker thread since `run_agent()`'s own remote-tool dispatch bridges
    to async via a fresh `anyio.run()` per call, which cannot nest inside
    an already-running event loop on the same thread.
    """
    async with mcp_app.router.lifespan_context(mcp_app):
        return await anyio.to_thread.run_sync(run_eval)


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

    def _run_eval(mcp_app: Any | None = None) -> dict[str, Any]:
        return evaluate_agent(
            pipeline,
            vectorstore,
            embedder,
            llm,
            examples,
            dataset_id,
            config,
            verbose=True,
            mcp_app=mcp_app,
        )

    if config.mcp.client.enabled:
        # Mirrors rag.api.deps.get_mcp_asgi_app(): the same in-process server
        # object the agent's MCP client dispatches remote business-tool calls
        # against, so get_customer_case/get_case_status/update_case_status are
        # actually reachable from this CLI harness, not just from the live API.
        from rag.mcp.asgi import build_mcp_asgi_app

        mcp_app = build_mcp_asgi_app(config, pipeline, vectorstore, embedder)
        result = anyio.run(_evaluate_agent_with_mcp_lifespan, mcp_app, lambda: _run_eval(mcp_app))
    else:
        result = _run_eval()

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
