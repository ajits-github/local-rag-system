"""Prometheus metrics: bounded label sets, a dedicated registry, defensive recording.

Uses its own `CollectorRegistry` rather than `prometheus_client`'s
process-wide default, so re-importing this module (e.g. across `pytest`
test collection) never raises a "Duplicated timeseries" error.

Every label is drawn from a small, fixed vocabulary: `tool_name` and
`node` are fixed literal names (`observe_tool_call` takes a plain `str`,
but every call site passes one of the fixed tool names, never an
arbitrary argument), `route`/`reason` are fixed enums, and
`method`/`path`/`status_code` come from registered HTTP routes, never the
raw request URL. No query text, tenant id, document id, or chunk id is
ever used as a label value.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()

_LATENCY_BUCKETS_SECONDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
_COUNT_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)

HTTP_REQUESTS_TOTAL = Counter(
    "rag_http_requests_total",
    "Total HTTP requests handled.",
    ["method", "path", "status_code"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)

AGENT_REQUESTS_TOTAL = Counter(
    "rag_agent_requests_total",
    "Total agent-graph runs, by route actually taken.",
    ["route"],
    registry=REGISTRY,
)
AGENT_TOTAL_LATENCY_SECONDS = Histogram(
    "rag_agent_total_latency_seconds",
    "Total agent-graph run latency in seconds, by route.",
    ["route"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)
AGENT_STEPS = Histogram(
    "rag_agent_steps",
    "Agent graph node-execution steps per run.",
    buckets=_COUNT_BUCKETS,
    registry=REGISTRY,
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "rag_agent_tool_calls_total",
    "Total agent tool dispatches, by tool name and outcome.",
    ["tool_name", "success"],
    registry=REGISTRY,
)
AGENT_TOOL_LATENCY_SECONDS = Histogram(
    "rag_agent_tool_latency_seconds",
    "Agent tool dispatch latency in seconds, by tool name.",
    ["tool_name"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)

AGENT_NODE_LATENCY_SECONDS = Histogram(
    "rag_agent_node_latency_seconds",
    "Agent graph node total wall-clock latency in seconds, by node.",
    ["node"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)
AGENT_NODE_LLM_LATENCY_SECONDS = Histogram(
    "rag_agent_node_llm_latency_seconds",
    "Agent graph node LLM-inference-only latency in seconds, by node "
    "(the subset of node latency spent inside llm.generate(), excluding "
    "JSON parsing/validation/template-rendering overhead).",
    ["node"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "rag_retrieval_latency_seconds",
    "RetrievalPipeline retrieval latency in seconds, by provider.",
    ["provider"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)

AUTHORIZATION_DENIALS_TOTAL = Counter(
    "rag_authorization_denials_total",
    "Retrieval queries where the authorization predicate excluded at least "
    "one otherwise-matching chunk (detected only on a thin/empty result set; "
    "see RetrievalPipeline._retrieve_timed_inner).",
    registry=REGISTRY,
)

AGENT_TERMINATION_REASON_TOTAL = Counter(
    "rag_agent_termination_reason_total",
    "Agent run termination reasons.",
    ["reason"],
    registry=REGISTRY,
)
AGENT_EVIDENCE_SUFFICIENCY_TOTAL = Counter(
    "rag_agent_evidence_sufficiency_total",
    "evaluate_evidence decisions, by outcome.",
    ["sufficient"],
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Errors by component.",
    ["component"],
    registry=REGISTRY,
)

GRAPH_QUERY_TOTAL = Counter(
    "rag_graph_query_total",
    "Total graph retrieval queries, by intent and outcome.",
    ["intent", "success"],
    registry=REGISTRY,
)
GRAPH_QUERY_LATENCY_SECONDS = Histogram(
    "rag_graph_query_latency_seconds",
    "Graph retrieval query latency in seconds, by intent.",
    ["intent"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)
GRAPH_FACTS_RETURNED = Histogram(
    "rag_graph_facts_returned",
    "Number of graph facts returned per graph retrieval query, by intent.",
    ["intent"],
    buckets=_COUNT_BUCKETS,
    registry=REGISTRY,
)
GRAPH_RETRIEVAL_FAILURES_TOTAL = Counter(
    "rag_graph_retrieval_failures_total",
    "Graph retrieval failures (Neo4j unavailable, query error, generated-Cypher "
    "rejection), by reason.",
    ["reason"],
    registry=REGISTRY,
)
GRAPHRAG_ROUTE_TOTAL = Counter(
    "rag_graphrag_route_total",
    "Answer requests, by which evidence source(s) the configured GraphRAG "
    "fusion strategy actually used.",
    ["strategy"],
    registry=REGISTRY,
)
GRAPH_EVIDENCE_IN_CONTEXT = Histogram(
    "rag_graph_evidence_in_context",
    "Number of graph-origin SearchResults that reached the generation context for one answer.",
    buckets=_COUNT_BUCKETS,
    registry=REGISTRY,
)

CONTEXT_CANDIDATE_TOKENS = Histogram(
    "rag_context_candidate_tokens",
    "Total candidate evidence tokens gathered before Context Engineering dedup/packing.",
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
    registry=REGISTRY,
)
CONTEXT_PACKED_TOKENS = Histogram(
    "rag_context_packed_tokens",
    "Total tokens actually packed into the final rendered context.",
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
    registry=REGISTRY,
)
CONTEXT_DROPPED_ITEMS_TOTAL = Counter(
    "rag_context_dropped_items_total",
    "Candidate context items excluded by packing, by reason.",
    ["reason"],
    registry=REGISTRY,
)
CONTEXT_DEDUPLICATED_ITEMS_TOTAL = Counter(
    "rag_context_deduplicated_items_total",
    "Candidate context items collapsed by deduplication, by match kind.",
    ["match_kind"],
    registry=REGISTRY,
)
CONTEXT_COMPRESSED_ITEMS_TOTAL = Counter(
    "rag_context_compressed_items_total",
    "Context items rewritten by compression before packing.",
    registry=REGISTRY,
)
CONTEXT_BUILD_LATENCY_SECONDS = Histogram(
    "rag_context_build_latency_seconds",
    "Time spent in the Context Engineering build pipeline (normalize through packing).",
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)
CONTEXT_OVERFLOW_PREVENTED_TOTAL = Counter(
    "rag_context_overflow_prevented_total",
    "Requests where packing had to drop or compress at least one eligible item to stay "
    "within the token budget (i.e. graceful degradation actually engaged).",
    registry=REGISTRY,
)
CONTEXT_UTILIZATION_RATIO = Histogram(
    "rag_context_utilization_ratio",
    "packed_context_tokens / available_context_budget, by packing strategy.",
    ["strategy"],
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    registry=REGISTRY,
)

FEEDBACK_SUBMISSIONS_TOTAL = Counter(
    "rag_feedback_submissions_total",
    "Total POST /feedback submissions, by rating and outcome.",
    ["rating", "outcome"],
    registry=REGISTRY,
)
FEEDBACK_SUBMISSION_LATENCY_SECONDS = Histogram(
    "rag_feedback_submission_latency_seconds",
    "POST /feedback handling latency in seconds (store write only).",
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)

_F = TypeVar("_F", bound=Callable[..., None])


def _defensive(fn: _F) -> _F:
    """Wrap a metric-recording function so it can never raise.

    A broken/misbehaving metric object must never surface as a request
    failure; any exception is logged once and swallowed.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.warning("Failed to record metric via %s", fn.__name__, exc_info=True)

    return wrapper  # type: ignore[return-value]


@_defensive
def observe_http_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Record one completed HTTP request."""
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=str(status_code)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration_seconds)


@_defensive
def observe_agent_request(route: str, total_seconds: float, step_count: int) -> None:
    """Record one completed `run_agent()` call."""
    AGENT_REQUESTS_TOTAL.labels(route=route).inc()
    AGENT_TOTAL_LATENCY_SECONDS.labels(route=route).observe(total_seconds)
    AGENT_STEPS.observe(step_count)


@_defensive
def observe_tool_call(tool_name: str, success: bool, latency_seconds: float) -> None:
    """Record one agent tool dispatch."""
    AGENT_TOOL_CALLS_TOTAL.labels(tool_name=tool_name, success=str(success).lower()).inc()
    AGENT_TOOL_LATENCY_SECONDS.labels(tool_name=tool_name).observe(latency_seconds)


@_defensive
def observe_node_latency(node: str, total_seconds: float, llm_seconds: float | None) -> None:
    """Record one agent graph node invocation's timing."""
    AGENT_NODE_LATENCY_SECONDS.labels(node=node).observe(total_seconds)
    if llm_seconds is not None:
        AGENT_NODE_LLM_LATENCY_SECONDS.labels(node=node).observe(llm_seconds)


@_defensive
def observe_retrieval_latency(provider: str, latency_seconds: float) -> None:
    """Record one `RetrievalPipeline` retrieval call."""
    RETRIEVAL_LATENCY_SECONDS.labels(provider=provider).observe(latency_seconds)


@_defensive
def observe_authorization_denial() -> None:
    """Record one retrieval query where authorization excluded at least one matching chunk."""
    AUTHORIZATION_DENIALS_TOTAL.inc()


@_defensive
def observe_termination_reason(reason: str) -> None:
    """Record one agent run's termination reason."""
    AGENT_TERMINATION_REASON_TOTAL.labels(reason=reason).inc()


@_defensive
def observe_evidence_sufficiency(sufficient: bool) -> None:
    """Record one `evaluate_evidence` decision outcome."""
    AGENT_EVIDENCE_SUFFICIENCY_TOTAL.labels(sufficient=str(sufficient).lower()).inc()


@_defensive
def observe_error(component: str) -> None:
    """Record one error, by originating component."""
    ERRORS_TOTAL.labels(component=component).inc()


@_defensive
def observe_graph_query(intent: str, latency_seconds: float, fact_count: int) -> None:
    """Record one `GraphRetriever.search` call.

    Parameters
    ----------
    intent : str
        One of `rag.graph.retrieval.GraphQueryIntent`'s fixed values
        (e.g. `"dependents_of"`, `"impact_of"`, `"unresolved"`), never a
        raw query string or entity id.
    latency_seconds : float
        Call duration.
    fact_count : int
        Number of `GraphFact`s returned.
    """
    GRAPH_QUERY_TOTAL.labels(
        intent=intent, success=str(fact_count > 0 or intent == "unresolved").lower()
    ).inc()
    GRAPH_QUERY_LATENCY_SECONDS.labels(intent=intent).observe(latency_seconds)
    GRAPH_FACTS_RETURNED.labels(intent=intent).observe(fact_count)


@_defensive
def observe_graph_retrieval_failure(reason: str) -> None:
    """Record one graph retrieval failure, by a fixed, bounded reason string."""
    GRAPH_RETRIEVAL_FAILURES_TOTAL.labels(reason=reason).inc()


@_defensive
def observe_graphrag_route(strategy: str) -> None:
    """Record one answer request's `config.graph.fusion.strategy`."""
    GRAPHRAG_ROUTE_TOTAL.labels(strategy=strategy).inc()


@_defensive
def observe_graph_evidence_in_context(count: int) -> None:
    """Record how many graph-origin `SearchResult`s reached one answer's generation context."""
    GRAPH_EVIDENCE_IN_CONTEXT.observe(count)


@_defensive
def observe_context_build(
    *,
    candidate_tokens: int,
    packed_tokens: int,
    build_seconds: float,
    strategy: str,
    utilization: float,
    dropped_by_reason: dict[str, int],
    dedup_by_kind: dict[str, int],
    compressed_item_count: int,
    overflow_prevented: bool,
) -> None:
    """Record one Context Engineering build (`rag.context.engine.ContextEngine.build`).

    Parameters
    ----------
    candidate_tokens, packed_tokens : int
        Pre-dedup candidate token total and final packed token total.
    build_seconds : float
        Wall-clock time for the whole build pipeline.
    strategy : str
        `config.context.packing_strategy` (a fixed, bounded vocabulary).
    utilization : float
        `packed_tokens / available_context_tokens`, already computed by
        the caller (never recomputed here, so this function can never
        divide by zero).
    dropped_by_reason : dict[str, int]
        Counts from `PackingTrace`, keyed by its fixed `reason` strings.
    dedup_by_kind : dict[str, int]
        `{"exact": n, "near": n}`.
    compressed_item_count : int
    overflow_prevented : bool
        Whether at least one eligible item had to be dropped or
        compressed to respect the budget.
    """
    CONTEXT_CANDIDATE_TOKENS.observe(candidate_tokens)
    CONTEXT_PACKED_TOKENS.observe(packed_tokens)
    CONTEXT_BUILD_LATENCY_SECONDS.observe(build_seconds)
    CONTEXT_UTILIZATION_RATIO.labels(strategy=strategy).observe(utilization)
    for reason, count in dropped_by_reason.items():
        CONTEXT_DROPPED_ITEMS_TOTAL.labels(reason=reason).inc(count)
    for kind, count in dedup_by_kind.items():
        CONTEXT_DEDUPLICATED_ITEMS_TOTAL.labels(match_kind=kind).inc(count)
    if compressed_item_count:
        CONTEXT_COMPRESSED_ITEMS_TOTAL.inc(compressed_item_count)
    if overflow_prevented:
        CONTEXT_OVERFLOW_PREVENTED_TOTAL.inc()


@_defensive
def observe_feedback_submission(rating: str, outcome: str, latency_seconds: float) -> None:
    """Record one `POST /feedback` submission.

    Parameters
    ----------
    rating : str
        `"positive"` or `"negative"` (a fixed, bounded vocabulary).
    outcome : str
        `"created"`, `"updated"`, or `"failed"` (a fixed, bounded
        vocabulary). Never the request's query/comment/tenant id, which
        would be unbounded-cardinality label values.
    latency_seconds : float
        Time spent in the feedback-store write.
    """
    FEEDBACK_SUBMISSIONS_TOTAL.labels(rating=rating, outcome=outcome).inc()
    FEEDBACK_SUBMISSION_LATENCY_SECONDS.observe(latency_seconds)


def render_metrics() -> bytes:
    """Render every metric in `REGISTRY` as Prometheus text exposition.

    Returns
    -------
    bytes
        The exposition-format payload for `GET /metrics`.
    """
    return generate_latest(REGISTRY)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "REGISTRY",
    "observe_agent_request",
    "observe_authorization_denial",
    "observe_context_build",
    "observe_error",
    "observe_evidence_sufficiency",
    "observe_feedback_submission",
    "observe_graph_evidence_in_context",
    "observe_graph_query",
    "observe_graph_retrieval_failure",
    "observe_graphrag_route",
    "observe_http_request",
    "observe_node_latency",
    "observe_retrieval_latency",
    "observe_termination_reason",
    "observe_tool_call",
    "render_metrics",
]
