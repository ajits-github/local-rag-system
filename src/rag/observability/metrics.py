"""Prometheus metrics: bounded label sets, a dedicated registry, defensive recording.

A dedicated `CollectorRegistry` (never `prometheus_client`'s process-wide
default) so re-importing this module. Which happens routinely across
`pytest`'s test collection. Never hits prometheus_client's "Duplicated
timeseries in CollectorRegistry" error the default registry is prone to.

Every metric here has an explicitly bounded label set (checked against
this milestone's requirement directly): `tool_name` (4 literal agent
tools plus, since the MCP milestone, 2 literal MCP-only business-case
tools -- `observe_tool_call` takes a plain `str`, but every call site is
one of these 6 fixed literals, never an arbitrary tool argument),
`node` (6 literal graph nodes), `route` (`classic_rag`/`agent`), `reason`
(5 literal termination reasons), `method`/`path`/`status_code` (a fixed
set of registered HTTP routes. Never the raw request URL). Never a
query string, tenant id, document id, chunk id, or arbitrary tool
argument.
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
    "observe_error",
    "observe_evidence_sufficiency",
    "observe_http_request",
    "observe_node_latency",
    "observe_retrieval_latency",
    "observe_termination_reason",
    "observe_tool_call",
    "render_metrics",
]
