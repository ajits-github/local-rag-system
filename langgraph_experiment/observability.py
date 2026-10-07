"""Prometheus counters and per-node OpenTelemetry spans for this experiment.

Mirrors `rag.observability.metrics`'s dedicated-`CollectorRegistry`
pattern (never the process-wide default, so re-importing this module
across `pytest` collection never raises "Duplicated timeseries") as its
own independent registry. Not additional counters bolted onto
`rag/observability/metrics.py`, for the same isolation reason
`audit.py` doesn't extend `rag.audit`'s event vocabulary. Tracing itself
*is* fully reused (`rag.observability.tracing.start_span`/
`configure_tracing`): that module is generic infrastructure with no
agent-specific code in it, unlike the metrics module's `rag_*`-prefixed,
already-agent-shaped counters.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphBubbleUp
from prometheus_client import CollectorRegistry, Counter, Histogram

from rag.observability import tracing

REGISTRY = CollectorRegistry()

_LATENCY_BUCKETS_SECONDS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

WORKFLOW_RUNS_TOTAL = Counter(
    "langgraph_experiment_workflow_runs_total",
    "Total graph invocations, by route and outcome.",
    ["route", "outcome"],
    registry=REGISTRY,
)
WORKFLOW_INTERRUPTED_TOTAL = Counter(
    "langgraph_experiment_workflow_interrupted_total",
    "Total runs that paused at the approval interrupt.",
    registry=REGISTRY,
)
WORKFLOW_FAILED_TOTAL = Counter(
    "langgraph_experiment_workflow_failed_total",
    "Total runs where a node raised an uncaught exception, by node name.",
    ["node"],
    registry=REGISTRY,
)
NODE_LATENCY_SECONDS = Histogram(
    "langgraph_experiment_node_latency_seconds",
    "Per-node-invocation latency in seconds.",
    ["node"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)


def traced_node(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a node function with an OpenTelemetry span and a latency observation.

    Applied once per node at `graph.build_graph` time, rather than
    editing every node function body. The node functions in `nodes.py`
    stay pure `(state) -> dict` with no tracing/metrics code mixed in.
    Failures are never swallowed here: an exception still propagates
    (the same "understandable workflow state" contract
    `test_failed_node_state.py` relies on), just after the span is
    closed with the error recorded on it (see `tracing.start_span`).

    `wait_for_approval`'s `interrupt()` call raises `langgraph.errors.
    GraphInterrupt` internally to unwind the stack on every pause. A
    real `Exception` subclass, not a genuine node failure. Caught and
    re-raised separately, *before* the generic `except Exception` branch,
    specifically so a normal pause is never counted in
    `WORKFLOW_FAILED_TOTAL` (confirmed directly against
    `langgraph.errors`'s exception hierarchy, not assumed: `GraphInterrupt`
    subclasses `GraphBubbleUp`, itself an `Exception` subclass).

    Parameters
    ----------
    name : str
        The node name, used as both the span name and the
        `NODE_LATENCY_SECONDS`/`WORKFLOW_FAILED_TOTAL` label value.
    fn : Callable[[dict], dict]
        The real node function.

    Returns
    -------
    Callable[[dict], dict]
        A wrapped node function with identical behavior plus tracing.
    """

    def wrapped(state: dict) -> dict:
        t0 = time.perf_counter()
        try:
            with tracing.start_span(f"langgraph_experiment.{name}") as span:
                tracing.set_attributes(span, {"node": name})
                result = fn(state)
        except GraphBubbleUp:
            raise
        except Exception:
            WORKFLOW_FAILED_TOTAL.labels(node=name).inc()
            raise
        finally:
            NODE_LATENCY_SECONDS.labels(node=name).observe(time.perf_counter() - t0)
        return result

    return wrapped


def record_run_outcome(result: dict[str, Any]) -> None:
    """Increment the run-level completion/interrupt counters for one `graph.invoke()` result.

    Called once by each CLI entrypoint after `graph.invoke()` returns --
    a run-level concern, not a per-node one, mirroring
    `rag.agent.graph.run_agent`'s own `_finish()` helper observing
    run-level metrics only after the whole run completes.

    Parameters
    ----------
    result : dict[str, Any]
        The dict `graph.invoke()` returned (or the state values from
        `graph.get_state(config).values`, same shape).
    """
    if "__interrupt__" in result:
        WORKFLOW_INTERRUPTED_TOTAL.inc()
        return
    route = result.get("route") or "unknown"
    outcome = result.get("termination_reason") or "unknown"
    WORKFLOW_RUNS_TOTAL.labels(route=route, outcome=outcome).inc()
