"""Compare all three architectures: custom harness, single-agent LangGraph, multi-agent LangGraph.

Usage
-----
    python -m multi_agent_experiment.cli.compare_harness rag "<query>" [--dataset-id ID]
    python -m multi_agent_experiment.cli.compare_harness mixed "<query>" [--dataset-id ID]
    python -m multi_agent_experiment.cli.compare_harness write-action

`rag` times a pure-knowledge question through all three: (A) `rag.agent.
graph.run_agent` (`route="classic_rag"`), (B) `langgraph_experiment`'s
`read_only` branch, (C) this package's `knowledge_only` route -- needs a
running Postgres (`make up`) and native Ollama, same as `langgraph_
experiment.cli.compare_harness rag`.

`mixed` times a real mixed (knowledge + business) question through the
multi-agent graph's actual parallel dispatch, then estimates what a
*sequential* dispatch would have cost by summing two separate,
independently-timed knowledge-only and business-only runs against the
same query's components -- see `cmd_mixed`'s docstring for exactly why
this is an honest estimate, not a fabricated one. Architectures A and B
have no equivalent "mixed, evidence from two independently-toolable
sources, merged into one grounded answer" path to compare against at
all: A's agent route *can* call more than one tool in sequence within its
own bounded loop, but has no notion of two specialist agents or parallel
dispatch; B is single-route by design (see that package's README). This
command is therefore multi-agent-only, and says so.

`write-action` extends `langgraph_experiment.cli.compare_harness
write-action`'s qualitative table with a third column for this package --
still no real timing comparison is possible for A (no pause/resume
primitive to time at all).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from langgraph_experiment.checkpointer import sqlite_checkpointer as single_agent_checkpointer
from langgraph_experiment.graph import build_graph as build_single_agent_graph
from langgraph_experiment.wiring import build_default_deps
from multi_agent_experiment.checkpointer import sqlite_checkpointer
from multi_agent_experiment.graph import build_graph
from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.retrieval.authorization import AuthorizationContext

# GraphDeps (langgraph_experiment.wiring) is reused wholesale by this
# package -- see multi_agent_experiment/__init__.py's module docstring --
# so build_default_deps() is exactly as valid for wiring the multi-agent
# graph as it is for the single-agent one. A separate call per graph
# (rather than sharing one GraphDeps instance) mirrors this file's own
# `rag` command building two independent deps objects below.
build_multi_agent_deps = build_default_deps

#: Rows 1-9 are `langgraph_experiment.cli.compare_harness`'s own table,
#: reproduced verbatim (single source of truth would require importing a
#: private module-level constant across packages for a cosmetic reason;
#: kept in sync by hand, same tradeoff that module's own docstring
#: already accepts). Row 10 (Coordination) is new to this file.
_QUALITATIVE_COMPARISON: list[tuple[str, str, str, str]] = [
    (
        "Code complexity",
        "~150 lines for the write-action path incl. one Python while-loop; "
        "directly steppable in a debugger.",
        "~110 lines across nodes.py/graph.py for the same path; control flow "
        "lives in the Pregel runtime, not a visible loop.",
        "~40 lines of new orchestration code (coordinator/merge/critic) plus "
        "the business branch reused unmodified from B -- more topology, but "
        "almost no new business logic.",
    ),
    (
        "State visibility",
        "AgentState is one pydantic object; no built-in history within a run.",
        "GraphState is checkpointed after every node -- full history via get_state_history().",
        "Same as B, plus per-specialist fields (knowledge_status, "
        "business_status) make it visible *which* agent contributed what, "
        "not just what the final state was.",
    ),
    (
        "Branching",
        "Plain if/elif; not queryable at runtime.",
        "add_conditional_edges() -- introspectable via get_graph().",
        "Same, plus Send-based dynamic fan-out -- the actual *set* of "
        "specialists dispatched is a runtime decision, not just which of a "
        "fixed set of edges was taken.",
    ),
    (
        "Retry handling",
        "None built in.",
        "RetryPolicy on a node -- declarative retry with backoff.",
        "RetryPolicy (reused, business branch) plus a second, higher-level "
        "retry: evidence_critic can bounce a run back through the "
        "coordinator to retry one specialist with widened parameters -- a "
        "retry loop a single node's RetryPolicy cannot express.",
    ),
    (
        "Checkpointing",
        "None.",
        "Every node transition durably checkpointed (SQLite here).",
        "Same, with one added gotcha this package hit for real: a downstream "
        "join node fed by parallel branches of *different* length must be "
        "declared with add_node(..., defer=True) or it fires once per "
        "branch instead of once overall -- see ISSUES.md.",
    ),
    (
        "Pause/resume",
        "Not supported.",
        "interrupt()/Command(resume=...) -- the core primitive B exists to demonstrate.",
        "Same mechanism, reused unmodified from B's business-write branch, "
        "now reachable from a coordinator that also routes to two other "
        "specialists.",
    ),
    (
        "Human approval",
        "Must be supplied before the run starts.",
        "A run can pause mid-flight and durably wait for a decision.",
        "Same as B (identical code).",
    ),
    (
        "Latency",
        "One process, one call stack.",
        "Small per-step checkpoint-write overhead.",
        "Same per-step overhead as B, offset by genuine parallel dispatch on "
        "a mixed query -- see `mixed` subcommand for a real number.",
    ),
    (
        "Debugging",
        "Standard Python debugger end to end.",
        "get_state(config).tasks[i].error surfaces a failed node's "
        "exception without a live debugger.",
        "Same, plus tool_call_log/critic_notes narrate *why* a specific "
        "specialist's contribution was or wasn't used -- useful precisely "
        "because there's more than one agent's output to reconcile.",
    ),
    (
        "Testability",
        "Tests call run_agent() directly with fakes; fully synchronous.",
        "Tests inject fakes via GraphDeps, assert on checkpoint state directly.",
        "Same GraphDeps injection (literally the same dataclass), plus a "
        "static-import-boundary test (tests/test_tool_isolation.py) that a "
        "single-agent graph has no equivalent need for.",
    ),
    (
        "Coordination",
        "N/A -- one agent, one tool-selection loop; 'coordination' is "
        "just sequential tool calls within one bounded loop.",
        "N/A -- single route per run, no specialist-to-specialist handoff.",
        "A real coordinator/router decision, parallel fan-out for "
        "independent specialists, and a merge step that reconciles two "
        "genuinely different evidence shapes (retrieved chunks vs. a "
        "business-case record) into one grounded answer.",
    ),
]


def _print_qualitative_comparison() -> None:
    header = (
        f"{'Axis':<18} | {'A: custom harness':<38} | "
        f"{'B: LangGraph single-agent':<38} | C: LangGraph multi-agent"
    )
    print(header)
    print("-" * len(header))
    for axis, a, b, c in _QUALITATIVE_COMPARISON:
        print(f"{axis:<18} | {a:<38} | {b:<38} | {c}")


def cmd_write_action(_: argparse.Namespace) -> None:
    """Print the qualitative A/B/C table; no timing is possible for this scenario."""
    print(
        "A has no pause/resume primitive, so there is nothing to time for the "
        "write-action/approval scenario. B and C share the identical, unmodified "
        "interrupt/resume code (langgraph_experiment.nodes.wait_for_approval), so "
        "their timing is the same mechanism at a different call depth -- not a "
        "meaningful A/B/C number either. Qualitative comparison:\n"
    )
    _print_qualitative_comparison()


def cmd_rag(args: argparse.Namespace) -> None:
    """Time a knowledge-only query through the custom harness, B, and C."""
    # with_ledger=False: this comparison never reaches the write branch, so
    # the production-hardening idempotency ledger's own Postgres connection
    # is unneeded scope for a timing-only run.
    single_deps = build_default_deps(with_retrieval=True, with_ledger=False)
    multi_deps = build_default_deps(with_retrieval=True, with_ledger=False)
    auth = AuthorizationContext(tenant_id=args.tenant, roles=args.role or [])
    filters = {"dataset_id": args.dataset_id} if args.dataset_id else None

    print("--- A: custom harness (rag.agent.graph.run_agent, classic_rag route) ---")
    state = AgentState(original_query=args.query, authorization_context=auth, filters=filters)
    t0 = time.perf_counter()
    custom_result = run_agent(
        state,
        pipeline=single_deps.pipeline,
        vectorstore=single_deps.pipeline._vectorstore,  # noqa: SLF001 -- read-only, comparison-only
        embedder=single_deps.pipeline._embedder,  # noqa: SLF001
        llm=single_deps.llm,
        config=single_deps.config,
    )
    custom_ms = (time.perf_counter() - t0) * 1000
    print(f"answer: {custom_result.state.final_answer}")
    print(f"total_ms: {custom_ms:.1f}")

    print("\n--- B: LangGraph single-agent (read_only branch) ---")
    t0 = time.perf_counter()
    with single_agent_checkpointer() as checkpointer:
        b_graph = build_single_agent_graph(single_deps, checkpointer)
        b_result = b_graph.invoke(
            {
                "original_query": args.query,
                "caller_subject": "compare-harness",
                "tenant_id": args.tenant,
                "roles": args.role or [],
                "dataset_id": args.dataset_id,
            },
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    b_ms = (time.perf_counter() - t0) * 1000
    print(f"answer: {b_result.get('final_answer')}")
    print(f"total_ms: {b_ms:.1f}")

    print("\n--- C: LangGraph multi-agent (knowledge_only route) ---")
    t0 = time.perf_counter()
    with sqlite_checkpointer() as checkpointer:
        c_graph = build_graph(multi_deps, checkpointer)
        c_result = c_graph.invoke(
            {
                "original_query": args.query,
                "caller_subject": "compare-harness",
                "tenant_id": args.tenant,
                "roles": args.role or [],
                "dataset_id": args.dataset_id,
            },
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    c_ms = (time.perf_counter() - t0) * 1000
    print(f"answer: {c_result.get('final_answer')}")
    print(f"total_ms: {c_ms:.1f}")

    print(f"\nB overhead vs A: {b_ms - custom_ms:+.1f} ms")
    print(f"C overhead vs A: {c_ms - custom_ms:+.1f} ms")
    print(
        f"C overhead vs B: {c_ms - b_ms:+.1f} ms "
        "(coordinator + merge + evidence_critic's extra hops, for a route "
        "that only ever needed one specialist)"
    )


def cmd_mixed(args: argparse.Namespace) -> None:
    """Time C's real parallel dispatch, then estimate sequential cost from two solo runs.

    The estimate is built from two *separately measured* real runs (a
    knowledge-only and a business-only call against the same identity/
    dataset), not a guess -- summing them is the honest cost a purely
    sequential coordinator (dispatch knowledge, wait, then dispatch
    business, wait) would have paid, since neither call depends on the
    other's result in this route (see README.md's "Parallel execution"
    section for why that independence is exactly what makes fan-out safe
    here). It is still an estimate, not a second real measurement of a
    sequential graph, because this package deliberately does not build a
    second, throwaway graph variant just to produce one number -- the
    qualitative reasoning is the same either way.
    """
    deps = build_multi_agent_deps(with_retrieval=True, with_ledger=False)
    auth_kwargs = {
        "tenant_id": args.tenant,
        "roles": args.role or [],
        "dataset_id": args.dataset_id,
    }

    print("--- C: mixed route, real parallel dispatch ---")
    t0 = time.perf_counter()
    with sqlite_checkpointer() as checkpointer:
        graph = build_graph(deps, checkpointer)
        result = graph.invoke(
            {"original_query": args.query, **auth_kwargs},
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    parallel_ms = (time.perf_counter() - t0) * 1000
    print(f"selected_specialists: {result.get('selected_specialists')}")
    print(f"answer: {result.get('final_answer')}")
    print(f"parallel total_ms: {parallel_ms:.1f}")

    if result.get("selected_specialists") != ["knowledge", "business"]:
        print("\nQuery did not route to both specialists; skipping the sequential estimate.")
        return

    print("\n--- component timings, for the sequential estimate ---")
    # A second invocation of the same mixed query would itself already be
    # parallel, so it cannot measure a sequential cost; component timings
    # instead come from a knowledge-only phrasing and a caller-supplied
    # business-only phrasing of the same underlying case, each run solo.
    knowledge_only_ms = _time_route(deps, args.knowledge_only_query or args.query, auth_kwargs)
    business_only_ms = (
        _time_route(deps, args.business_only_query, auth_kwargs)
        if args.business_only_query
        else None
    )

    if business_only_ms is None:
        print(
            "\n--no-op-- pass --business-only-query to compute a real sequential "
            "estimate (a business-only phrasing of the same case, e.g. "
            "'What is the status of CASE-1001?')."
        )
        return

    sequential_estimate_ms = knowledge_only_ms + business_only_ms
    print(f"knowledge-only component: {knowledge_only_ms:.1f} ms")
    print(f"business-only component: {business_only_ms:.1f} ms")
    print(f"sequential estimate (sum): {sequential_estimate_ms:.1f} ms")
    print(f"parallel actual: {parallel_ms:.1f} ms")
    print(f"parallel saved: {sequential_estimate_ms - parallel_ms:+.1f} ms")


def _time_route(deps, query: str, auth_kwargs: dict) -> float:
    t0 = time.perf_counter()
    with sqlite_checkpointer() as checkpointer:
        graph = build_graph(deps, checkpointer)
        graph.invoke(
            {"original_query": query, **auth_kwargs},
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    return (time.perf_counter() - t0) * 1000


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: dispatch to rag/mixed/write-action."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rag_cmd = sub.add_parser("rag", help="Time a knowledge-only query through A, B, and C")
    rag_cmd.add_argument("query")
    rag_cmd.add_argument("--tenant", default=None)
    rag_cmd.add_argument("--role", action="append", default=None)
    rag_cmd.add_argument("--dataset-id", default=None)
    rag_cmd.set_defaults(func=cmd_rag)

    mixed_cmd = sub.add_parser("mixed", help="Time C's parallel dispatch vs. a sequential estimate")
    mixed_cmd.add_argument("query")
    mixed_cmd.add_argument("--tenant", default=None)
    mixed_cmd.add_argument("--role", action="append", default=None)
    mixed_cmd.add_argument("--dataset-id", default=None)
    mixed_cmd.add_argument("--knowledge-only-query", default=None)
    mixed_cmd.add_argument("--business-only-query", default=None)
    mixed_cmd.set_defaults(func=cmd_mixed)

    write_cmd = sub.add_parser("write-action", help="Print the qualitative A/B/C comparison table")
    write_cmd.set_defaults(func=cmd_write_action)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
