"""Regression test for the three LangGraph primitives this package's design depends on.

Originally a throwaway probe script, run once against the actually-
installed `langgraph==1.2.11` *before* any of `graph.py`/`routing.py` was
written, to de-risk three assumptions the whole multi-agent topology rests
on (see `routing.route_after_coordinator`'s docstring):

1. A conditional-edge function can return a single-element list of
   `Send` objects to dispatch to exactly one dynamically-chosen node.
2. A conditional-edge function can return a *multi*-element list of
   `Send` objects to fan out to genuinely parallel branches, which join
   correctly at a shared downstream node once every dispatched branch
   completes -- and `Annotated[..., operator.add]` reducer fields
   accumulate correctly across those parallel writes.
3. A conditional-edge function can mix a bare node-name string (for a
   single, statically-known destination) with `Send`-based dispatch in
   the same graph, and `interrupt()`/`Command(resume=...)` work
   identically through a bare-string-routed node as they do in the
   single-agent experiment.

Kept as a permanent, minimal test (independent of this package's own
graph) rather than deleted after use, so a future `langgraph` upgrade
that changes any of this behavior fails here first, with a two-node
graph, rather than as a confusing failure inside a real multi-agent
scenario test.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


class _ProbeState(TypedDict, total=False):
    route: str
    a_result: str
    b_result: str
    log: Annotated[list[str], operator.add]
    approved: bool


def _node_a(state: _ProbeState) -> dict:
    return {"a_result": "A_DONE", "log": ["a"]}


def _node_b(state: _ProbeState) -> dict:
    return {"b_result": "B_DONE", "log": ["b"]}


def _node_c(state: _ProbeState) -> dict:
    decision = interrupt({"ask": "approve?"})
    return {"approved": bool(decision), "log": [f"c:{decision}"]}


def _merge(state: _ProbeState) -> dict:
    return {"log": ["merge"]}


def _route(state: _ProbeState):
    if state.get("route") == "solo_c":
        return "node_c"
    targets = [Send("node_a", state)]
    if state.get("route") == "mixed":
        targets.append(Send("node_b", state))
    return targets


def _build_probe_graph():
    g = StateGraph(_ProbeState)
    g.add_node("coordinator", lambda state: {})
    g.add_node("node_a", _node_a)
    g.add_node("node_b", _node_b)
    g.add_node("node_c", _node_c)
    g.add_node("merge", _merge)
    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", _route, {"node_c": "node_c"})
    g.add_edge("node_a", "merge")
    g.add_edge("node_b", "merge")
    g.add_edge("merge", END)
    g.add_edge("node_c", END)
    return g.compile(checkpointer=InMemorySaver())


def test_single_element_send_list_dispatches_to_exactly_one_node():
    """A conditional edge returning one Send dispatches to exactly that one node."""
    graph = _build_probe_graph()
    result = graph.invoke({"route": "solo_a"}, {"configurable": {"thread_id": "probe-1"}})
    assert result["a_result"] == "A_DONE"
    assert "b_result" not in result


def test_two_element_send_list_runs_parallel_branches_that_join_with_reducer_accumulation():
    """Two Sends run as parallel branches that join with operator.add reducer accumulation."""
    graph = _build_probe_graph()
    result = graph.invoke({"route": "mixed"}, {"configurable": {"thread_id": "probe-2"}})
    assert result["a_result"] == "A_DONE"
    assert result["b_result"] == "B_DONE"
    # operator.add accumulated both parallel branches' contributions plus merge's own.
    assert sorted(result["log"]) == ["a", "b", "merge"]


def test_bare_string_conditional_edge_supports_interrupt_and_resume():
    """A bare-string-routed node still supports interrupt()/Command(resume=...)."""
    graph = _build_probe_graph()
    config = {"configurable": {"thread_id": "probe-3"}}
    paused = graph.invoke({"route": "solo_c"}, config)
    assert "__interrupt__" in paused

    resumed = graph.invoke(Command(resume=True), config)
    assert resumed["approved"] is True
    assert resumed["log"] == ["c:True"]


# ---------------------------------------------------------------------------
# A fourth assumption, found the hard way while building the real graph (see
# ISSUES.md): when two parallel Send branches have UNEQUAL length before
# reaching a shared join node, LangGraph's fan-in triggers the join as soon
# as EITHER branch's edge fires -- not once after both complete. graph.py's
# "merge" node has exactly this shape (knowledge_agent is 1 hop,
# business_select_tool -> business_execute_read -> business_evaluate_read is
# 3), and this bug was caught by `tests/test_case3_mixed_parallel.py` seeing
# a doubled LLM call before `merge` was given `defer=True`. Reproduced here
# with a minimal two-branch graph, once without the fix (documenting the bug
# is real, not a phantom) and once with it.
# ---------------------------------------------------------------------------


class _UnevenState(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def _short_branch(state: _UnevenState) -> dict:
    return {"log": ["short"]}


def _long_branch_step_1(state: _UnevenState) -> dict:
    return {"log": ["long-1"]}


def _long_branch_step_2(state: _UnevenState) -> dict:
    return {"log": ["long-2"]}


def _uneven_merge(state: _UnevenState) -> dict:
    return {"log": ["merge"]}


def _build_uneven_branch_graph(*, defer_merge: bool):
    g = StateGraph(_UnevenState)
    g.add_node("coordinator", lambda state: {})
    g.add_node("short_branch", _short_branch)
    g.add_node("long_branch_1", _long_branch_step_1)
    g.add_node("long_branch_2", _long_branch_step_2)
    g.add_node("merge", _uneven_merge, defer=defer_merge)
    g.add_edge(START, "coordinator")
    g.add_conditional_edges(
        "coordinator",
        lambda state: [Send("short_branch", state), Send("long_branch_1", state)],
    )
    g.add_edge("short_branch", "merge")
    g.add_edge("long_branch_1", "long_branch_2")
    g.add_edge("long_branch_2", "merge")
    g.add_edge("merge", END)
    return g.compile(checkpointer=InMemorySaver())


def test_uneven_branch_lengths_without_defer_run_the_join_node_twice():
    """Documents the bug: with no `defer`, "merge" fires once per branch, not once overall."""
    graph = _build_uneven_branch_graph(defer_merge=False)
    result = graph.invoke({}, {"configurable": {"thread_id": "uneven-buggy"}})
    assert result["log"].count("merge") == 2


def test_uneven_branch_lengths_with_defer_run_the_join_node_exactly_once():
    """The fix: `defer=True` makes "merge" wait for every pending task, whatever the length."""
    graph = _build_uneven_branch_graph(defer_merge=True)
    result = graph.invoke({}, {"configurable": {"thread_id": "uneven-fixed"}})
    assert result["log"].count("merge") == 1
    assert sorted(result["log"]) == ["long-1", "long-2", "merge", "short"]
