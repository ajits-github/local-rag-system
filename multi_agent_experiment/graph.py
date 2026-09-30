"""Builds the multi-agent `StateGraph`: coordinator + two specialists + critic.

```
START
  |
coordinator (round 0: classify; round >=1: retry-adjust only)
  |
  +-- business mutation (bare string, no Send) ----------------+
  |   business_validate_write                                  |
  |     +-- reject_invalid -> business_synthesize_write -> END |
  |     +-- await_approval -> business_wait_approval (INTERRUPT)
  |                              +-- rejected -> business_synthesize_write -> END
  |                              +-- execute  -> business_execute_write
  |                                                -> business_synthesize_write -> END
  |
  +-- knowledge only (Send x1) --> knowledge_agent ------+
  |                                                       |
  +-- business read only (Send x1)                        |
  |     --> business_select_tool                          |
  |           -> business_execute_read                    +--> merge --> evidence_critic
  |                 -> business_evaluate_read -------------+                  |
  |                                                                +-- "retry" (knowledge
  +-- mixed (Send x2, PARALLEL)                                    |    only, bounded --
  |     --> knowledge_agent ----------------------------------------+    see limits.py --
  |     +-> business_select_tool                                        back to coordinator)
  |           -> business_execute_read                              |
  |                 -> business_evaluate_read ------------------------+
  |                                                                   +-- "proceed"
  |                                                                        |
  |                                                                   final_synthesis --> END
```

One thread (`config["configurable"]["thread_id"]`) is one query's
lifecycle, matching `langgraph_experiment.graph`'s own convention -- see
that module's docstring.

**Module map** (see each module's own docstring for the full reasoning):
`knowledge_agent.py` and `business_agent.py` are the two specialists,
each importable and testable in isolation with no dependency on the
other. `orchestrator.py` is the "sees both" coordination tier
(coordinator, merge, critic, final synthesis). `business_agent.py`'s
read/write nodes are, in turn, `langgraph_experiment.nodes`/`routing`
functions reused **unmodified** -- this file is the only place that
wires all of it into one compiled graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from langgraph_experiment import routing as business_routing
from langgraph_experiment.wiring import GraphDeps, TransientCaseStoreError
from multi_agent_experiment import business_agent, knowledge_agent, orchestrator, routing
from multi_agent_experiment.state import MultiAgentState

#: Identical to `langgraph_experiment.graph._CASE_READ_RETRY_POLICY` --
#: retries only a simulated transient case-store failure, never a genuine
#: deterministic outcome (not-found, not-authorized).
_BUSINESS_READ_RETRY_POLICY = RetryPolicy(
    max_attempts=3, initial_interval=0.05, retry_on=(TransientCaseStoreError,)
)


def build_graph(deps: GraphDeps, checkpointer: object) -> CompiledStateGraph:
    """Wire every node/edge and compile against the given checkpointer.

    Parameters
    ----------
    deps : GraphDeps
        The same `langgraph_experiment.wiring.GraphDeps` the single-agent
        experiment uses -- no new dependency-injection type for this
        package. See that class's docstring.
    checkpointer : object
        Any LangGraph `BaseCheckpointSaver` -- `checkpointer.
        sqlite_checkpointer` (this package's own, pointed at a sibling
        `data/` directory) for the durable demo path, or an in-memory
        `InMemorySaver` for tests.

    Returns
    -------
    CompiledStateGraph
        Ready for `.invoke(...)`/`.get_state(...)`.
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("coordinator", orchestrator.coordinator)
    graph.add_node("knowledge_agent", knowledge_agent.make_knowledge_agent_node(deps))
    graph.add_node("business_select_tool", business_agent.select_tool)
    graph.add_node(
        "business_execute_read",
        business_agent.make_execute_read_node(deps),
        retry_policy=_BUSINESS_READ_RETRY_POLICY,
    )
    graph.add_node("business_evaluate_read", business_agent.evaluate_read)
    graph.add_node("business_validate_write", business_agent.validate_write_request)
    graph.add_node("business_wait_approval", business_agent.make_wait_for_approval_node(deps))
    graph.add_node("business_execute_write", business_agent.make_execute_write_action_node(deps))
    graph.add_node("business_synthesize_write", business_agent.synthesize_write_result)
    # defer=True is load-bearing, not cosmetic: the knowledge branch is one
    # hop (knowledge_agent -> merge) but the business-read branch is three
    # (select_tool -> execute_read -> evaluate_read -> merge). Without
    # `defer`, LangGraph's fan-in triggers "merge" as soon as EITHER
    # predecessor's edge fires -- for the "mixed" route this ran merge
    # (and, transitively, evidence_critic/final_synthesis) once right after
    # knowledge_agent alone finished, and again after the business branch
    # caught up, doubling the LLM call and briefly answering from partial
    # evidence. Confirmed directly against the installed langgraph (see
    # ISSUES.md and tests/test_probe_langgraph_primitives.py) before this
    # fix, not assumed: `defer=True` makes "merge" wait for every pending
    # task to finish before it runs, regardless of branch length.
    graph.add_node("merge", orchestrator.merge, defer=True)
    graph.add_node("evidence_critic", orchestrator.evidence_critic)
    graph.add_node("final_synthesis", orchestrator.make_final_synthesis_node(deps))

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        routing.route_after_coordinator,
        # Only the bare-string ("non-Send") destination needs a path_map
        # entry; Send-dispatched targets (knowledge_agent,
        # business_select_tool) name their own destination explicitly and
        # need no map entry -- confirmed against the installed langgraph
        # in tests/test_probe_langgraph_primitives.py.
        {"business_validate_write": "business_validate_write"},
    )

    # Knowledge and business-read branches both join at "merge" --
    # LangGraph waits for every dispatched branch in a superstep to
    # finish before running a shared downstream node, which is what makes
    # this a real fan-out/join, not a race.
    graph.add_edge("knowledge_agent", "merge")
    graph.add_edge("business_select_tool", "business_execute_read")
    graph.add_edge("business_execute_read", "business_evaluate_read")
    graph.add_edge("business_evaluate_read", "merge")

    graph.add_edge("merge", "evidence_critic")
    graph.add_conditional_edges(
        "evidence_critic",
        routing.route_after_critic,
        {"retry": "coordinator", "proceed": "final_synthesis"},
    )
    graph.add_edge("final_synthesis", END)

    graph.add_conditional_edges(
        "business_validate_write",
        business_routing.route_after_validate,
        {"await_approval": "business_wait_approval", "reject_invalid": "business_synthesize_write"},
    )
    graph.add_conditional_edges(
        "business_wait_approval",
        business_routing.route_after_approval,
        {"execute": "business_execute_write", "rejected": "business_synthesize_write"},
    )
    graph.add_edge("business_execute_write", "business_synthesize_write")
    graph.add_edge("business_synthesize_write", END)

    return graph.compile(checkpointer=checkpointer)
