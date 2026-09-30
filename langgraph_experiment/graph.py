"""Builds the experimental `StateGraph`.

```
START
  |
classify
  |
  +-- read_only  -> retrieve -> synthesize_rag -> END
  |
  +-- case_read  -> select_case_tool -> execute_case_read_tool -> evaluate_case_read
  |                 -> synthesize_case_read -> END
  |
  +-- case_write -> validate_write_request
                       +-- reject_invalid  -> synthesize_write_result -> END
                       +-- await_approval  -> wait_for_approval (INTERRUPT)
                                                +-- rejected -> synthesize_write_result -> END
                                                |   (approval_state: rejected / expired /
                                                |    workflow_timeout / denied_insufficient_role /
                                                |    cancelled -- see state.ApprovalState)
                                                +-- execute  -> execute_write_action
                                                                  -> synthesize_write_result -> END
```

One thread (`config["configurable"]["thread_id"]`) is one query's
lifecycle in this experiment, not a multi-turn conversation -- see
README.md's "Scope decisions" section.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from langgraph_experiment import nodes, observability, routing
from langgraph_experiment.state import GraphState
from langgraph_experiment.wiring import GraphDeps, TransientCaseStoreError

#: Retries only a simulated transient case-store failure (see
#: `wiring.TransientCaseStoreError`'s docstring), never a genuine
#: deterministic outcome (not-found, not-authorized) -- those are not
#: that exception type and are never retried.
_CASE_READ_RETRY_POLICY = RetryPolicy(
    max_attempts=3, initial_interval=0.05, retry_on=(TransientCaseStoreError,)
)

#: Retries a transient case-store failure on the *write* path too. Safe
#: only because `nodes.make_execute_write_action_node` checks
#: `GraphDeps.action_ledger` *first* on every attempt, including a
#: RetryPolicy-triggered one -- a retry policy on a write node with no
#: such guard would risk double-mutating on exactly the failure class
#: it's meant to recover from.
_WRITE_ACTION_RETRY_POLICY = RetryPolicy(
    max_attempts=3, initial_interval=0.05, retry_on=(TransientCaseStoreError,)
)


def build_graph(deps: GraphDeps, checkpointer: object) -> CompiledStateGraph:
    """Wire every node/edge and compile against the given checkpointer.

    Parameters
    ----------
    deps : GraphDeps
        Injected pipeline/LLM/case-store dependencies (see `wiring.py`).
    checkpointer : object
        Any LangGraph `BaseCheckpointSaver`: `langgraph_experiment.
        checkpointer.postgres_checkpointer` (the preferred, production-
        shaped durable backend), `sqlite_checkpointer` (a lighter-weight
        durable fallback with no Postgres dependency), or an in-memory
        `langgraph.checkpoint.memory.InMemorySaver` for a test that
        doesn't need restart/resume across processes.

    Returns
    -------
    CompiledStateGraph
        Ready for `.invoke(...)`/`.get_state(...)`.
    """
    graph = StateGraph(GraphState)

    def traced(name: str, fn):
        return observability.traced_node(name, fn)

    graph.add_node("classify", traced("classify", nodes.classify))
    graph.add_node("retrieve", traced("retrieve", nodes.make_retrieve_node(deps)))
    graph.add_node("synthesize_rag", traced("synthesize_rag", nodes.make_synthesize_rag_node(deps)))
    graph.add_node("select_case_tool", traced("select_case_tool", nodes.select_case_tool))
    graph.add_node(
        "execute_case_read_tool",
        traced("execute_case_read_tool", nodes.make_execute_case_read_node(deps)),
        retry_policy=_CASE_READ_RETRY_POLICY,
    )
    graph.add_node("evaluate_case_read", traced("evaluate_case_read", nodes.evaluate_case_read))
    graph.add_node(
        "synthesize_case_read", traced("synthesize_case_read", nodes.synthesize_case_read)
    )
    graph.add_node(
        "validate_write_request",
        traced("validate_write_request", nodes.make_validate_write_request_node(deps)),
    )
    graph.add_node(
        "wait_for_approval", traced("wait_for_approval", nodes.make_wait_for_approval_node(deps))
    )
    graph.add_node(
        "execute_write_action",
        traced("execute_write_action", nodes.make_execute_write_action_node(deps)),
        retry_policy=_WRITE_ACTION_RETRY_POLICY,
    )
    graph.add_node(
        "synthesize_write_result", traced("synthesize_write_result", nodes.synthesize_write_result)
    )

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        routing.route_after_classify,
        {
            "read_only": "retrieve",
            "case_read": "select_case_tool",
            "case_write": "validate_write_request",
        },
    )

    graph.add_edge("retrieve", "synthesize_rag")
    graph.add_edge("synthesize_rag", END)

    graph.add_edge("select_case_tool", "execute_case_read_tool")
    graph.add_edge("execute_case_read_tool", "evaluate_case_read")
    graph.add_edge("evaluate_case_read", "synthesize_case_read")
    graph.add_edge("synthesize_case_read", END)

    graph.add_conditional_edges(
        "validate_write_request",
        routing.route_after_validate,
        {"await_approval": "wait_for_approval", "reject_invalid": "synthesize_write_result"},
    )
    graph.add_conditional_edges(
        "wait_for_approval",
        routing.route_after_approval,
        {"execute": "execute_write_action", "rejected": "synthesize_write_result"},
    )
    graph.add_edge("execute_write_action", "synthesize_write_result")
    graph.add_edge("synthesize_write_result", END)

    return graph.compile(checkpointer=checkpointer)
