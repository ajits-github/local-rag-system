"""Conditional-edge decision functions for the experimental graph.

Each function reads already-set state and returns one of the string keys
`graph.build_graph`'s `add_conditional_edges(..., path_map)` maps to a
node name. Kept separate from `nodes.py` so a routing decision (a pure
function of state, no side effects, no injected deps) is never confused
with a node that can call out to `RetrievalPipeline`/the case store.
"""

from __future__ import annotations

from langgraph_experiment.state import GraphState


def route_after_classify(state: GraphState) -> str:
    """Dispatch on `classify`'s routing decision."""
    return state.get("route") or "read_only"


def route_after_validate(state: GraphState) -> str:
    """`"await_approval"` if the request was accepted, else `"reject_invalid"`."""
    return "await_approval" if state.get("pending_action") is not None else "reject_invalid"


def route_after_approval(state: GraphState) -> str:
    """`"execute"` only if the resumed decision was approved by a role-holding identity."""
    pending = state.get("pending_action") or {}
    return "execute" if pending.get("approval_state") == "approved" else "rejected"
