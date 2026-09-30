"""Conditional-edge decision functions for the multi-agent graph.

`route_after_coordinator` is the one function in this package that
returns `langgraph.types.Send` objects -- see `graph.py`'s module
docstring for why `Send`, specifically, is what makes the "mixed" route's
two specialists run as genuinely parallel branches rather than
sequentially, and `tests/test_probe_langgraph_primitives.py` for the
throwaway-script-turned-permanent-regression-test that first confirmed
this against the actually-installed `langgraph==1.2.11` before any of the
rest of this package was written.

`route_after_validate`/`route_after_approval` (the business-write branch's
two internal decisions) are reused unmodified from `langgraph_experiment.
routing` -- see `graph.py` for where they're wired in directly, not
reimported here.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from multi_agent_experiment.state import MultiAgentState


def route_after_coordinator(state: MultiAgentState) -> list[Any] | str:
    """Dispatch the coordinator's routing decision.

    Three shapes, matching README.md's architecture diagram exactly:

    - **Business mutation**: a plain string (`"business_validate_write"`),
      not wrapped in `Send` -- this is always a single target, and keeping
      it a bare conditional-edge return (rather than
      `[Send("business_validate_write", state)]`) means the mutation
      branch's interrupt/resume mechanics are byte-identical to the
      already-proven single-agent experiment's own (unwrapped) write
      branch, with no `Send`-specific behavior to re-verify for that path.
    - **Solo** (`knowledge_only` or `business_read_only`): a one-element
      `Send` list -- still routed through `Send` (not a bare string) so
      the solo and mixed cases share one code path/mental model, per this
      function's own "uniform dispatch" design choice.
    - **Mixed**: a two-element `Send` list -- `knowledge_agent` and
      `business_select_tool` are dispatched in the same superstep and run
      concurrently; both `add_edge(..., "merge")` in `graph.py`, so
      `merge` only runs once both have completed (LangGraph's standard
      fan-out/join semantics, confirmed empirically in `tests/
      test_probe_langgraph_primitives.py`).

    On a retry round (`coordinator` forces `needs_business=False`; see
    that node's docstring), this naturally produces a one-element
    `[Send("knowledge_agent", state)]` list -- the business branch is
    never re-dispatched.

    Each `Send` carries the **entire current state** as its payload
    (rather than a narrowed partial state) because the target node
    functions read several keys by name (`tenant_id`, `roles`, `case_id`,
    `dataset_id`, ...) that were not necessarily set by the immediately
    preceding node -- see `state.py`'s docstring for the full field list.
    """
    if state.get("is_case_mutation"):
        return "business_validate_write"

    targets: list[Any] = []
    if state.get("needs_knowledge"):
        targets.append(Send("knowledge_agent", state))
    if state.get("needs_business"):
        targets.append(Send("business_select_tool", state))
    if not targets:
        # Defensive fallback only -- `coordinator` always sets at least one
        # of `needs_knowledge`/`needs_business` to `True` on round 0, and a
        # retry round only ever narrows `needs_business`, never clears
        # `needs_knowledge` too. Kept so a future change to `coordinator`
        # fails safe (a knowledge-only default) rather than dead-ending
        # the graph with zero dispatched tasks.
        targets.append(Send("knowledge_agent", state))
    return targets


def route_after_critic(state: MultiAgentState) -> str:
    """`"retry"` if `evidence_critic` requested one more knowledge attempt, else `"proceed"`."""
    return "retry" if state.get("critic_wants_retry") else "proceed"
