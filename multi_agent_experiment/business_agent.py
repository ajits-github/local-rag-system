"""The Business Agent: three reused node functions plus one thin wrapper, isolated from RAG tools.

Tool isolation, symmetric to `knowledge_agent.py`'s claim: this module
imports `langgraph_experiment.nodes` (which itself imports `rag.mcp.
business.*`) and nothing from `rag.retrieval.pipeline`/`rag.agent.tools`.
It never imports `RetrievalPipeline`, so there is no code path here that
could call `search_knowledge_base` even by accident. Verified statically
in `tests/test_tool_isolation.py`.

Every read/write node below is **imported and used unmodified** from
`langgraph_experiment.nodes` -- the single-agent experiment's already-
tested business branch, including its production-hardening additions
(the idempotency ledger `execute_write_action` guards mutations through,
server-side approval-expiry/workflow-timeout checks in
`wait_for_approval`; see that module's docstring: it reuses
`rag.mcp.business.store` unmodified in turn, which is what actually
enforces tenant/role authorization, transition validity, and sensitive-
transition approval; none of those rules are reimplemented at any layer
here). The only new code in this file is `evaluate_read`, a thin wrapper
adding the one field (`business_status`) the multi-agent orchestrator's
critic/merge nodes need that a single-agent, single-branch graph never
had to expose to a sibling node.
"""

from __future__ import annotations

from langgraph_experiment import nodes as _single_agent_nodes
from multi_agent_experiment.state import MultiAgentState

# Reused, unmodified. Rebinding these names (rather than re-wrapping them)
# means a future change to the single-agent experiment's business branch
# is picked up here automatically, with no risk of the two drifting apart.
select_tool = _single_agent_nodes.select_case_tool
make_execute_read_node = _single_agent_nodes.make_execute_case_read_node
make_validate_write_request_node = _single_agent_nodes.make_validate_write_request_node
make_wait_for_approval_node = _single_agent_nodes.make_wait_for_approval_node
make_execute_write_action_node = _single_agent_nodes.make_execute_write_action_node
synthesize_write_result = _single_agent_nodes.synthesize_write_result


def evaluate_read(state: MultiAgentState) -> dict:
    """Check read sufficiency for the Business Agent: reused logic plus one added field.

    Delegates entirely to `langgraph_experiment.nodes.evaluate_case_read`
    for `termination_reason` (only set on a miss/denial), then adds
    `business_status` (`"ok"`/`"not_found_or_denied"`) and one
    `ToolCallRecord` -- the two signals `orchestrator.evidence_critic`/
    `orchestrator.merge` need that the single-agent experiment's terminal,
    non-multi-agent read branch never had to expose to a sibling node.
    """
    result = dict(_single_agent_nodes.evaluate_case_read(state))
    result["business_status"] = "ok" if state.get("case_found") else "not_found_or_denied"
    result["business_tool_calls"] = [
        {
            "specialist": "business",
            "tool_name": state.get("case_tool_name") or "get_customer_case",
            "success": bool(state.get("case_found")),
            "latency_ms": 0.0,
        }
    ]
    return result
