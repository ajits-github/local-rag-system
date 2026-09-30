"""The Knowledge Agent: one specialist, one tool, structurally unable to mutate business state.

Tool isolation here is not a runtime permission check -- it is a fact
about this module's import graph, verified statically in `tests/
test_tool_isolation.py`: this file imports `rag.retrieval.*` and
`langgraph_experiment.wiring`/`state` only. It never imports
`rag.mcp.business` (the business-case store), `rag.agent.mcp_client`, or
`business_agent.py`. There is no code path in this module that could call
`update_case_status` even if a bug tried to construct one, because the
function that would do so was never written here.

Allowed tool: `search_knowledge_base` (`RetrievalPipeline.retrieve()`),
matching production's own `rag.agent.tools.search_knowledge_base` and
`langgraph_experiment.nodes.retrieve`. This experiment's Knowledge Agent
does not additionally exercise `get_document`/`get_latest_document`/
`get_related_context` -- a deliberate scope decision (see README.md's
"Scope decisions"): those three are already demonstrated, against the
same underlying `rag.agent.tools` module, by production's agent; this
experiment's teaching value is specialist routing/parallelism/retry
across *agents*, not tool selection *within* one agent.
"""

from __future__ import annotations

import time

from langgraph_experiment.state import EvidenceItem
from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.limits import DEFAULT_KNOWLEDGE_TOP_K
from multi_agent_experiment.state import MultiAgentState, ToolCallRecord
from rag.retrieval.authorization import AuthorizationContext


def _tool_call(tool_name: str, success: bool, started_at: float) -> ToolCallRecord:
    return {
        "specialist": "knowledge",
        "tool_name": tool_name,
        "success": success,
        "latency_ms": (time.perf_counter() - started_at) * 1000,
    }


def make_knowledge_agent_node(deps: GraphDeps):
    """Build the Knowledge Agent node.

    Contains its own failures: a raised exception from `pipeline.
    retrieve()` (simulating a timeout or a downstream error -- see
    `tests/test_specialist_failures.py`) is caught here and reported as
    `knowledge_status="failed"` with only the exception's class name
    recorded, never propagated to crash the run and never leaking an
    internal message. `orchestrator.evidence_critic`/`final_synthesis`
    both treat `"failed"` the same as `"no_evidence"`: never fabricate a
    knowledge answer either way.

    Reads `state["knowledge_top_k"]` (set by `orchestrator.coordinator`,
    widened on a critic-requested retry -- see that node's docstring)
    rather than a fixed constant, so the same function serves both the
    first attempt and the retry without needing to know which one it is.
    """

    def knowledge_agent(state: MultiAgentState) -> dict:
        started_at = time.perf_counter()
        top_k = state.get("knowledge_top_k") or DEFAULT_KNOWLEDGE_TOP_K
        if deps.pipeline is None:
            return {
                "knowledge_status": "failed",
                "knowledge_evidence": [],
                "knowledge_error": "retrieval_unavailable",
                "knowledge_tool_calls": [_tool_call("search_knowledge_base", False, started_at)],
            }
        auth = AuthorizationContext(
            tenant_id=state.get("tenant_id"), roles=list(state.get("roles") or [])
        )
        filters = {"dataset_id": state["dataset_id"]} if state.get("dataset_id") else None
        try:
            results = deps.pipeline.retrieve(
                state["original_query"], filters=filters, candidate_k=top_k, auth=auth
            )
        except Exception as exc:  # noqa: BLE001 -- contain a specialist failure, never crash the run
            return {
                "knowledge_status": "failed",
                "knowledge_evidence": [],
                "knowledge_error": type(exc).__name__,
                "knowledge_tool_calls": [_tool_call("search_knowledge_base", False, started_at)],
            }
        evidence: list[EvidenceItem] = [
            {
                "chunk_id": r.chunk.metadata.chunk_id,
                "source": r.chunk.metadata.source,
                "content": r.chunk.content,
                "score": r.score,
            }
            for r in results
        ]
        return {
            "knowledge_status": "ok" if evidence else "no_evidence",
            "knowledge_evidence": evidence,
            "knowledge_error": None,
            "knowledge_tool_calls": [_tool_call("search_knowledge_base", True, started_at)],
        }

    return knowledge_agent
