"""Typed graph state for the experimental LangGraph agent.

Deliberately plain `TypedDict`s of JSON-safe values (`str`/`int`/`float`/
`bool`/`None`/`dict`/`list`), not the real `SearchResult`/`Chunk`/
`CustomerCase` pydantic models `rag.*` already defines. LangGraph's
SQLite checkpointer will happily serialize richer objects too, but a
plain-dict state keeps every checkpoint human-readable when inspected
directly (`sqlite3 langgraph_experiment/data/checkpoints.sqlite`) and
keeps the JSON shape stable across a `rag` pydantic-model change --
useful properties for a *learning* graph, even though production's own
`AgentState` (`rag/agent/state.py`) is real pydantic throughout.

`GraphState.__annotations__` is the graph's schema: LangGraph builds the
graph's channels from it, and every node function receives (a subset of)
these keys and returns a partial update of them. There is no shared
mutable object nodes reach into, unlike `rag.agent.graph`'s `AgentState`
being threaded through and mutated in place. See README.md's
"StateGraph vs ordinary Python control flow" section for why that
difference is the whole point of `StateGraph`.
"""

from __future__ import annotations

from typing import Literal, TypedDict

Route = Literal["read_only", "case_read", "case_write"]
ApprovalState = Literal[
    "pending",
    "approved",
    "rejected",
    "denied_insufficient_role",
    "expired",
    "cancelled",
    "workflow_timeout",
]


class EvidenceItem(TypedDict):
    """One retrieved knowledge-base chunk, trimmed to what synthesis/citation needs."""

    chunk_id: str
    source: str
    content: str
    score: float | None


class Citation(TypedDict):
    """One citation attached to a final answer, chunk- or case-sourced."""

    chunk_id: str | None
    case_id: str | None
    source: str
    score: float | None


class PendingCaseAction(TypedDict):
    """The one write-action request this graph can pause on.

    Set by `validate_write_request` *before* `wait_for_approval` calls
    `interrupt()`, so it is durably checkpointed as part of the paused
    state: exactly the fields the prompt spec asked to persist across a
    pause (action type, safe identifiers, requested transition, approval
    state), plus two production-hardening additions: `operation_id` and
    `expires_at`. Never a JWT, chain-of-thought, or raw secret.

    Attributes
    ----------
    operation_id : str
        A UUID4 minted once, by `validate_write_request`, and never
        re-minted on resume/retry. The idempotency key
        `langgraph_experiment.idempotency.ActionLedger` correlates a
        mutation attempt against, so a crash-and-resume or an automatic
        `RetryPolicy` retry of `execute_write_action` can recognize "this
        exact request already ran" rather than re-deriving identity from
        `(case_id, new_status)` alone (which would conflate two genuinely
        different requests for the same transition).
    expires_at : str
        ISO timestamp after which `wait_for_approval` refuses to honor a
        resume, regardless of the decision it carries. An approval
        request does not wait forever. Checked server-side, inside the
        node, the same trust posture already applied to `approver_roles`.
    """

    action_type: Literal["update_case_status"]
    case_id: str
    new_status: str
    requested_at: str
    expires_at: str
    operation_id: str
    approval_state: ApprovalState


class GraphState(TypedDict, total=False):
    """The experimental graph's full state schema.

    Attributes
    ----------
    original_query : str
        The caller's question, unmodified. Required at graph start.
    thread_id : str
        A plain copy of the LangGraph `config["configurable"]["thread_id"]`
        this run was invoked with, passed in by the caller alongside
        `original_query`, kept in state (not read out of LangGraph's own
        config inside a node) so `idempotency.ActionLedger` rows can
        record which thread an operation belongs to without any node
        needing a `(state, config)` signature. Purely a display/audit
        field; never used for routing or authorization.
    caller_subject : str
        A display-only identifier for the calling demo identity (never a
        real JWT `sub` claim. See `langgraph_experiment.identity`).
    tenant_id : str | None
        Demo caller's tenant, reused as-is by `AuthorizationContext`/
        `VerifiedIdentity` for both the RAG and case-store branches.
    roles : list[str]
        Demo caller's roles.
    dataset_id : str | None
        Retrieval dataset scope for the read-only RAG branch.
    workflow_started_at : str
        ISO timestamp set once, by `classify`, the graph's first node --
        the per-workflow timeout ceiling (`GraphDeps.
        max_workflow_duration_seconds`) is measured from this, distinct
        from `PendingCaseAction.expires_at`'s narrower approval-specific
        window (see `nodes.wait_for_approval`).
    route : Route | None
        `classify`'s routing decision.
    is_case_mutation : bool
        Whether `classify` detected a write-action request.
    case_id : str | None
        Extracted case identifier, for both the read and write branches.
    case_tool_name : str | None
        `select_case_tool`'s decision (`get_customer_case` or
        `get_case_status`), read-branch only.
    case_result : dict | None
        The read tool's result (a `CustomerCase`/`CaseStatusResult`
        `model_dump()`), or `None` for "not found or not authorized".
    case_found : bool
        Whether `case_result` is present, set by `execute_case_read_tool`
        and consumed by `evaluate_case_read`.
    requested_new_status : str | None
        Extracted target status, write branch only.
    pending_action : PendingCaseAction | None
        The one write action this run may act on. See that class's
        docstring for why this exists.
    approval_decision : dict | None
        The raw resume payload `wait_for_approval`'s `interrupt()` call
        received back, kept for observability. Never trusted on its own:
        `pending_action["approval_state"]` is the actual, role-checked
        verdict; see `langgraph_experiment.nodes.wait_for_approval`.
    case_action_outcome : dict | None
        `rag.mcp.business.store.update_case_status`'s result
        (`CaseActionOutcome.model_dump()`), write branch only.
    retrieved_evidence : list[EvidenceItem]
        Read-only RAG branch evidence.
    final_answer : str | None
        The terminal answer text, set by exactly one synthesize node.
    citations : list[Citation]
        Sources backing `final_answer`.
    termination_reason : str | None
        Why the run ended; see each node's docstring for the values it
        can set. `None` only while a run is still in progress or paused.
    """

    original_query: str
    thread_id: str
    caller_subject: str
    tenant_id: str | None
    roles: list[str]
    dataset_id: str | None
    workflow_started_at: str
    route: Route | None
    is_case_mutation: bool
    case_id: str | None
    case_tool_name: str | None
    case_result: dict | None
    case_found: bool
    requested_new_status: str | None
    pending_action: PendingCaseAction | None
    approval_decision: dict | None
    case_action_outcome: dict | None
    retrieved_evidence: list[EvidenceItem]
    final_answer: str | None
    citations: list[Citation]
    termination_reason: str | None
