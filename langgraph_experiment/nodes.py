"""Node functions for the experimental LangGraph agent.

Every node is a plain function `(state) -> dict`, returning only the
keys it changes -- LangGraph merges the return value into the checkpointed
state (see `state.py`'s module docstring for why this is the core
difference from `rag.agent.graph`'s in-place-mutated `AgentState`). A node
that needs injected dependencies (`GraphDeps`) is a small factory
(`make_*_node(deps) -> node_fn`) that closes over them, exactly mirroring
how `rag.agent.graph._execute_tool` receives `pipeline`/`vectorstore`/
`embedder` as explicit parameters rather than reaching for a module-level
singleton.

Business rules (case transition validity, sensitive-transition approval,
tenant/role authorization) are never reimplemented here -- every case-store
call goes through `rag.mcp.business.store` unmodified via `GraphDeps`.
Retrieval authorization goes through the real `AuthorizationContext`/
`RetrievalPipeline.retrieve()`, also unmodified. This module only adds
graph-shaped glue: routing decisions, request extraction, and
checkpoint-safe result shaping.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.types import interrupt

from langgraph_experiment import audit
from langgraph_experiment.idempotency import _ReplayedOutcome
from langgraph_experiment.identity import DemoIdentity
from langgraph_experiment.state import Citation, EvidenceItem, GraphState, PendingCaseAction
from langgraph_experiment.wiring import GraphDeps
from rag.agent.graph import _looks_like_case_mutation_request
from rag.mcp.business.schemas import CaseApproval, CaseStatus
from rag.retrieval.authorization import AuthorizationContext

_CASE_ID_RE = re.compile(r"\bCASE-\d+\b", re.IGNORECASE)
_VALID_STATUSES: frozenset[CaseStatus] = frozenset({"open", "in_progress", "resolved", "closed"})

# Deliberately simple, deterministic keyword mapping -- not an LLM call.
# The experiment's routing/extraction nodes are kept fully deterministic so
# the interrupt/checkpoint/resume mechanics this package exists to teach
# are never confounded by LLM JSON-parsing flakiness (the existing custom
# harness, rag/agent/decisions.py, already demonstrates LLM-driven
# structured decisions; duplicating that here would teach nothing new
# about LangGraph itself). See README.md's "Scope decisions" section.
_MUTATION_VERB_TO_STATUS: dict[str, CaseStatus] = {
    "close": "closed",
    "closes": "closed",
    "closing": "closed",
    "resolve": "resolved",
    "resolves": "resolved",
    "resolving": "resolved",
    "reopen": "open",
    "reopens": "open",
    "reopening": "open",
}
_DIRECTIVE_STATUS_RE = re.compile(r"\b(open|in.progress|resolved|closed)\b", re.IGNORECASE)


def _extract_case_id(query: str) -> str | None:
    """Return the first `CASE-<digits>` token in `query`, uppercased, or `None`."""
    match = _CASE_ID_RE.search(query)
    return match.group(0).upper() if match else None


def _extract_target_status(query: str) -> CaseStatus | None:
    """Best-effort extraction of the requested target status from a mutation query.

    Checks mutation verbs (close/resolve/reopen) first, then falls back
    to an explicit status word (open/in_progress/resolved/closed) for
    directive phrasings ("set case X to resolved"). Returns `None` if
    neither matches; `validate_write_request` treats that as an invalid
    request rather than guessing.
    """
    lowered = query.lower()
    for verb, status in _MUTATION_VERB_TO_STATUS.items():
        if re.search(rf"\b{verb}\b", lowered):
            return status
    match = _DIRECTIVE_STATUS_RE.search(lowered)
    if match:
        normalized = match.group(1).replace("-", "_").replace(" ", "_")
        if normalized in _VALID_STATUSES:
            return normalized  # type: ignore[return-value]
    return None


def _identity_from_state(state: GraphState) -> Any:
    """Rebuild a `VerifiedIdentity` from the checkpointed tenant/roles claims."""
    return DemoIdentity(
        subject=state.get("caller_subject") or "demo-caller",
        tenant_id=state.get("tenant_id"),
        roles=tuple(state.get("roles") or ()),
    ).to_verified_identity()


def classify(state: GraphState) -> dict:
    """`classify` node: deterministic routing into read-only RAG, case-read, or case-write.

    Mirrors `rag.agent.graph._looks_like_case_mutation_request`'s deterministic
    override (reused, not reimplemented) for detecting a write-action
    request; this experiment applies the same rule as the very first
    routing decision, rather than as an override on top of an LLM
    classification, since this graph has no LLM classify step at all.

    Also stamps `workflow_started_at` (once; `state.get(...) or ...`
    guards a re-entry from ever resetting it) -- the origin point
    `GraphDeps.max_workflow_duration_seconds` is measured from, checked
    later in `wait_for_approval`.
    """
    workflow_started_at = state.get("workflow_started_at") or datetime.now(UTC).isoformat()
    query = state["original_query"]
    if _looks_like_case_mutation_request(query):
        return {
            "workflow_started_at": workflow_started_at,
            "route": "case_write",
            "is_case_mutation": True,
            "case_id": _extract_case_id(query),
            "requested_new_status": _extract_target_status(query),
        }
    case_id = _extract_case_id(query)
    if case_id is not None:
        return {
            "workflow_started_at": workflow_started_at,
            "route": "case_read",
            "is_case_mutation": False,
            "case_id": case_id,
        }
    return {
        "workflow_started_at": workflow_started_at,
        "route": "read_only",
        "is_case_mutation": False,
    }


def make_retrieve_node(deps: GraphDeps):
    """Build the `retrieve` node: authorized search via the real `RetrievalPipeline`."""

    def retrieve(state: GraphState) -> dict:
        if deps.pipeline is None:
            return {"retrieved_evidence": [], "termination_reason": "retrieval_unavailable"}
        auth = AuthorizationContext(
            tenant_id=state.get("tenant_id"), roles=list(state.get("roles") or [])
        )
        filters = {"dataset_id": state["dataset_id"]} if state.get("dataset_id") else None
        results = deps.pipeline.retrieve(state["original_query"], filters=filters, auth=auth)
        evidence: list[EvidenceItem] = [
            {
                "chunk_id": r.chunk.metadata.chunk_id,
                "source": r.chunk.metadata.source,
                "content": r.chunk.content,
                "score": r.score,
            }
            for r in results
        ]
        return {"retrieved_evidence": evidence}

    return retrieve


def make_synthesize_rag_node(deps: GraphDeps):
    """Build the `synthesize_rag` node: render accumulated evidence into a cited answer."""

    def synthesize_rag(state: GraphState) -> dict:
        evidence = state.get("retrieved_evidence") or []
        if not evidence:
            return {
                "final_answer": (
                    "I don't have enough authorized, retrieved evidence to answer this question."
                ),
                "citations": [],
                "termination_reason": state.get("termination_reason") or "insufficient_evidence",
            }
        if deps.llm is None or deps.rag_prompt_template is None:
            return {
                "final_answer": "Retrieval/generation dependencies unavailable in this demo run.",
                "citations": [],
                "termination_reason": "retrieval_unavailable",
            }
        context = "\n\n---\n\n".join(
            f"[Source {i}: {item['source']}]\n{item['content']}"
            for i, item in enumerate(evidence, start=1)
        )
        system, user = deps.rag_prompt_template.render(
            context=context, query=state["original_query"]
        )
        answer = deps.llm.generate(system, user)
        citations: list[Citation] = [
            {
                "chunk_id": item["chunk_id"],
                "case_id": None,
                "source": item["source"],
                "score": item["score"],
            }
            for item in evidence
        ]
        return {"final_answer": answer, "citations": citations, "termination_reason": "synthesized"}

    return synthesize_rag


def select_case_tool(state: GraphState) -> dict:
    """`select_case_tool` node: choose the narrow status read vs. the full case read.

    Deterministic on query wording ("status" without "detail"/"details"),
    the same simplicity tradeoff as `classify`.
    """
    query = state["original_query"].lower()
    wants_status_only = "status" in query and "detail" not in query
    return {"case_tool_name": "get_case_status" if wants_status_only else "get_customer_case"}


def make_execute_case_read_node(deps: GraphDeps):
    """Build the `execute_case_read_tool` node: dispatch the selected read tool.

    Registered with a `RetryPolicy` in `graph.build_graph` (retrying only
    `TransientCaseStoreError`) -- a genuine, deterministic node failure
    (case not found, not authorized) is not that exception type and is
    never retried; only a simulated transient backend failure is.
    """

    def execute_case_read_tool(state: GraphState) -> dict:
        case_id = state.get("case_id")
        if not case_id:
            return {"case_result": None, "case_found": False}
        identity = _identity_from_state(state)
        cross_tenant_roles = list(deps.cross_tenant_support_roles)
        if state.get("case_tool_name") == "get_case_status":
            result = deps.get_case_status_fn(case_id, identity, cross_tenant_roles)
        else:
            result = deps.get_customer_case_fn(case_id, identity, cross_tenant_roles)
        return {
            "case_result": result.model_dump(mode="json") if result is not None else None,
            "case_found": result is not None,
        }

    return execute_case_read_tool


def evaluate_case_read(state: GraphState) -> dict:
    """`evaluate_case_read` node: the trivial sufficiency check for a single-fetch read tool."""
    if not state.get("case_found"):
        return {"termination_reason": "case_not_found_or_denied"}
    return {}


def synthesize_case_read(state: GraphState) -> dict:
    """`synthesize_case_read` node: render the fetched case data as a final answer.

    Deterministic templating, not an LLM call -- see `README.md`'s
    "Scope decisions" section for why: it keeps the case-store branch
    (including the whole interrupt/approval demo) runnable with zero
    external services.
    """
    case_id = state.get("case_id")
    if not state.get("case_found"):
        return {
            "final_answer": (
                f"I could not find case {case_id}, or you are not authorized to view it."
            ),
            "citations": [],
            "termination_reason": state.get("termination_reason") or "case_not_found_or_denied",
        }
    result = state["case_result"]
    assert result is not None
    if state.get("case_tool_name") == "get_case_status":
        answer = (
            f"Case {result['case_id']} is currently '{result['status']}' "
            f"(priority: {result['priority']}), last updated {result['updated_at']}."
        )
    else:
        answer = (
            f"Case {result['case_id']} ({result['subject']}) is '{result['status']}', "
            f"priority {result['priority']}, assigned to {result['assigned_team']}. "
            f"{result['description']}"
        )
    citation: Citation = {
        "chunk_id": None,
        "case_id": result["case_id"],
        "source": f"case:{result['case_id']}",
        "score": None,
    }
    return {"final_answer": answer, "citations": [citation], "termination_reason": "synthesized"}


def make_validate_write_request_node(deps: GraphDeps):
    """Build the `validate_write_request` node: deterministic pre-checks before ever pausing.

    Sets `pending_action` (with `approval_state="pending"`) only when the
    request has a real case id and a recognized target status -- this is
    the field that gets checkpointed just before `wait_for_approval`
    pauses, so it must already be complete and correct here, including
    the two production-hardening fields `operation_id` (minted once, a
    fresh `uuid.uuid4()`, never re-minted on a later resume/retry --
    see `state.PendingCaseAction`'s docstring) and `expires_at`
    (`now + deps.approval_timeout_seconds`).

    Note this node does *not* decide whether the transition is
    "sensitive" (that is `rag.mcp.business.store`'s own
    `_SENSITIVE_TRANSITIONS` rule, deliberately not duplicated here) --
    every well-formed write request pauses for human confirmation,
    regardless of which transition it names. See README.md's "Scope
    decisions" section for why: it lets the interrupt/resume demo run
    reliably against any case/transition, and makes explicit that the
    graph-level pause is a UX control layered on top of, never a
    substitute for, `update_case_status`'s own independent
    transition/approval enforcement (still applied, unmodified, when
    `execute_write_action` finally calls it).
    """

    def validate_write_request(state: GraphState) -> dict:
        case_id = state.get("case_id")
        new_status = state.get("requested_new_status")
        if not case_id or new_status not in _VALID_STATUSES:
            audit.log_workflow_event(
                "write_action_invalid_request", case_id=case_id, requested_new_status=new_status
            )
            return {
                "termination_reason": "invalid_request",
                "final_answer": (
                    "I could not determine a valid case id and target status from that "
                    "request; no action was taken."
                ),
                "citations": [],
            }
        now = datetime.now(UTC)
        operation_id = str(uuid.uuid4())
        pending: PendingCaseAction = {
            "action_type": "update_case_status",
            "case_id": case_id,
            "new_status": new_status,
            "requested_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=deps.approval_timeout_seconds)).isoformat(),
            "operation_id": operation_id,
            "approval_state": "pending",
        }
        audit.log_workflow_event(
            "write_action_requested",
            operation_id=operation_id,
            case_id=case_id,
            requested_new_status=new_status,
        )
        return {"pending_action": pending}

    return validate_write_request


def make_wait_for_approval_node(deps: GraphDeps):
    """Build the `wait_for_approval` node: the one interrupt point in this graph.

    `interrupt(payload)` pauses the run and durably checkpoints it; the
    value it returns on resume is whatever `Command(resume=...)` supplied
    (see `langgraph_experiment.cli.approval_cli`, the only code path that
    constructs that `Command`). That payload is trusted only if it also
    carries a role in `deps.approval_roles` -- a resume payload claiming
    `"decision": "approve"` is never honored on its own, matching the
    prompt spec's "do not accept approved=true simply because the LLM
    generated it" requirement, and mirroring the production MCP path's
    own belt-and-suspenders role re-check in
    `rag.mcp.business.approvals.resolve_case_action_approvals`. No LLM
    node in this graph ever produces or sees this payload at all.

    Two production-hardening checks run before the decision itself is
    even looked at, both server-side (never trusting a CLI-side
    pre-check alone, the same posture already applied to
    `approver_roles`): `workflow_timed_out` (elapsed time since
    `state["workflow_started_at"]` against `deps.
    max_workflow_duration_seconds`) and `approval_expired` (`now` against
    `pending["expires_at"]`). A `"decision": "cancel"` resume payload is
    its own terminal branch, distinct from `"reject"` -- see
    `state.ApprovalState`'s values.
    """

    def wait_for_approval(state: GraphState) -> dict:
        pending = state["pending_action"]
        assert pending is not None
        raw_decision = interrupt(
            {
                "action_type": pending["action_type"],
                "case_id": pending["case_id"],
                "new_status": pending["new_status"],
                "requested_at": pending["requested_at"],
                "expires_at": pending["expires_at"],
                "operation_id": pending["operation_id"],
                "requested_by_tenant": state.get("tenant_id"),
                "requested_by_roles": list(state.get("roles") or []),
                "prompt": (
                    f"Approve changing case {pending['case_id']} to "
                    f"'{pending['new_status']}'? (approve/reject/cancel)"
                ),
            }
        )
        decision = raw_decision if isinstance(raw_decision, dict) else {}
        now = datetime.now(UTC)
        updated_pending = dict(pending)

        workflow_started_at = datetime.fromisoformat(state["workflow_started_at"])
        expires_at = datetime.fromisoformat(pending["expires_at"])
        workflow_timed_out = (
            now - workflow_started_at
        ).total_seconds() > deps.max_workflow_duration_seconds
        approval_expired = now > expires_at

        if workflow_timed_out:
            updated_pending["approval_state"] = "workflow_timeout"
            audit.log_workflow_event(
                "workflow_timed_out",
                operation_id=pending["operation_id"],
                case_id=pending["case_id"],
            )
        elif approval_expired:
            updated_pending["approval_state"] = "expired"
            audit.log_workflow_event(
                "approval_expired",
                operation_id=pending["operation_id"],
                case_id=pending["case_id"],
                expires_at=pending["expires_at"],
            )
        elif decision.get("decision") == "cancel":
            updated_pending["approval_state"] = "cancelled"
            audit.log_workflow_event(
                "workflow_cancelled",
                operation_id=pending["operation_id"],
                case_id=pending["case_id"],
            )
        else:
            approver_roles = set(decision.get("approver_roles") or [])
            wants_approve = decision.get("decision") == "approve"
            is_approved = wants_approve and bool(approver_roles & set(deps.approval_roles))
            if is_approved:
                updated_pending["approval_state"] = "approved"
                audit.log_workflow_event(
                    "approval_granted",
                    operation_id=pending["operation_id"],
                    case_id=pending["case_id"],
                )
            elif wants_approve:
                # Claimed approval, but the resuming identity held no
                # approval role -- fails closed rather than trusting the
                # claim.
                updated_pending["approval_state"] = "denied_insufficient_role"
                audit.log_workflow_event(
                    "approval_denied_insufficient_role",
                    operation_id=pending["operation_id"],
                    case_id=pending["case_id"],
                )
            else:
                updated_pending["approval_state"] = "rejected"
                audit.log_workflow_event(
                    "approval_rejected",
                    operation_id=pending["operation_id"],
                    case_id=pending["case_id"],
                )
        return {"pending_action": updated_pending, "approval_decision": decision}

    return wait_for_approval


def make_execute_write_action_node(deps: GraphDeps):
    """Build the `execute_write_action` node: only reached once approval is confirmed.

    Calls `rag.mcp.business.store.update_case_status` unmodified, with
    the same identity/cross-tenant rules every other case-store caller
    uses, plus the one `CaseApproval` this exact request was approved
    for. `update_case_status` re-derives whether that approval was even
    necessary (only `resolved -> closed` requires one) and re-checks the
    transition table itself; nothing here assumes the transition is
    valid just because a human approved pausing on it.

    When `deps.action_ledger` is set, the mutation is dispatched through
    `ActionLedger.record_attempt` rather than called directly -- guards
    against a duplicate resume/automatic-retry re-running the mutation,
    including recovering safely from a crash between the mutation
    executing and the ledger recording it (see `idempotency.py`'s module
    docstring for the full guarantee). Registered with its own
    `RetryPolicy` in `graph.build_graph` -- safe specifically *because*
    the ledger check happens first on every attempt, including automatic
    retries; a `RetryPolicy` on a write node with no such guard would be
    a real double-mutation risk, not a convenience (see that policy's own
    comment in `graph.py`).
    """

    def execute_write_action(state: GraphState) -> dict:
        pending = state["pending_action"]
        assert pending is not None
        identity = _identity_from_state(state)
        approval = CaseApproval(case_id=pending["case_id"], new_status=pending["new_status"])

        def mutate() -> Any:
            return deps.update_case_status_fn(
                pending["case_id"],
                pending["new_status"],
                identity,
                list(deps.cross_tenant_support_roles),
                [approval],
            )

        if deps.action_ledger is not None:
            result = deps.action_ledger.record_attempt(
                pending["operation_id"],
                thread_id=state.get("thread_id") or "",
                case_id=pending["case_id"],
                new_status=pending["new_status"],
                mutate=mutate,
            )
        else:
            result = mutate()

        if result is None:
            return {"case_action_outcome": None}
        if isinstance(result, _ReplayedOutcome):
            audit.log_workflow_event(
                "write_action_ledger_replay",
                operation_id=pending["operation_id"],
                case_id=pending["case_id"],
                outcome=result.outcome,
            )
            outcome_dict = {
                "outcome": result.outcome,
                "case_id": pending["case_id"],
                "previous_status": result.previous_status,
                "new_status": pending["new_status"],
            }
        else:
            outcome_dict = result.model_dump(mode="json")
        audit.log_workflow_event(
            "write_action_executed",
            operation_id=pending["operation_id"],
            case_id=pending["case_id"],
            outcome=outcome_dict["outcome"],
        )
        return {"case_action_outcome": outcome_dict}

    return execute_write_action


_OUTCOME_TEXT = {
    "executed": "Case {case_id} was updated from '{previous_status}' to '{new_status}'.",
    "already_in_status": "Case {case_id} is already '{new_status}'; no change was needed.",
    "invalid_transition": (
        "Cannot move case {case_id} from '{previous_status}' to '{new_status}'; no change was made."
    ),
    "approval_required": (
        "Case {case_id}'s transition to '{new_status}' requires prior approval; no change was made."
    ),
}


def synthesize_write_result(state: GraphState) -> dict:
    """`synthesize_write_result` node: the single terminal formatter for the write branch.

    Reached from three different predecessors (an invalid request that
    never paused, a rejected/role-denied approval, or a real
    `execute_write_action` outcome); handles all three without
    overwriting a `final_answer` `validate_write_request` already set for
    the first case.
    """
    if state.get("termination_reason") == "invalid_request":
        return {}
    pending = state.get("pending_action") or {}
    approval_state = pending.get("approval_state")
    if approval_state == "rejected":
        return {
            "final_answer": (
                f"The request to update case {pending.get('case_id')} was rejected; "
                "no change was made."
            ),
            "citations": [],
            "termination_reason": "rejected",
        }
    if approval_state == "denied_insufficient_role":
        return {
            "final_answer": (
                f"The approval for case {pending.get('case_id')} was not honored: the "
                "approving identity did not hold an approval role. No change was made."
            ),
            "citations": [],
            "termination_reason": "approval_denied_insufficient_role",
        }
    if approval_state == "expired":
        return {
            "final_answer": (
                f"The approval window for case {pending.get('case_id')} expired before a "
                "decision was recorded; no change was made."
            ),
            "citations": [],
            "termination_reason": "approval_expired",
        }
    if approval_state == "workflow_timeout":
        return {
            "final_answer": (
                f"This workflow exceeded its maximum allowed duration before a decision "
                f"was recorded for case {pending.get('case_id')}; no change was made."
            ),
            "citations": [],
            "termination_reason": "workflow_timed_out",
        }
    if approval_state == "cancelled":
        return {
            "final_answer": (
                f"The request to update case {pending.get('case_id')} was cancelled; "
                "no change was made."
            ),
            "citations": [],
            "termination_reason": "cancelled",
        }
    outcome = state.get("case_action_outcome")
    if outcome is None:
        case_id = pending.get("case_id")
        return {
            "final_answer": (
                f"Case {case_id} was not found, or you are not authorized to modify it."
            ),
            "citations": [],
            "termination_reason": "case_not_found_or_denied",
        }
    citation: Citation = {
        "chunk_id": None,
        "case_id": outcome["case_id"],
        "source": f"case:{outcome['case_id']}",
        "score": None,
    }
    return {
        "final_answer": _OUTCOME_TEXT[outcome["outcome"]].format(**outcome),
        "citations": [citation],
        "termination_reason": outcome["outcome"],
    }
