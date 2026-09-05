"""Synthetic customer-support-case shapes.

Separate from `rag.schemas`/`rag.mcp.schemas`: a case is not a
retrieved chunk and has no `chunk_id`/`score`/`origin`/embedding
concept. `CaseStatusResult` is a distinct, narrower model rather than
`CustomerCase` with fields omitted at the call site, so
`get_case_status` demonstrates a genuinely narrow-scope read tool over
the same resource `get_customer_case` reads in full.

`CaseApproval` and `CaseActionOutcome` back the `update_case_status`
write-action tool. `MAX_CASE_APPROVALS` is the hard schema-level ceiling
on how many approvals a single request/token may carry; see
`rag.config.BusinessActionConfig.max_case_approvals_per_request` for the
smaller, operator-configurable limit enforced at the API boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CaseStatus = Literal["open", "in_progress", "resolved", "closed"]
CasePriority = Literal["low", "medium", "high", "urgent"]
CaseActionOutcomeType = Literal[
    "executed", "already_in_status", "invalid_transition", "approval_required"
]

MAX_CASE_APPROVALS = 10


class CustomerCase(BaseModel):
    """One synthetic customer-support case, already authorized for the caller.

    Attributes
    ----------
    case_id : str
        Stable identifier, e.g. `"CASE-1001"`.
    tenant_id : str
        Owning tenant. Never `None`: unlike document chunks, every case
        has a concrete tenant, with no legacy "visible to everyone"
        state to preserve.
    """

    case_id: str
    tenant_id: str
    customer_name: str
    subject: str
    description: str
    status: CaseStatus
    priority: CasePriority
    assigned_team: str
    created_at: datetime
    updated_at: datetime


class CaseStatusResult(BaseModel):
    """The narrow, status-only projection of a `CustomerCase`."""

    case_id: str
    status: CaseStatus
    priority: CasePriority
    updated_at: datetime


class CaseApproval(BaseModel):
    """A trusted, pre-authorized grant to perform one exact case status transition.

    Never accepted from a tool argument. Names an exact `(case_id,
    new_status)` pair, so an approval minted for one case/transition
    cannot authorize a different one; there is no separate "used"
    flag or nonce needed for that guarantee.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=64)
    new_status: CaseStatus


class CaseActionOutcome(BaseModel):
    """Result of one `update_case_status` request.

    `outcome` is always one of a fixed set: `"executed"` (the mutation
    happened), `"already_in_status"` (no-op, deterministic), `"invalid_
    transition"` (rejected, no mutation), or `"approval_required"`
    (rejected pending approval, no mutation). `previous_status` is the
    case's status before this request was evaluated.
    """

    outcome: CaseActionOutcomeType
    case_id: str
    previous_status: CaseStatus
    new_status: CaseStatus
    updated_at: datetime
