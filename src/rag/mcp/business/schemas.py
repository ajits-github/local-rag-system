"""Synthetic customer-support-case shapes.

Deliberately separate from `rag.schemas`/`rag.mcp.schemas`: a case is not
a retrieved chunk and carries no `chunk_id`/`score`/`origin`/embedding
concept at all. `CaseStatusResult` is a second, narrower shape (not just
`CustomerCase` with fields omitted at the call site) so `get_case_status`
demonstrates a real narrow-scope read tool over the same backend resource
`get_customer_case` reads in full -- a deliberate two-tool example, not
one tool with an optional "fields" argument.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

CaseStatus = Literal["open", "in_progress", "resolved", "closed"]
CasePriority = Literal["low", "medium", "high", "urgent"]


class CustomerCase(BaseModel):
    """One synthetic customer-support case, already authorized for the caller.

    Attributes
    ----------
    case_id : str
        Stable identifier, e.g. `"CASE-1001"`.
    tenant_id : str
        Owning tenant. Never `None` -- unlike document chunks, every
        synthetic case has a concrete tenant, so there is no
        pre-governance "visible to everyone" legacy state to preserve.
    customer_name : str
        Synthetic customer name.
    subject : str
        One-line case subject.
    description : str
        Case body text.
    status : CaseStatus
        Current case status.
    priority : CasePriority
        Case priority.
    assigned_team : str
        Synthetic owning team name.
    created_at, updated_at : datetime
        Case timestamps.
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
    """The narrow, status-only projection of a `CustomerCase`.

    Attributes
    ----------
    case_id : str
        Stable identifier.
    status : CaseStatus
        Current case status.
    priority : CasePriority
        Case priority.
    updated_at : datetime
        When the case was last updated.
    """

    case_id: str
    status: CaseStatus
    priority: CasePriority
    updated_at: datetime
