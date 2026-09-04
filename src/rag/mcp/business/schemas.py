"""Synthetic customer-support-case shapes.

Separate from `rag.schemas`/`rag.mcp.schemas`: a case is not a
retrieved chunk and has no `chunk_id`/`score`/`origin`/embedding
concept. `CaseStatusResult` is a distinct, narrower model rather than
`CustomerCase` with fields omitted at the call site, so
`get_case_status` demonstrates a genuinely narrow-scope read tool over
the same resource `get_customer_case` reads in full.
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
