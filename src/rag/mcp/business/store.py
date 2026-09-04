"""Synthetic customer-support-case backend, with its own tenant/role authorization.

Read-only, in-memory, and self-contained (no Postgres/network
dependency): a stand-in for a separate backend system an MCP server
might front, not a real case-management system.

Authorization here does not reuse
`rag.retrieval.authorization.AuthorizationContext` (no document/
freshness/trust concept applies), and is unconditional (no config
kill-switch), since every case has a concrete tenant with no legacy
"unrestricted" state to preserve. The rule mirrors the document
predicate: own tenant plus a matching `allowed_roles` entry, or a
`cross_tenant_support_roles` role that is also on the case's own
`allowed_roles`.

An unauthorized case and a nonexistent case both resolve to `None`, so
a caller can never distinguish the two from the response. Unlike
document-level ACL (a silent SQL filter), the lookup happens in Python,
so a real denial is still audit-logged as `authorization_denied`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.audit import log_audit_event, pseudonymous_subject
from rag.mcp.business.schemas import CasePriority, CaseStatus, CaseStatusResult, CustomerCase


class _CaseRecord(BaseModel):
    """Internal seed record: every `CustomerCase` field plus the ACL-only `allowed_roles`.

    `allowed_roles` is never serialized back to a caller; it exists only
    for `_is_authorized` to check against.
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
    allowed_roles: list[str]

    def to_public(self) -> CustomerCase:
        """Return the full, client-facing case shape (no `allowed_roles`)."""
        return CustomerCase(**self.model_dump(exclude={"allowed_roles"}))

    def to_status(self) -> CaseStatusResult:
        """Return the narrow, status-only client-facing shape."""
        return CaseStatusResult(
            case_id=self.case_id,
            status=self.status,
            priority=self.priority,
            updated_at=self.updated_at,
        )


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_SYNTHETIC_CASES: dict[str, _CaseRecord] = {
    record.case_id: record
    for record in (
        _CaseRecord(
            case_id="CASE-1001",
            tenant_id="tenant_alpha",
            customer_name="Acme Corp",
            subject="Login failures after SSO migration",
            description=(
                "Users report intermittent login failures since the SAML "
                "provider cutover on 2026-08-20. Affects roughly 5% of "
                "sign-in attempts, concentrated in the EU region."
            ),
            status="in_progress",
            priority="high",
            assigned_team="Platform Support",
            created_at=_dt("2026-08-21T09:15:00"),
            updated_at=_dt("2026-08-27T14:02:00"),
            allowed_roles=["tenant_alpha_operator", "tenant_alpha_admin"],
        ),
        _CaseRecord(
            case_id="CASE-1002",
            tenant_id="tenant_alpha",
            customer_name="Acme Corp",
            subject="Invoice discrepancy for March billing cycle",
            description=(
                "Customer disputes a $340 overcharge on the March invoice, "
                "tied to a seat-count reconciliation error. Escalated to "
                "billing; admin-only per this tenant's finance-data policy."
            ),
            status="open",
            priority="medium",
            assigned_team="Billing",
            created_at=_dt("2026-08-25T11:40:00"),
            updated_at=_dt("2026-08-26T08:10:00"),
            # Admin-only within tenant_alpha, deliberately excluding
            # tenant_alpha_operator: proves same-tenant access still
            # requires a matching role, not just a matching tenant.
            allowed_roles=["tenant_alpha_admin"],
        ),
        _CaseRecord(
            case_id="CASE-1003",
            tenant_id="tenant_alpha",
            customer_name="Acme Corp",
            subject="Feature request: bulk export API",
            description="Customer requests a bulk CSV export endpoint. Closed as planned work.",
            status="closed",
            priority="low",
            assigned_team="Product",
            created_at=_dt("2026-07-30T10:00:00"),
            updated_at=_dt("2026-08-05T16:45:00"),
            allowed_roles=["tenant_alpha_operator", "tenant_alpha_admin"],
        ),
        _CaseRecord(
            case_id="CASE-2001",
            tenant_id="tenant_beta",
            customer_name="Globex Ltd",
            subject="API rate limit increase request",
            description="Requested a rate limit increase from 100 to 500 req/min. Approved.",
            status="resolved",
            priority="low",
            assigned_team="Platform Support",
            created_at=_dt("2026-08-10T13:20:00"),
            updated_at=_dt("2026-08-12T09:00:00"),
            allowed_roles=["tenant_beta_operator", "tenant_beta_admin"],
        ),
        _CaseRecord(
            case_id="CASE-2002",
            tenant_id="tenant_beta",
            customer_name="Globex Ltd",
            subject="Data export failing for large datasets",
            description=(
                "Exports over 2GB time out at the load balancer. Under "
                "active investigation; also visible to cross-tenant "
                "support since it may share a root cause with CASE-1001."
            ),
            status="open",
            priority="urgent",
            assigned_team="Platform Support",
            created_at=_dt("2026-08-28T07:55:00"),
            updated_at=_dt("2026-08-29T17:30:00"),
            # Includes techfusion_support: a support engineer holding that
            # role (and appearing in security.authorization.cross_tenant_
            # support_roles) may access this case from outside tenant_beta.
            allowed_roles=["tenant_beta_operator", "tenant_beta_admin", "techfusion_support"],
        ),
    )
}


def _is_authorized(
    case: _CaseRecord, identity: VerifiedIdentity | None, cross_tenant_support_roles: list[str]
) -> bool:
    """Mirror the document ACL rule: own tenant + matching role, or an allowed cross-tenant role.

    `identity is None` means unrestricted, the same "no identity
    asserted, no restriction" convention every authorization surface in
    this codebase uses.
    """
    if identity is None:
        return True
    role_set = set(identity.roles)
    allowed_role_set = set(case.allowed_roles)
    if identity.tenant_id == case.tenant_id:
        return bool(role_set & allowed_role_set)
    support_roles = role_set & set(cross_tenant_support_roles)
    return bool(support_roles & allowed_role_set)


def _lookup_authorized(
    case_id: str,
    identity: VerifiedIdentity | None,
    cross_tenant_support_roles: list[str],
    *,
    action: str,
) -> _CaseRecord | None:
    """Look up a case and enforce authorization, audit-logging a denial."""
    case = _SYNTHETIC_CASES.get(case_id)
    if case is None:
        return None
    if not _is_authorized(case, identity, cross_tenant_support_roles):
        assert identity is not None  # _is_authorized(..., None, ...) is always True
        log_audit_event(
            "authorization_denied",
            subject=pseudonymous_subject(identity.subject),
            action=action,
            tenant_id=identity.tenant_id,
            case_id=case_id,
        )
        return None
    return case


def get_customer_case(
    case_id: str, identity: VerifiedIdentity | None, cross_tenant_support_roles: list[str]
) -> CustomerCase | None:
    """Fetch a case's full detail, or `None` if it doesn't exist or the caller may not see it.

    Parameters
    ----------
    case_id : str
        The case identifier, e.g. `"CASE-1001"`.
    identity : VerifiedIdentity | None
        The resolved caller identity (see `rag.mcp.identity`), or `None`
        when `security.auth.enabled` is `False`.
    cross_tenant_support_roles : list[str]
        `config.security.authorization.cross_tenant_support_roles`,
        reused as-is, not a second privilege list.

    Returns
    -------
    CustomerCase | None
        `None` for both "no such case" and "case exists, caller not
        authorized", indistinguishable by design.
    """
    case = _lookup_authorized(
        case_id, identity, cross_tenant_support_roles, action="get_customer_case"
    )
    return case.to_public() if case is not None else None


def get_case_status(
    case_id: str, identity: VerifiedIdentity | None, cross_tenant_support_roles: list[str]
) -> CaseStatusResult | None:
    """Fetch a case's narrow status projection. Same authorization rule as `get_customer_case`.

    Parameters
    ----------
    case_id, identity, cross_tenant_support_roles
        Same as `get_customer_case`.

    Returns
    -------
    CaseStatusResult | None
        `None` for both "no such case" and "not authorized".
    """
    case = _lookup_authorized(
        case_id, identity, cross_tenant_support_roles, action="get_case_status"
    )
    return case.to_status() if case is not None else None
