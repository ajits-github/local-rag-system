"""Unit tests for `rag.mcp.business.store`: the synthetic business-case backend's own ACL.

No MCP protocol or transport machinery needed here -- `get_customer_case`/
`get_case_status` are plain functions taking a `case_id`, a resolved
`VerifiedIdentity | None`, and `cross_tenant_support_roles`. Mirrors
`test_mcp_identity.py`'s "no protocol needed, plain-function unit test"
convention. Real-wire-protocol coverage (including that an unauthorized
case and a nonexistent one are indistinguishable over the actual MCP
transport) lives in `tests/integration/test_mcp_end_to_end.py`.
"""

from __future__ import annotations

import logging

from rag.api.auth import VerifiedIdentity
from rag.mcp.business.store import get_case_status, get_customer_case

_SUPPORT_ROLES = ["techfusion_support"]


def _identity(tenant_id: str, roles: list[str]) -> VerifiedIdentity:
    return VerifiedIdentity(subject="alice", tenant_id=tenant_id, roles=roles)


def test_get_customer_case_returns_none_for_an_unknown_case_id():
    """A case_id with no matching record at all is a plain None, not an exception."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert get_customer_case("CASE-DOES-NOT-EXIST", identity, _SUPPORT_ROLES) is None


def test_get_customer_case_same_tenant_matching_role_is_authorized():
    """Own tenant plus a role on the case's allowed_roles: granted."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    case = get_customer_case("CASE-1001", identity, _SUPPORT_ROLES)
    assert case is not None
    assert case.case_id == "CASE-1001"
    assert case.tenant_id == "tenant_alpha"


def test_get_customer_case_same_tenant_role_mismatch_is_denied():
    """CASE-1002 is admin-only within tenant_alpha; an operator in the same tenant is denied.

    Proves tenant match alone is never sufficient, mirroring
    `test_role_mismatch_within_same_tenant_is_still_denied` for document
    ACL.
    """
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert get_customer_case("CASE-1002", identity, _SUPPORT_ROLES) is None


def test_get_customer_case_same_tenant_correct_role_is_authorized():
    """The matching role (admin, not operator) for CASE-1002 is granted."""
    identity = _identity("tenant_alpha", ["tenant_alpha_admin"])
    case = get_customer_case("CASE-1002", identity, _SUPPORT_ROLES)
    assert case is not None
    assert case.case_id == "CASE-1002"


def test_get_customer_case_cross_tenant_without_support_role_is_denied():
    """A plain operator role from another tenant, with no support role at all, is denied."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert get_customer_case("CASE-2001", identity, _SUPPORT_ROLES) is None


def test_get_customer_case_cross_tenant_support_role_listed_on_case_is_authorized():
    """techfusion_support matches both the config allow-list and CASE-2002's own ACL."""
    identity = _identity("tenant_alpha", ["techfusion_support"])
    case = get_customer_case("CASE-2002", identity, _SUPPORT_ROLES)
    assert case is not None
    assert case.case_id == "CASE-2002"
    assert case.tenant_id == "tenant_beta"


def test_get_customer_case_cross_tenant_support_role_not_listed_on_case_is_denied():
    """techfusion_support is in the config allow-list but CASE-2001 doesn't list it -- still denied.

    Mirrors `test_support_role_not_listed_on_document_cannot_access_it`
    for document ACL: a config-level allow-list role only works when the
    target resource's own ACL also names it.
    """
    identity = _identity("tenant_alpha", ["techfusion_support"])
    assert get_customer_case("CASE-2001", identity, _SUPPORT_ROLES) is None


def test_get_customer_case_unrestricted_when_identity_is_none():
    """No verified identity (auth disabled): fully unrestricted, matching every other surface."""
    case = get_customer_case("CASE-1002", None, _SUPPORT_ROLES)
    assert case is not None
    assert case.case_id == "CASE-1002"


def test_get_case_status_applies_the_same_authorization_rule():
    """get_case_status is denied/authorized exactly like get_customer_case for the same case."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert get_case_status("CASE-1002", identity, _SUPPORT_ROLES) is None
    status = get_case_status("CASE-1001", identity, _SUPPORT_ROLES)
    assert status is not None
    assert status.case_id == "CASE-1001"


def test_get_case_status_projection_has_no_subject_or_description_fields():
    """The narrow shape genuinely doesn't carry the fields get_customer_case exposes."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    status = get_case_status("CASE-1001", identity, _SUPPORT_ROLES)
    assert status is not None
    assert set(status.model_dump().keys()) == {"case_id", "status", "priority", "updated_at"}


def test_denied_access_logs_an_authorization_denied_audit_event(caplog):
    """Unlike document-level ACL's documented gap, a real per-case denial IS observable here."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        result = get_customer_case("CASE-1002", identity, _SUPPORT_ROLES)

    assert result is None
    denial_records = [r for r in caplog.records if r.getMessage() == "authorization_denied"]
    assert len(denial_records) == 1
    assert denial_records[0].action == "get_customer_case"
    assert denial_records[0].case_id == "CASE-1002"
    assert denial_records[0].tenant_id == "tenant_alpha"


def test_nonexistent_case_id_logs_no_authorization_denied_event(caplog):
    """A case that simply doesn't exist is not a security event -- only a real denial is."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        result = get_customer_case("CASE-DOES-NOT-EXIST", identity, _SUPPORT_ROLES)

    assert result is None
    assert not [r for r in caplog.records if r.getMessage() == "authorization_denied"]
