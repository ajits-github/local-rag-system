"""Unit tests for `rag.mcp.business.store.update_case_status`, the one write action.

Mirrors `test_mcp_business_case_store.py`'s "plain function, no MCP
protocol needed" convention. `_reset_case_store` restores the shared
in-memory `_SYNTHETIC_CASES` dict after every test, since
`update_case_status` is the first function in this module that mutates
it and other test modules assume its seeded, unmutated state.
"""

from __future__ import annotations

import copy
import logging

import pytest

from rag.api.auth import VerifiedIdentity
from rag.mcp.business import store as store_module
from rag.mcp.business.schemas import CaseApproval
from rag.mcp.business.store import update_case_status

_SUPPORT_ROLES = ["techfusion_support"]


def _identity(tenant_id: str, roles: list[str]) -> VerifiedIdentity:
    return VerifiedIdentity(subject="alice", tenant_id=tenant_id, roles=roles)


@pytest.fixture(autouse=True)
def _reset_case_store():
    original = copy.deepcopy(store_module._SYNTHETIC_CASES)
    yield
    store_module._SYNTHETIC_CASES.clear()
    store_module._SYNTHETIC_CASES.update(original)


def _status_of(case_id: str) -> str:
    return store_module._SYNTHETIC_CASES[case_id].status


# --- authorization -------------------------------------------------------------


def test_same_tenant_correct_role_succeeds():
    """CASE-1001 (in_progress) is a valid open transition an operator may make."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = update_case_status("CASE-1001", "resolved", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "executed"
    assert _status_of("CASE-1001") == "resolved"


def test_wrong_role_within_same_tenant_is_denied():
    """CASE-1002 is admin-only; an operator in the same tenant gets None, no mutation."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = update_case_status("CASE-1002", "in_progress", identity, _SUPPORT_ROLES)
    assert outcome is None
    assert _status_of("CASE-1002") == "open"


def test_cross_tenant_without_support_role_is_denied():
    """A caller from another tenant, with no support role, cannot touch CASE-2001."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES)
    assert outcome is None
    assert _status_of("CASE-2001") == "resolved"


def test_privileged_cross_tenant_role_listed_on_case_succeeds():
    """techfusion_support is on CASE-2002's own allowed_roles: cross-tenant access granted."""
    identity = _identity("tenant_alpha", ["techfusion_support"])
    outcome = update_case_status("CASE-2002", "in_progress", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "executed"
    assert _status_of("CASE-2002") == "in_progress"


def test_unauthenticated_caller_is_denied():
    """identity=None is a hard denial here, unlike the two read tools' unrestricted default."""
    outcome = update_case_status("CASE-1001", "resolved", None, _SUPPORT_ROLES)
    assert outcome is None
    assert _status_of("CASE-1001") == "in_progress"


def test_unknown_case_id_returns_none():
    """A case_id with no matching record is a plain None, indistinguishable from a denial."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert update_case_status("CASE-DOES-NOT-EXIST", "resolved", identity, _SUPPORT_ROLES) is None


# --- business rules --------------------------------------------------------------


def test_invalid_transition_is_rejected_deterministically():
    """CASE-1003 is closed (terminal); closed -> open is not in the transition table."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = update_case_status("CASE-1003", "open", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "invalid_transition"
    assert outcome.previous_status == "closed"
    assert _status_of("CASE-1003") == "closed"


def test_already_in_target_status_is_a_deterministic_no_op():
    """Requesting the case's current status is a defined no-op, not an error or a mutation."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = update_case_status("CASE-1001", "in_progress", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "already_in_status"
    assert _status_of("CASE-1001") == "in_progress"


def test_skipping_a_state_is_rejected():
    """Open -> resolved skips in_progress; not a valid single transition."""
    identity = _identity("tenant_alpha", ["tenant_alpha_admin"])
    outcome = update_case_status("CASE-1002", "resolved", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "invalid_transition"
    assert _status_of("CASE-1002") == "open"


# --- approval ----------------------------------------------------------------------


def test_sensitive_transition_without_approval_requires_approval_and_does_not_mutate():
    """CASE-2001 is resolved; resolved -> closed is sensitive and needs prior approval."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    outcome = update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "approval_required"
    assert _status_of("CASE-2001") == "resolved"


def test_matching_approval_permits_execution():
    """An approval naming the exact (case_id, new_status) pair allows the mutation."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    approvals = [CaseApproval(case_id="CASE-2001", new_status="closed")]
    outcome = update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES, approvals)
    assert outcome is not None
    assert outcome.outcome == "executed"
    assert _status_of("CASE-2001") == "closed"


def test_approval_for_a_different_case_does_not_authorize_this_one():
    """An approval scoped to a different case_id never applies here."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    approvals = [CaseApproval(case_id="CASE-9999", new_status="closed")]
    outcome = update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES, approvals)
    assert outcome is not None
    assert outcome.outcome == "approval_required"
    assert _status_of("CASE-2001") == "resolved"


def test_approval_for_a_different_transition_does_not_authorize_this_one():
    """An approval for the same case but a different target status never applies here."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    approvals = [CaseApproval(case_id="CASE-2001", new_status="resolved")]
    outcome = update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES, approvals)
    assert outcome is not None
    assert outcome.outcome == "approval_required"
    assert _status_of("CASE-2001") == "resolved"


def test_update_case_status_args_has_no_approval_field():
    """Schema-level proof: the LLM cannot smuggle an approval flag into tool arguments."""
    import pydantic

    from rag.agent.tool_schemas import UpdateCaseStatusArgs

    for field in ("approved", "approval", "approval_token", "case_approvals", "auth"):
        assert field not in UpdateCaseStatusArgs.model_fields
    with pytest.raises(pydantic.ValidationError):
        UpdateCaseStatusArgs(case_id="CASE-2001", new_status="closed", approved=True)


# --- audit ---------------------------------------------------------------------------


def test_executed_transition_logs_requested_and_executed_events(caplog):
    """A successful mutation logs both case_action_requested and case_action_executed."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        update_case_status("CASE-1001", "resolved", identity, _SUPPORT_ROLES)

    messages = [r.getMessage() for r in caplog.records]
    assert "case_action_requested" in messages
    assert "case_action_executed" in messages
    executed = next(r for r in caplog.records if r.getMessage() == "case_action_executed")
    assert executed.case_id == "CASE-1001"
    assert executed.from_status == "in_progress"
    assert executed.to_status == "resolved"


def test_approval_required_logs_the_matching_event(caplog):
    """A sensitive transition without approval logs case_action_approval_required, not executed."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES)

    messages = [r.getMessage() for r in caplog.records]
    assert "case_action_approval_required" in messages
    assert "case_action_executed" not in messages


def test_invalid_transition_logs_the_matching_event(caplog):
    """A rejected transition logs case_action_invalid_transition, not executed."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        update_case_status("CASE-1003", "open", identity, _SUPPORT_ROLES)

    messages = [r.getMessage() for r in caplog.records]
    assert "case_action_invalid_transition" in messages
    assert "case_action_executed" not in messages


def test_unauthenticated_denial_logs_authorization_denied_not_case_action_requested(caplog):
    """An unauthenticated caller never reaches the case_action_requested log line."""
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        update_case_status("CASE-1001", "resolved", None, _SUPPORT_ROLES)

    messages = [r.getMessage() for r in caplog.records]
    assert "authorization_denied" in messages
    assert "case_action_requested" not in messages


def test_audit_events_never_carry_case_description_or_subject_text(caplog):
    """Only ids/status names/a pseudonymous subject hash are logged, never case content."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        update_case_status("CASE-1001", "resolved", identity, _SUPPORT_ROLES)

    for record in caplog.records:
        text = repr(record.__dict__)
        assert "Login failures" not in text  # CASE-1001's real description text
        assert "alice" not in text  # the raw subject, never logged unhashed
