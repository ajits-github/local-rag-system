"""Unit tests for `rag.mcp.business.approvals.resolve_case_action_approvals`.

No MCP protocol machinery needed -- a plain function taking a headers
mapping (or nothing) and an `AppConfig`, mirroring `test_mcp_identity.py`'s
JWT-construction conventions. Covers the fix for a real bypass: a caller
presenting an ordinary (non-internal-service) token that happens to carry
a `case_approvals` claim must never have it honored, since that would
let a direct MCP caller skip `rag.api.request_auth.resolve_case_approvals`'s
role gate entirely by never going through `POST /agent/query` at all.
"""

from __future__ import annotations

import time

import jwt

from rag.config import load_config
from rag.mcp.business.approvals import resolve_case_action_approvals

_SECRET = "mcp-business-approvals-unit-test-secret"

_APPROVAL = {"case_id": "CASE-2001", "new_status": "closed"}


def _internal_service_token(**claim_overrides):
    """Build a token shaped exactly like `mint_internal_token`'s output."""
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "rag-agent-internal",
        "tenant_id": "tenant_beta",
        "roles": ["case_status_approver"],
        "iat": now,
        "exp": now + 60,
        "iss": "rag-agent-internal",
        "token_use": "mcp_internal_service",
        "case_approvals": [_APPROVAL],
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _ordinary_user_token(**claim_overrides):
    """Build a token shaped like a normal end-user's own credential."""
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_beta",
        "roles": ["tenant_beta_operator"],
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _auth_config(**overrides):
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    security = config.security.model_copy(update={"auth": auth_config})
    config = config.model_copy(update={"security": security})
    return config.model_copy(update=overrides) if overrides else config


def _headers(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_returns_empty_list_when_auth_disabled():
    """auth.enabled=False: no token check at all, always no approvals."""
    config = load_config()
    assert config.security.auth.enabled is False
    assert resolve_case_action_approvals(_headers(_internal_service_token()), config) == []


def test_returns_empty_list_with_no_header(monkeypatch):
    """No Authorization header at all: nothing to resolve."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    assert resolve_case_action_approvals(None, config) == []
    assert resolve_case_action_approvals({}, config) == []


def test_returns_empty_list_for_an_unverifiable_token(monkeypatch):
    """A malformed token fails to decode; treated as no approvals, not an error."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    assert resolve_case_action_approvals(_headers("not-a-real-jwt"), config) == []


def test_accepts_a_genuine_internal_service_token(monkeypatch):
    """A token shaped exactly like mint_internal_token's output is honored."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    approvals = resolve_case_action_approvals(_headers(_internal_service_token()), config)
    assert len(approvals) == 1
    assert approvals[0].case_id == "CASE-2001"
    assert approvals[0].new_status == "closed"


def test_rejects_an_ordinary_user_token_carrying_a_case_approvals_claim(monkeypatch):
    """The exact bypass this fix closes: a non-internal token must never be honored.

    Without the sub/token_use check, a caller able to mint themselves an
    ordinary token (this deployment's tokens are minted and verified in
    the same trust domain) could attach case_approvals directly and call
    the MCP server without ever going through the role-gated
    POST /agent/query boundary at all.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    token = _ordinary_user_token(
        roles=["case_status_approver"], case_approvals=[_APPROVAL]
    )  # even holding the approval role and the claim itself, sub is wrong
    assert resolve_case_action_approvals(_headers(token), config) == []


def test_rejects_a_token_with_the_right_subject_but_no_token_use_marker(monkeypatch):
    """Sub alone is not sufficient; the token_use marker must also match."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    now = int(time.time())
    claims = {
        "sub": "rag-agent-internal",
        "tenant_id": "tenant_beta",
        "roles": ["case_status_approver"],
        "iat": now,
        "exp": now + 60,
        "case_approvals": [_APPROVAL],
    }
    token = jwt.encode(claims, _SECRET, algorithm="HS256")
    assert resolve_case_action_approvals(_headers(token), config) == []


def test_rejects_an_internal_shaped_token_whose_roles_lack_an_approval_role(monkeypatch):
    """Defense in depth: even a genuinely internal token is re-checked against approval_roles."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    token = _internal_service_token(roles=["tenant_beta_operator"])
    assert resolve_case_action_approvals(_headers(token), config) == []


def test_returns_empty_list_when_case_approvals_claim_is_absent(monkeypatch):
    """A genuine internal token with no case_approvals claim resolves to an empty list."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    token = _internal_service_token(case_approvals=[])
    assert resolve_case_action_approvals(_headers(token), config) == []


def test_malformed_approval_entries_are_skipped_not_raised(monkeypatch):
    """A malformed entry in an otherwise-valid list is skipped, not a hard failure."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    config = _auth_config()
    token = _internal_service_token(
        case_approvals=[_APPROVAL, {"case_id": "CASE-1"}, {"not": "an approval"}]
    )
    approvals = resolve_case_action_approvals(_headers(token), config)
    assert len(approvals) == 1
    assert approvals[0].case_id == "CASE-2001"
