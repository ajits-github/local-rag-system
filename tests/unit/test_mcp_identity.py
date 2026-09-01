"""Unit tests for `rag.mcp.identity`: transport-owned identity resolution.

No MCP protocol machinery needed here -- `resolve_http_identity`/
`resolve_stdio_identity` are plain functions taking a headers mapping (or
nothing) and an `AppConfig`. Mirrors `test_api_query_auth_boundary.py`'s
JWT-construction conventions.
"""

from __future__ import annotations

import logging
import time

import jwt
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from rag.config import load_config
from rag.mcp.identity import resolve_http_identity, resolve_stdio_identity

_SECRET = "mcp-identity-unit-test-secret"


def _token(**claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _auth_config(**overrides):
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    auth_config = auth_config.model_copy(update=overrides)
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)


def test_resolve_http_identity_returns_none_when_auth_disabled():
    """auth.enabled=False: no header check at all, always unrestricted."""
    config = load_config()
    assert config.security.auth.enabled is False
    assert resolve_http_identity(None, config) is None
    assert resolve_http_identity({"authorization": "Bearer garbage"}, config) is None


def test_resolve_http_identity_accepts_a_valid_bearer_token():
    """A well-formed, correctly-signed token resolves to its claimed identity."""
    config = _auth_config()
    token = _token()
    identity = resolve_http_identity({"authorization": f"Bearer {token}"}, config)
    assert identity is not None
    assert identity.tenant_id == "tenant_alpha"
    assert identity.roles == ["tenant_alpha_operator"]


def test_resolve_http_identity_header_lookup_is_case_insensitive_key():
    """A client sending the canonical 'Authorization' capitalization still works."""
    config = _auth_config()
    token = _token()
    identity = resolve_http_identity({"Authorization": f"Bearer {token}"}, config)
    assert identity is not None
    assert identity.tenant_id == "tenant_alpha"


def test_resolve_http_identity_missing_header_fails_closed_by_default():
    """No Authorization header, auth enabled, insecure_dev_mode off: always rejected."""
    config = _auth_config()
    with pytest.raises(ToolError, match="Missing Authorization header"):
        resolve_http_identity(None, config)
    with pytest.raises(ToolError, match="Missing Authorization header"):
        resolve_http_identity({}, config)


def test_resolve_http_identity_missing_header_allowed_under_insecure_dev_mode():
    """insecure_dev_mode relaxes only the "no header at all" case, matching get_current_identity."""
    config = _auth_config(insecure_dev_mode=True)
    assert resolve_http_identity(None, config) is None


def test_resolve_http_identity_insecure_dev_mode_never_rescues_an_invalid_token():
    """A present-but-invalid token is always rejected, regardless of insecure_dev_mode."""
    config = _auth_config(insecure_dev_mode=True)
    bad_token = jwt.encode(
        {"sub": "eve", "tenant_id": "tenant_alpha", "roles": [], "exp": int(time.time()) + 3600},
        "wrong-secret",
        algorithm="HS256",
    )
    with pytest.raises(ToolError, match="Invalid or expired token"):
        resolve_http_identity({"authorization": f"Bearer {bad_token}"}, config)


def test_resolve_http_identity_malformed_header_is_rejected():
    """A non-Bearer scheme or an empty token value is rejected as malformed."""
    config = _auth_config()
    with pytest.raises(ToolError, match="Malformed Authorization header"):
        resolve_http_identity({"authorization": "NotBearer sometoken"}, config)
    with pytest.raises(ToolError, match="Malformed Authorization header"):
        resolve_http_identity({"authorization": "Bearer "}, config)


def test_resolve_http_identity_expired_token_is_rejected():
    """An expired token fails closed, never a silent fallback to unrestricted access."""
    config = _auth_config()
    now = int(time.time())
    expired = jwt.encode(
        {
            "sub": "alice",
            "tenant_id": "tenant_alpha",
            "roles": [],
            "iat": now - 7200,
            "exp": now - 3600,
        },
        _SECRET,
        algorithm="HS256",
    )
    with pytest.raises(ToolError, match="Invalid or expired token"):
        resolve_http_identity({"authorization": f"Bearer {expired}"}, config)


def test_resolve_stdio_identity_returns_none_when_auth_disabled():
    """auth.enabled=False: no MCP_AUTH_TOKEN check at all, matching the HTTP transport."""
    config = load_config()
    assert resolve_stdio_identity(config) is None


def test_resolve_stdio_identity_requires_env_var_when_auth_enabled(monkeypatch):
    """Auth enabled but MCP_AUTH_TOKEN unset: the stdio process refuses to start unrestricted."""
    config = _auth_config()
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        resolve_stdio_identity(config)


def test_resolve_stdio_identity_accepts_a_valid_token_from_env(monkeypatch):
    """A valid MCP_AUTH_TOKEN resolves once, at process startup, into a fixed identity."""
    config = _auth_config()
    monkeypatch.setenv("MCP_AUTH_TOKEN", _token())
    identity = resolve_stdio_identity(config)
    assert identity is not None
    assert identity.tenant_id == "tenant_alpha"


def test_resolve_stdio_identity_rejects_an_invalid_token_from_env(monkeypatch):
    """An unparseable MCP_AUTH_TOKEN fails process startup rather than starting unrestricted."""
    config = _auth_config()
    monkeypatch.setenv("MCP_AUTH_TOKEN", "not-a-real-jwt")
    with pytest.raises(RuntimeError, match="failed verification"):
        resolve_stdio_identity(config)


def _assert_no_secret_in_logs(records, *secrets: str) -> None:
    for record in records:
        rendered = f"{record.getMessage()!r} {record.__dict__!r}"
        for secret in secrets:
            assert secret not in rendered, f"log record leaked a raw secret: {rendered!r}"


def test_resolve_http_identity_success_path_never_logs_the_raw_token(caplog):
    """A successful mcp_auth_success audit event never carries the raw JWT string."""
    config = _auth_config()
    token = _token()

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        identity = resolve_http_identity({"authorization": f"Bearer {token}"}, config)

    assert identity is not None
    assert caplog.records
    _assert_no_secret_in_logs(caplog.records, token)


def test_resolve_http_identity_failure_path_never_logs_the_raw_token_or_signing_secret(caplog):
    """A failed mcp_auth_failure audit event never carries the raw token or its signing secret."""
    config = _auth_config()
    bad_token = jwt.encode(
        {"sub": "eve", "tenant_id": "tenant_alpha", "roles": [], "exp": int(time.time()) + 3600},
        "totally-wrong-secret",
        algorithm="HS256",
    )

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        with pytest.raises(ToolError):
            resolve_http_identity({"authorization": f"Bearer {bad_token}"}, config)

    assert caplog.records
    _assert_no_secret_in_logs(caplog.records, bad_token, "totally-wrong-secret")
