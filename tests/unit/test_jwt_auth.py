from __future__ import annotations

import time

import jwt
import pytest

from rag.api.auth import AuthenticationError, verify_jwt
from rag.config import load_config


def _config(monkeypatch, **jwt_overrides):
    """Load config with a fixed HS256 test secret and any `security.auth.jwt` overrides."""
    monkeypatch.setenv("JWT_HS256_SECRET", "unit-test-only-not-a-real-secret")
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update=jwt_overrides)
    security = config.security.model_copy(
        update={"auth": config.security.auth.model_copy(update={"jwt": jwt_config})}
    )
    return config.model_copy(update={"security": security})


def _token(secret="unit-test-only-not-a-real-secret", algorithm="HS256", **claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, secret, algorithm=algorithm)


def test_valid_jwt_produces_verified_identity(monkeypatch):
    """A correctly signed, unexpired token with all required claims verifies successfully."""
    config = _config(monkeypatch)
    token = _token()

    identity = verify_jwt(token, config)

    assert identity.subject == "alice"
    assert identity.tenant_id == "tenant_alpha"
    assert identity.roles == ["tenant_alpha_operator"]


def test_expired_jwt_is_rejected(monkeypatch):
    """A token whose exp claim is in the past is rejected with reason='expired'."""
    config = _config(monkeypatch)
    now = int(time.time())
    token = _token(iat=now - 7200, exp=now - 3600)

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "expired"


def test_tampered_signature_is_rejected(monkeypatch):
    """A token signed with a different secret fails signature verification."""
    config = _config(monkeypatch)
    token = _token(secret="a-completely-different-secret-value-here")

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "invalid_signature"


def test_malformed_token_is_rejected(monkeypatch):
    """A string that isn't a JWT at all is rejected as malformed."""
    config = _config(monkeypatch)

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt("not-a-jwt-at-all", config)
    assert exc_info.value.reason == "malformed"


def test_wrong_issuer_is_rejected(monkeypatch):
    """When an issuer is configured, a token with a different iss claim is rejected."""
    config = _config(monkeypatch, issuer="https://trusted-issuer.example")
    token = _token(iss="https://untrusted-issuer.example")

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "invalid_issuer"


def test_wrong_audience_is_rejected(monkeypatch):
    """When an audience is configured, a token with a different aud claim is rejected."""
    config = _config(monkeypatch, audience="rag-api")
    token = _token(aud="some-other-service")

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "invalid_audience"


def test_missing_required_claim_is_rejected(monkeypatch):
    """A token missing a configured required_claims entry (e.g. tenant_id) is rejected."""
    config = _config(monkeypatch)
    now = int(time.time())
    token = jwt.encode(
        {"sub": "alice", "roles": [], "iat": now, "exp": now + 3600},
        "unit-test-only-not-a-real-secret",
        algorithm="HS256",
    )

    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "missing_claim"


def test_issuer_not_checked_when_unconfigured(monkeypatch):
    """With issuer left None (the default), any iss claim value verifies fine.

    A token asserting an `aud` claim is deliberately not exercised here:
    pyjwt itself requires `audience=` to be passed to `decode()` whenever
    the token carries an `aud` claim at all (regardless of this project's
    own config). consistent with how `verify_jwt`/`issue_dev_token.py`
    only ever add an `aud` claim when `jwt.audience` is actually configured.
    """
    config = _config(monkeypatch)
    token = _token(iss="anything")

    identity = verify_jwt(token, config)

    assert identity.subject == "alice"
