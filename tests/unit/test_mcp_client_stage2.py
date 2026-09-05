"""Unit tests for `rag.agent.mcp_client` that need no real MCP server/transport.

Covers startup validation, internal-token minting (issuer/audience/expiry
behavior, and the deliberate never-forward-the-caller's-token invariant),
the fail-closed-without-identity guard, and the synthetic-evidence
mapping's structural isolation from document-specific behaviors
(freshness/relationship-expansion/document ACL). End-to-end dispatch
against a real MCP server object is covered separately in
`tests/integration/test_agent_mcp_client_stage2.py`.
"""

from __future__ import annotations

import time

import jwt
import pytest

from rag.agent import mcp_client
from rag.agent.tool_schemas import GetCaseStatusArgs, GetCustomerCaseArgs, UpdateCaseStatusArgs
from rag.agent.tools import ToolExecutionError
from rag.config import load_config
from rag.mcp.business.schemas import CaseApproval
from rag.retrieval.authorization import AuthorizationContext

_SECRET = "mcp-client-unit-test-secret-not-real"


def _config(**overrides):
    """Return `load_config()` with `mcp.client`/`security.auth` overrides applied."""
    config = load_config().model_copy(deep=True)
    for path, value in overrides.items():
        obj = config
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config


def _secure_config(**overrides):
    """Return a config with HS256 auth enabled against `_SECRET`, plus any overrides."""
    config = _config(
        **{
            "security.auth.enabled": True,
            "security.auth.jwt.secret_env_var": "MCP_CLIENT_UNIT_TEST_SECRET",
            **overrides,
        }
    )
    return config


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_UNIT_TEST_SECRET", _SECRET)


# --- validate_startup_config -------------------------------------------------


def test_validate_startup_config_is_a_noop_when_client_disabled():
    """The default config (mcp.client.enabled=False) never raises."""
    mcp_client.validate_startup_config(load_config())


def test_validate_startup_config_rejects_client_enabled_with_auth_disabled():
    """mcp.client.enabled=True + security.auth.enabled=False fails startup, per the approval."""
    config = _config(**{"mcp.client.enabled": True, "security.auth.enabled": False})
    with pytest.raises(RuntimeError, match="security.auth.enabled=True"):
        mcp_client.validate_startup_config(config)


def test_validate_startup_config_rejects_non_hs256_algorithm():
    """Internal token minting needs a symmetric signing key; RS256/ES256 fails startup."""
    config = _secure_config(**{"security.auth.jwt.algorithm": "RS256", "mcp.client.enabled": True})
    with pytest.raises(RuntimeError, match="HS256"):
        mcp_client.validate_startup_config(config)


def test_validate_startup_config_rejects_asgi_transport_without_mcp_server():
    """transport='asgi' (the default) requires mcp.enabled=True to have a server to bind to."""
    config = _secure_config(**{"mcp.client.enabled": True, "mcp.enabled": False})
    assert config.mcp.client.transport == "asgi"
    with pytest.raises(RuntimeError, match="mcp.enabled=True"):
        mcp_client.validate_startup_config(config)


def test_validate_startup_config_accepts_a_fully_valid_configuration():
    """Auth enabled, HS256, mcp server mounted: no error."""
    config = _secure_config(**{"mcp.client.enabled": True, "mcp.enabled": True})
    mcp_client.validate_startup_config(config)


def test_validate_startup_config_accepts_http_transport_without_mcp_server():
    """transport='http' needs no local server object, so mcp.enabled=False is fine for it."""
    config = _secure_config(
        **{"mcp.client.enabled": True, "mcp.enabled": False, "mcp.client.transport": "http"}
    )
    mcp_client.validate_startup_config(config)


# --- mint_internal_token ------------------------------------------------------


def test_mint_internal_token_never_reuses_a_caller_subject():
    """The minted token's sub is always the fixed internal-service value, never a real caller id.

    There is no caller subject on AuthorizationContext at all (by design;
    see rag.agent.state's module docstring) -- this test proves the
    minted token's sub is the configured synthetic marker, not derived
    from anything request-specific, so it can never coincide with or
    imply a real end user's identity.
    """
    config = _secure_config()
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["sub"] == config.mcp.client.internal_token_subject
    assert claims["sub"] == "rag-agent-internal"
    assert claims["token_use"] == "mcp_internal_service"


def test_mint_internal_token_carries_resolved_tenant_and_roles():
    """The minted token's tenant_id/roles claims match the resolved AuthorizationContext exactly."""
    config = _secure_config()
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_admin", "x"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["tenant_id"] == "tenant_beta"
    assert claims["roles"] == ["tenant_beta_admin", "x"]


def test_mint_internal_token_has_a_short_expiry_and_it_is_enforced():
    """Exp is genuinely short-lived and genuinely checked by the real verify_jwt.

    `leeway_seconds` is zeroed out for this test: the default 30s
    clock-skew tolerance would otherwise swallow a 1s-TTL token's
    expiration for the 2s this test sleeps, which would make this test
    pass for the wrong reason (leeway masking a genuinely broken
    expiration check).
    """
    from rag.api.auth import AuthenticationError, verify_jwt

    config = _secure_config(
        **{"mcp.client.internal_token_ttl_seconds": 1, "security.auth.jwt.leeway_seconds": 0}
    )
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)

    # Fresh: verifies fine.
    identity = verify_jwt(token, config)
    assert identity.tenant_id == "tenant_alpha"

    time.sleep(2)
    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, config)
    assert exc_info.value.reason == "expired"


def test_mint_internal_token_iss_is_informational_when_jwt_issuer_unset():
    """Iss is always present (documentation/audit value) even when jwt_config.issuer is unset.

    Confirmed this is genuinely safe against the installed pyjwt: an
    unchecked iss claim on the token never causes verify_jwt to reject
    it, unlike aud (see the next two tests).
    """
    config = _secure_config()
    assert config.security.auth.jwt.issuer is None
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["iss"] == config.mcp.client.internal_token_issuer

    from rag.api.auth import verify_jwt

    identity = verify_jwt(token, config)  # must not raise despite the unchecked iss claim
    assert identity.issuer == config.mcp.client.internal_token_issuer


def test_mint_internal_token_uses_configured_issuer_when_jwt_issuer_is_set():
    """When security.auth.jwt.issuer IS configured, the mint uses that exact value.

    So the unmodified, shared verify_jwt actively validates it, exactly
    as it would for a real end-user token -- not a special relaxation
    for this token.
    """
    config = _secure_config(**{"security.auth.jwt.issuer": "https://issuer.example"})
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(
        token,
        _SECRET,
        algorithms=["HS256"],
        issuer="https://issuer.example",
        options={"verify_aud": False},
    )
    assert claims["iss"] == "https://issuer.example"


def test_mint_internal_token_omits_aud_entirely_when_jwt_audience_unset():
    """No aud claim at all when jwt_config.audience is unset -- not an unchecked fallback value.

    Real, reproduced PyJWT behavior (not assumed by analogy with iss):
    a token carrying ANY aud claim fails verification when the verifier
    passes no expected audience, regardless of that claim's value. See
    mint_internal_token's docstring for the full explanation.
    """
    config = _secure_config()
    assert config.security.auth.jwt.audience is None
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert "aud" not in claims

    from rag.api.auth import verify_jwt

    verify_jwt(token, config)  # must not raise invalid_audience


def test_mint_internal_token_uses_and_enforces_configured_audience_when_set():
    """A configured jwt.audience is used verbatim and actively enforced against a mismatch."""
    config = _secure_config(**{"security.auth.jwt.audience": "rag-mcp-business-real"})
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], audience="rag-mcp-business-real")
    assert claims["aud"] == "rag-mcp-business-real"

    from rag.api.auth import AuthenticationError, verify_jwt

    # A verifier expecting a DIFFERENT audience correctly rejects it.
    wrong_aud_config = config.model_copy(deep=True)
    wrong_aud_config.security.auth.jwt.audience = "someone-else"
    with pytest.raises(AuthenticationError) as exc_info:
        verify_jwt(token, wrong_aud_config)
    assert exc_info.value.reason == "invalid_audience"


def test_mint_internal_token_rejects_non_hs256_algorithm():
    """Defense in depth: mint_internal_token itself also refuses non-HS256, not just startup."""
    config = _secure_config(**{"security.auth.jwt.algorithm": "RS256"})
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    with pytest.raises(RuntimeError, match="HS256"):
        mcp_client.mint_internal_token(auth, config)


def test_mint_internal_token_omits_case_approvals_claim_when_none_supplied():
    """No case_approvals claim at all for an ordinary read-tool call."""
    config = _secure_config()
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    token = mcp_client.mint_internal_token(auth, config)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert "case_approvals" not in claims


def test_mint_internal_token_carries_case_approvals_claim_when_supplied():
    """A supplied case_approvals list is embedded verbatim as a signed claim."""
    config = _secure_config()
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    approvals = [CaseApproval(case_id="CASE-1002", new_status="resolved")]
    token = mcp_client.mint_internal_token(auth, config, approvals)
    claims = jwt.decode(token, _SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["case_approvals"] == [{"case_id": "CASE-1002", "new_status": "resolved"}]


def test_mint_internal_token_rejects_case_approvals_over_the_configured_maximum():
    """Defense in depth: the token minter itself also bounds case_approvals, not just the API."""
    config = _secure_config(**{"mcp.business_actions.max_case_approvals_per_request": 2})
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["op"])
    approvals = [CaseApproval(case_id=f"CASE-{i}", new_status="closed") for i in range(3)]
    with pytest.raises(RuntimeError, match="case_approvals"):
        mcp_client.mint_internal_token(auth, config, approvals)


# --- fail-closed without identity --------------------------------------------


def test_dispatch_remote_tool_sync_fails_closed_with_no_auth_context():
    """No AuthorizationContext at all: rejected before any dispatch is attempted."""
    config = _secure_config(**{"mcp.client.enabled": True, "mcp.enabled": True})
    with pytest.raises(ToolExecutionError, match="authenticated caller identity"):
        mcp_client.dispatch_remote_tool_sync(
            "get_customer_case",
            GetCustomerCaseArgs(case_id="CASE-1001"),
            auth=None,
            config=config,
            mcp_app=None,
        )


def test_dispatch_remote_tool_sync_fails_closed_with_no_tenant_id():
    """An AuthorizationContext with no tenant_id is treated the same as no identity at all."""
    config = _secure_config(**{"mcp.client.enabled": True, "mcp.enabled": True})
    auth = AuthorizationContext(tenant_id=None, roles=["some_role"])
    with pytest.raises(ToolExecutionError, match="authenticated caller identity"):
        mcp_client.dispatch_remote_tool_sync(
            "get_case_status",
            GetCaseStatusArgs(case_id="CASE-1001"),
            auth=auth,
            config=config,
            mcp_app=None,
        )


def test_dispatch_remote_tool_sync_fails_closed_for_update_case_status_with_no_auth():
    """The write-action tool has no anonymous path either, exactly like the two read tools."""
    config = _secure_config(**{"mcp.client.enabled": True, "mcp.enabled": True})
    with pytest.raises(ToolExecutionError, match="authenticated caller identity"):
        mcp_client.dispatch_remote_tool_sync(
            "update_case_status",
            UpdateCaseStatusArgs(case_id="CASE-1002", new_status="closed"),
            auth=None,
            config=config,
            mcp_app=None,
        )


# --- synthetic evidence: structurally inert for document-specific behaviors --


def test_business_case_evidence_has_no_document_governance_fields_set():
    """Freshness/relationship-expansion/document-ACL fields are all unset by construction.

    Proves the approval's item 3 requirement structurally: this synthetic
    evidence can never be mistaken for a real, versioned, ACL-governed
    document, since every field those subsystems read is None.
    """
    result = mcp_client._business_result_to_search_result(
        "get_customer_case",
        "CASE-1001",
        {
            "case_id": "CASE-1001",
            "tenant_id": "tenant_alpha",
            "customer_name": "Acme Corp",
            "subject": "s",
            "description": "d",
            "status": "open",
            "priority": "low",
            "assigned_team": "Team",
            "created_at": "2026-08-21T09:15:00Z",
            "updated_at": "2026-08-27T14:02:00Z",
        },
    )
    meta = result.chunk.metadata
    assert result.origin == "mcp_remote"
    assert meta.source == "mcp://business/CASE-1001"
    assert meta.chunk_id == "mcp:get_customer_case:CASE-1001"
    assert meta.document_version is None
    assert meta.status is None
    assert meta.effective_from is None
    assert meta.supersedes_source is None
    assert meta.allowed_roles is None
    assert meta.classification is None
    assert meta.trust_level is None
    assert meta.dataset_id == "mcp_business"
    assert meta.tenant_id == "tenant_alpha"  # provenance/display only, not re-checked ACL


def test_case_status_evidence_carries_no_tenant_id():
    """CaseStatusResult has no tenant_id field; the synthetic chunk reflects that honestly."""
    result = mcp_client._business_result_to_search_result(
        "get_case_status",
        "CASE-1001",
        {
            "case_id": "CASE-1001",
            "status": "open",
            "priority": "low",
            "updated_at": "2026-08-27T14:02:00Z",
        },
    )
    assert result.chunk.metadata.tenant_id is None
    assert result.origin == "mcp_remote"


@pytest.mark.parametrize(
    ("outcome", "expect_in_text"),
    [
        ("executed", "was changed from resolved to closed"),
        ("already_in_status", "already closed; no change was made"),
        ("invalid_transition", "not a valid transition"),
        ("approval_required", "requires approval before it can be applied"),
    ],
)
def test_update_case_status_evidence_states_the_true_outcome(outcome, expect_in_text):
    """Every outcome's rendered evidence text is unambiguous about whether a mutation happened."""
    result = mcp_client._business_result_to_search_result(
        "update_case_status",
        "CASE-1002",
        {
            "outcome": outcome,
            "case_id": "CASE-1002",
            "previous_status": "resolved",
            "new_status": "closed",
            "updated_at": "2026-08-27T14:02:00Z",
        },
    )
    assert expect_in_text in result.chunk.content
    assert result.origin == "mcp_remote"
    assert result.chunk.metadata.tenant_id is None
    assert result.chunk.metadata.document_version is None
    assert result.chunk.metadata.allowed_roles is None
