from __future__ import annotations

from rag.config import load_config
from rag.eval.egress_policy import apply_egress_policy


def _enabled_config(**overrides):
    config = load_config()
    egress_config = config.security.egress_policy.model_copy(update={"enabled": True, **overrides})
    security = config.security.model_copy(update={"egress_policy": egress_config})
    return config.model_copy(update={"security": security})


def _source(**overrides) -> dict:
    base = {
        "content": "the retry delay is 45 seconds",
        "tenant_id": None,
        "classification": None,
        "trust_level": None,
        "sensitive_field_ids": None,
        "redacted_field_ids": [],
    }
    base.update(overrides)
    return base


def test_disabled_by_default_passes_content_through_unchanged():
    """A no-op when egress_policy.enabled is False. content passes through as-is."""
    config = load_config()
    assert config.security.egress_policy.enabled is False

    decision = apply_egress_policy(_source(content="anything at all"), config)

    assert decision.allowed is True
    assert decision.redacted_context == "anything at all"


def test_unredacted_sensitive_field_blocked_from_egress():
    """A source tagged with a sensitive field that wasn't redacted is blocked outright."""
    config = _enabled_config()
    source = _source(
        content="the admin key is SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE",
        sensitive_field_ids=["synthetic_admin_credential"],
        redacted_field_ids=[],
    )

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.redacted_context == ""
    assert decision.blocked_reason == "unredacted_sensitive_field"


def test_fully_redacted_sensitive_field_is_allowed_through():
    """A source whose sensitive field WAS redacted (present in redacted_field_ids) passes."""
    config = _enabled_config()
    source = _source(
        content="the admin key is [REDACTED:SENSITIVE_FIELD]",
        sensitive_field_ids=["synthetic_admin_credential"],
        redacted_field_ids=["synthetic_admin_credential"],
    )

    decision = apply_egress_policy(source, config)

    assert decision.allowed is True
    assert decision.redacted_context == source["content"]


def test_authorized_source_passes_through_unchanged():
    """A source with no sensitive fields and an allowed classification passes through untouched."""
    config = _enabled_config()
    source = _source(content="the retry delay is 45 seconds", classification="internal")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is True
    assert decision.redacted_context == source["content"]


def test_unauthorized_tenant_source_blocked_from_egress():
    """A source whose tenant_id is on the blocked-tenant list is blocked outright."""
    config = _enabled_config(blocked_tenant_ids=["tenant_beta"])
    source = _source(content="beta-internal detail", tenant_id="tenant_beta")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.blocked_reason == "blocked_tenant"


def test_confidential_classification_blocked_by_default():
    """A confidential-classified source is blocked by the default classification_policy."""
    config = _enabled_config()
    source = _source(content="confidential detail", classification="confidential")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.blocked_reason == "classification"


def test_restricted_classification_blocked_by_default():
    """A restricted-classified source is blocked by the default classification_policy."""
    config = _enabled_config()
    source = _source(content="restricted detail", classification="restricted")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.blocked_reason == "classification"


def test_unknown_classification_fails_closed():
    """A classification with no entry in classification_policy is blocked, not silently allowed."""
    config = _enabled_config()
    source = _source(content="something", classification="not_a_known_classification")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.blocked_reason == "classification"


def test_missing_classification_is_allowed_for_backward_compatibility():
    """None/missing classification (pre-governance corpus) is allowed, matching 'internal'."""
    config = _enabled_config()
    source = _source(content="ungoverned legacy content", classification=None)

    decision = apply_egress_policy(source, config)

    assert decision.allowed is True


def test_require_authoritative_trust_blocks_non_authoritative_source():
    """With require_authoritative_trust=True, a non-authoritative source is blocked."""
    config = _enabled_config(require_authoritative_trust=True)
    source = _source(content="community wiki note", trust_level="community")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is False
    assert decision.blocked_reason == "trust_level"


def test_require_authoritative_trust_allows_authoritative_source():
    """With require_authoritative_trust=True, an authoritative source is allowed through."""
    config = _enabled_config(require_authoritative_trust=True)
    source = _source(content="official policy text", trust_level="authoritative")

    decision = apply_egress_policy(source, config)

    assert decision.allowed is True
