from __future__ import annotations

from datetime import date

from rag.retrieval.authorization import AuthorizationContext


def test_authorization_context_defaults_are_fully_permissive():
    """A bare AuthorizationContext() has no tenant/roles/as_of/exclusions set."""
    auth = AuthorizationContext()
    assert auth.tenant_id is None
    assert auth.roles == []
    assert auth.as_of is None
    assert auth.include_superseded is False
    assert auth.resolved_excluded_document_ids == []


def test_authorization_context_accepts_explicit_fields():
    """Every field can be set explicitly and round-trips unchanged."""
    auth = AuthorizationContext(
        tenant_id="tenant_alpha",
        roles=["tenant_alpha_operator", "techfusion_support"],
        as_of=date(2026, 3, 15),
        include_superseded=True,
    )
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["tenant_alpha_operator", "techfusion_support"]
    assert auth.as_of == date(2026, 3, 15)
    assert auth.include_superseded is True
