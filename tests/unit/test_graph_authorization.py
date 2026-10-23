from __future__ import annotations

from rag.graph.authorization import build_graph_authorization_predicate, is_fact_authorized
from rag.retrieval.authorization import AuthorizationContext


def test_build_predicate_none_auth_is_unrestricted():
    """`None` auth produces an empty clause and no parameters."""
    clause, params = build_graph_authorization_predicate(
        None, node_var="e", cross_tenant_support_roles=[]
    )
    assert clause == ""
    assert params == {}


def test_build_predicate_references_node_var_and_namespaces_params():
    """The predicate references the given node variable and namespaces every parameter."""
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["employee"])
    clause, params = build_graph_authorization_predicate(
        auth, node_var="neighbor", cross_tenant_support_roles=[]
    )
    assert "neighbor.tenant_id" in clause
    assert "neighbor.allowed_roles" in clause
    assert all(key.startswith("_auth_") for key in params)


def test_build_predicate_includes_freshness_exclusion_only_when_present():
    """The excluded-document clause only appears once `resolved_excluded_document_ids` is set."""
    auth = AuthorizationContext(tenant_id="t", roles=[])
    clause_without, _ = build_graph_authorization_predicate(
        auth, node_var="e", cross_tenant_support_roles=[]
    )
    assert "_auth_excluded_document_ids" not in clause_without

    auth_with_exclusions = auth.model_copy(update={"resolved_excluded_document_ids": ["doc-1"]})
    clause_with, params_with = build_graph_authorization_predicate(
        auth_with_exclusions, node_var="e", cross_tenant_support_roles=[]
    )
    assert "source_document_id" in clause_with
    assert params_with["_auth_excluded_document_ids"] == ["doc-1"]


# --- is_fact_authorized: mirrors pgvector's own positive/negative authorization matrix ---


def test_is_fact_authorized_none_auth_is_unrestricted():
    """`None` auth authorizes any fact."""
    assert is_fact_authorized({"tenant_id": "tenant_alpha"}, None, cross_tenant_support_roles=[])


def test_is_fact_authorized_untenanted_fact_always_visible():
    """A fact with `tenant_id=None` is never gated, regardless of caller tenant."""
    fact = {"tenant_id": None, "allowed_roles": None}
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=[])
    assert is_fact_authorized(fact, auth, cross_tenant_support_roles=[])


def test_is_fact_authorized_same_tenant_no_roles_required_visible():
    """A same-tenant fact with no `allowed_roles` restriction is visible to any role."""
    fact = {"tenant_id": "tenant_alpha", "allowed_roles": None}
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=[])
    assert is_fact_authorized(fact, auth, cross_tenant_support_roles=[])


def test_is_fact_authorized_cross_tenant_denied_by_default():
    """Tenant B's caller cannot see tenant A's fact without a cross-tenant support role."""
    fact = {"tenant_id": "tenant_alpha", "allowed_roles": None}
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["employee"])
    assert not is_fact_authorized(fact, auth, cross_tenant_support_roles=["techfusion_support"])


def test_is_fact_authorized_cross_tenant_support_role_must_also_be_on_the_fact():
    """Holding the support role alone is not enough; the fact's allowed_roles must list it too."""
    fact = {"tenant_id": "tenant_alpha", "allowed_roles": ["some_other_role"]}
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["techfusion_support"])
    assert not is_fact_authorized(fact, auth, cross_tenant_support_roles=["techfusion_support"])


def test_is_fact_authorized_cross_tenant_support_role_explicitly_listed_grants_access():
    """A support role listed on the fact's own `allowed_roles` grants cross-tenant access."""
    fact = {"tenant_id": "tenant_alpha", "allowed_roles": ["techfusion_support"]}
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["techfusion_support"])
    assert is_fact_authorized(fact, auth, cross_tenant_support_roles=["techfusion_support"])


def test_is_fact_authorized_role_mismatch_within_same_tenant_still_denied():
    """Same tenant alone is not sufficient when the fact restricts allowed_roles."""
    fact = {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_admin"]}
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["employee"])
    assert not is_fact_authorized(fact, auth, cross_tenant_support_roles=[])


def test_is_fact_authorized_freshness_exclusion_denies_a_superseded_document():
    """A fact sourced from an excluded document id is denied."""
    fact = {"tenant_id": None, "allowed_roles": None, "source_document_id": "doc-old"}
    auth = AuthorizationContext(
        tenant_id=None, roles=[], resolved_excluded_document_ids=["doc-old"]
    )
    assert not is_fact_authorized(fact, auth, cross_tenant_support_roles=[])


def test_is_fact_authorized_trust_level_mismatch_denied():
    """A fact whose `trust_level` doesn't match `require_trust_level` is denied."""
    fact = {"tenant_id": None, "allowed_roles": None, "trust_level": "untrusted"}
    auth = AuthorizationContext(tenant_id=None, roles=[], require_trust_level="authoritative")
    assert not is_fact_authorized(fact, auth, cross_tenant_support_roles=[])


def test_is_fact_authorized_unset_trust_level_never_gated_by_require_trust_level():
    """A fact with no `trust_level` set is never denied by `require_trust_level`."""
    fact = {"tenant_id": None, "allowed_roles": None, "trust_level": None}
    auth = AuthorizationContext(tenant_id=None, roles=[], require_trust_level="authoritative")
    assert is_fact_authorized(fact, auth, cross_tenant_support_roles=[])
