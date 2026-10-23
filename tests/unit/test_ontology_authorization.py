"""Tests: tenant/role authorization, including that inference never bypasses it.

`rag.ontology.authorization` is a thin pass-through to
`rag.graph.authorization.is_fact_authorized`, so these tests exercise
the exact same authorization function the GraphRAG milestone's own
`tests/unit/test_graph_authorization.py` exercises, just fed from RDF
instead of a Neo4j node's properties.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")

ALPHA_DB = "http://techfusion.example/graph/techfusion_graph_demo/database-alpha-console-database"
RAG_SERVICE = "http://techfusion.example/graph/techfusion_graph_demo/service-rag-service"


def test_no_identity_sees_everything(ontology_valid_store):
    """No identity sees everything."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized

    props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    assert is_entity_authorized(props, None) is True


def test_same_tenant_same_role_is_authorized(ontology_valid_store):
    """Same tenant same role is authorized."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.retrieval.authorization import AuthorizationContext

    props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    assert is_entity_authorized(props, auth) is True


def test_wrong_tenant_is_denied(ontology_valid_store):
    """Wrong tenant is denied."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.retrieval.authorization import AuthorizationContext

    props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    assert is_entity_authorized(props, auth) is False


def test_cross_tenant_support_role_not_listed_on_entity_is_still_denied(ontology_valid_store):
    """Cross tenant support role not listed on entity is still denied."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.retrieval.authorization import AuthorizationContext

    props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["techfusion_support"])
    assert (
        is_entity_authorized(props, auth, cross_tenant_support_roles=["techfusion_support"])
        is False
    )


def test_cross_tenant_support_role_listed_on_entity_is_authorized(ontology_valid_store):
    """Cross tenant support role listed on entity is authorized."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.retrieval.authorization import AuthorizationContext

    props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_alpha_operator"])
    assert (
        is_entity_authorized(props, auth, cross_tenant_support_roles=["tenant_alpha_operator"])
        is True
    )


def test_unscoped_entity_is_visible_to_any_tenant(ontology_valid_store):
    """Unscoped entity is visible to any tenant."""
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.retrieval.authorization import AuthorizationContext

    props = fetch_entity_auth_properties(ontology_valid_store, RAG_SERVICE)
    assert props["tenant_id"] is None
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    assert is_entity_authorized(props, auth) is True


def test_authorization_outcome_is_unchanged_by_running_inference_first(ontology_valid_store):
    """The central security property: inference never makes an unauthorized entity visible.

    See docs/ontology.md section 17.
    """
    from rag.ontology.authorization import fetch_entity_auth_properties, is_entity_authorized
    from rag.ontology.reasoning import materialize_owl_rl
    from rag.retrieval.authorization import AuthorizationContext

    before_props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)
    materialize_owl_rl(ontology_valid_store)
    after_props = fetch_entity_auth_properties(ontology_valid_store, ALPHA_DB)

    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
    before_result = is_entity_authorized(before_props, auth)
    after_result = is_entity_authorized(after_props, auth)
    assert before_result is False
    assert after_result is False
