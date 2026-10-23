"""Tests: semantic interoperability mapping and canonical entity resolution."""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")


def test_all_three_customer_shaped_classes_map_to_customer(ontology_valid_store):
    """All three customer shaped classes map to customer."""
    from rag.ontology.interoperability import find_equivalent_external_classes

    mappings = find_equivalent_external_classes(ontology_valid_store)
    external_to_canonical = {
        m.external_class.rsplit("#", 1)[-1]: m.canonical_class.rsplit("#", 1)[-1] for m in mappings
    }
    assert external_to_canonical["Client"] == "Customer"
    assert external_to_canonical["Customer"] == "Customer"
    assert external_to_canonical["BusinessPartner"] == "Customer"


def test_case_shaped_classes_map_to_customercase_not_incident(ontology_valid_store):
    """Case shaped classes map to customercase not incident."""
    from rag.ontology.interoperability import find_equivalent_external_classes

    mappings = find_equivalent_external_classes(ontology_valid_store)
    external_to_canonical = {
        m.external_class.rsplit("#", 1)[-1]: m.canonical_class.rsplit("#", 1)[-1] for m in mappings
    }
    assert external_to_canonical["ticket"] == "CustomerCase"
    assert external_to_canonical["support_case"] == "CustomerCase"
    assert external_to_canonical["incident"] == "CustomerCase"
    assert "Incident" not in external_to_canonical.values()


def test_sap_customer_is_confirmed(ontology_valid_store):
    """Sap customer is confirmed."""
    from rag.ontology.interoperability import summarize_entity_resolution

    links = summarize_entity_resolution(ontology_valid_store)
    sap_link = next(link for link in links if "customer_4711" in link.external_entity)
    assert sap_link.confirmed is True


def test_crm_account_is_proposed_not_confirmed(ontology_valid_store):
    """Crm account is proposed not confirmed."""
    from rag.ontology.interoperability import summarize_entity_resolution

    links = summarize_entity_resolution(ontology_valid_store)
    crm_link = next(link for link in links if "account_abc123" in link.external_entity)
    assert crm_link.confirmed is False
    assert crm_link.status == "proposed"


def test_servicenow_record_has_no_link_at_all(ontology_valid_store):
    """Servicenow record has no link at all."""
    from rag.ontology.interoperability import summarize_entity_resolution

    links = summarize_entity_resolution(ontology_valid_store)
    assert not any("organization_xyz" in link.external_entity for link in links)


def test_servicenow_record_is_still_type_compatible_via_schema_mapping_alone(ontology_valid_store):
    """Type-compatibility via owl:equivalentClass is independent of identity resolution.

    See docs/ontology.md section 11.
    """
    from rag.ontology.reasoning import materialize_owl_rl

    materialize_owl_rl(ontology_valid_store)
    q = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    ASK {
        ?customer a tech:Customer .
        FILTER(CONTAINS(STR(?customer), "organization_xyz"))
    }
    """
    assert bool(ontology_valid_store.query(q))
