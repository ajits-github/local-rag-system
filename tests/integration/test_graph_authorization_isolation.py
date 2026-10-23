"""Adversarial proof that graph-side tenant/role authorization actually holds.

Mirrors `test_authorization_isolation.py`'s shape for the graph: a
tenant_alpha-scoped fact must never be traversable/resolvable by a
tenant_beta caller, even when it's the closest, most relevant match for
the query. Self-skips without a reachable Postgres/Neo4j.
"""

from __future__ import annotations

import uuid

from rag.graph.client import Neo4jClient
from rag.graph.retrieval import GraphRetriever
from rag.graph.schema import GraphProvenance, GraphRelationship
from rag.graph.upsert import GraphWriter
from rag.retrieval.authorization import AuthorizationContext


def _graph_config(config):
    graphed = config.model_copy(deep=True)
    graphed.graph.enabled = True
    return graphed


def test_tenant_cannot_resolve_other_tenants_graph_entity(require_neo4j, config):
    """A tenant_beta caller cannot resolve a tenant_alpha-scoped entity, even by exact name."""
    ns = f"pytest-graph-authz-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        writer = GraphWriter(client)
        provenance = GraphProvenance(
            source_document_id="doc-alpha",
            source_chunk_id="doc-alpha_0",
            dataset_id=ns,
            tenant_id="tenant_alpha",
            allowed_roles=["tenant_alpha_operator"],
            extraction_method="deterministic",
            confidence=1.0,
            document_checksum="c1",
        )
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Alpha Admin Console",
                relationship_type="OWNED_BY",
                object_type="Team",
                object_name="Alpha Team",
                provenance=provenance,
            )
        )

        retriever = GraphRetriever(graphed, client)
        beta_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
        results = retriever.resolve_entities(
            "Alpha Admin Console", dataset_id=ns, auth=beta_auth, limit=5
        )
        assert results == []

        # Sanity check: the same query with no restriction (or the correct tenant) does resolve --
        # proving the empty result above is a real denial, not a broken query.
        unrestricted = retriever.resolve_entities(
            "Alpha Admin Console", dataset_id=ns, auth=None, limit=5
        )
        assert len(unrestricted) == 1
        alpha_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        same_tenant = retriever.resolve_entities(
            "Alpha Admin Console", dataset_id=ns, auth=alpha_auth, limit=5
        )
        assert len(same_tenant) == 1
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_cross_tenant_support_role_only_grants_access_when_listed_on_the_fact(
    require_neo4j, config
):
    """Holding the support role alone isn't enough; the fact's allowed_roles must list it too."""
    ns = f"pytest-graph-authz-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        writer = GraphWriter(client)
        # Fact WITHOUT the support role listed.
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Unlisted Support Target",
                relationship_type="OWNED_BY",
                object_type="Team",
                object_name="Alpha Team",
                provenance=GraphProvenance(
                    source_document_id="doc-1",
                    dataset_id=ns,
                    tenant_id="tenant_alpha",
                    allowed_roles=["tenant_alpha_operator"],
                    extraction_method="deterministic",
                    document_checksum="c1",
                ),
            )
        )
        # Fact WITH the support role explicitly listed.
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Listed Support Target",
                relationship_type="OWNED_BY",
                object_type="Team",
                object_name="Alpha Team",
                provenance=GraphProvenance(
                    source_document_id="doc-2",
                    dataset_id=ns,
                    tenant_id="tenant_alpha",
                    allowed_roles=["tenant_alpha_operator", "techfusion_support"],
                    extraction_method="deterministic",
                    document_checksum="c1",
                ),
            )
        )

        retriever = GraphRetriever(graphed, client)
        support_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["techfusion_support"])

        unlisted = retriever.resolve_entities(
            "Unlisted Support Target", dataset_id=ns, auth=support_auth, limit=5
        )
        assert unlisted == []

        listed = retriever.resolve_entities(
            "Listed Support Target", dataset_id=ns, auth=support_auth, limit=5
        )
        assert len(listed) == 1
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_dependency_path_with_an_unauthorized_destination_is_never_partially_surfaced(
    require_neo4j, config
):
    """A path whose destination is tenant-restricted returns no path at all, never a partial one.

    Both hops of the restricted chain share `tenant_id="tenant_alpha"` on
    the same relationship write -- `upsert_relationship` derives *both*
    endpoints' canonical id from that one write's `provenance.tenant_id`,
    so a single relationship can only ever connect two nodes within the
    same tenant scope (matching the real pipeline: one document's
    extraction applies its own `tenant_id` uniformly to everything it
    asserts). A separate, fully unrestricted chain proves the denial
    above is a real authorization effect, not a broken query.
    """
    ns = f"pytest-graph-authz-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        writer = GraphWriter(client)
        restricted_provenance = dict(
            dataset_id=ns,
            extraction_method="deterministic",
            document_checksum="c1",
            tenant_id="tenant_alpha",
            allowed_roles=["tenant_alpha_operator"],
        )
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Restricted Path A",
                relationship_type="DEPENDS_ON",
                object_type="Service",
                object_name="Restricted Path B",
                provenance=GraphProvenance(source_document_id="doc-1", **restricted_provenance),
            )
        )
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Restricted Path B",
                relationship_type="DEPENDS_ON",
                object_type="Database",
                object_name="Restricted Path C",
                provenance=GraphProvenance(source_document_id="doc-1", **restricted_provenance),
            )
        )
        # A fully independent, unrestricted control chain.
        open_provenance = dict(
            dataset_id=ns, extraction_method="deterministic", document_checksum="c1"
        )
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Open Path A",
                relationship_type="DEPENDS_ON",
                object_type="Service",
                object_name="Open Path B",
                provenance=GraphProvenance(source_document_id="doc-2", **open_provenance),
            )
        )
        writer.upsert_relationship(
            GraphRelationship(
                subject_type="Service",
                subject_name="Open Path B",
                relationship_type="DEPENDS_ON",
                object_type="Database",
                object_name="Open Path C",
                provenance=GraphProvenance(source_document_id="doc-2", **open_provenance),
            )
        )

        retriever = GraphRetriever(graphed, client)
        beta_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])

        denied = retriever.find_path(
            "Restricted Path A", "Restricted Path C", dataset_id=ns, auth=beta_auth
        )
        assert denied.facts == []

        alpha_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        authorized = retriever.find_path(
            "Restricted Path A", "Restricted Path C", dataset_id=ns, auth=alpha_auth
        )
        assert len(authorized.facts) == 1

        control = retriever.find_path("Open Path A", "Open Path C", dataset_id=ns, auth=beta_auth)
        assert len(control.facts) == 1
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()
