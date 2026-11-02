"""Adversarial proof that graph-side tenant/role authorization actually holds.

Mirrors `test_authorization_isolation.py`'s shape for the graph: a
tenant_alpha-scoped fact must never be traversable/resolvable by a
tenant_beta caller, even when it's the closest, most relevant match for
the query. Self-skips without a reachable Postgres/Neo4j.
"""

from __future__ import annotations

import uuid

import pytest

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


def _relationship(subject, verb, obj, ns, document, roles, *, object_type="Service", checksum="c1"):
    return GraphRelationship(
        subject_type="Service",
        subject_name=subject,
        relationship_type=verb,
        object_type=object_type,
        object_name=obj,
        provenance=GraphProvenance(
            source_document_id=document,
            source_chunk_id=f"{document}_0",
            dataset_id=ns,
            tenant_id="t",
            allowed_roles=roles,
            extraction_method="llm",
            confidence=1.0,
            document_checksum=checksum,
        ),
    )


_EVERYONE = ["viewer", "admin"]
_VIEWER = AuthorizationContext(tenant_id="t", roles=["viewer"])
_ADMIN = AuthorizationContext(tenant_id="t", roles=["admin"])


@pytest.fixture
def restricted_edge_graph(require_neo4j, config):
    """Public entities with an admin-only edge, a restricted hop, and a public control."""
    ns = f"pytest-graph-edges-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    writer = GraphWriter(client)
    public_runbook = {"object_type": "Runbook"}
    secret_team = {"object_type": "Team"}
    for fact in [
        # A public document mentions both services, so both nodes are viewer-visible.
        _relationship(
            "Alpha Service",
            "HAS_RUNBOOK",
            "Shared Runbook",
            ns,
            "public",
            _EVERYONE,
            **public_runbook,
        ),
        _relationship(
            "Beta Service",
            "HAS_RUNBOOK",
            "Shared Runbook",
            ns,
            "public",
            _EVERYONE,
            **public_runbook,
        ),
        # Only an admin-only document states how they relate.
        _relationship("Alpha Service", "DEPENDS_ON", "Beta Service", ns, "secret", ["admin"]),
        _relationship(
            "Alpha Service", "OWNED_BY", "Secret Team", ns, "secret", ["admin"], **secret_team
        ),
        # Hidden Service is first seen in the restricted document, so its node is restricted.
        _relationship(
            "Hidden Service", "OWNED_BY", "Secret Team", ns, "secret", ["admin"], **secret_team
        ),
        _relationship("Gamma Service", "DEPENDS_ON", "Hidden Service", ns, "public", _EVERYONE),
        _relationship("Hidden Service", "DEPENDS_ON", "Delta Service", ns, "public", _EVERYONE),
        # A public dependency, for proving a denial is not a broken query.
        _relationship("Epsilon Service", "DEPENDS_ON", "Zeta Service", ns, "public", _EVERYONE),
        # Two routes: a restricted direct edge, and a longer public one.
        _relationship("Kappa Service", "DEPENDS_ON", "Lambda Service", ns, "public", _EVERYONE),
        _relationship("Lambda Service", "DEPENDS_ON", "Mu Service", ns, "public", _EVERYONE),
        _relationship("Kappa Service", "DEPENDS_ON", "Mu Service", ns, "secret", ["admin"]),
    ]:
        writer.upsert_relationship(fact)
    try:
        yield GraphRetriever(graphed, client), ns
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


@pytest.mark.parametrize(
    "query",
    [
        "What does Alpha Service depend on?",
        "What depends on Beta Service?",
        "What is impacted if Beta Service fails?",
    ],
)
def test_restricted_edge_between_public_entities_is_hidden_from_a_viewer(
    restricted_edge_graph, query
):
    """Seeing both entities does not grant seeing the admin-only relationship between them."""
    retriever, ns = restricted_edge_graph

    viewer = retriever.search(query, dataset_id=ns, auth=_VIEWER)
    admin = retriever.search(query, dataset_id=ns, auth=_ADMIN)

    assert not any("Alpha Service" in f.text and "Beta Service" in f.text for f in viewer.facts)
    connected = [f for f in admin.facts if "Alpha Service" in f.text and "Beta Service" in f.text]
    assert connected, "the denial above must be a real denial, not a broken query"
    assert connected[0].source_document_id == "secret"


def test_entity_lookup_and_owner_queries_hide_restricted_edges(restricted_edge_graph):
    """The one-hop query shapes follow the same rule as the traversals."""
    retriever, ns = restricted_edge_graph

    lookup = retriever.get_entity("Alpha Service", dataset_id=ns, auth=_VIEWER)
    assert [f.text for f in lookup.facts] == ["Alpha Service HAS_RUNBOOK Shared Runbook"]

    owners = retriever.search("Who owns Alpha Service?", dataset_id=ns, auth=_VIEWER)
    assert owners.intent == "owners_of"
    assert owners.facts == []
    assert retriever.search("Who owns Alpha Service?", dataset_id=ns, auth=_ADMIN).facts


def test_get_neighbors_hides_restricted_edges_and_hops(restricted_edge_graph):
    """`get_neighbors` must not reach a node through an unauthorized relationship."""
    retriever, ns = restricted_edge_graph

    def names(name, auth):
        result = retriever.get_neighbors(name, dataset_id=ns, auth=auth, max_hops=2)
        return {e.name for f in result.facts for e in f.entities}

    assert "Beta Service" in names("Alpha Service", _VIEWER)  # via the public runbook
    assert "Secret Team" not in names("Alpha Service", _VIEWER)
    assert "Secret Team" in names("Alpha Service", _ADMIN)
    assert "Delta Service" not in names("Gamma Service", _VIEWER)


def test_traversal_through_a_restricted_node_is_hidden(restricted_edge_graph):
    """Public edges around a restricted intermediate node do not reveal what is behind it."""
    retriever, ns = restricted_edge_graph
    query = "What does Gamma Service depend on?"

    assert retriever.search(query, dataset_id=ns, auth=_VIEWER).facts == []
    admin_facts = retriever.search(query, dataset_id=ns, auth=_ADMIN).facts
    assert {f.entities[1].name for f in admin_facts} == {"Hidden Service", "Delta Service"}


def test_permitted_route_survives_when_the_shortest_route_is_restricted(restricted_edge_graph):
    """An authorized longer path is used, and the fact cites what that route rests on."""
    retriever, ns = restricted_edge_graph

    viewer = retriever.search("What does Kappa Service depend on?", dataset_id=ns, auth=_VIEWER)
    fact = next(f for f in viewer.facts if f.entities[1].name == "Mu Service")
    assert [[s.document_id for s in d] for d in fact.support_sets] == [["public", "public"]]

    path = retriever.find_path("Kappa Service", "Mu Service", dataset_id=ns, auth=_VIEWER)
    assert [f.text for f in path.facts] == [
        "Dependency path: Kappa Service -> Lambda Service -> Mu Service"
    ]
    direct = retriever.find_path("Kappa Service", "Mu Service", dataset_id=ns, auth=_ADMIN)
    assert [f.text for f in direct.facts] == ["Dependency path: Kappa Service -> Mu Service"]


def test_superseded_document_is_excluded_without_emptying_other_queries(restricted_edge_graph):
    """Freshness removes that document's edges; entity nodes have no source and stay visible."""
    retriever, ns = restricted_edge_graph
    query = "What does Epsilon Service depend on?"

    unrelated = _VIEWER.model_copy(update={"resolved_excluded_document_ids": ["other"]})
    assert retriever.search(query, dataset_id=ns, auth=unrelated).facts

    superseded = _VIEWER.model_copy(update={"resolved_excluded_document_ids": ["public"]})
    assert retriever.search(query, dataset_id=ns, auth=superseded).facts == []


def test_fact_support_carries_the_projection_checksum(restricted_edge_graph):
    """Fusion can only reject a stale fact if the checksum survives the query."""
    retriever, ns = restricted_edge_graph

    result = retriever.search("What does Epsilon Service depend on?", dataset_id=ns, auth=_VIEWER)

    [[source]] = result.facts[0].support_sets
    assert (source.document_id, source.document_checksum) == ("public", "c1")
