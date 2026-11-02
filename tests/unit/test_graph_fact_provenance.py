"""Graph facts are authorized and revalidated by the documents that state them.

Regressions for SEC-020 and SEC-021: neighbor queries checked only the returned
node, and a fact cited the document that first mentioned an entity rather than
the one asserting the relationship. A fake client supplies the query's row shape;
`tests/integration/test_graph_authorization_isolation.py` proves the Cypher.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.config import load_config
from rag.graph import queries
from rag.graph.authorization import (
    build_graph_authorization_predicate,
    build_graph_path_authorization_predicate,
    is_fact_authorized,
)
from rag.graph.fusion import graph_facts_to_search_results
from rag.graph.retrieval import GraphRetriever
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.egress_policy import apply_egress_policy
from rag.retrieval.pipeline import source_dict
from rag.schemas import ChunkMetadata

_AUTH = AuthorizationContext(tenant_id="t", roles=["viewer"])


def _metadata(document_id: str, **overrides) -> ChunkMetadata:
    now = datetime.now(UTC)
    fields = dict(
        document_id=document_id,
        chunk_id=f"{document_id}_0",
        source=f"{document_id}.md",
        source_type="markdown",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds",
    )
    fields.update(overrides)
    return ChunkMetadata(**fields)


class _Store:
    """Fixed visible documents and current checksums."""

    def __init__(self, visible: dict[str, ChunkMetadata], checksums: dict[str, str] | None = None):
        self._visible = visible
        self._checksums = checksums or {}
        self.asked: list[str] = []

    def get_document_governance(self, document_ids, dataset_id, auth=None):
        self.asked.extend(document_ids)
        return {d: self._visible[d] for d in document_ids if d in self._visible}

    def get_document_checksums_by_id(self, document_ids, dataset_id):
        return {d: self._checksums[d] for d in document_ids if d in self._checksums}


class _Client:
    """Resolves one entity, then returns `rows`, recording the Cypher."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.cypher: list[str] = []

    def run_read(self, cypher, params):
        self.cypher.append(cypher)
        if len(self.cypher) == 1:
            node = {"canonical_id": "alpha", "name": "Alpha Service"}
            return [{"e": node, "entity_labels": ["Service"]}]
        return self._rows


def _dependency_row(support_sets: list) -> dict:
    node = {"canonical_id": "beta", "name": "Beta Service", "first_seen_document_id": "public-doc"}
    return {
        "dependency": node,
        "dependency_labels": ["Service"],
        "hops": 1,
        "support_sets": support_sets,
    }


def _facts(support_sets: list):
    client = _Client([_dependency_row(support_sets)])
    result = GraphRetriever(load_config(), client).search(
        "What does Alpha Service depend on?", dataset_id="ds", auth=_AUTH
    )
    return result.facts


def test_relationship_fact_cites_the_edge_document_not_the_entity_first_mention():
    """The citation and Postgres lookup follow the asserting document."""
    store = _Store({"edge-doc": _metadata("edge-doc")})

    [result] = graph_facts_to_search_results(
        _facts([[["edge-doc", None]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )

    assert store.asked == ["edge-doc"]
    assert result.chunk.metadata.source == "edge-doc.md"


def test_fact_is_dropped_when_only_the_entity_document_is_still_visible():
    """A revoked or deleted edge document drops the fact, despite a public first mention."""
    store = _Store({"public-doc": _metadata("public-doc")})

    assert not graph_facts_to_search_results(
        _facts([[["revoked-doc", None]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )


def test_fact_survives_deletion_of_the_entity_first_mention():
    """A node outliving its first document must not lose facts stated elsewhere."""
    store = _Store({"edge-doc": _metadata("edge-doc")})

    assert graph_facts_to_search_results(
        _facts([[["edge-doc", None]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )


def test_fact_projected_from_an_older_version_of_the_document_is_dropped():
    """A document replaced while its projection failed must not keep asserting old facts."""
    store = _Store({"edge-doc": _metadata("edge-doc")}, {"edge-doc": "new-hash"})

    assert not graph_facts_to_search_results(
        _facts([[["edge-doc", "old-hash"]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )
    assert graph_facts_to_search_results(
        _facts([[["edge-doc", "new-hash"]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )


def test_a_second_derivation_keeps_the_fact_when_the_first_is_invalid():
    """A fact needs only one valid derivation, and cites that one."""
    store = _Store({"valid-doc": _metadata("valid-doc")})

    [result] = graph_facts_to_search_results(
        _facts([[["revoked-doc", None]], [["valid-doc", None]]]),
        vectorstore=store,
        dataset_id="ds",
        auth=_AUTH,
    )

    assert result.chunk.metadata.document_id == "valid-doc"


def test_multihop_fact_needs_every_document_on_its_path():
    """One invalid hop invalidates the whole derivation."""
    store = _Store({"hop-1": _metadata("hop-1")})

    assert not graph_facts_to_search_results(
        _facts([[["hop-1", None], ["hop-2", None]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )


def test_multihop_fact_takes_the_strictest_governance_of_its_documents():
    """A public first hop must not carry a confidential second hop past the egress gate."""
    config = load_config().model_copy(deep=True)
    config.security.egress_policy.enabled = True
    store = _Store(
        {
            "hop-1": _metadata(
                "hop-1", classification="public", trust_level="authoritative", status="active"
            ),
            "hop-2": _metadata(
                "hop-2", classification="confidential", trust_level="community", status="draft"
            ),
        }
    )

    [result] = graph_facts_to_search_results(
        _facts([[["hop-1", None], ["hop-2", None]]]), vectorstore=store, dataset_id="ds", auth=_AUTH
    )
    meta = result.chunk.metadata

    assert meta.source == "hop-1.md"
    assert (meta.classification, meta.trust_level, meta.status) == (
        "confidential",
        "community",
        "draft",
    )
    assert apply_egress_policy(source_dict(result), config).blocked_reason == "classification"


@pytest.mark.parametrize(
    ("builder", "extra"),
    [
        (queries.dependents_of_query, {"max_depth": 3}),
        (queries.dependencies_of_query, {"max_depth": 3}),
        (queries.impact_of_query, {"max_depth": 3}),
    ],
)
def test_traversal_templates_authorize_the_whole_path(builder, extra):
    """The path clause reaches the query and the row carries edge provenance."""
    cypher, _ = builder(
        dataset_id="ds",
        entity_canonical_id="x",
        auth_clause="",
        path_auth_clause="PATH_AUTH",
        limit=5,
        **extra,
    )
    assert "PATH_AUTH" in cypher
    assert "support_sets" in cypher


@pytest.mark.parametrize(
    "builder",
    [queries.runbooks_for_query, queries.owners_of_query, queries.policies_governing_query],
)
def test_one_hop_templates_authorize_the_edge(builder):
    """The edge clause reaches the query and the row carries edge provenance."""
    cypher, _ = builder(
        dataset_id="ds", entity_canonical_id="x", auth_clause="", edge_auth_clause="EDGE", limit=5
    )
    assert "EDGE" in cypher
    assert "support_sets" in cypher


def test_retriever_sends_edge_authorization_for_lookup_and_neighbors():
    """The two inline queries are held to the same rule as the templates."""
    config = load_config()
    client = _Client([])
    retriever = GraphRetriever(config, client)
    retriever.get_entity("Alpha Service", dataset_id="ds", auth=_AUTH)
    assert "r.allowed_roles" in client.cypher[1]

    client = _Client([])
    GraphRetriever(config, client).get_neighbors("Alpha Service", dataset_id="ds", auth=_AUTH)
    assert "relationships(path)" in client.cypher[1]


def test_path_predicate_is_empty_without_auth():
    """`None` auth stays unrestricted."""
    assert (
        build_graph_path_authorization_predicate(
            None, path_var="path", cross_tenant_support_roles=[]
        )
        == ""
    )


def test_exclusion_predicate_is_null_safe_for_nodes_without_a_source_document():
    """`NOT (null IN list)` would drop every entity node once any document is superseded."""
    auth = _AUTH.model_copy(update={"resolved_excluded_document_ids": ["old"]})
    clause, _ = build_graph_authorization_predicate(
        auth, node_var="n", cross_tenant_support_roles=[]
    )
    assert "coalesce(n.source_document_id, n.document_id) IS NULL OR NOT" in clause


def test_python_recheck_excludes_a_superseded_document_node():
    """A `Document` node has `document_id`, not `source_document_id`."""
    auth = _AUTH.model_copy(update={"resolved_excluded_document_ids": ["old"]})
    assert not is_fact_authorized({"document_id": "old"}, auth, cross_tenant_support_roles=[])
    assert is_fact_authorized({"canonical_id": "svc"}, auth, cross_tenant_support_roles=[])


class _Node(dict):
    labels = ("Service",)


def test_path_fact_shows_each_relationship_direction_and_needs_every_edge_document():
    """The search runs both ways, so a `->` for every hop would misstate who depends on whom."""
    alpha = _Node(canonical_id="a", name="Alpha", first_seen_document_id="public")
    beta = _Node(canonical_id="b", name="Beta", first_seen_document_id="public")
    gamma = _Node(canonical_id="c", name="Gamma", first_seen_document_id="public")

    class _PathResult:
        nodes = [alpha, beta, gamma]
        relationships = [{"source_document_id": "d1"}, {"source_document_id": "d2"}]

    class _PathClient:
        def __init__(self) -> None:
            self.calls = 0

        def run_read(self, cypher, params):
            self.calls += 1
            if self.calls == 1:
                return [{"e": alpha, "entity_labels": ["Service"]}]
            if self.calls == 2:
                return [{"e": gamma, "entity_labels": ["Service"]}]
            return [{"path": _PathResult, "edge_starts": ["a", "c"]}]

    result = GraphRetriever(load_config(), _PathClient()).find_path(
        "Alpha", "Gamma", dataset_id="ds", auth=None
    )

    [fact] = result.facts
    assert fact.text == "Dependency path: Alpha -> Beta <- Gamma"
    assert [s.document_id for s in fact.support_sets[0]] == ["d1", "d2"]
