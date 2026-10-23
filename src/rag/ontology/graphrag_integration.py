"""Ontology-aware GraphRAG retrieval (docs/ontology.md sections 12, 13, 18).

Two retrieval functions over the SAME instance data, differing only in
whether the queried type is expanded through the ontology first:

`naive_type_match`: exact `rdf:type` match only, the behavior a
property-graph query matching one Neo4j label (no subsumption
knowledge) would give.

`ontology_aware_type_match`: expands the queried class to its bounded
subclass set (`query_expansion.expand_concept`) first, then matches any
of them. This is "ordinary GraphRAG vs. ontology-aware GraphRAG"
(docs/ontology.md section 18), expressed as two functions over one
shared dataset instead of two separate systems, so the comparison is
exact and reproducible.

Both functions operate on this milestone's own RDF mirror of the real
Neo4j `techfusion_graph_demo` dataset
(`data/ontology/instances/instances_graph_demo.ttl`), not a live Neo4j connection. See the
module-level "Why no live Neo4j call" note below for why, and
`expand_and_query_live_neo4j` for the sketched (written, not executed)
live-graph integration path.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.ontology.query_expansion import expand_concept
from rag.ontology.store import RdflibStore

_EXACT_TYPE_QUERY = """
SELECT ?entity ?name WHERE {
    ?entity a ?type .
    OPTIONAL { ?entity <http://techfusion.example/ontology/v1#name> ?name }
}
"""


@dataclass(frozen=True)
class TypeMatchResult:
    """One type-based retrieval's result set.

    Attributes
    ----------
    queried_class : str
        The class IRI that was queried.
    expanded_classes : list[str]
        Every additional class actually matched against, besides
        `queried_class` itself. Empty for `naive_type_match`.
    entities : list[str]
        Matched entity IRIs, deduplicated, in discovery order.
    """

    queried_class: str
    expanded_classes: list[str]
    entities: list[str]


def naive_type_match(data_store: RdflibStore, class_iri: str) -> TypeMatchResult:
    """Match entities by an EXACT `rdf:type` triple only.

    Parameters
    ----------
    data_store : RdflibStore
        Instance data to search.
    class_iri : str
        The class to match, exactly.

    Returns
    -------
    TypeMatchResult
        Entities with this exact asserted type. `expanded_classes` is
        always empty.
    """
    entities = _entities_of_exact_type(data_store, class_iri)
    return TypeMatchResult(queried_class=class_iri, expanded_classes=[], entities=entities)


def ontology_aware_type_match(
    data_store: RdflibStore,
    ontology_store: RdflibStore,
    class_iri: str,
    *,
    max_depth: int = 2,
    max_results: int = 20,
) -> TypeMatchResult:
    """Match entities by the queried class OR any of its bounded subclasses.

    Parameters
    ----------
    data_store : RdflibStore
        Instance data to search.
    ontology_store : RdflibStore
        Holds the `rdfs:subClassOf` triples `query_expansion.expand_concept`
        walks. May be the same store as `data_store` if both are loaded
        together.
    class_iri : str
        The root class to expand from.
    max_depth, max_results : int, optional
        Forwarded to `query_expansion.expand_concept`.

    Returns
    -------
    TypeMatchResult
        Entities matching the root class or any expanded subclass,
        deduplicated.
    """
    expansion = expand_concept(
        ontology_store, class_iri, max_depth=max_depth, max_results=max_results
    )
    all_classes = [class_iri, *expansion.expanded_classes]

    seen: set[str] = set()
    entities: list[str] = []
    for one_class in all_classes:
        for entity in _entities_of_exact_type(data_store, one_class):
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)

    return TypeMatchResult(
        queried_class=class_iri,
        expanded_classes=expansion.expanded_classes,
        entities=entities,
    )


def _entities_of_exact_type(store: RdflibStore, class_iri: str) -> list[str]:
    """Return every entity with an exact `rdf:type class_iri` triple, by IRI."""
    from typing import cast

    from rdflib import URIRef
    from rdflib.query import ResultRow

    rows = cast(
        "list[ResultRow]",
        list(store.query(_EXACT_TYPE_QUERY, init_bindings={"type": URIRef(class_iri)})),
    )
    return [str(row.entity) for row in rows]


# ---------------------------------------------------------------------------
# Live Neo4j integration: imports are lazy (inside the function body) so
# that importing this module never requires `neo4j` or a running server.
# `rag.graph.client`/`rag.graph.retrieval` are cheap to import; the one
# genuinely expensive import in this codebase is `rag.factory` (which
# constructs the embedder/reranker), and this function never touches it.
# Requires a reachable Neo4j instance: `Neo4jClient.health_check()` fails
# fast and this function raises `GraphUnavailableError` rather than
# hanging or silently returning nothing; see
# `tests/unit/test_ontology_graphrag_integration.py` for what is covered
# without a live server.
# ---------------------------------------------------------------------------


def expand_and_query_live_neo4j(
    ontology_store: RdflibStore,
    class_iri: str,
    *,
    dataset_id: str,
    config_path: str | None = None,
    max_depth: int = 2,
    max_results: int = 20,
) -> list[dict[str, object]]:
    """Expand `class_iri` via the ontology, then query live Neo4j for matching entities.

    Maps every expanded class (plus `class_iri` itself) back to its
    `rag.graph.schema.EntityType` label via the `tech:neo4jLabel`
    annotation already present on every class in
    `data/ontology/schema/techfusion_ontology_v1.ttl`, then runs one
    bounded, parameterized Cypher query matching any of those labels --
    the live-graph counterpart of `ontology_aware_type_match` above.
    Classes with no real Neo4j label (`tech:neo4jLabel` starting with
    `"none"`, e.g. `tech:Vendor`) are skipped; they can never match a
    real node.

    Parameters
    ----------
    ontology_store : RdflibStore
        Holds the ontology, for expansion and `tech:neo4jLabel` lookup.
    class_iri : str
        The root class to expand from.
    dataset_id : str
        The `rag.graph` dataset to query (e.g. `"techfusion_graph_demo"`).
    config_path : str | None, optional
        Passed to `rag.config.load_config`; `None` uses its own default
        (`config/default.yaml`).
    max_depth, max_results : int, optional
        Forwarded to `query_expansion.expand_concept`.

    Returns
    -------
    list[dict[str, object]]
        One dict per matched node (`name`, `labels`), bounded by
        `max_results`.

    Raises
    ------
    rag.graph.client.GraphUnavailableError
        If Neo4j is unreachable, propagated from `Neo4jClient.run_read`
        (or raised directly here if `health_check()` fails first).
    """
    from rag.config import load_config
    from rag.graph.client import GraphUnavailableError, Neo4jClient

    expansion = expand_concept(
        ontology_store, class_iri, max_depth=max_depth, max_results=max_results
    )
    neo4j_labels = _resolve_neo4j_labels(ontology_store, [class_iri, *expansion.expanded_classes])
    if not neo4j_labels:
        return []

    config = load_config(config_path) if config_path else load_config()
    client = Neo4jClient(config)
    try:
        if not client.health_check():
            raise GraphUnavailableError("Neo4j health check failed")
        cypher = """
        MATCH (n)
        WHERE n.dataset_id = $dataset_id AND any(l IN labels(n) WHERE l IN $labels)
        RETURN n.name AS name, labels(n) AS labels
        LIMIT $limit
        """
        rows = client.run_read(
            cypher,
            {"dataset_id": dataset_id, "labels": neo4j_labels, "limit": max_results},
        )
    finally:
        client.close()
    return rows


def _resolve_neo4j_labels(ontology_store: RdflibStore, class_iris: list[str]) -> list[str]:
    """Map ontology class IRIs to their real `tech:neo4jLabel` values, skipping any with none."""
    from typing import cast

    from rdflib import URIRef
    from rdflib.query import ResultRow

    neo4j_label_property = URIRef("http://techfusion.example/ontology/v1#neo4jLabel")
    labels: list[str] = []
    for class_iri in class_iris:
        rows = cast(
            "list[ResultRow]",
            list(
                ontology_store.query(
                    "SELECT ?label WHERE { ?cls ?prop ?label }",
                    init_bindings={"cls": URIRef(class_iri), "prop": neo4j_label_property},
                )
            ),
        )
        for row in rows:
            first_word = str(row.label).split()[0]
            if first_word.lower() != "none":
                labels.append(first_word)
    return labels
