"""Fixed, parameterized Cypher templates for the supported graph intents.

Each builder returns `(cypher, parameters)`. Caller-supplied values are
always query parameters. The only interpolated values are relationship
types, depth bounds and support limits, which come from code-controlled
constants. Templates AND their `auth_clause` (the returned node) and
`path_auth_clause`/`edge_auth_clause` (everything traversed) into `WHERE`.
Each returns `support_sets`: the `[document_id, checksum]` pairs of the
relationships behind a fact, one list per derivation, shortest first.
"""

from __future__ import annotations

from typing import Any

_DEPENDENCY_REL_TYPES = "DEPENDS_ON|RUNS_ON|CONNECTS_TO"
_MAX_SUPPORT_SETS = 5
_PATH_SUPPORT = "[rel IN relationships(path) | [rel.source_document_id, rel.document_checksum]]"
_PATH_ORDER = "length(path) ASC, [rel IN relationships(path) | rel.source_document_id] ASC"
EDGE_SUPPORT_SETS = (
    f"collect(DISTINCT [[r.source_document_id, r.document_checksum]])[..{_MAX_SUPPORT_SETS}]"
)


def _where(*clauses: str) -> str:
    """Join non-empty clauses with `AND`, or return `"true"` if all are empty."""
    non_empty = [c for c in clauses if c]
    return " AND ".join(non_empty) if non_empty else "true"


def resolve_entities_query(
    *,
    dataset_id: str,
    name_query: str,
    entity_types: list[str] | None,
    auth_clause: str,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the query that links free-text `name_query` to graph entities.

    Exact name matches sort first. Alias and substring matches follow. The
    query is always scoped to `dataset_id`.

    Parameters
    ----------
    dataset_id : str
        The dataset namespace to search within.
    name_query : str
        Free-text entity name (or fragment) to resolve.
    entity_types : list[str] | None
        Restrict to these entity types (already validated against
        `rag.graph.schema.ENTITY_TYPES` by the caller), or `None` for any
        type.
    auth_clause : str
        Authorization predicate fragment from
        `build_graph_authorization_predicate(..., node_var="e")`.
    limit : int
        Maximum number of candidate entities to return.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    type_clause = "any(l IN labels(e) WHERE l IN $entity_types)" if entity_types else ""
    cypher = f"""
    MATCH (e)
    WHERE e.dataset_id = $dataset_id
      AND {_where(type_clause, auth_clause)}
      AND (
        toLower(e.name) = toLower($name_query)
        OR toLower($name_query) IN [a IN coalesce(e.aliases, []) | toLower(a)]
        OR toLower(e.name) CONTAINS toLower($name_query)
      )
    RETURN e, labels(e) AS entity_labels,
      CASE WHEN toLower(e.name) = toLower($name_query) THEN 0 ELSE 1 END AS rank
    ORDER BY rank ASC, e.name ASC
    LIMIT $limit
    """
    parameters: dict[str, Any] = {
        "dataset_id": dataset_id,
        "name_query": name_query,
        "limit": limit,
    }
    if entity_types:
        parameters["entity_types"] = entity_types
    return cypher, parameters


def path_support_with(result_var: str) -> str:
    """Build the `WITH` clauses aggregating `path` into `hops` and `support_sets` for `result_var`.

    `result_var` must be a hardcoded literal.
    """
    return f"""WITH {result_var}, path ORDER BY {_PATH_ORDER}
    WITH {result_var}, min(length(path)) AS hops,
        collect({_PATH_SUPPORT})[..{_MAX_SUPPORT_SETS}] AS support_sets"""


def _edge_support_with(result_var: str) -> str:
    """Build the `WITH` clauses aggregating one-hop edge `r` into `support_sets`."""
    return f"""WITH {result_var}, r ORDER BY r.source_document_id ASC
    WITH {result_var}, {EDGE_SUPPORT_SETS} AS support_sets"""


def dependents_of_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    path_auth_clause: str,
    max_depth: int,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "what depends on X" query, traversing dependency relationships in reverse.

    Bounded to `max_depth` hops. `auth_clause` must use `node_var="dependent"`,
    and `path_auth_clause` must use `path_var="path"`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH path = (dependent)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]->(target)
    WHERE {_where(auth_clause, path_auth_clause)}
    {path_support_with("dependent")}
    RETURN dependent, labels(dependent) AS dependent_labels, hops, support_sets
    ORDER BY hops ASC, dependent.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def dependencies_of_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    path_auth_clause: str,
    max_depth: int,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "what does X depend on" query, traversing dependency relationships forward.

    Bounded to `max_depth` hops. `auth_clause` must use `node_var="dependency"`,
    and `path_auth_clause` must use `path_var="path"`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    cypher = f"""
    MATCH (source {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH path = (source)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]->(dependency)
    WHERE {_where(auth_clause, path_auth_clause)}
    {path_support_with("dependency")}
    RETURN dependency, labels(dependency) AS dependency_labels, hops, support_sets
    ORDER BY hops ASC, dependency.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def runbooks_for_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    edge_auth_clause: str,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "which runbooks apply to X" query (`HAS_RUNBOOK`, either direction).

    `auth_clause` must be built with `node_var="runbook"`, and `edge_auth_clause`
    with `node_var="r"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[r:HAS_RUNBOOK]-(runbook:Runbook)
    WHERE {_where(auth_clause, edge_auth_clause)}
    {_edge_support_with("runbook")}
    RETURN runbook, labels(runbook) AS runbook_labels, support_sets
    ORDER BY runbook.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def owners_of_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    edge_auth_clause: str,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "which team/system owns X" query (`OWNED_BY`).

    `auth_clause` must be built with `node_var="owner"`, and `edge_auth_clause`
    with `node_var="r"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[r:OWNED_BY]->(owner)
    WHERE {_where(auth_clause, edge_auth_clause)}
    {_edge_support_with("owner")}
    RETURN owner, labels(owner) AS owner_labels, support_sets
    ORDER BY owner.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def policies_governing_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    edge_auth_clause: str,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "which policy governs X" query (`GOVERNED_BY`/`HAS_POLICY`, either direction).

    `auth_clause` must be built with `node_var="policy"`, and `edge_auth_clause`
    with `node_var="r"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[r:GOVERNED_BY|HAS_POLICY]-(policy:Policy)
    WHERE {_where(auth_clause, edge_auth_clause)}
    {_edge_support_with("policy")}
    RETURN policy, labels(policy) AS policy_labels, support_sets
    ORDER BY policy.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def impact_of_query(
    *,
    dataset_id: str,
    entity_canonical_id: str,
    auth_clause: str,
    path_auth_clause: str,
    max_depth: int,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build the "what is impacted if X fails" query.

    Returns the reverse dependency chain, bounded to `max_depth`, together
    with the target. `auth_clause` must use `node_var="impacted_candidate"`,
    and `path_auth_clause` must use `path_var="path"`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`. The target's own row has empty `support_sets`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    OPTIONAL MATCH path = (dependent)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]->(target)
    WHERE {_where(path_auth_clause)}
    WITH target, dependent, path ORDER BY {_PATH_ORDER}
    WITH target, dependent, collect({_PATH_SUPPORT})[..{_MAX_SUPPORT_SETS}] AS support_sets
    WITH target, collect(
        CASE WHEN dependent IS NULL THEN null
        ELSE {{node: dependent, support_sets: support_sets}} END
    ) AS entries
    UNWIND entries + [{{node: target, support_sets: []}}] AS entry
    WITH entry.node AS impacted_candidate, entry.support_sets AS support_sets
    WHERE {_where(auth_clause)}
    RETURN impacted_candidate AS impacted, labels(impacted_candidate) AS impacted_labels,
        support_sets
    ORDER BY impacted.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def dependency_path_query(
    *,
    dataset_id: str,
    source_canonical_id: str,
    target_canonical_id: str,
    path_auth_clause: str,
    max_depth: int,
) -> tuple[str, dict[str, Any]]:
    """Build the shortest-path query between two entities, bounded by `max_depth`.

    Searches in both directions. `path_auth_clause` must use `path_var="path"`,
    so a restricted path never hides an authorized longer one. The caller also
    re-checks every node and relationship on the returned path in Python.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`. `edge_starts` holds each relationship's start
        node id, which gives its direction.
    """
    cypher = f"""
    MATCH (source {{canonical_id: $source_id, dataset_id: $dataset_id}})
    MATCH (destination {{canonical_id: $target_id, dataset_id: $dataset_id}})
    MATCH path = shortestPath(
        (source)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]-(destination)
    )
    WHERE {_where(path_auth_clause)}
    RETURN path, [rel IN relationships(path) | startNode(rel).canonical_id] AS edge_starts
    LIMIT 1
    """
    return cypher, {
        "dataset_id": dataset_id,
        "source_id": source_canonical_id,
        "target_id": target_canonical_id,
    }
