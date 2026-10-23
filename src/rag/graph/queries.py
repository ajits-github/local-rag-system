"""Fixed, parameterized Cypher templates for the supported graph intents.

Each builder returns `(cypher, parameters)`. Caller-supplied values are
always query parameters. The only interpolated values are relationship
types and depth bounds, which come from code-controlled constants. Templates
that take an `auth_clause` AND it into their `WHERE` clause. The
`dependency_path` template takes none, and its results are checked in Python.
"""

from __future__ import annotations

from typing import Any

_DEPENDENCY_REL_TYPES = "DEPENDS_ON|RUNS_ON|CONNECTS_TO"


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


def dependents_of_query(
    *, dataset_id: str, entity_canonical_id: str, auth_clause: str, max_depth: int, limit: int
) -> tuple[str, dict[str, Any]]:
    """Build the "what depends on X" query, traversing dependency relationships in reverse.

    Bounded to `max_depth` hops. `auth_clause` must use `node_var="dependent"`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH path = (dependent)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]->(target)
    WHERE {_where(auth_clause)}
    WITH dependent, min(length(path)) AS hops
    RETURN dependent, labels(dependent) AS dependent_labels, hops
    ORDER BY hops ASC, dependent.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def runbooks_for_query(
    *, dataset_id: str, entity_canonical_id: str, auth_clause: str, limit: int
) -> tuple[str, dict[str, Any]]:
    """Build the "which runbooks apply to X" query (`HAS_RUNBOOK`, either direction).

    `auth_clause` must be built with `node_var="runbook"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[:HAS_RUNBOOK]-(runbook:Runbook)
    WHERE {_where(auth_clause)}
    RETURN DISTINCT runbook, labels(runbook) AS runbook_labels
    ORDER BY runbook.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def owners_of_query(
    *, dataset_id: str, entity_canonical_id: str, auth_clause: str, limit: int
) -> tuple[str, dict[str, Any]]:
    """Build the "which team/system owns X" query (`OWNED_BY`).

    `auth_clause` must be built with `node_var="owner"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[:OWNED_BY]->(owner)
    WHERE {_where(auth_clause)}
    RETURN DISTINCT owner, labels(owner) AS owner_labels
    ORDER BY owner.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def policies_governing_query(
    *, dataset_id: str, entity_canonical_id: str, auth_clause: str, limit: int
) -> tuple[str, dict[str, Any]]:
    """Build the "which policy governs X" query (`GOVERNED_BY`/`HAS_POLICY`, either direction).

    `auth_clause` must be built with `node_var="policy"`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    MATCH (target)-[:GOVERNED_BY|HAS_POLICY]-(policy:Policy)
    WHERE {_where(auth_clause)}
    RETURN DISTINCT policy, labels(policy) AS policy_labels
    ORDER BY policy.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def impact_of_query(
    *, dataset_id: str, entity_canonical_id: str, auth_clause: str, max_depth: int, limit: int
) -> tuple[str, dict[str, Any]]:
    """Build the "what is impacted if X fails" query.

    Returns the reverse dependency chain, bounded to `max_depth`, together
    with the target and any `Incident` nodes that `AFFECTS` the target.
    `auth_clause` must use `node_var="impacted_candidate"`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    cypher = f"""
    MATCH (target {{canonical_id: $entity_id, dataset_id: $dataset_id}})
    OPTIONAL MATCH depPath = (dependent)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]->(target)
    WITH target, collect(DISTINCT dependent) AS dependents
    OPTIONAL MATCH (incident:Incident)-[:AFFECTS]->(target)
    WITH target, dependents, collect(DISTINCT incident) AS direct_incidents
    UNWIND (dependents + [target]) AS impacted_candidate
    WITH impacted_candidate, direct_incidents
    WHERE {_where(auth_clause)}
    RETURN DISTINCT impacted_candidate AS impacted, labels(impacted_candidate) AS impacted_labels,
        direct_incidents
    ORDER BY impacted.name ASC
    LIMIT $limit
    """
    return cypher, {"dataset_id": dataset_id, "entity_id": entity_canonical_id, "limit": limit}


def dependency_path_query(
    *, dataset_id: str, source_canonical_id: str, target_canonical_id: str, max_depth: int
) -> tuple[str, dict[str, Any]]:
    """Build the shortest-path query between two entities, bounded by `max_depth`.

    No `auth_clause` is taken. The caller resolves both endpoints with
    authorization applied first, and every node on the returned path is
    re-checked in Python before it is used.

    Returns
    -------
    tuple[str, dict[str, Any]]
        `(cypher, parameters)`.
    """
    cypher = f"""
    MATCH (source {{canonical_id: $source_id, dataset_id: $dataset_id}})
    MATCH (destination {{canonical_id: $target_id, dataset_id: $dataset_id}})
    MATCH path = shortestPath(
        (source)-[:{_DEPENDENCY_REL_TYPES}*1..{max_depth}]-(destination)
    )
    RETURN path
    LIMIT 1
    """
    return cypher, {
        "dataset_id": dataset_id,
        "source_id": source_canonical_id,
        "target_id": target_canonical_id,
    }
