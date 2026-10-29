"""Graph retrieval: entity linking, intent classification, and bounded traversal.

`GraphRetriever` resolves entities named in a query and runs one bounded
query to answer it. It first tries a deterministic template matched by
keyword intent. If that finds nothing and LLM Cypher generation is enabled,
it falls back to generated Cypher. Results are returned as `GraphFact`s with
provenance, after the authorization re-check. This class does not read
`config.graph.enabled`; the caller checks that.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.config import AppConfig
from rag.generation.base import LLM
from rag.graph import queries
from rag.graph.authorization import build_graph_authorization_predicate, is_fact_authorized
from rag.graph.client import GraphUnavailableError, Neo4jClient
from rag.graph.cypher_guard import CypherValidationError, validate_generated_cypher
from rag.graph.schema import ENTITY_TYPES, RELATIONSHIP_TYPES
from rag.observability import metrics as observability_metrics
from rag.observability import tracing
from rag.prompts.loader import load_prompt_template
from rag.retrieval.authorization import AuthorizationContext

logger = logging.getLogger(__name__)

GraphQueryIntent = Literal[
    "dependents_of",
    "runbooks_for",
    "owners_of",
    "policies_governing",
    "impact_of",
    "dependency_path",
    "entity_lookup",
    "llm_generated",
    "unresolved",
]


class GraphEntityRef(BaseModel):
    """One resolved graph entity, as surfaced to a caller (never a bare Neo4j node id)."""

    canonical_id: str
    name: str
    entity_type: str


class GraphFact(BaseModel):
    """One normalized graph-derived fact: a fully human-readable, cited statement.

    The graph-side counterpart of a `SearchResult`'s chunk text. `rag.graph.fusion`
    converts it into a `SearchResult` for the shared context, citation, and
    redaction code.

    Attributes
    ----------
    text : str
        A human-readable rendering of the fact (e.g. "Document Service
        DEPENDS_ON PostgreSQL"), never a raw property dump.
    fact_type : {"entity", "relationship", "path"}
        What kind of graph structure this fact represents.
    entities : list[GraphEntityRef]
        Every entity mentioned in this fact.
    source_document_id, source_chunk_id : str | None
        Provenance, from the underlying relationship/node's own properties
        (`None` only for a fact with no single owning document, which
        should not occur for anything this codebase writes).
    dataset_id : str
        The dataset this fact was resolved within.
    confidence : float
        The underlying fact's extraction confidence (`1.0` for
        deterministic facts).
    extraction_method : str
        `"deterministic"` or `"llm"`.
    """

    text: str
    fact_type: Literal["entity", "relationship", "path"]
    entities: list[GraphEntityRef] = Field(default_factory=list)
    source_document_id: str | None = None
    source_chunk_id: str | None = None
    dataset_id: str
    confidence: float = 1.0
    extraction_method: str = "deterministic"


class GraphRetrievalResult(BaseModel):
    """The outcome of one `GraphRetriever.search` call.

    Attributes
    ----------
    facts : list[GraphFact]
        Normalized, authorized, bounded facts.
    intent : GraphQueryIntent
        Which query strategy answered this call.
    resolved_entities : list[GraphEntityRef]
        Entities the query text was linked to.
    truncated : bool
        Whether the underlying query hit `max_nodes_returned`/
        `max_paths_returned` and may have more (unreturned) matches.
    latency_ms : float
        Wall-clock time for this call.
    degraded : bool
        `True` when Neo4j was unreachable and this result is an empty
        fallback, not a genuine "no facts" answer.
    """

    facts: list[GraphFact] = Field(default_factory=list)
    intent: GraphQueryIntent
    resolved_entities: list[GraphEntityRef] = Field(default_factory=list)
    truncated: bool = False
    latency_ms: float = 0.0
    degraded: bool = False


# Literal keyword matching, so classification is reproducible. A wrong match
# only selects a different bounded, authorized query.
_DEPENDENTS_RE = re.compile(r"\b(depends? on|relies? on|what depends|who depends)\b", re.IGNORECASE)
_RUNBOOK_RE = re.compile(r"\brunbooks?\b", re.IGNORECASE)
_OWNER_RE = re.compile(r"\b(owns?|owner|ownership)\b", re.IGNORECASE)
_POLICY_RE = re.compile(r"\bpolic(y|ies)\b", re.IGNORECASE)
_IMPACT_RE = re.compile(
    r"\b(impact|affect|unavailable|fails?|failure|outage|down|goes down)\b", re.IGNORECASE
)
_PATH_RE = re.compile(
    r"\b(path between|relationship between|how (is|are) .+ related to|connects? .+ to)\b",
    re.IGNORECASE,
)
# Capitalized-phrase heuristic for candidate entity names. This is not NER.
_CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][\w&/-]*(?:\s+[A-Z][\w&/-]*)*\b")


def _extract_candidate_names(query: str) -> list[str]:
    """Extract candidate entity-name spans from free text query, longest first."""
    spans = {
        m.group(0).strip() for m in _CAPITALIZED_SPAN_RE.finditer(query) if len(m.group(0)) > 2
    }
    return sorted(spans, key=len, reverse=True)


def _classify_intent(query: str) -> GraphQueryIntent:
    """Classify a query's graph-retrieval intent from literal keyword cues."""
    if _PATH_RE.search(query):
        return "dependency_path"
    if _IMPACT_RE.search(query):
        return "impact_of"
    if _RUNBOOK_RE.search(query):
        return "runbooks_for"
    if _OWNER_RE.search(query):
        return "owners_of"
    if _POLICY_RE.search(query):
        return "policies_governing"
    if _DEPENDENTS_RE.search(query):
        return "dependents_of"
    return "entity_lookup"


class GraphRetriever:
    """Resolves entities and runs bounded graph queries, deterministic or LLM-assisted."""

    def __init__(self, config: AppConfig, client: Neo4jClient, llm: LLM | None = None) -> None:
        """Wire up from config and an injected `Neo4jClient`.

        Parameters
        ----------
        config : AppConfig
            Application configuration; `config.graph` supplies every bound.
        client : Neo4jClient
            The graph client to query through.
        llm : LLM | None, optional
            Used only when `config.graph.cypher_generation.enabled=True`.
        """
        self._config = config
        self._client = client
        self._llm = llm
        self._cross_tenant_support_roles = config.security.authorization.cross_tenant_support_roles
        self._cypher_gen_template = None
        if config.graph.cypher_generation.enabled:
            self._cypher_gen_template = load_prompt_template(
                str(config.agent_prompt_template_path(config.graph.cypher_generation.prompt_path))
            )

    def resolve_entities(
        self,
        name_query: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        entity_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[GraphEntityRef]:
        """Resolve free-text `name_query` to authorized candidate graph entities.

        Parameters
        ----------
        name_query : str
            Free-text entity name or fragment.
        dataset_id : str
            Dataset namespace to search within.
        auth : AuthorizationContext | None
            Caller authorization context.
        entity_types : list[str] | None, optional
            Restrict to these types; each must already be in
            `ENTITY_TYPES` (silently dropped otherwise, never passed
            through to Cypher unvalidated).
        limit : int, optional
            Maximum candidates to return.

        Returns
        -------
        list[GraphEntityRef]
            Authorized candidates, best match first.
        """
        validated_types = (
            [t for t in entity_types if t in ENTITY_TYPES] if entity_types else None
        ) or None
        auth_clause, auth_params = build_graph_authorization_predicate(
            auth, node_var="e", cross_tenant_support_roles=self._cross_tenant_support_roles
        )
        cypher, params = queries.resolve_entities_query(
            dataset_id=dataset_id,
            name_query=name_query,
            entity_types=validated_types,
            auth_clause=auth_clause,
            limit=limit,
        )
        params.update(auth_params)
        rows = self._client.run_read(cypher, params)
        results = []
        for row in rows:
            node = row["e"]
            if not is_fact_authorized(
                dict(node), auth, cross_tenant_support_roles=self._cross_tenant_support_roles
            ):
                continue
            entity_label = next(
                (label for label in row["entity_labels"] if label in ENTITY_TYPES), "Document"
            )
            results.append(
                GraphEntityRef(
                    canonical_id=node["canonical_id"], name=node["name"], entity_type=entity_label
                )
            )
        return results

    def get_entity(
        self,
        name: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        entity_type: str | None = None,
    ) -> GraphRetrievalResult:
        """Resolve a named entity and summarize its direct (1-hop) neighbors.

        The `get_entity` agent tool's implementation (`rag.agent.tools.get_entity`).

        Parameters
        ----------
        name : str
            The entity's name (exact, alias, or substring).
        dataset_id : str
            Dataset namespace to search within.
        auth : AuthorizationContext | None
            Caller authorization context.
        entity_type : str | None, optional
            Restrict resolution to this type, if known.

        Returns
        -------
        GraphRetrievalResult
            `intent="entity_lookup"`, `resolved_entities` with 0 or 1
            entry, and one fact per authorized direct neighbor.
        """
        cfg = self._config.graph.traversal
        resolved = self.resolve_entities(
            name,
            dataset_id=dataset_id,
            auth=auth,
            entity_types=[entity_type] if entity_type else None,
            limit=1,
        )
        if not resolved:
            return GraphRetrievalResult(intent="entity_lookup")
        return self._entity_lookup_result(resolved[0], dataset_id=dataset_id, auth=auth, cfg=cfg)

    def get_neighbors(
        self,
        entity_name: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        relationship_type: str | None = None,
        max_hops: int = 1,
    ) -> GraphRetrievalResult:
        """Traverse up to `max_hops` from a named entity, optionally by relationship type.

        `max_hops` is clamped to `config.graph.traversal.max_traversal_depth`.

        Parameters
        ----------
        entity_name : str
            The starting entity's name.
        dataset_id : str
            Dataset namespace to search within.
        auth : AuthorizationContext | None
            Caller authorization context.
        relationship_type : str | None, optional
            Restrict traversal to this relationship type; must already be
            in `RELATIONSHIP_TYPES` (silently ignored otherwise, never
            interpolated into Cypher unvalidated).
        max_hops : int, optional
            Upper bound on traversal depth, clamped to
            `config.graph.traversal.max_traversal_depth`.

        Returns
        -------
        GraphRetrievalResult
            `intent="entity_lookup"`, one fact per authorized node found
            within `max_hops`.
        """
        cfg = self._config.graph.traversal
        resolved = self.resolve_entities(entity_name, dataset_id=dataset_id, auth=auth, limit=1)
        if not resolved:
            return GraphRetrievalResult(intent="entity_lookup")
        entity = resolved[0]
        bounded_hops = min(max(max_hops, 1), cfg.max_traversal_depth)
        rel_filter = f":{relationship_type}" if relationship_type in RELATIONSHIP_TYPES else ""
        auth_clause, auth_params = build_graph_authorization_predicate(
            auth, node_var="neighbor", cross_tenant_support_roles=self._cross_tenant_support_roles
        )
        cypher = f"""
        MATCH (e {{canonical_id: $entity_id, dataset_id: $dataset_id}})
        MATCH path = (e)-[{rel_filter}*1..{bounded_hops}]-(neighbor)
        WHERE {auth_clause or "true"}
        WITH DISTINCT neighbor, labels(neighbor) AS neighbor_labels, min(length(path)) AS hops
        RETURN neighbor, neighbor_labels, hops
        ORDER BY hops ASC, neighbor.name ASC
        LIMIT $limit
        """
        params = {
            "entity_id": entity.canonical_id,
            "dataset_id": dataset_id,
            "limit": cfg.max_nodes_returned,
            **auth_params,
        }
        rows = self._client.run_read(cypher, params)
        facts = []
        for row in rows:
            neighbor = dict(row["neighbor"])
            if not is_fact_authorized(
                neighbor, auth, cross_tenant_support_roles=self._cross_tenant_support_roles
            ):
                continue
            facts.append(
                GraphFact(
                    text=f"{neighbor['name']} is within {row['hops']} hop(s) of {entity.name}"
                    + (
                        f" via {relationship_type}"
                        if relationship_type in RELATIONSHIP_TYPES
                        else ""
                    ),
                    fact_type="relationship",
                    entities=[
                        entity,
                        GraphEntityRef(
                            canonical_id=neighbor["canonical_id"],
                            name=neighbor["name"],
                            entity_type=_first_entity_type(row["neighbor_labels"]),
                        ),
                    ],
                    source_document_id=neighbor.get("first_seen_document_id"),
                    dataset_id=dataset_id,
                )
            )
        return GraphRetrievalResult(
            facts=facts,
            intent="entity_lookup",
            resolved_entities=[entity],
            truncated=len(rows) >= cfg.max_nodes_returned,
        )

    def find_path(
        self,
        source_entity_name: str,
        target_entity_name: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
    ) -> GraphRetrievalResult:
        """Bounded shortest path between two named entities.

        Skips the name-extraction heuristic used by `search()`, since the caller
        names both entities.

        Returns
        -------
        GraphRetrievalResult
            `intent="dependency_path"`, at most one `fact_type="path"` fact.
        """
        cfg = self._config.graph.traversal
        return self._search_dependency_path(
            source_entity_name, target_entity_name, dataset_id=dataset_id, auth=auth, cfg=cfg
        )

    def search(
        self,
        query: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None = None,
    ) -> GraphRetrievalResult:
        """Answer one natural-language query via entity linking and bounded traversal.

        Returns a `degraded=True` empty result on a Neo4j outage. Never raises
        for that case.

        Parameters
        ----------
        query : str
            The natural-language question.
        dataset_id : str
            Dataset namespace to search within.
        auth : AuthorizationContext | None, optional
            Caller authorization context.

        Returns
        -------
        GraphRetrievalResult
            Normalized, authorized, bounded facts.
        """
        t0 = time.perf_counter()
        cfg = self._config.graph.traversal
        try:
            with tracing.start_span("graph_retrieval") as span:
                result = self._search_inner(query, dataset_id=dataset_id, auth=auth, cfg=cfg)
                tracing.set_attributes(
                    span,
                    {
                        "intent": result.intent,
                        "fact_count": len(result.facts),
                        "resolved_entity_count": len(result.resolved_entities),
                    },
                )
        except GraphUnavailableError:
            logger.warning("Graph retrieval degraded: Neo4j unavailable", exc_info=True)
            observability_metrics.observe_error("graph")
            result = GraphRetrievalResult(intent="unresolved", degraded=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        observability_metrics.observe_graph_query(
            result.intent, latency_ms / 1000, len(result.facts)
        )
        return result.model_copy(update={"latency_ms": latency_ms})

    def _search_inner(
        self, query: str, *, dataset_id: str, auth: AuthorizationContext | None, cfg: Any
    ) -> GraphRetrievalResult:
        """Do `search`'s work: classify intent, resolve entities, run the matching template."""
        intent = _classify_intent(query)
        candidate_names = _extract_candidate_names(query)

        if intent == "dependency_path" and len(candidate_names) >= 2:
            return self._search_dependency_path(
                candidate_names[0], candidate_names[1], dataset_id=dataset_id, auth=auth, cfg=cfg
            )

        resolved: list[GraphEntityRef] = []
        for name in candidate_names:
            resolved = self.resolve_entities(name, dataset_id=dataset_id, auth=auth, limit=1)
            if resolved:
                break

        if not resolved:
            if self._cypher_gen_template is not None and self._llm is not None:
                return self._search_via_generated_cypher(query, dataset_id=dataset_id, auth=auth)
            return GraphRetrievalResult(intent="unresolved")

        entity = resolved[0]
        template_map = {
            "dependents_of": (queries.dependents_of_query, "dependent"),
            "runbooks_for": (queries.runbooks_for_query, "runbook"),
            "owners_of": (queries.owners_of_query, "owner"),
            "policies_governing": (queries.policies_governing_query, "policy"),
        }

        if intent == "impact_of":
            return self._run_impact_of(entity, dataset_id=dataset_id, auth=auth, cfg=cfg)
        if intent in template_map:
            builder, result_var = template_map[intent]
            return self._run_neighbor_template(
                builder, result_var, entity, intent, dataset_id=dataset_id, auth=auth, cfg=cfg
            )
        return self._entity_lookup_result(entity, dataset_id=dataset_id, auth=auth, cfg=cfg)

    def _run_neighbor_template(
        self,
        builder: Any,
        result_var: str,
        entity: GraphEntityRef,
        intent: GraphQueryIntent,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        cfg: Any,
    ) -> GraphRetrievalResult:
        """Run a one-hop neighbor template (`dependents_of`/`runbooks_for`/`owners_of`/`policies_governing`)."""  # noqa: E501
        auth_clause, auth_params = build_graph_authorization_predicate(
            auth, node_var=result_var, cross_tenant_support_roles=self._cross_tenant_support_roles
        )
        kwargs: dict[str, Any] = {
            "dataset_id": dataset_id,
            "entity_canonical_id": entity.canonical_id,
            "auth_clause": auth_clause,
            "limit": cfg.max_nodes_returned,
        }
        if intent == "dependents_of":
            kwargs["max_depth"] = cfg.max_traversal_depth
        cypher, params = builder(**kwargs)
        params.update(auth_params)
        rows = self._client.run_read(cypher, params)
        facts = []
        for row in rows:
            node = dict(row[result_var])
            if not is_fact_authorized(
                node, auth, cross_tenant_support_roles=self._cross_tenant_support_roles
            ):
                continue
            neighbor_type = _first_entity_type(row.get(f"{result_var}_labels", []))
            verb = _INTENT_VERBS[intent]
            facts.append(
                GraphFact(
                    text=f"{node['name']} {verb} {entity.name}",
                    fact_type="relationship",
                    entities=[
                        GraphEntityRef(
                            canonical_id=node["canonical_id"],
                            name=node["name"],
                            entity_type=neighbor_type,
                        ),
                        entity,
                    ],
                    source_document_id=node.get("first_seen_document_id"),
                    dataset_id=dataset_id,
                )
            )
        return GraphRetrievalResult(
            facts=facts,
            intent=intent,
            resolved_entities=[entity],
            truncated=len(rows) >= cfg.max_nodes_returned,
        )

    def _run_impact_of(
        self,
        entity: GraphEntityRef,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        cfg: Any,
    ) -> GraphRetrievalResult:
        """Run the impact-analysis template and normalize its two-part result shape."""
        auth_clause, auth_params = build_graph_authorization_predicate(
            auth,
            node_var="impacted_candidate",
            cross_tenant_support_roles=self._cross_tenant_support_roles,
        )
        cypher, params = queries.impact_of_query(
            dataset_id=dataset_id,
            entity_canonical_id=entity.canonical_id,
            auth_clause=auth_clause,
            max_depth=cfg.max_traversal_depth,
            limit=cfg.max_nodes_returned,
        )
        params.update(auth_params)
        rows = self._client.run_read(cypher, params)
        facts = []
        for row in rows:
            node = dict(row["impacted"])
            if not is_fact_authorized(
                node, auth, cross_tenant_support_roles=self._cross_tenant_support_roles
            ):
                continue
            if node["canonical_id"] == entity.canonical_id:
                text = (
                    f"{node['name']} would be directly impacted if "
                    f"{entity.name} fails or is unavailable"
                )
            else:
                text = (
                    f"{node['name']} could be impacted, since it depends "
                    f"(directly or transitively) on {entity.name}"
                )
            facts.append(
                GraphFact(
                    text=text,
                    fact_type="relationship",
                    entities=[
                        GraphEntityRef(
                            canonical_id=node["canonical_id"],
                            name=node["name"],
                            entity_type=_first_entity_type(row.get("impacted_labels", [])),
                        ),
                        entity,
                    ],
                    source_document_id=node.get("first_seen_document_id"),
                    dataset_id=dataset_id,
                )
            )
        return GraphRetrievalResult(
            facts=facts,
            intent="impact_of",
            resolved_entities=[entity],
            truncated=len(rows) >= cfg.max_nodes_returned,
        )

    def _search_dependency_path(
        self,
        name_a: str,
        name_b: str,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        cfg: Any,
    ) -> GraphRetrievalResult:
        """Resolve two entities and run the bounded shortest-path template between them."""
        entities_a = self.resolve_entities(name_a, dataset_id=dataset_id, auth=auth, limit=1)
        entities_b = self.resolve_entities(name_b, dataset_id=dataset_id, auth=auth, limit=1)
        if not entities_a or not entities_b:
            return GraphRetrievalResult(intent="dependency_path")
        entity_a, entity_b = entities_a[0], entities_b[0]
        cypher, params = queries.dependency_path_query(
            dataset_id=dataset_id,
            source_canonical_id=entity_a.canonical_id,
            target_canonical_id=entity_b.canonical_id,
            max_depth=cfg.max_traversal_depth,
        )
        rows = self._client.run_read(cypher, params)
        if not rows:
            return GraphRetrievalResult(
                intent="dependency_path", resolved_entities=[entity_a, entity_b]
            )
        path = rows[0]["path"]
        path_entries = [(dict(n), list(n.labels)) for n in path.nodes]
        # Permission to know two entities exist does not grant permission
        # to know they are connected: a relationship carries its own
        # allowed_roles/trust_level/source_document_id (see upsert.py),
        # independent of either endpoint's. Every node AND every edge on
        # the path must be authorized before any of it is surfaced.
        path_relationship_props = [dict(r) for r in path.relationships]
        if not all(
            is_fact_authorized(n, auth, cross_tenant_support_roles=self._cross_tenant_support_roles)
            for n, _labels in path_entries
        ) or not all(
            is_fact_authorized(r, auth, cross_tenant_support_roles=self._cross_tenant_support_roles)
            for r in path_relationship_props
        ):
            # A path with any unauthorized hop (node or relationship) is
            # never partially surfaced.
            return GraphRetrievalResult(
                intent="dependency_path", resolved_entities=[entity_a, entity_b]
            )
        chain = " -> ".join(n["name"] for n, _labels in path_entries)
        fact = GraphFact(
            text=f"Dependency path: {chain}",
            fact_type="path",
            entities=[
                GraphEntityRef(
                    canonical_id=n["canonical_id"],
                    name=n["name"],
                    entity_type=_first_entity_type(labels),
                )
                for n, labels in path_entries
            ],
            source_document_id=path_entries[0][0].get("first_seen_document_id"),
            dataset_id=dataset_id,
        )
        return GraphRetrievalResult(
            facts=[fact], intent="dependency_path", resolved_entities=[entity_a, entity_b]
        )

    def _entity_lookup_result(
        self,
        entity: GraphEntityRef,
        *,
        dataset_id: str,
        auth: AuthorizationContext | None,
        cfg: Any,
    ) -> GraphRetrievalResult:
        """Fall back to a plain 1-hop neighbor summary when no specific intent matched."""
        auth_clause, auth_params = build_graph_authorization_predicate(
            auth, node_var="neighbor", cross_tenant_support_roles=self._cross_tenant_support_roles
        )
        cypher = f"""
        MATCH (e {{canonical_id: $entity_id, dataset_id: $dataset_id}})
        MATCH (e)-[r]-(neighbor)
        WHERE {auth_clause or "true"}
        RETURN type(r) AS rel_type, startNode(r) = e AS outgoing, neighbor,
            labels(neighbor) AS neighbor_labels
        LIMIT $limit
        """
        params = {
            "entity_id": entity.canonical_id,
            "dataset_id": dataset_id,
            "limit": cfg.max_nodes_returned,
            **auth_params,
        }
        rows = self._client.run_read(cypher, params)
        facts = []
        for row in rows:
            neighbor = dict(row["neighbor"])
            if not is_fact_authorized(
                neighbor, auth, cross_tenant_support_roles=self._cross_tenant_support_roles
            ):
                continue
            subj, obj = (
                (entity.name, neighbor["name"])
                if row["outgoing"]
                else (
                    neighbor["name"],
                    entity.name,
                )
            )
            facts.append(
                GraphFact(
                    text=f"{subj} {row['rel_type']} {obj}",
                    fact_type="relationship",
                    entities=[
                        entity,
                        GraphEntityRef(
                            canonical_id=neighbor["canonical_id"],
                            name=neighbor["name"],
                            entity_type=_first_entity_type(row.get("neighbor_labels", [])),
                        ),
                    ],
                    source_document_id=neighbor.get("first_seen_document_id"),
                    dataset_id=dataset_id,
                )
            )
        return GraphRetrievalResult(
            facts=facts,
            intent="entity_lookup",
            resolved_entities=[entity],
            truncated=len(rows) >= cfg.max_nodes_returned,
        )

    def _search_via_generated_cypher(
        self, query: str, *, dataset_id: str, auth: AuthorizationContext | None
    ) -> GraphRetrievalResult:
        """Generate, validate, run, and authorize-filter Cypher for an unresolved query.

        Applies the text checks in `cypher_guard`, runs the query read-only, and
        re-checks each returned node.
        """
        assert self._llm is not None and self._cypher_gen_template is not None
        cfg = self._config.graph.cypher_generation
        system, user = self._cypher_gen_template.render(
            query=query, relationship_types=", ".join(sorted(RELATIONSHIP_TYPES))
        )
        raw = self._llm.generate(system, user)
        try:
            validated = validate_generated_cypher(
                raw, max_depth=cfg.max_depth, max_result_limit=cfg.max_result_limit
            )
        except CypherValidationError as exc:
            logger.warning("Rejected LLM-generated Cypher: %s", exc)
            observability_metrics.observe_error("graph_cypher_generation")
            return GraphRetrievalResult(intent="unresolved")

        try:
            rows = self._client.run_read(
                validated, {"dataset_id": dataset_id}, timeout=cfg.timeout_seconds
            )
        except GraphUnavailableError:
            logger.warning("Generated Cypher execution failed", exc_info=True)
            return GraphRetrievalResult(intent="unresolved", degraded=True)

        facts = []
        for row in rows:
            for _key, value in row.items():
                coerced = _coerce_node_dict(value)
                if coerced is None:
                    continue
                node_dict, entity_type = coerced
                if not is_fact_authorized(
                    node_dict, auth, cross_tenant_support_roles=self._cross_tenant_support_roles
                ):
                    continue
                facts.append(
                    GraphFact(
                        text=f"{node_dict.get('name', '(unnamed)')} ({entity_type})",
                        fact_type="entity",
                        entities=[
                            GraphEntityRef(
                                canonical_id=node_dict.get("canonical_id", "unknown"),
                                name=node_dict.get("name", "(unnamed)"),
                                entity_type=entity_type,
                            )
                        ],
                        source_document_id=node_dict.get("first_seen_document_id"),
                        dataset_id=dataset_id,
                    )
                )
        return GraphRetrievalResult(facts=facts, intent="llm_generated")


_INTENT_VERBS: dict[GraphQueryIntent, str] = {
    "dependents_of": "depends on",
    "runbooks_for": "has runbook applicable to",
    "owners_of": "is owned by",
    "policies_governing": "is governed by",
}


def _first_entity_type(labels: list[str]) -> str:
    """Return the first label that's in `ENTITY_TYPES`, or `"Document"` as a safe fallback.

    A node written by `GraphWriter` always has exactly one label (its
    `entity_type`), so `labels` normally has length 1; the allowlist
    filter is defense in depth against an unexpected extra label.
    """
    return next((label for label in labels if label in ENTITY_TYPES), "Document")


def _coerce_node_dict(value: Any) -> tuple[dict[str, Any], str] | None:
    """Return `(dict(value), entity_type)` for a Neo4j `Node`, else `None`.

    Only nodes count as entity evidence. Relationships and scalars are skipped.
    """
    if hasattr(value, "labels") and hasattr(value, "items"):
        return dict(value), _first_entity_type(list(value.labels))
    return None
