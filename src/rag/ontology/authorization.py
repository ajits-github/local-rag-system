"""Tenant/role authorization for ontology facts (docs/ontology.md section 17, ontology milestone).

Reuses `rag.graph.authorization.is_fact_authorized` and
`rag.retrieval.authorization.AuthorizationContext` directly, rather
than re-implementing the rule: this module's only job is reading an RDF
entity's tenant/role properties into the same `dict` shape
`is_fact_authorized` already expects from a Neo4j node's properties, so
the exact same authorization function, not a parallel, independently
maintained copy of its logic, governs both the property graph and
this semantic layer.

The security property this enforces (docs/ontology.md section 17's explicit
requirement): ontology reasoning (subclass/equivalence/inverse
entailment) is a read-only, server-side-only operation over facts that
were already subject to this same authorization check before a
reasoner ever ran. Inference can make an ALREADY-authorized fact
visible under an additional inferred type or inverse relationship, but
it can never make an UNAUTHORIZED fact visible at all, because
`is_entity_authorized` below re-checks every result again, after
inference, independent of whatever a query's own `FILTER` clause
expressed. A semantic relationship (e.g. "X is a Service") never, by
itself, grants access to instance data about X.
"""

from __future__ import annotations

from typing import cast

from rdflib.query import ResultRow

from rag.graph.authorization import is_fact_authorized
from rag.ontology.store import RdflibStore
from rag.retrieval.authorization import AuthorizationContext

_PROPERTIES_QUERY = """
PREFIX tech: <http://techfusion.example/ontology/v1#>
SELECT ?tenantId ?role ?trustLevel ?sourceDocumentId WHERE {
    OPTIONAL { ?entity tech:tenantId ?tenantId }
    OPTIONAL { ?entity tech:allowedRole ?role }
    OPTIONAL { ?entity tech:trustLevel ?trustLevel }
    OPTIONAL { ?entity tech:sourceDocumentId ?sourceDocumentId }
}
"""


def fetch_entity_auth_properties(store: RdflibStore, entity_iri: str) -> dict[str, object]:
    """Read one entity's tenant/role/trust/provenance properties off the graph.

    Returns the same `dict` shape `rag.graph.authorization.is_fact_authorized`
    already expects from a Neo4j node's properties (`tenant_id`,
    `allowed_roles`, `trust_level`, `source_document_id`), so that
    function can be called unmodified against RDF-sourced data too.

    Parameters
    ----------
    store : RdflibStore
        The graph to read from.
    entity_iri : str
        The entity's full IRI (not a prefixed name).

    Returns
    -------
    dict[str, object]
        `{"tenant_id": str | None, "allowed_roles": list[str] | None,
        "trust_level": str | None, "source_document_id": str | None}`.
        `tenant_id`/`trust_level`/`source_document_id` are `None` when
        no corresponding triple exists (unrestricted/unset, mirroring
        `rag.graph.authorization`'s `tenant_id IS NULL` convention
        exactly); `allowed_roles` is `None` (not an empty list) when no
        `tech:allowedRole` triple exists, since `is_fact_authorized`
        treats an empty/absent list as "no role restriction" either way
        but a few call sites elsewhere in this codebase rely on `None`
        specifically meaning "field absent."
    """
    from rdflib import URIRef

    rows = cast(
        "list[ResultRow]",
        list(store.query(_PROPERTIES_QUERY, init_bindings={"entity": URIRef(entity_iri)})),
    )
    tenant_id: str | None = None
    trust_level: str | None = None
    source_document_id: str | None = None
    roles: list[str] = []
    for row in rows:
        if row.tenantId is not None:
            tenant_id = str(row.tenantId)
        if row.trustLevel is not None:
            trust_level = str(row.trustLevel)
        if row.sourceDocumentId is not None:
            source_document_id = str(row.sourceDocumentId)
        if row.role is not None:
            roles.append(str(row.role))

    return {
        "tenant_id": tenant_id,
        "allowed_roles": roles or None,
        "trust_level": trust_level,
        "source_document_id": source_document_id,
    }


def is_entity_authorized(
    properties: dict[str, object],
    auth: AuthorizationContext | None,
    *,
    cross_tenant_support_roles: list[str] | None = None,
) -> bool:
    """Decide whether `auth` may see an entity with these properties.

    A thin, documented pass-through to
    `rag.graph.authorization.is_fact_authorized`, kept as a separate
    function in this module only so call sites read "ontology
    authorization," and so this is the one place a future, genuinely
    ontology-specific rule would be added, not because the underlying
    check is reimplemented here.

    Parameters
    ----------
    properties : dict[str, object]
        As returned by `fetch_entity_auth_properties`.
    auth : AuthorizationContext | None
        `None` means unrestricted (always authorized). This is the caller
        asserting no identity at all, used only for unauthenticated/
        internal demos, never a real caller's default.
    cross_tenant_support_roles : list[str] | None, optional
        Server policy (`config.security.authorization.cross_tenant_support_roles`);
        defaults to `[]` (no cross-tenant support role at all) when
        omitted.

    Returns
    -------
    bool
        `True` if `auth` may see this entity.
    """
    return is_fact_authorized(
        properties,
        auth,
        cross_tenant_support_roles=cross_tenant_support_roles or [],
    )


def filter_authorized_entities(
    store: RdflibStore,
    entity_iris: list[str],
    auth: AuthorizationContext | None,
    *,
    cross_tenant_support_roles: list[str] | None = None,
) -> list[str]:
    """Apply `is_entity_authorized` to a list of candidate entity IRIs.

    Parameters
    ----------
    store : RdflibStore
        The graph to read tenant/role properties from.
    entity_iris : list[str]
        Candidate entity IRIs, e.g. from a prior SPARQL query's results.
    auth : AuthorizationContext | None
        The caller's resolved context.
    cross_tenant_support_roles : list[str] | None, optional
        Forwarded to `is_entity_authorized`.

    Returns
    -------
    list[str]
        The subset of `entity_iris` that `auth` may see, in the same
        order they were given.
    """
    return [
        iri
        for iri in entity_iris
        if is_entity_authorized(
            fetch_entity_auth_properties(store, iri),
            auth,
            cross_tenant_support_roles=cross_tenant_support_roles,
        )
    ]
