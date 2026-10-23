"""Semantic interoperability and entity-resolution helpers (docs/ontology.md sections 10, 11).

Two concerns, kept in two functions, because they operate on two
different kinds of mapping triple:

- `find_equivalent_external_classes`: schema-level terminology mapping
  (`owl:equivalentClass` between an external system's own class and a
  TechFusion ontology class), from
  `data/ontology/instances/interoperability_mappings.ttl`.
- `summarize_entity_resolution`: instance-level identity mapping
  (`owl:sameAs` for confirmed matches, `tech:candidateSameAs` for
  proposed-but-unconfirmed ones), from
  `data/ontology/instances/canonical_entity_resolution.ttl`. Deliberately reports
  `owl:sameAs` and `tech:candidateSameAs` links separately and never
  silently treats a candidate as confirmed: "do not merge entities
  automatically without evidence" (this milestone's own instruction)
  is enforced here by simply never having a code path that promotes one
  to the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from rdflib.query import ResultRow

from rag.ontology.store import RdflibStore

_EQUIVALENT_CLASSES_QUERY = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT ?externalClass ?canonicalClass WHERE {
    ?externalClass owl:equivalentClass ?canonicalClass .
}
"""

_CONFIRMED_SAME_AS_QUERY = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX tech: <http://techfusion.example/ontology/v1#>
SELECT ?canonical ?external ?sourceSystem ?externalId ?resolvedBy WHERE {
    ?canonical owl:sameAs ?external .
    OPTIONAL { ?external tech:sourceSystem ?sourceSystem }
    OPTIONAL { ?external tech:externalIdentifier ?externalId }
    OPTIONAL { ?external tech:resolvedBy ?resolvedBy }
}
"""

_CANDIDATE_SAME_AS_QUERY = """
PREFIX tech: <http://techfusion.example/ontology/v1#>
SELECT ?canonical ?external ?status ?resolvedBy ?sourceSystem WHERE {
    ?canonical tech:candidateSameAs ?external .
    OPTIONAL { ?external tech:resolutionStatus ?status }
    OPTIONAL { ?external tech:resolvedBy ?resolvedBy }
    OPTIONAL { ?external tech:sourceSystem ?sourceSystem }
}
"""


@dataclass(frozen=True)
class ClassMapping:
    """One external-system-class-to-canonical-class mapping.

    Attributes
    ----------
    external_class : str
        The external system's own class IRI (e.g. a SAP-shaped `Client` class).
    canonical_class : str
        The TechFusion ontology class it was mapped to.
    """

    external_class: str
    canonical_class: str


@dataclass(frozen=True)
class EntityResolutionLink:
    """One instance-level identity link, confirmed or proposed.

    Attributes
    ----------
    canonical_entity : str
        The canonical TechFusion individual's IRI.
    external_entity : str
        The external system's own record IRI.
    confirmed : bool
        `True` for an `owl:sameAs` link, `False` for a
        `tech:candidateSameAs` link not yet promoted.
    status : str | None
        `tech:resolutionStatus` on the candidate link ("proposed" or
        "confirmed"), `None` for a confirmed `owl:sameAs` link (which
        carries no such property; its confirmation IS the `owl:sameAs`
        triple itself).
    resolved_by : str | None
        Who or what proposed/confirmed this link.
    source_system : str | None
        Which external system this record came from.
    """

    canonical_entity: str
    external_entity: str
    confirmed: bool
    status: str | None
    resolved_by: str | None
    source_system: str | None


def find_equivalent_external_classes(store: RdflibStore) -> list[ClassMapping]:
    """List every `owl:equivalentClass` mapping from an external system's class.

    Parameters
    ----------
    store : RdflibStore
        Should include `data/ontology/instances/interoperability_mappings.ttl`.

    Returns
    -------
    list[ClassMapping]
        Every external-to-canonical class mapping found.
    """
    rows = cast("list[ResultRow]", list(store.query(_EQUIVALENT_CLASSES_QUERY)))
    return [
        ClassMapping(external_class=str(row.externalClass), canonical_class=str(row.canonicalClass))
        for row in rows
    ]


def summarize_entity_resolution(store: RdflibStore) -> list[EntityResolutionLink]:
    """List every confirmed (`owl:sameAs`) and proposed (`tech:candidateSameAs`) identity link.

    Parameters
    ----------
    store : RdflibStore
        Should include `data/ontology/instances/canonical_entity_resolution.ttl`.

    Returns
    -------
    list[EntityResolutionLink]
        Confirmed links first, then proposed ones, in query order.
    """
    confirmed_rows = cast("list[ResultRow]", list(store.query(_CONFIRMED_SAME_AS_QUERY)))
    confirmed = [
        EntityResolutionLink(
            canonical_entity=str(row.canonical),
            external_entity=str(row.external),
            confirmed=True,
            status=None,
            resolved_by=str(row.resolvedBy) if row.resolvedBy is not None else None,
            source_system=str(row.sourceSystem) if row.sourceSystem is not None else None,
        )
        for row in confirmed_rows
    ]
    proposed_rows = cast("list[ResultRow]", list(store.query(_CANDIDATE_SAME_AS_QUERY)))
    proposed = [
        EntityResolutionLink(
            canonical_entity=str(row.canonical),
            external_entity=str(row.external),
            confirmed=False,
            status=str(row.status) if row.status is not None else None,
            resolved_by=str(row.resolvedBy) if row.resolvedBy is not None else None,
            source_system=str(row.sourceSystem) if row.sourceSystem is not None else None,
        )
        for row in proposed_rows
    ]
    return confirmed + proposed
