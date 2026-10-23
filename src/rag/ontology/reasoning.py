"""RDFS and OWL-RL inference over an `RdflibStore` (docs/ontology.md sections 4, 5, 14).

Wraps `owlrl`'s forward-chaining closure algorithms. `materialize_rdfs`
and `materialize_owl_rl` both mutate the graph they are given IN PLACE:
this is how `owlrl` works, and how a real RDFS/OWL reasoner's
"materialization" strategy generally works (inferred triples are added
directly into the graph, rather than kept in a separate side table
re-derived at query time). Every caller that wants to compare "before"
and "after" inference takes an `RdflibStore.copy()` first
(`store.py`'s `copy()` docstring explains why), so it is always
possible to show, concretely, which triples were explicitly asserted
and which were only ever inferred, the distinction docs/ontology.md
section 4 asks to be taught, not just stated.

Open-world assumption (docs/ontology.md section 14): none of the functions below
ever conclude that a fact is FALSE because it is absent. `ask_boolean`
only ever answers "is this triple pattern entailed", never "is this
triple pattern false"; the two are not the same question under
RDF/OWL semantics, unlike a closed-world database query, where an
absent row conventionally does mean "false". This is why
`find_disjoint_class_violations` below is phrased as "report every
individual asserted under two classes declared disjoint", a positive
check for a stated contradiction, rather than "prove the graph is
consistent" (which would require a real OWL DL reasoner capable of a
full consistency proof, out of scope here; see the module docstring
in data/ontology/instances/reasoning_fixtures.ttl).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from owlrl import DeductiveClosure, OWLRL_Semantics, RDFS_Semantics
from rdflib.query import ResultRow

from rag.ontology.store import RdflibStore

DISJOINT_VIOLATION_QUERY = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT DISTINCT ?individual ?classA ?classB WHERE {
    ?classA owl:disjointWith ?classB .
    ?individual a ?classA .
    ?individual a ?classB .
}
"""


@dataclass(frozen=True)
class InferenceReport:
    """Triple counts before and after a materialization pass.

    Attributes
    ----------
    explicit_triple_count : int
        Number of triples present before this reasoner ran.
    total_triple_count : int
        Number of triples present after this reasoner ran (always
        `>= explicit_triple_count`; a reasoner only ever adds triples).
    """

    explicit_triple_count: int
    total_triple_count: int

    @property
    def inferred_triple_count(self) -> int:
        """Return how many NEW triples this materialization pass added."""
        return self.total_triple_count - self.explicit_triple_count


def materialize_rdfs(store: RdflibStore) -> InferenceReport:
    """Run RDFS closure over `store.graph`, in place.

    Entails `rdfs:subClassOf` transitivity, `rdfs:domain`/`rdfs:range`
    type propagation, and `rdfs:subPropertyOf`, per the RDFS semantics.

    Parameters
    ----------
    store : RdflibStore
        Mutated in place. Call `store.copy()` first if the pre-inference
        graph is still needed.

    Returns
    -------
    InferenceReport
        Triple counts before and after.
    """
    explicit_count = len(store)
    DeductiveClosure(RDFS_Semantics).expand(store.graph)
    return InferenceReport(explicit_triple_count=explicit_count, total_triple_count=len(store))


def materialize_owl_rl(store: RdflibStore) -> InferenceReport:
    """Run the OWL-RL profile's closure over `store.graph`, in place.

    A superset of RDFS closure that additionally entails
    `owl:equivalentClass`/`owl:equivalentProperty` membership,
    `owl:inverseOf`, `owl:SymmetricProperty`,
    `owl:TransitiveProperty`, and `owl:FunctionalProperty`'s
    forced-`owl:sameAs` rule, among others.

    Parameters
    ----------
    store : RdflibStore
        Mutated in place. Call `store.copy()` first if the pre-inference
        graph is still needed.

    Returns
    -------
    InferenceReport
        Triple counts before and after.
    """
    explicit_count = len(store)
    DeductiveClosure(OWLRL_Semantics).expand(store.graph)
    return InferenceReport(explicit_triple_count=explicit_count, total_triple_count=len(store))


def inferred_only_triples(before: RdflibStore, after: RdflibStore) -> set[tuple]:
    """Return every triple present in `after` but not in `before`.

    Parameters
    ----------
    before : RdflibStore
        A store captured (via `.copy()`) before a materialization pass.
    after : RdflibStore
        The same store, after `materialize_rdfs`/`materialize_owl_rl` ran.

    Returns
    -------
    set[tuple]
        The set of newly-inferred `(subject, predicate, object)` triples.
    """
    before_triples = set(before.graph)
    after_triples = set(after.graph)
    return after_triples - before_triples


def find_disjoint_class_violations(store: RdflibStore) -> list[tuple[str, str, str]]:
    """Report every individual typed under two classes declared `owl:disjointWith`.

    A lightweight, bounded SPARQL-ASK-shaped consistency check (see
    `data/ontology/instances/reasoning_fixtures.ttl`'s module docstring for why this,
    rather than a full OWL DL reasoner's inconsistency proof, is the
    practical mechanism used here).

    Parameters
    ----------
    store : RdflibStore
        Should already include both the ontology (for the
        `owl:disjointWith` axioms) and the instance data (for the
        `rdf:type` assertions) being checked.

    Returns
    -------
    list[tuple[str, str, str]]
        `(individual, classA, classB)` for every violation found.
    """
    return [
        (str(row.individual), str(row.classA), str(row.classB))
        for row in cast("list[ResultRow]", list(store.query(DISJOINT_VIOLATION_QUERY)))
    ]
