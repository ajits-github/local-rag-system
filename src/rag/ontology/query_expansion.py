"""Bounded ontology-aware concept expansion (docs/ontology.md section 13).

`expand_concept` walks `rdfs:subClassOf` in the subclass direction
(querying "what points TO this class", i.e. `?sub rdfs:subClassOf
<root>`), breadth-first, so a query concept like `tech:Service` expands
to `{APIService, DatabaseService, MessagingService}` before retrieval
runs. Bounded by BOTH a hop-depth cap and a result-count cap, so
expansion never walks through an entire ontology; the two caps are
independent, so one alone is never relied on (a very wide but shallow
hierarchy is still capped by result count; a very deep but narrow one
is still capped by depth).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

from rdflib.query import ResultRow

from rag.ontology.store import RdflibStore

_SUBCLASSES_OF_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?sub WHERE {
    ?sub rdfs:subClassOf ?parent .
}
"""


@dataclass(frozen=True)
class ExpansionResult:
    """The outcome of one bounded concept expansion.

    Attributes
    ----------
    root_class : str
        The concept that was expanded, as given.
    expanded_classes : list[str]
        Every subclass found, in breadth-first discovery order. Does
        NOT include `root_class` itself.
    hit_depth_cap : bool
        `True` if the BFS stopped because `max_depth` was reached while
        candidates remained unexplored (as opposed to simply running
        out of subclasses before the cap).
    hit_result_cap : bool
        `True` if the BFS stopped because `max_results` was reached.
    """

    root_class: str
    expanded_classes: list[str]
    hit_depth_cap: bool
    hit_result_cap: bool


def expand_concept(
    store: RdflibStore,
    root_class_iri: str,
    *,
    max_depth: int = 2,
    max_results: int = 20,
) -> ExpansionResult:
    """Breadth-first expand `root_class_iri` to its subclasses, bounded.

    Parameters
    ----------
    store : RdflibStore
        Must hold the ontology's `rdfs:subClassOf` triples (the T-box);
        instance data is irrelevant to this function.
    root_class_iri : str
        The class to expand, as a full IRI.
    max_depth : int, optional
        Maximum number of subclass hops to follow, by default 2. Depth
        1 is "direct subclasses only".
    max_results : int, optional
        Maximum number of subclasses to return, by default 20,
        regardless of how many more the ontology actually has.

    Returns
    -------
    ExpansionResult
        The expanded class set plus which bound (if any) stopped it.
    """
    from rdflib import URIRef

    root = URIRef(root_class_iri)
    seen = {root}
    expanded: list[str] = []
    frontier: deque[tuple[object, int]] = deque([(root, 0)])
    hit_depth_cap = False
    hit_result_cap = False

    while frontier:
        current, depth = frontier.popleft()
        if depth >= max_depth:
            hit_depth_cap = hit_depth_cap or _has_any_subclass(store, current)
            continue

        for sub in _direct_subclasses(store, current):
            if sub in seen:
                continue
            seen.add(sub)
            expanded.append(str(sub))
            if len(expanded) >= max_results:
                hit_result_cap = True
                return ExpansionResult(
                    root_class=root_class_iri,
                    expanded_classes=expanded,
                    hit_depth_cap=hit_depth_cap,
                    hit_result_cap=hit_result_cap,
                )
            frontier.append((sub, depth + 1))

    return ExpansionResult(
        root_class=root_class_iri,
        expanded_classes=expanded,
        hit_depth_cap=hit_depth_cap,
        hit_result_cap=hit_result_cap,
    )


def _direct_subclasses(store: RdflibStore, parent) -> list:
    """Return every class with an `rdfs:subClassOf parent` triple."""
    rows = cast(
        "list[ResultRow]", list(store.query(_SUBCLASSES_OF_QUERY, init_bindings={"parent": parent}))
    )
    return [row.sub for row in rows]


def _has_any_subclass(store: RdflibStore, parent) -> bool:
    """Return whether `parent` has at least one direct subclass, for cap reporting."""
    return bool(_direct_subclasses(store, parent))
