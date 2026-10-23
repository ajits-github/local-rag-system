"""SHACL validation via `pyshacl` (docs/ontology.md section 6).

Run in two modes, because they can legitimately disagree, for a reason
verified directly against pyshacl in this milestone rather than
assumed: `sh:targetClass`/`sh:class` are subclass-aware per the W3C
SHACL Recommendation's own built-in "SHACL instance" definition (it
bakes in `rdf:type`/`rdfs:subClassOf*` closure as core SHACL semantics)
even with `inference=None`. PostgreSQL and OpenSearch, both typed only
`tech:DatabaseService` and never directly `tech:Service`, are already
flagged by `ServiceShape`'s owner requirement in raw (`inference=None`)
mode. Confirmed by running this exact module against this exact data,
not inferred from the spec text alone.

`inference=None` (raw): validates the data graph exactly as asserted,
with that subclass-aware target matching as the only "extra" reasoning
involved.

`inference="rdfs"`: pyshacl's own pre-pass additionally materializes
full RDFS axiomatic closure (`rdfs:domain`/`rdfs:range` type
propagation, `rdfs:subPropertyOf`, etc.) into a working copy of the
data graph before shapes are checked. This is strictly broader than,
and a different concept from, the subclass-aware target matching
above. See `scripts/ontology_demo.py --demo shacl` for a worked example where
this produced a real, caught side effect during this milestone: a
narrow `rdfs:domain tech:Service` once declared on `tech:dependsOn`
caused `inference="rdfs"` mode to re-type an Application individual as
`tech:Service` purely from that domain axiom (since it was also a
`dependsOn` subject in the real seed data), fixed by removing that
domain declaration and moving the real constraint into SHACL
(`data/ontology/shapes/techfusion_shapes.ttl`'s `DependsOnEndpointsShape`), which
validates without ever manufacturing a new triple.

This module never conflates OWL reasoning (which ADDS facts) with
SHACL validation (which only CHECKS facts, inferred or not, and never
itself adds a triple). `inference="rdfs"` here is pyshacl calling
into the same `owlrl` RDFS closure this package's own `reasoning.py`
module uses directly elsewhere, not a SHACL-specific inference concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pyshacl import validate
from rdflib import RDF
from rdflib.namespace import Namespace

from rag.ontology.store import RdflibStore

SH = Namespace("http://www.w3.org/ns/shacl#")

InferenceMode = Literal[None, "rdfs"]


@dataclass(frozen=True)
class ShaclReport:
    """One SHACL validation run's outcome.

    Attributes
    ----------
    conforms : bool
        `True` iff zero violations were found.
    violation_count : int
        Number of `sh:ValidationResult` nodes in the results graph.
    results_text : str
        pyshacl's own human-readable report text, for direct display.
    violation_summaries : list[str]
        One short line per violation: `"<focus node>: <message>"`,
        extracted from the results graph for compact printing (the CLI
        demos print these instead of the full `results_text` by
        default, which is useful but verbose).
    """

    conforms: bool
    violation_count: int
    results_text: str
    violation_summaries: list[str]


def run_shacl_validation(
    data_store: RdflibStore,
    shapes_store: RdflibStore,
    *,
    inference: InferenceMode = None,
) -> ShaclReport:
    """Validate `data_store` against `shapes_store`'s SHACL shapes.

    Parameters
    ----------
    data_store : RdflibStore
        The data graph to validate. Not mutated: pyshacl validates a
        copy internally regardless of `inference`.
    shapes_store : RdflibStore
        The SHACL shapes graph (e.g. `data/ontology/shapes/techfusion_shapes.ttl`).
    inference : {None, "rdfs"}, optional
        `None` (default) validates the data exactly as asserted.
        `"rdfs"` runs RDFS closure over a working copy of the data
        first. See the module docstring for why these can disagree.

    Returns
    -------
    ShaclReport
        The validation outcome.
    """
    conforms, results_graph, results_text = validate(
        data_store.graph,
        shacl_graph=shapes_store.graph,
        inference=inference,
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
    )

    violation_nodes = list(results_graph.subjects(RDF.type, SH.ValidationResult))
    summaries: list[str] = []
    for node in violation_nodes:
        focus_node = results_graph.value(node, SH.focusNode)
        message = results_graph.value(node, SH.resultMessage)
        summaries.append(f"{focus_node}: {message}")

    return ShaclReport(
        conforms=conforms,
        violation_count=len(violation_nodes),
        results_text=str(results_text),
        violation_summaries=summaries,
    )
