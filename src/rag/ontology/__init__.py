"""Ontology / semantic-layer milestone: RDF/RDFS/OWL/SHACL/SPARQL over the GraphRAG domain.

Additive, like the GraphRAG milestone it sits beside: `src/rag/graph/`
(Neo4j, property graph, production retrieval path) is untouched by
this package. `rag.ontology` is reachable today only through
`scripts/ontology_demo.py` and `rag.eval.run_ontology_eval`, not wired
into `rag.api`/`rag.agent`. See `docs/ontology.md` for the full
design and `CLAUDE.md`'s "Ontology and semantic layer" section for the
condensed milestone record.
"""

from __future__ import annotations
