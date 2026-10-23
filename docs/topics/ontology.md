# Ontology and Semantic Layer

`rag.ontology` adds an RDF/RDFS/OWL/SHACL/SPARQL semantic layer over the same
TechFusion domain the [Knowledge Graph / GraphRAG](graphrag.md) milestone
already models in Neo4j. It is additive, like GraphRAG itself: nothing in
`src/rag/graph/` changes, and `rag.ontology` is not wired into `rag.api` or
`rag.agent`. Today it is reachable through `scripts/ontology_demo.py` and
`python -m rag.eval.run_ontology_eval`.

The point is the distinction GraphRAG's own flat Neo4j node labels do not
express: `tech:DatabaseService rdfs:subClassOf tech:Service` lets a query
for "every Service" also match something only ever typed `DatabaseService`,
a real recall improvement (naive 6/10 entities vs. ontology-aware 10/10
on a `Service` query, measured in `data/eval/ontology_gold.jsonl`), not a
hypothetical one.

Security is unchanged, not re-derived. `rag.ontology.authorization` reuses
`rag.graph.authorization.is_fact_authorized` and
`rag.retrieval.authorization.AuthorizationContext` directly, so ontology
reasoning (subclass/equivalence/inverse entailment) runs over facts already
subject to the same tenant/role check document-level retrieval uses, and an
inferred relationship never grants access on its own. See
[Security](security.md) for the control-by-control breakdown this reuses.

## What RDF adds that the property graph does not

Neo4j's node labels (`Service`, `Database`, `API`, ...) are flat; there is
no Cypher statement anywhere that declares one a kind of another. This
ontology adds that subsumption explicitly (`rdfs:subClassOf`), plus a
handful of selective OWL characteristics. `tech:ownedBy` is
`owl:FunctionalProperty`, `tech:governedBy`/`tech:appliesTo` are mutual
`owl:inverseOf`, `tech:partOf` is the one `owl:TransitiveProperty` (chosen
deliberately: `tech:dependsOn`/`tech:supersedes` stay non-transitive,
matching `rag.graph.queries`'s bounded-depth traversal and
`rag.retrieval.freshness`'s union-find resolution respectively, not a
default).

## SHACL found a real bug, not a demo bug

Validating with and without RDFS inference disagreed once. A narrow
`rdfs:domain tech:Service` on `tech:dependsOn` let RDFS closure silently
retype an `Application` individual as a `Service`, since the real seed data
also uses `Application` as a `DEPENDS_ON` subject. Fixed by removing the
domain axiom and expressing the real constraint in SHACL instead, which
validates without manufacturing a new triple. That is the concrete answer to
"OWL reasoning and SHACL validation are different concepts." Full story in
`ISSUES.md` and `data/ontology/CHANGELOG.md`.

## Semantic interoperability and entity resolution

Three synthetic external systems' own class names
(`sap:Client`/`crm:Customer`/`svcnow:BusinessPartner`) map via real
`owl:equivalentClass` triples into `tech:Customer`, never a silent
rename. Entity resolution for one canonical customer produces three
different, real outcomes (a confirmed `owl:sameAs` link, a proposed-only
`tech:candidateSameAs` link not auto-promoted, and one deliberately
unresolved record), because "do not merge entities automatically without
evidence" is enforced by there being no code path that promotes a candidate
to confirmed.

The full design (class hierarchy, every RDF/RDFS/OWL/SHACL/SPARQL
concept worked through against real data, the Cypher-vs-SPARQL comparison,
the deterministic evaluation, and 25 interview questions) is in
[`docs/ontology.md`](../ontology.md).

API reference: [Ontology](../reference/ontology.md).
