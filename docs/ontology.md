# Ontology and Semantic Layer

An enterprise-style ontology and semantic layer (`rag.ontology`) built on top
of the Knowledge Graph / GraphRAG milestone's Neo4j property graph
(`src/rag/graph/`, `docs/graphrag.md`). Learning goal: understand and
demonstrate, concretely and against the same real data, the difference
between raw graph data, a knowledge graph, an ontology, a semantic model,
RDF, RDFS, OWL, SHACL, SPARQL, and property graphs/Cypher.

This is **additive**, like GraphRAG itself: the Neo4j/property-graph
implementation is untouched and remains the production retrieval path
(`POST /graph/query`, the agent's four graph tools). `rag.ontology` is
reachable today only through `scripts/ontology_demo.py` and
`python -m rag.eval.run_ontology_eval`, not wired into `rag.api`/`rag.agent`.

## Switches

None yet. `rag.ontology` has no `config.ontology.enabled` flag, because
nothing in the serving path calls it; unlike `graph.enabled`, which gates a
real code path inside `RetrievalPipeline`/the agent tool list, this milestone
is a standalone CLI/eval layer. Install the optional dependency group first:
`pip install -e .[ontology]` (`rdflib`, `owlrl`, `pyshacl`).

## Architecture

```
                data/ontology/schema/techfusion_ontology_v1*.ttl   (B: schema knowledge)
                                    |
                        rdfs:subClassOf / owl:* axioms
                                    |
     data/ontology/instances/instances_graph_demo.ttl --- mirrors --- Neo4j techfusion_graph_demo
     data/ontology/instances/instances_customer_domain.ttl   (A: instance data, same facts, two models)
     data/ontology/instances/interoperability_mappings.ttl
     data/ontology/instances/canonical_entity_resolution.ttl
                                    |
                      rag.ontology.store.RdflibStore
                                    |
        +---------------+----------+----------+------------------+
        |               |                     |                  |
  reasoning.py    validation.py       query_expansion.py   authorization.py
  (RDFS/OWL-RL)   (pyshacl/SHACL)     (bounded subclass     (reuses rag.graph.
                                       expansion)            authorization.is_fact_authorized)
        |               |                     |                  |
        +---------------+----------+----------+------------------+
                                    |
                      graphrag_integration.py
            (naive_type_match vs ontology_aware_type_match,
             the same comparison rag.eval.run_ontology_eval scores;
             expand_and_query_live_neo4j for a real Neo4j connection)
```

## Directory map

```
src/rag/ontology/
  __init__.py
  paths.py               resolves data/ontology/* paths from rag.config.REPO_ROOT
  store.py                backend-agnostic RDF store abstraction
  reasoning.py             RDFS/OWL-RL inference (owlrl)
  validation.py            SHACL validation (pyshacl), raw vs RDFS-closed
  query_expansion.py       bounded ontology-aware concept expansion
  authorization.py         reuses rag.graph.authorization.is_fact_authorized +
                           rag.retrieval.authorization.AuthorizationContext directly
  interoperability.py      class-mapping + entity-resolution query helpers
  graphrag_integration.py naive vs ontology-aware retrieval, + a real (needs
                           a running Neo4j instance, see "Known limitations")
                           live-Neo4j integration function
  sparql_runner.py         load/run/format the .rq files
src/rag/eval/run_ontology_eval.py   deterministic eval harness (python -m rag.eval.run_ontology_eval)
data/ontology/
  schema/techfusion_ontology_v1.ttl, techfusion_ontology_v1_1.ttl
  shapes/techfusion_shapes.ttl
  instances/instances_graph_demo.ttl        RDF mirror of the real Neo4j dataset
  instances/instances_customer_domain.ttl   synthetic Customer/CustomerCase/Incident/Environment
  instances/instances_invalid.ttl           deliberately broken, SHACL demo only
  instances/reasoning_fixtures.ttl          deliberately constructed, OWL-RL demo only
  instances/interoperability_mappings.ttl   external-system class/terminology mapping
  instances/canonical_entity_resolution.ttl instance-level entity resolution
  queries/sparql/*.rq      8 representative SPARQL queries
  queries/cypher/*.cypher + NOTE.txt   5 Cypher counterparts, plus where none exists
  CHANGELOG.md             ontology versioning record
data/eval/ontology_gold.jsonl   12-question deterministic eval set
scripts/ontology_demo.py        one consolidated, --demo-selectable CLI (see "Exact run commands")
tests/unit/test_ontology_*.py   40 pytest tests covering every module above
```

## 1. Understand the current graph: instance data (A) vs. schema knowledge (B)

`src/rag/graph/schema.py` defines 17 node labels (`EntityType`) and 13
relationship types (`RelationshipType`) as flat `Literal` sets; there is no
Cypher statement, constraint, or schema object anywhere in that codebase that
declares one label a kind of another. Every node and relationship also
carries `dataset_id`/`tenant_id`/`allowed_roles` for scoping
(`GraphProvenance`).

- **A. Instance data**, e.g. `Document Service DEPENDS_ON PostgreSQL`,
  written by `scripts/seed_graph_demo_data.py`. Mirrored byte-for-byte (same
  entities, same relationships) in
  `data/ontology/instances/instances_graph_demo.ttl`, so every Cypher/SPARQL
  comparison here runs over the identical facts.
- **B. Semantic/schema knowledge**, e.g. "`DatabaseService` is a subclass
  of `Service`", "`dependsOn` connects a `Service` to a `Service`",
  modeled in `data/ontology/schema/techfusion_ontology_v1.ttl`, which this
  milestone adds. Every class carries a `tech:neo4jLabel` annotation naming
  the exact `EntityType` it corresponds to (cross-checked against
  `rag.graph.schema.ENTITY_TYPES` by
  `tests/unit/test_ontology_store_and_loading.py`), so the mapping is
  explicit, not implied, and stays correct if `EntityType` ever changes.

## 2. The enterprise ontology (class hierarchy + relationships)

```
tech:EnterpriseEntity
  tech:Service
    tech:APIService              (<- Neo4j label "API")
    tech:DatabaseService         (<- Neo4j label "Database")
    tech:MessagingService        (<- semantic refinement; Neo4j has no such label; see note below)
  tech:InfrastructureComponent
    tech:Technology              (<- Neo4j label "Technology")
  tech:Application
  tech:Team
  tech:OperationalDocumentation  (new umbrella, section 12's own example)
    tech:Policy
    tech:Runbook
  tech:Incident
  tech:Customer
  tech:CustomerCase
  tech:Environment
  tech:Document
  tech:Category
  tech:Tenant
  tech:Role
```

(v1.1.0 adds, non-breaking: `tech:System` deprecated -> equivalent to
`tech:InfrastructureComponent`; `tech:Vendor` new. See `data/ontology/CHANGELOG.md`.)

**A semantic refinement the property graph does not have**: Kafka is typed
`"Technology"` (a flat label) in the real Neo4j graph, but this ontology
additionally types it `tech:MessagingService` in
`data/ontology/instances/instances_graph_demo.ttl`; the same real-world
entity, two models, one more precise than the other.

Relationships (object properties): `dependsOn`, `ownedBy`,
`governedBy`/`appliesTo` (mutual inverses), `hasRunbook`/`runbookFor`,
`affects`/`isAffectedBy`, `runsOn`/`hosts`, `connectsTo` (symmetric),
`references`, `supersedes`, `partOf` (the one `owl:TransitiveProperty` in
this ontology), `relatedTo` (symmetric), `raisedBy`/`hasCase`,
`candidateSameAs`. Each one's comment in
`data/ontology/schema/techfusion_ontology_v1.ttl` explains its domain/range
and transitivity choice; see sections 4/5/14 below for why those choices
were made deliberately, not by default.

## 3. RDF fundamentals

```turtle
inst:service-rag-service
    a tech:Service ;
    tech:name "RAG Service" ;
    tech:dependsOn inst:database-postgresql ,
                   inst:database-opensearch ,
                   inst:service-document-service .
```

- **Subject** `inst:service-rag-service`, **predicate** `tech:dependsOn`,
  **object** `inst:database-postgresql`; one **triple** per
  subject-predicate-object combination above.
- **URI/IRI**: `inst:service-rag-service` expands to the full IRI
  `http://techfusion.example/graph/techfusion_graph_demo/service-rag-service`.
- **Literal**: `"RAG Service"` (a plain string value, not an IRI).
- **Namespace**/**prefix**: `tech:` abbreviates
  `http://techfusion.example/ontology/v1#`; `inst:` abbreviates
  `http://techfusion.example/graph/techfusion_graph_demo/`.

All instance data uses real entities from the techfusion_graph_demo dataset
(`data/ontology/instances/instances_graph_demo.ttl`) or clearly-labeled
synthetic additions (`data/ontology/instances/instances_customer_domain.ttl`).

## 4. RDFS: classes, subclasses, inference, and what "explicit vs. inferred" means

`tech:DatabaseService rdfs:subClassOf tech:Service` is the RDFS axiom. Given
`inst:database-postgresql a tech:DatabaseService` (explicit), an RDFS
reasoner entails `inst:database-postgresql a tech:Service` (inferred, never
written anywhere in the source file).

Run it: `python scripts/ontology_demo.py --demo rdfs`:

```
Explicitly asserted (tech:Service), before RDFS: False
True after RDFS closure (via tech:DatabaseService rdfs:subClassOf tech:Service): True
```

`rag.ontology.reasoning.materialize_rdfs` mutates a graph **in place** (a
real RDFS/OWL reasoner's "materialization" strategy: inferred triples are
added directly into the graph). `inferred_only_triples(before, after)` shows
the exact diff, after filtering out the RDFS-mandated but content-free `"X a
rdfs:Resource"` fact every term gets (own RDFS rule `rdfs4a`/`rdfs4b`,
true but not interesting for a demo).

A real caveat, verified empirically, not assumed: `sh:targetClass`/`sh:class`
in SHACL are **already** subclass-aware per the W3C Recommendation's own
built-in semantics, with no separate RDFS inference pass required; see
section 6.

## 5. OWL: functional properties, equivalence, disjointness

Kept small and selective, per this milestone's own instruction. Three uses,
each demonstrated live:

**`owl:FunctionalProperty` forcing `owl:sameAs`.** `tech:ownedBy` is
functional (every Service in this domain has exactly one owning Team).
`data/ontology/instances/reasoning_fixtures.ttl` asserts two different
owners for one fictitious service. Run
`python scripts/ontology_demo.py --demo owl`:

```
inst:team-billing-ops owl:sameAs -> ['...team-billing-ops', '...team-finance-platform']
```

A reasoner did not reject this (OWL has no closed-world rejection); it
concluded the two named teams must be the **same individual**, because a
functional property's object is logically unique. No Cypher cardinality
rule concludes two differently-IRI'd nodes denote the same entity; that
would need bespoke application code.

**`owl:equivalentClass` across heterogeneous systems**; the headline
answer to "demonstrate an inference difficult to express as simple
property-graph traversal" (section 10's worked example, below).

**`owl:disjointWith` as a lightweight consistency check.**
`tech:Service owl:disjointWith tech:Team` (among others).
`reasoning.find_disjoint_class_violations` runs a bounded SPARQL query
joining `owl:disjointWith` axioms against asserted+inferred `rdf:type` facts
(not a full OWL DL reasoner's inconsistency proof; see that function's
docstring for why this is still real OWL semantics, a lighter-weight
mechanism for checking it). Confirmed against
`data/ontology/instances/reasoning_fixtures.ttl`'s deliberately dual-typed
individual.

**OWL ontology vs. database schema**: a database schema (or Neo4j's own
label set) only ever *constrains* what can be written; it never *infers*
new facts from what is already there. An OWL ontology does both:
`tech:DatabaseService rdfs:subClassOf tech:Service` behaves like a schema
constraint in one direction (a `DatabaseService` individual must be
well-formed as a `Service` too) and an inference rule in the other (a
reasoner adds the `Service` type automatically). A schema cannot do the
latter.

## 6. SHACL validation: a real, caught bug, not a scripted demo

Run `python scripts/ontology_demo.py --demo shacl`. Two real findings, both
verified by running `pyshacl`, not assumed (see `KNOWN_ISSUES.md`):

1. **`sh:targetClass` is subclass-aware even with `inference=None`.**
   PostgreSQL and OpenSearch (typed only `tech:DatabaseService`, never
   `tech:Service` directly) are *already* flagged by `ServiceShape`'s owner
   requirement in raw validation. This is the W3C SHACL Recommendation's own
   built-in "SHACL instance" definition (`rdf:type`/`rdfs:subClassOf*`
   closure, baked into core SHACL semantics), not a separate inference
   option.
2. **A real bug, found via this exact validation run, fixed before the
   ontology was used anywhere else.** An earlier `tech:dependsOn` had a
   narrow `rdfs:domain tech:Service`. Because the real seed data also has an
   `Application` subject for `DEPENDS_ON` (Alpha Internal Console),
   `inference="rdfs"` mode silently re-typed that Application as a
   `Service`, changing which SHACL shapes applied to it; raw and
   RDFS-closed modes disagreed (6 violations each, different reasons).
   Fixed by removing the domain axiom and expressing the real constraint in
   SHACL instead (`DependsOnEndpointsShape`, `sh:or` over
   `Service`/`Application`). See `data/ontology/CHANGELOG.md` and
   `python scripts/ontology_demo.py --demo domain-range-pitfall` (a clean,
   isolated reproduction using a toy ontology).

After the fix, both modes agree:

```
A. Valid data, raw:         conforms=False  violations=5
A. Valid data, RDFS-closed: conforms=False  violations=5
```

Both report the *same* 5 entities (PostgreSQL, OpenSearch, Alpha Console
Database, RAG Service, Kafka) missing a `tech:ownedBy` triple. This is a
genuine data-quality finding in the demo graph, kept honestly rather than
patched to look clean.

```
B. Deliberately invalid data: conforms=False  violations=7
```

Covers every shape at least once: missing name/owner, two customers on one
case, an out-of-vocabulary status, a missing Document source/provenance, and
a non-Service/Application `dependsOn` subject.

**OWL reasoning vs. SHACL validation, the actual distinction, grounded in
the above**: OWL reasoning (section 4/5) **adds** facts a reasoner is
entitled to conclude; it never rejects anything (open-world, no such concept
as "invalid" in OWL/RDFS). SHACL validation **only checks** facts (inferred
or not) against shape constraints and *never itself adds a triple*;
`pyshacl`'s own `inference="rdfs"` parameter means "run an RDFS
reasoner first, then check," not a SHACL concept. A `tech:dependsOn` triple
with no domain/range axiom is still checked by `DependsOnEndpointsShape`;
the checking (SHACL) and the concluding (OWL/RDFS) are different
operations, and this ontology deliberately now uses each for the job it is
suited to.

## 7. SPARQL

8 queries in `data/ontology/queries/sparql/*.rq`, run live by
`python scripts/ontology_demo.py --demo sparql` (raw and after OWL-RL
closure, side by side). Covers
`SELECT`/`WHERE`/`FILTER`/`OPTIONAL`/`UNION`/property paths (`+`,
`*`)/`LIMIT`:

| # | Query | Demonstrates |
|---|---|---|
| 01 | services owned by Workflow | `SELECT`/`WHERE`/`ORDER BY` |
| 02 | RAG Service's direct dependencies | `OPTIONAL` |
| 03 | policies applying to a service | `UNION` (collapses post-inference; see file) |
| 04 | RAG Service's transitive dependencies | property path `+` |
| 05 | runbooks for incident-affected services | 3-hop join |
| 06 | subclasses of `tech:Service` | property path `*`, and the `owl:Nothing` caveat (below) |
| 07 | individuals typed `tech:Customer` via inference | `OPTIONAL`+`FILTER`, raw vs. reasoned |
| 08 | dependents of PostgreSQL via the inverse-only property | raw=0 rows, reasoned=2 rows |

**A found-while-building caveat, fixed in the query file itself**: running
query 06 after OWL-RL closure originally also returned `owl:Nothing` as a
"subclass" of `tech:Service`; correct OWL semantics (the empty class is
trivially a subclass of everything), not a bug, but not useful for this
demo, so the file now filters it with `FILTER(?subclass != owl:Nothing)` and
documents why.

## 8. RDF store

`rag.ontology.store.RdflibStore` wraps an in-process `rdflib.Graph` behind
the small `RDFStoreBackend` protocol (`load`, `query`, `__len__`,
`serialize`). Chosen over a real triple-store server per this milestone's
own instruction to use RDFLib first and stay lightweight; the same "no
infrastructure without a demonstrated requirement now" pattern the main
project already applies to Redis and MCP. A future `FusekiStore`/
`GraphDBStore` implementing the same protocol over SPARQL HTTP (rdflib's own
`SPARQLStore`, or `SPARQLWrapper`) would let every script/test here run
unmodified against a real server; not built, since no concrete learning need
for one was identified.

## 9. Property graph vs. RDF: the same facts, modeled twice

`python scripts/ontology_demo.py --demo property-graph-vs-rdf` prints each
Cypher query (`data/ontology/queries/cypher/*.cypher`, written for this
comparison, **not** executed against live Neo4j; no Neo4j instance was
running when this milestone was built) next to its SPARQL counterpart's
live result.

| Aspect | Property graph (Neo4j/Cypher) | RDF (triple store/SPARQL) |
|---|---|---|
| Graph model | Labeled property graph: nodes with labels+properties, directed typed edges with their own properties | Triples: subject-predicate-object; properties are triples too, with no node/edge-level property bag distinction |
| Relationship properties | First-class (an edge can carry `confidence`, `extracted_at`, ...) | Requires reification or a separate predicate per "edge property"; this ontology sidesteps it by putting that provenance on named individuals instead |
| Schema | Flat node labels, no declared subsumption (`Service`/`Database`/`API` are independent) | Explicit class hierarchy (`rdfs:subClassOf`), schema is itself queryable data |
| Inference | None built in; "Database is a Service" must be hardcoded into every query that needs it | RDFS/OWL entailment: subclass, equivalence, inverse, transitive; declared once, used everywhere |
| Constraints | Encoded in application code (this project's own `ALLOWED_FILTER_FIELDS`/schema.py allowlists) | SHACL: a separate, declarative, re-usable validation layer |
| Query language | Cypher: pattern-matching, very readable for path traversal, bounded `*1..N` variable-length | SPARQL: triple-pattern matching, `UNION`/`OPTIONAL`/property paths, a W3C standard |
| Interoperability | Proprietary per vendor (Cypher dialects vary) | A single W3C standard; `owl:equivalentClass` gives real cross-system interoperability (section 10) with no Cypher equivalent |
| Performance (this demo's scale) | Not measured here (no live Neo4j connection available) | Sub-15ms per query against an in-memory ~460-triple graph; not representative of production scale |
| Developer ergonomics | Very approachable for "find a path" questions | Steeper learning curve (prefixes, triples, reasoning modes) for the same questions |
| Enterprise standards | De facto per product (Neo4j, TigerGraph, ...) | W3C standard (RDF/RDFS/OWL/SPARQL/SHACL), common in data-integration/MDM/regulatory contexts (this is why it recurs in SAP/enterprise-data interviews) |

**Where there is no Cypher counterpart at all**
(`data/ontology/queries/cypher/06_query_subclasses_NOTE.txt`): queries 06-08
depend on `rdfs:subClassOf`/`owl:equivalentClass`/`owl:inverseOf`
entailment, none of which exist in Cypher/Neo4j. The closest equivalent is a
hardcoded label list a human updates by hand every time the modeled
hierarchy changes; this is the concrete answer to section 5's
"demonstrate an inference difficult to express as simple property-graph
traversal."

## 10. Semantic interoperability

`data/ontology/instances/interoperability_mappings.ttl` maps three synthetic
external systems' own class names into this ontology, as real, inspectable
`owl:equivalentClass` triples (never a silent rename):

| External system | Its own class | Maps to |
|---|---|---|
| SAP-shaped | `sap:Client` | `tech:Customer` |
| CRM-shaped | `crm:Customer` | `tech:Customer` |
| ServiceNow-shaped | `svcnow:BusinessPartner` | `tech:Customer` |
| SAP-shaped | `sap:ticket` | `tech:CustomerCase` |
| CRM-shaped | `crm:support_case` | `tech:CustomerCase` |
| ServiceNow-shaped | `svcnow:incident` | `tech:CustomerCase`, **not** `tech:Incident` |

That last row is deliberate: despite the shared English word "incident",
the operational sense (`tech:Incident`, an event affecting a Service) and
the customer-support sense (a case) are different concepts in this domain;
naive field-renaming would have merged them.

Run `python scripts/ontology_demo.py --demo interop`. A genuine,
non-obvious finding from running this live: once all three systems' classes
are pairwise `owl:equivalentClass` to `tech:Customer`, OWL-RL entails they
are *mutually* equivalent to each other too (equivalence is symmetric +
transitive), so any individual typed under *any one* of the four class
names is entailed to carry *all four* types. This is exactly what
"reasoning across heterogeneous systems" means concretely: a query for `a
tech:Customer` transparently matches records from all three external
systems, with zero per-system relabeling ETL step; the Cypher equivalent
would need a real, maintained relabeling pipeline.

## 11. Canonical entity model (entity resolution)

`data/ontology/instances/canonical_entity_resolution.ttl` resolves one
real-world customer ("Northwind Logistics") as seen by three source
systems, with three deliberately different outcomes; real entity
resolution never produces one blanket answer:

| External record | Outcome | Mechanism | Evidence |
|---|---|---|---|
| `sap:customer_4711` | **Confirmed** | `owl:sameAs` | Matching legal name + tax id, human-reviewed |
| `crm:account_abc123` | **Proposed, not merged** | `tech:candidateSameAs` (plain object property, not `owl:sameAs`) | Fuzzy name match, 0.72 confidence, unconfirmed |
| `svcnow:organization_xyz` | **Unresolved** | no link at all | Name is close but not exact; "(APAC)" suggests a possibly distinct subsidiary |

`tech:candidateSameAs` exists specifically so "do not merge entities
automatically without evidence" is enforced by construction: there is no
code path anywhere in this package that promotes a candidate link to
`owl:sameAs`. `tech:resolutionStatus`/`tech:resolvedBy`/`tech:sourceSystem`
carry the provenance a real review would need.

**A subtlety found while building the demo, not designed in advance**:
`svcnow:organization_xyz`, despite having no identity link at all, *still*
shows up typed `tech:Customer` after OWL-RL closure (section 10 above),
purely because its own asserted class (`svcnow:BusinessPartner`) is
schema-mapped to `tech:Customer`, independent of instance-level identity
resolution. "Type-compatible with Customer" and "IS this specific customer"
are different claims; conflating them would be exactly the unjustified
automatic merge this section warns against. See
`python scripts/ontology_demo.py --demo interop`'s printed output.

## 12. Ontology + GraphRAG integration

`rag.ontology.graphrag_integration.ontology_aware_type_match` expands a
queried class through the ontology (`query_expansion.expand_concept`,
bounded) before matching instance data, versus `naive_type_match` (exact
type only; what a property-graph query matching one label would give).
Both run over `data/ontology/instances/instances_graph_demo.ttl`, the RDF
mirror of the real Neo4j dataset.

```
naive:            6 entities (the 6 un-subclassed Service individuals)
ontology-aware:  10 entities (+ PostgreSQL, OpenSearch, Alpha Console Database, Kafka)
```

This is the literal "What infrastructure services depend on X" story from
this milestone's own prompt: a naive label match misses infrastructure
typed under a more specific subclass; expansion recovers it.
`tech:OperationalDocumentation` (new in this ontology, umbrella over
`Policy`/`Runbook`) is the other worked example this section's prompt names
("map Runbook/Policy/Procedure into a broader concept"); queryable the
same way via `query_expansion.expand_concept`.

`rag.ontology.graphrag_integration.expand_and_query_live_neo4j` does the
same expansion, then maps each expanded class to its real
`rag.graph.schema.EntityType` label (via `tech:neo4jLabel`) and runs one
bounded, parameterized Cypher query through the real
`rag.graph.client.Neo4jClient`. Importing the modules it needs
(`rag.graph.client`, `rag.graph.retrieval`) was measured directly at well
under 1 second each; this is NOT the expensive `rag.factory` import path
documented elsewhere in this codebase. The one reason this function has not
been executed end-to-end is simpler: it needs a running Neo4j instance,
and none was available (`docker ps` found no containers at all). Its
fail-closed behavior
(a clear `GraphUnavailableError`, confirmed by manually running it against
an unreachable port) is what's verified.

## 13. Ontology-aware query expansion

`rag.ontology.query_expansion.expand_concept(store, class_iri, max_depth=2,
max_results=20)` walks `rdfs:subClassOf` breadth-first, bounded by **both**
a hop-depth cap and a result-count cap independently (never through the
whole ontology). Verified against a synthetic 5-level chain in
`tests/unit/test_ontology_query_expansion.py`: `max_depth=2` stops after 2
levels regardless of how many more exist; `max_results=2` stops after 2
results regardless of depth remaining.

Measured recall improvement (from `python -m rag.eval.run_ontology_eval`,
full numbers in section 18): **naive mean recall 0.867 -> ontology-aware
mean recall 1.0** across the 3 type-expansion gold questions, with the
improvement landing *exactly* on the one question designed to need it
(+0.4 recall) and *zero* change on the two questions designed not to need it
-- confirming expansion helps selectively, not universally.

## 14. Reasoning: subclass, inverse, transitive, and the open-world assumption

- **Subclass inference**: section 4.
- **Inverse inference**: `tech:isDependencyOf` is never asserted anywhere in
  instance data, only `tech:dependsOn` is. Query 08 (section 7) returns 0
  rows on the raw graph and the correct 2 dependents only after
  `owl:inverseOf` entailment runs.
- **Transitive inference, used deliberately once, withheld deliberately
  twice**: `tech:partOf` is the one `owl:TransitiveProperty` in this
  ontology; textbook-valid (a part of a part is a part of the whole).
  `tech:dependsOn` and `tech:supersedes` are both deliberately **not**
  transitive, even though each looks like it should be at first glance:
  - `rag.graph.queries` already does *bounded-depth* Cypher traversal
    (`*1..N`) for multi-hop dependency questions, not unbounded transitive
    inference, because indirect dependency and direct dependency are not
    the same claim, and an unbounded closure risks cycles and runaway
    result sets.
  - `rag.retrieval.freshness` already resolves "current version" via an
    explicit union-find over the `supersedes` chain in Python, not blind
    transitivity, because a query asking "what does C directly replace"
    must return the immediate predecessor, not every ancestor.

  "Do not declare a relationship transitive merely because it is
  convenient" (this milestone's own instruction) is enforced here by
  matching two real, pre-existing design decisions in the main codebase,
  not by arbitrary choice.

- **Open-world assumption**: none of this package's functions ever conclude
  a fact is *false* because it is absent. `find_disjoint_class_violations`
  reports a stated contradiction (two disjoint types on one individual),
  never "the graph is consistent" (which needs a full OWL DL reasoner, out
  of scope here). Absence of a fact (e.g. no `owl:sameAs` link for
  `svcnow:organization_xyz`) means "not established," never "confirmed not
  the same entity"; the entity-resolution demo (section 11) is the
  concrete illustration of this: unresolved is its own, distinct outcome,
  not a negative answer.

## 15. Governance / versioning

See `data/ontology/CHANGELOG.md` in full. Summary: the term namespace
(`.../ontology/v1#`) stays fixed across non-breaking releases;
`owl:versionIRI`/`owl:versionInfo` mark the document's own version. v1.1.0
deprecates `tech:System` (kept, `owl:equivalentClass`-bridged to
`tech:InfrastructureComponent`, never removed) and adds `tech:Vendor` (pure
addition). A breaking change (redefining an existing term) would require a
new major namespace; not done here, since nothing in this ontology's
evolution so far has needed it.

## 16. Provenance: ontology/schema vs. instance data

Kept structurally separate, never conflated:

- **Schema provenance** (the ontology itself): `owl:versionIRI`,
  `owl:versionInfo`, `dct:creator`, `dct:created`, and each class's own
  `rdfs:comment` as its definition; all in
  `data/ontology/schema/*.ttl`'s header and per-term annotations.
- **Instance provenance** (a fact): `tech:sourceDocumentId`,
  `tech:extractionMethod`, `tech:confidence`, `tech:extractedAt`,
  `tech:datasetId`, `tech:trustLevel`; mirroring
  `rag.graph.schema.GraphProvenance`'s fields exactly, field for field, on
  every individual in `data/ontology/instances/instances_graph_demo.ttl`.

## 17. Security: ontology reasoning never bypasses authorization

`rag.ontology.authorization` is a thin pass-through to
`rag.graph.authorization.is_fact_authorized`, reusing it (and
`rag.retrieval.authorization.AuthorizationContext`) directly rather than
re-implementing the rule; the exact same function, not a parallel,
independently maintained copy of its logic, governs both the property graph
and this semantic layer. `python scripts/ontology_demo.py --demo auth` and
`tests/unit/test_ontology_authorization.py` prove the actual security
property this section asks for: **inference runs first, authorization is
checked second, on every test**:

```
Alpha Console Database (tenant_alpha, role tenant_alpha_operator):
  no identity:                      visible=True   (unrestricted, by design)
  tenant_alpha + matching role:      visible=True
  tenant_beta:                       visible=False
  tenant_beta + a role LISTED on the entity (cross-tenant support): visible=True
```

run with OWL-RL closure already materialized beforehand; an inferred fact
about an entity (e.g. its entailed `tech:Service` membership) changes
nothing about who may see the entity. A semantic relationship never grants
access to the instance data on its own. An unscoped entity (no
`tech:tenantId` triple, e.g. RAG Service) stays visible to every tenant,
inference or not; matching `rag.graph.authorization`'s own "`tenant_id IS
NULL` means unrestricted" rule exactly, because it is the same rule.

## 18. Evaluation: ordinary vs. ontology-aware GraphRAG

`python -m rag.eval.run_ontology_eval` against `data/eval/ontology_gold.jsonl`
(12 deterministic questions: subclass expansion x2, graph traversal x5,
terminology mapping x2, entity resolution x2, one deliberately
unanswerable). No LLM judge, no RAGAS; every expected answer is exact and
deterministic, matching `rag.eval.run_graph_eval`'s own precedent for the
GraphRAG milestone's small gold set.

```
question_count: 12
answer_correctness: 1.0          (12/12 exact-match correct)
retrieval_recall:
  naive:           0.867
  ontology_aware:  1.0
ontology_awareness_impact:
  rows_where_expansion_should_help: 1          (recall delta: +0.4)
  rows_where_expansion_should_not_change_result: 2  (recall delta: 0.0)
graph_path_correctness: 1.0
sparql_recall: 1.0
semantic_mapping_accuracy: 1.0   (both class-mapping and entity-resolution questions)
```

**Do not assume ontology always improves results (this milestone's own
instruction), honored concretely**: `q02` (list every `DatabaseService`) is
a designed negative control; `DatabaseService` has no subclasses of its
own, so naive and ontology-aware retrieval return *identical* results. The
aggregate recall improvement above is driven entirely by `q01` (the
`Service` expansion), not by every question uniformly improving.

**Latency**: reported in the raw JSON report but not treated as a
meaningful signal here; at this scale (an in-memory, ~460-triple graph),
per-query latency is dominated by `rdflib`'s own cold-start/parse overhead,
not by any real algorithmic cost difference between naive and
ontology-aware matching (mirrors this project's own documented finding that
host-level noise can swamp a real signal at small scale, e.g. the
context/token-budget A/B/C experiment in `CLAUDE.md`). Not a claim that
ontology-aware retrieval is faster in general.

## Exact run commands

```bash
# one-time setup
pip install -e ".[ontology]"

# run every demo in one teaching-ordered pass
python scripts/ontology_demo.py --demo all

# or individually
python scripts/ontology_demo.py --demo rdfs
python scripts/ontology_demo.py --demo domain-range-pitfall
python scripts/ontology_demo.py --demo owl
python scripts/ontology_demo.py --demo shacl
python scripts/ontology_demo.py --demo sparql
python scripts/ontology_demo.py --demo property-graph-vs-rdf
python scripts/ontology_demo.py --demo interop
python scripts/ontology_demo.py --demo auth

# deterministic evaluation (ordinary vs. ontology-aware GraphRAG)
python -m rag.eval.run_ontology_eval --output /tmp/ontology_eval_report.json

# tests (40 tests, no external services needed; part of the normal tests/unit run too)
pytest tests/unit/test_ontology_*.py -v
```

## Deliverables checklist

- [x] Ontology architecture diagram: "Architecture" above- [x] Ontology class hierarchy: section 2- [x] Relationship/property model: section 2, full detail in `data/ontology/schema/techfusion_ontology_v1.ttl`- [x] Turtle ontology file: `data/ontology/schema/techfusion_ontology_v1.ttl` (+ `techfusion_ontology_v1_1.ttl`)- [x] SHACL shapes: `data/ontology/shapes/techfusion_shapes.ttl`- [x] Representative RDF instance data: `data/ontology/instances/instances_graph_demo.ttl`, `instances_customer_domain.ttl`- [x] SPARQL queries: `data/ontology/queries/sparql/*.rq` (8)- [x] Corresponding Cypher queries where useful: `data/ontology/queries/cypher/*.cypher` (5) + the "no counterpart" note- [x] Valid and invalid SHACL examples: section 6- [x] Reasoning examples: sections 4, 5, 14- [x] GraphRAG integration example: section 12- [x] Ontology-aware retrieval evaluation: section 18- [x] RDF vs. property-graph comparison: section 9- [x] Exact run commands: above- [x] 25 interview questions with concise answers: below
## 25 interview questions

1. **What is an ontology?** A formal, explicit specification of a shared
   conceptualization: the classes, properties, and axioms describing a
   domain, written so both humans and machines can reason over it, not only
   store it.
2. **Ontology vs. taxonomy?** A taxonomy is a hierarchy (is-a links only,
   usually one tree). An ontology includes a taxonomy but also defines
   non-hierarchical relationships, properties, domains/ranges, and axioms
   (equivalence, disjointness, cardinality) that let you infer new facts,
   not only browse a tree.
3. **Ontology vs. Knowledge Graph?** A knowledge graph is instance data
   (entities and their relationships: "Document Service DEPENDS_ON
   PostgreSQL"). An ontology is the schema/semantic layer describing what
   kinds of entities and relationships are possible and what they mean. A KG
   can exist with no ontology (raw graph data); an ontology with no
   instances is a schema.
4. **Ontology vs. database schema?** A schema constrains what can be stored
   and is usually closed-world (absence = false). An ontology also *infers*
   new facts from what is stored (open-world: absence = unknown) and is
   itself queryable data, not a separate DDL artifact.
5. **What is RDF?** Resource Description Framework: a data model expressing
   all facts as subject-predicate-object triples, where subjects/predicates
   are (usually) IRIs and objects are IRIs or literals.
6. **What is a triple?** One atomic fact: subject, predicate, object, e.g.
   `(RAGService, dependsOn, PostgreSQL)`.
7. **URI vs. IRI?** A URI is an ASCII-restricted identifier; an IRI
   (Internationalized Resource Identifier) is the same idea generalized to
   Unicode. RDF uses IRIs; in practice most tooling treats "URI" and "IRI"
   interchangeably in casual speech.
8. **What is RDFS?** RDF Schema: a light vocabulary layered on RDF for
   declaring classes, subclass/subproperty relationships, and domain/range,
   enough for basic taxonomic inference.
9. **What is OWL?** Web Ontology Language: a much richer vocabulary on top
   of RDFS, adding equivalence, disjointness, cardinality, property
   characteristics (functional, inverse, symmetric, transitive), and (in
   its fuller profiles) full description-logic reasoning.
10. **What is SHACL?** Shapes Constraint Language: a W3C standard for
    validating RDF graphs against shapes (cardinality, datatype, value
    ranges, enumerations). It checks; it never infers or adds triples.
11. **What is SPARQL?** The W3C standard query language for RDF:
    `SELECT`/`ASK`/`CONSTRUCT`/`DESCRIBE` over triple patterns, with
    `FILTER`, `OPTIONAL`, `UNION`, and property paths.
12. **What is an RDF triple store?** A database purpose-built to store and
    query triples (natively or via SPARQL), e.g. GraphDB, Fuseki,
    Blazegraph; or, for this milestone, `rdflib`'s in-memory graph.
13. **What is a property graph?** A graph model where nodes carry
    labels+properties and edges carry types+properties directly (no
    reification needed for edge attributes). Neo4j's model.
14. **Cypher vs. SPARQL?** Cypher is pattern-matching over a labeled
    property graph, optimized for path traversal readability. SPARQL is
    triple-pattern matching over RDF, standardized, and uniquely able to
    query the schema (T-box) the same way it queries data (A-box).
15. **Domain and range?** `rdfs:domain P C` entails that anything used as
    P's subject is a C; `rdfs:range P C` does the same for P's object. They
    are **inference triggers**, not validation constraints; see section
    6's real, caught bug.
16. **Subclass inference, worked example?** `DatabaseService
    rdfs:subClassOf Service` + `PostgreSQL a DatabaseService` entails
    `PostgreSQL a Service`, without that triple ever being written.
17. **Open-world assumption?** The absence of a fact does not mean it is
    false, only that it is not (yet) known/asserted; contrast with a
    typical closed-world database, where a missing row conventionally
    means "no." This is why entity resolution here has three outcomes
    (confirmed / proposed / unresolved), not two.
18. **Semantic interoperability?** The ability for independently-built
    systems using different terminology for the same concepts to be queried
    uniformly, via explicit mappings (`owl:equivalentClass`) rather than ad
    hoc renaming (section 10).
19. **Entity resolution?** Determining whether records from different
    systems refer to the same real-world entity, with explicit
    confidence/provenance, never an automatic merge without evidence
    (section 11).
20. **Ontology-aware RAG?** Retrieval that reasons over an ontology's class
    hierarchy/equivalences before matching, so a query for a broad concept
    (`Service`) also retrieves instances only typed under a narrower one
    (`DatabaseService`), improving recall on exactly the questions where
    naive exact-type matching would miss relevant results (section 12/18).
21. **Reasoning vs. validation?** Reasoning (RDFS/OWL) *adds* facts a
    reasoner is logically entitled to conclude; it never rejects anything.
    Validation (SHACL) *checks* facts (inferred or not) against constraints
    and never adds a triple. Conflating the two is a common, real mistake;
    see section 6.
22. **Knowledge graph vs. semantic layer?** A knowledge graph is the data; a
    semantic layer (ontology + mappings + reasoning) is what makes that
    data *interoperable and inferable* across systems/teams, independent of
    how any one system happens to store it.
23. **Why is `tech:dependsOn` not `owl:TransitiveProperty`?** Because the
    real codebase already handles multi-hop dependency via bounded Cypher
    traversal (`*1..N`), for a documented reason (indirect != direct
    dependency, and unbounded closure risks cycles/runaway results); this
    ontology matches that existing, deliberate design rather than
    introducing a second, conflicting notion of transitivity.
24. **Why does `owl:FunctionalProperty` entail `owl:sameAs` instead of
    raising an error?** Because OWL has no concept of rejecting data; a
    functional property's semantics (its object is unique) are satisfied by
    concluding the two asserted objects are identical. Whether that is
    desirable is a validation question (SHACL's job, via
    `sh:maxCount 1`), not a reasoning question.
25. **When would an enterprise pick RDF/OWL over a property graph, or vice
    versa?** RDF/OWL when cross-system interoperability, a standardized
    query language across vendors, or formal inference/consistency
    checking is the primary requirement (data integration hubs,
    regulatory/MDM contexts, SAP-style enterprise semantic layers). A
    property graph when the primary requirement is fast, readable path
    traversal over one system's own data with first-class edge properties,
    and a small, fixed set of relationship types a team controls directly;
    this project's own production GraphRAG path is exactly that second
    case, which is why this ontology sits *beside* it rather than replacing
    it.

## Known limitations

- `rag.ontology.graphrag_integration.expand_and_query_live_neo4j` is
  implemented for real (not a stub) but has not been executed against a
  live Neo4j instance, since none was available. Its fail-closed
  behavior against an unreachable port was confirmed manually.
- `data/ontology/queries/cypher/*.cypher` were written for the Cypher-vs-
  SPARQL comparison and were not executed against a live Neo4j instance
  either; only their SPARQL counterparts were run.
- No full OWL DL consistency reasoner is used anywhere (`owlrl`'s OWL-RL
  profile is a sound-but-incomplete subset, chosen deliberately for being
  fast and dependency-light); disjointness is checked via a bounded SPARQL
  query instead of a true inconsistency proof; see
  `reasoning.find_disjoint_class_violations`'s own docstring.
- No GraphDB/Fuseki (or any real SPARQL HTTP endpoint) was stood up;
  `RdflibStore` is the only backend implemented, per this milestone's own
  instruction to prefer RDFLib first and add a server only with a
  demonstrated learning need (none was identified).
