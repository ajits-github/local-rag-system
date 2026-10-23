# TechFusion Enterprise Ontology changelog

Governance record for `ontology/*.ttl` (docs/ontology.md section 15). Namespace
and versioning rule followed here: the term namespace
(`http://techfusion.example/ontology/v1#`, prefix `tech:`) stays fixed
across every non-breaking release; only `owl:versionIRI`/
`owl:versionInfo` on the ontology document itself change. A breaking
change (redefining what an EXISTING term means) would require bumping
to a new namespace (`tech2:` at `.../ontology/v2#`), so that old
triples are never silently reinterpreted under new semantics. Nothing
in this changelog has done that yet.

## v1.1.0 (`ontology/techfusion_ontology_v1_1.ttl`), 2026-10-07

Non-breaking. Loads on top of v1.0.0 via `owl:imports` (informational;
this experiment's own loader (`store.py`) loads both files by path).

- **Deprecated** `tech:System` (`owl:deprecated true`), in favor of
  `tech:InfrastructureComponent`. Kept, not removed: any existing data
  typed `tech:System` keeps parsing and keeps classifying correctly,
  via `tech:System owl:equivalentClass tech:InfrastructureComponent`.
  `rag.graph.schema.EntityType` still lists `"System"` as a legal Neo4j
  label; this ontology continues to model it for exactly that reason.
- **Added** `tech:Vendor` (a third-party supplier). Pure addition, no
  interaction with any existing term.

## v1.0.0 (`ontology/techfusion_ontology_v1.ttl`), 2026-10-07

Initial release. 17 classes (one per `rag.graph.schema.EntityType`
label, plus `tech:EnterpriseEntity` as an implicit root and
`tech:OperationalDocumentation`/`tech:APIService`/`tech:DatabaseService`/
`tech:MessagingService` as semantic refinements the property graph does
not itself declare), 18 object properties (one per
`rag.graph.schema.RelationshipType`, plus `tech:raisedBy`/`tech:hasCase`
and `tech:candidateSameAs` for the entity-resolution layer), 11
datatype properties mirroring `GraphProvenance`'s fields plus a handful
of SHACL-facing attributes.

**In-session fix, same release, before this ontology was otherwise
used anywhere**: `tech:dependsOn` originally declared `rdfs:domain
tech:Service`. Running SHACL validation in both raw and RDFS-closed
modes against the real demo graph (which also has an `Application`
subject for `DEPENDS_ON`, per `scripts/seed_graph_demo_data.py`'s
Alpha Internal Console relationship) showed the two modes disagreeing:
RDFS closure silently re-typed that `Application` individual as a
`tech:Service`, purely from the domain axiom, which then changed which
SHACL shapes applied to it. Fixed by removing the domain declaration
entirely and moving the real "subject must be a Service or Application"
constraint into SHACL (`data/ontology/shapes/techfusion_shapes.ttl`'s
`DependsOnEndpointsShape`, `sh:or` over both classes), which validates
without ever manufacturing a new triple. See
`ontology/techfusion_ontology_v1.ttl`'s comment on `tech:dependsOn` and
`scripts/ontology_demo.py --demo domain-range-pitfall` for an isolated reproduction of the
same mechanism. Not treated as a separate dot-release: caught and fixed
before v1.0.0 was used in any downstream file, so there was never a
"wrong" v1.0.0 in circulation.
