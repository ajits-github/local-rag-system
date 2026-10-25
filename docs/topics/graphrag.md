# Knowledge Graph and GraphRAG

--8<-- "README.md:docs-graphrag"

The graph uses the same security checks as the rest of the system. Tenant
and role checks are applied in the Cypher query and again in Python on every
returned fact. See [Security](security.md) for the other controls, and
[Agentic RAG](agentic-rag.md) for how the graph tools fit into the agent loop.

The full design, schema, query guardrails, fusion strategies, failure
behavior, and verification status are in [`docs/graphrag.md`](../graphrag.md).

API reference: [Knowledge Graph / GraphRAG](../reference/graph.md).

## Knowledge Graph and GraphRAG (graphrag milestone)

Adds a Neo4j-backed property graph (entities/relationships derived from the
document corpus) and a GraphRAG retrieval mode, as an **additional**
retrieval capability layered on top of the existing dense/hybrid/RRF/
rerank/relationship-expansion pipeline, never a replacement. See
`docs/graphrag.md` for the full architecture, schema diagram, Cypher
examples, run commands, evaluation results, and a 20-question interview
reference; this section is the condensed version.

- **`config.graph.enabled`** (default `false`) is a true no-op matching
  every other milestone's kill-switch convention: no `Neo4jClient` is ever
  constructed, `IngestionPipeline` never runs graph extraction,
  `rag.graph.pipeline.GraphRAGPipeline.answer()` always delegates
  byte-for-byte to the unmodified `RetrievalPipeline.answer()`, and the
  four graph agent tools are never offered to the LLM. `POST /query` and
  the agent's `classic_rag` route are **not modified at all** by this
  milestone, graph capability is reachable only via the four explicit
  agent tools (`search_knowledge_graph`/`get_entity`/`get_neighbors`/
  `find_dependency_path`, gated on `graph.enabled`, a fifth
  `agent_tool_select_v6_graph.yaml` prompt variant) and a new, separate
  `POST /graph/query` endpoint, a deliberately conservative integration
  choice that makes "preserve existing retrieval architecture" true by
  construction, not convention.
- **`rag/graph/`** (new package): `schema.py` (the closed `ENTITY_TYPES`/
  `RELATIONSHIP_TYPES` allowlists, `canonical_entity_id`, scoped by
  `(entity_type, dataset_id, tenant_id, normalized_name)`, mirroring
  `document_id`'s own `(source, dataset_id)` scoping precedent) +
  `client.py` (`Neo4jClient`, the one module that imports `neo4j`
  directly, connection-pooled/timeout-bounded, mirroring `PgVectorStore`'s
  relationship to `psycopg2`) + `authorization.py`
  (`build_graph_authorization_predicate`, the Cypher mirror of
  `pgvector.py`'s SQL predicate, plus `is_fact_authorized`, a
  Python-side post-execution re-check applied to every result regardless
  of the query's own `WHERE` clause) + `canonicalization.py` (a small,
  hand-curated alias table, deliberately not fuzzy matching) +
  `cypher_guard.py` (`validate_generated_cypher`: read-only-only,
  single-statement, bounded-depth, trusted-`LIMIT` validation for the
  optional LLM-Cypher-generation path, explicitly documented as
  defense-in-depth, not the actual security boundary; see below) +
  `queries.py` (six hand-written, parameterized, bounded deterministic
  Cypher templates) + `extraction.py` (deterministic Document/Category/
  Supersedes facts, plus optional `extra="forbid"`-validated LLM
  extraction against the closed type allowlists) + `upsert.py`
  (`GraphWriter`, idempotent `MERGE`-based writes with defense-in-depth
  allowlist re-validation) + `retrieval.py` (`GraphRetriever`: literal
  keyword intent classification, capitalized-noun-phrase entity linking,
  bounded traversal) + `fusion.py` (`GraphFact` -> `SearchResult` with
  `origin="graph"`, resolving a real citeable document path from the
  fact's own `Document` node rather than a bare Neo4j id) + `pipeline.py`
  (`GraphRAGPipeline`, four fusion strategies) + `constraints.py`
  (per-label uniqueness constraints/indexes, run via
  `scripts/init_graph_db.py`).
- **`SearchResult.origin` gained a fifth value, `"graph"`** (joining
  `"retrieved"`/`"expanded"`/`"tool_fetched"`/`"mcp_remote"`), mirroring
  exactly the precedent `"mcp_remote"` already set: a graph fact is never
  real `VectorStore` content, so it's never eligible for freshness
  resolution or relationship expansion, but unlike MCP business evidence
  it *does* resolve to a real, citeable document path (via
  `rag.graph.fusion.resolve_document_sources`, reading each fact's
  `source_document_id`'s `source` property straight off its `Document`
  node). `Citation.origin`/`SourceItem.origin` were extended to match.
- **Graph authorization is enforced twice, independently**: the Cypher
  `WHERE` predicate (mirroring `pgvector.py`'s tenant/role/trust rule
  exactly) on every deterministic template, *and* a Python-side
  post-execution re-check (`is_fact_authorized`) applied to every
  returned row regardless of what the query's own predicate did,
  the actual security boundary for the LLM-generated-Cypher path, since a
  text-level validator can never be proven complete. A path with any
  unauthorized hop is discarded whole, never partially surfaced.
  Verified end-to-end against a live Neo4j instance (cross-
  tenant denial, cross-tenant-support-role gating requiring the role be
  listed on the fact itself, and unauthorized-destination path denial,
  8 chained scenarios, ~6 seconds total).
- **A real, load-bearing design property found while writing an
  adversarial test, not just reasoned through**: `GraphWriter.
  upsert_relationship` derives *both* endpoints' `canonical_id` from that
  one write's own `provenance.tenant_id`, so a single relationship can
  only ever connect two nodes within one tenant scope. A non-issue for
  the real extraction pipeline (one document's `tenant_id` applies
  uniformly to everything it asserts), but it means a hand-constructed
  cross-tenant relationship isn't expressible in one write the way it
  might be if node identity and edge provenance were independent. See
  `KNOWN_ISSUES.md`'s "A relationship write can't span two tenant scopes..."
  entry.
- **Consistency**: Postgres remains the sole source of truth; graph
  extraction always runs *after* `replace_document_chunks` commits, and
  a graph-write failure is always caught/logged/counted, never re-raised
  into `ingest_file`. Re-ingesting a changed document always calls
  `GraphWriter.delete_facts_by_document` before re-extracting (deletes
  every relationship sourced from that document, then any now-orphaned
  node), so stale relationship instances never accumulate. `IngestionPipeline.
  clear_dataset` and the directory-walk deleted-document path both also
  clean up graph facts, resolving `document_id`s to delete *before* the
  Postgres-side delete runs. **A real gap found building the demo
  dataset**: `ingest_file`'s checksum-based "unchanged" shortcut runs
  *before* graph extraction, so a Neo4j-only data loss is invisible to a
  plain re-ingest (Postgres still sees the checksum as unchanged and
  skips the graph stage entirely), recovering requires `clear_dataset`
  (clears both sides) or a real content edit, not just a re-run. See
  `KNOWN_ISSUES.md`'s "Clearing only the graph side of a dataset didn't
  actually clear it..." entry.
- **Two real Neo4j-specific bugs found and fixed**: (1)
  `server.metrics.enabled` isn't a real Neo4j 5.28 setting name under
  strict config validation, crash-looped the container; removed rather
  than guessed at again. (2) `MERGE`'s inline property map rejects a
  literal `null` (`GraphRelationship.provenance.source_chunk_id` is
  legitimately `str | None`) with a hard `SemanticError`, not a silent
  no-match, fixed by coalescing to a `""` sentinel before it reaches the
  `MERGE` pattern. (3) `Neo4jClient.run_read` originally built rows with
  the driver's own `Record.data()`, which recursively flattens `Node`/
  `Relationship`/`Path` values into plain dicts/lists, discarding
  `.labels`/`.type`, switched to `dict(record)`, which preserves the
  real graph-typed values every `rag.graph.retrieval` caller needs. See
  `KNOWN_ISSUES.md` for all three.
- **LLM-based graph extraction is implemented and schema-validated**
  (`extra="forbid"`, closed type allowlists), confirmed to execute
  correctly against a real Ollama call including its failure path, but
  running it at any real scale is impractical on this project's 8GB-RAM
  host (model-load time alone dominates a single call; see `KNOWN_ISSUES.md`).
  The graph demo dataset (`scripts/seed_graph_demo_data.py`,
  `techfusion_graph_demo`) uses hand-curated relationships instead, each
  traceable to an exact sentence in the real corpus documents it also
  ingests, with `extraction_method` honestly recorded as `"deterministic"`
  rather than `"llm"` since no model call actually produced them.
- **`eval/run_graph_eval.py`** compares `vector_only`/`graph_only`/
  `vector_plus_graph`/`graph_seeded_vector` fusion strategies against
  `data/eval/graph_gold.jsonl` (9 questions covering direct relationship,
  multi-hop, impact analysis, policy relationship, supersession,
  graph-unanswerable, and cross-tenant denial, item 9's full checklist),
  reusing `KeywordOverlapScorer` and `source_matches_relevant` rather than
  duplicating `eval/run_eval.py`'s much larger multimodal/safety/agentic
  apparatus. `scripts/benchmark_graph.py` benchmarks direct-lookup/1-hop/
  2-hop/3-hop/path-query latency at a configurable synthetic node count.
  See `docs/graphrag.md` section 16 for which of these were actually run
  live and at what scale.
- **`GoldExample` gained six additive, optional fields**
  (`requires_graph_traversal`, `graph_traversal_depth`, `graph_category`,
  `expected_graph_entities`, `expected_graph_relationship_types`,
  `graph_unanswerable`), same pattern as every prior milestone's
  gold-schema extension, old gold files keep parsing unchanged.
