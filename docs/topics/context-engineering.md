# Context Engineering

`rag.context` is an explicit, measurable, token-budget-aware context-assembly
layer above the existing retrieval/GraphRAG/agentic pipelines. See
[Context Engineering](../context_engineering.md) for the full design
writeup (algorithms, security analysis, configuration reference, measured
benchmark numbers, known limitations).

API reference: [Context Engineering](../reference/context.md).

## Context Engineering (context-engineering milestone)

Adds an explicit, measurable, token-budget-aware context-assembly layer
(`rag.context`) above the existing retrieval/GraphRAG/agentic pipelines.
Before this milestone the entire context-sizing story was one fixed
cutoff, `retrieval.generation_context_top_n`, with no token counting
anywhere (Ollama only reports `prompt_eval_count`/`eval_count` *after* a
call returns), no model-context-window concept, no deduplication beyond
what fusion already did implicitly, no compression, and no conversation
history at all (every route stays fully stateless per request). See
`docs/context_engineering.md` for the full design writeup (algorithms,
security analysis, configuration reference, measured benchmark numbers,
known limitations); this section is the condensed version.

- **`config.context.enabled`** (default `false`) is a true no-op
  matching every other milestone's kill-switch convention: every call
  site falls back to the original `rag.retrieval.pipeline.
  build_context()` behavior, byte-identical, when this subsystem is off.
- **One shared `ContextEngine`, not three independent implementations.**
  `RetrievalPipeline` owns the one instance (built once from `config.
  context`, exposed via a new `build_context_text()` method and a
  `context_engine` property, the same reuse pattern `llm`/
  `prompt_template` already use). `GraphRAGPipeline._generate_from_results`
  and `agent/graph.py`'s `_synthesize` node both call `pipeline.
  build_context_text(...)` instead of `build_context()` directly, so
  none of the three decides independently how much context to include.
- **`ContextItem`** (`rag.context.types`) is the normalized unit every
  evidence source (vector/hybrid chunks, graph facts, MCP business
  results, agent tool-fetched chunks, relationship-expanded chunks,
  conversation turns) is converted into before anything else runs.
  `rag.context.normalize` only ever runs on evidence that already
  passed through `RetrievalPipeline.sanitize_evidence` (or an
  equivalent already-authorized source, e.g. MCP business results,
  authorized inside `rag.mcp.business.store` before conversion); this
  subsystem adds no authorization logic of its own, it repackages
  already-safe data. Deliberately excludes JWTs, secrets, internal
  service tokens, hidden chain-of-thought, raw `AuthorizationContext`
  objects, and unrestricted tool payloads. `reranker_score` always
  reads `None`, a documented, not fabricated, gap: the existing
  `Reranker.rerank()` overwrites `SearchResult.score` in place rather
  than keeping pre-/post-rerank scores separately.
- **Budget algorithm** (`rag.context.planner`): `available_context_tokens
  = model_context_window - reserved_output_tokens - system_prompt_tokens
  - user_query_tokens - framework_overhead_tokens - safety_margin_tokens`,
  floored at 0 (a negative raw value is recorded as `budget_deficit_tokens`
  and surfaced as `budget_infeasible`, logged and counted, never silently
  treated as an empty-context success), then allocated across four buckets (`document`/
  `graph_path`/`mcp_tool`/`conversation_history`) by `SourceBudgetConfig`'s
  four shares (validated to sum to 1.0 at config-load time). `config.
  context.model_limits` is a per-`generation.model_name` context-window
  table (`qwen2.5:*` pre-seeded at 32768), since Ollama itself defaults
  every model to a 2048-token `num_ctx` unless set explicitly and this
  codebase had no such table before. `config.context.adaptive.enabled`
  (default `true`) classifies each request's dominant evidence
  composition from the evidence actually gathered, deterministically,
  never a second LLM call (`classify_context_query_type`), into
  `knowledge_only`/`relationship`/`business`/`conversational`, each with
  its own budget-share override.
- **`rag.context.tokens.TokenCounter`**: no pre-call tokenizer exists
  anywhere else in this codebase, so this is a documented estimate, not
  an exact count. `HeuristicTokenCounter` (default, `ceil(len(text) /
  4.0)`, always rounds up) needs no dependency; `TiktokenTokenCounter`
  (optional, lazy-imported like `ragas`/`datasets`) uses `cl100k_base` as
  a tighter but still explicitly approximate proxy, never the real
  tokenizer for any model this codebase actually runs.
  `safety_margin_tokens` exists specifically to absorb the gap between
  either estimate and the real tokenizer.
- **Relevance vs. authority vs. freshness** (`rag.context.prioritize`)
  are three different signals, never collapsed into one fabricated
  confidence score. Relevance (`relevance_score`) is kept for
  observability but *not* used as a cross-source sort key, since dense
  cosine similarity, BM25/RRF scores, and graph/tool evidence's fixed
  `1.0` constants are not on comparable scales, the same dense-vs-BM25
  incomparability `RetrievalPipeline`'s own docstrings already document
  for retrieval itself. Authority (`authority_score`) comes only from
  `trust_level` (`1.0` authoritative / `0.0` untrusted / `0.5` neutral
  otherwise, the three bins this corpus's data actually encodes).
  Freshness (`is_current`) comes only from `status`. Final ordering is a
  stable sort: freshness tier, then authority tier, then each item's
  original position, deterministic for the same input every time.
- **Deduplication** (`rag.context.dedup`): exact-id, exact-content-hash,
  then word-shingle Jaccard near-duplicate matching (scoped per
  `source_type`, so a document chunk and a graph fact restating the same
  sentence are never merged across source types). Keeps the first
  occurrence, records the discarded item's id onto the kept item's
  `merged_from`, never silently losing provenance. Two genuinely
  different facts about the same topic are never merged, only
  byte-identical or near-byte-identical text triggers a merge.
- **Compression** (`rag.context.compression`): structural, never an
  arbitrary character-offset cut. Prose uses sentence-boundary
  extractive trimming (query-term overlap scoring, sentences kept in
  original order); table/code/configuration use line-boundary trimming,
  always keeping the header/opening line; graph paths drop whole
  trailing hops (`rag.context.graph_packing.shrink_graph_path`), never
  mid-name. `compression.threshold_tokens` gates on the *remaining
  headroom*, not the candidate's own size, a real bug (CTX-001 in
  `KNOWN_ISSUES.md`) found by this milestone's own overflow-regression
  tests: gating on item size instead caused graceful degradation to
  silently return an empty context under a tight budget with many
  individually-modest items. Compression never reintroduces redacted
  data, it only ever selects a subset of already-existing, already-
  sanitized text.
- **Packing: three comparable strategies** (`rag.context.packing`), so
  the design can be measured against itself. `"top_n"` reproduces the
  pre-milestone fixed-chunk-count baseline (comparison only, not for
  production use once this is enabled). `"naive_token_first"` packs by
  rank order until the global budget is exhausted, ignoring authority/
  freshness/per-bucket allocation. `"priority_aware"` (the default when
  enabled) is the real algorithm: rank via `prioritize_items`, walk once,
  include whole items that fit both their bucket and global headroom,
  attempt compression toward headroom for items that don't, exclude and
  record why otherwise, continuing to smaller candidates rather than
  stopping the walk. Never spends past `available_context_tokens`, so a
  packing decision can never touch generation's own reserved budget.
- **Graph-path handling**: a multi-hop fact already arrives as *one*
  `GraphFact`/`SearchResult` (`rag.graph.fusion` never surfaces one
  `SearchResult` per hop), so "never split a path into meaningless
  fragments" is satisfied structurally one layer below `ContextItem`.
  `rag.context.graph_packing.parse_graph_fact_text` adds a best-effort
  regex parse of the rendered fact text back into structured
  `GraphHop`s, built from the closed `rag.graph.schema.
  RELATIONSHIP_TYPES` vocabulary (not a generic "looks uppercase"
  pattern, a real bug, CTX-002, found because several real entity names
  in this corpus are themselves acronyms, e.g. "Alpha API"), purely for
  display/metadata and the hop-aware compression shrink; it never
  affects correctness, since `ContextItem.content` always carries the
  original rendered text regardless of parse success.
- **Conversation history** (`rag.context.history`): this system has no
  server-side session store, a multi-turn caller re-sends prior turns
  each request (the same "caller asserts, server bounds/uses it"
  pattern `tenant_id`/`roles`/`case_approvals` already use), via a new
  optional `conversation_history` field on `POST /query`/`POST
  /agent/query`/`POST /agent/query/stream`, bounded by new
  `security.dos_limits.max_history_turns`/`max_history_turn_chars`.
  Four strategies: `full`, `sliding_window` (default), `summarized`
  (a deterministic, extractive sentence-selection summary of older
  turns, never an LLM call, kept recent turns raw), `retrieval`
  (keyword-overlap-selects the most query-relevant older turns). A
  `ConversationTurn.role` is only ever `"user"`/`"assistant"`, never
  `"system"`; history turns become `ContextItem`s rendered inside the
  *user* turn's context block, never the system turn, so a poisoned
  prior turn is data, not an instruction channel, exactly like a
  retrieved document. `AgentState.conversation_history` is consumed only
  by `_synthesize`/`_run_classic_rag`, deliberately not by the earlier
  decision prompts (`classify`/`decompose`/`tool_select`/
  `evidence_sufficiency`), a scoping decision: what evidence to gather
  for the *current* question doesn't change based on how a prior turn
  was phrased.
- **Render order ("lost in the middle")** (`rag.context.ordering`,
  `config.context.ordering_strategy`, default `begin_end_sandwich`):
  separate from packing, decides what order *survivors* are rendered in.
  `begin_end_sandwich` places the highest-priority items at both the
  start and end of the context block, motivated by published findings
  that decoder-only LLMs attend most reliably to the beginning/end of a
  long context. **Not verified against this project's own local model**:
  no live Ollama instance was reachable while this milestone was built;
  the four strategies are directly pluggable and the experiment is ready
  to run, just not yet executed. Treat the default as a reasonable,
  literature-motivated choice, not a measured conclusion for this system.
- **Security**: context poisoning (a document, graph fact, MCP result,
  or conversation turn containing "ignore previous instructions") stays
  data, never an instruction, tested directly
  (`tests/unit/test_context_engine.py`'s injection tests): the fixed
  system prompt string is measured for token cost but never mutated by,
  or concatenated with, any evidence text anywhere in this pipeline.
  Redaction markers already in retrieved text are never reconstructed by
  compression (selection only, no decoding/regeneration/LLM call). The
  optional token-count cache (`rag.context.caching.TokenCountCache`, off
  by default, no measurable benefit demonstrated for this single-
  instance deployment) keys every entry on `(tenant_id, text_hash,
  token-counter strategy)`, never raw text, so no entry can cross a
  tenant boundary. System instructions are never represented as a
  `ContextItem` at all, so no packing decision can ever drop them to
  make room for evidence.
- **Observability**: new bounded-label Prometheus metrics
  (`rag_context_candidate_tokens`/`rag_context_packed_tokens`/
  `rag_context_dropped_items_total{reason}`/
  `rag_context_deduplicated_items_total{match_kind}`/
  `rag_context_compressed_items_total`/
  `rag_context_build_latency_seconds`/
  `rag_context_overflow_prevented_total`/
  `rag_context_utilization_ratio{strategy}` (unclamped)/
  `rag_context_budget_exceeded_total{strategy}`/
  `rag_context_budget_infeasible_total`) and new OTel spans
  (`context_normalize`/`context_deduplication`/`context_packing`).
  `QueryResponse`/`AgentQueryResponse` both gained an optional
  `context_engineering` field (token counts, utilization, reduction,
  dedup/compression counts, truncation, build latency; never raw
  evidence content, already available, redacted, in `sources`).
- **Evaluation**: `rag.eval.run_context_eval` is a synthetic,
  corpus-free harness (no Postgres/Ollama/Neo4j needed) covering all 15
  required scenario categories (short context, long document, many
  relevant/irrelevant chunks, duplicates, graph+document, MCP+document,
  conflicting evidence, stale-vs-current, long history, prompt
  injection, over-budget volume, one giant chunk, a multi-agent-scoped
  case, an unanswerable query) across 4 candidate-token-size targets (2k/
  8k/16k/32k) and the 4 packing strategies (`scripts/
  benchmark_context_engineering.py` renders the 240-run sweep into
  `experiments/results/context_engineering/report.json` and
  `experiments/reports/context_engineering_benchmark.md`). Deliberately
  computes only deterministic, non-generation metrics (tokens,
  utilization, reduction, dedup/compression counts, truncation,
  latency); answer-correctness/citation-correctness/context-precision-
  recall need a real LLM call against the real corpus and stay
  `run_eval.py`'s/RAGAS's job, not duplicated here with a fabricated
  judge. Measured in this sandbox: the full pipeline strategy dropped 0
  eligible items (compressing 6 instead) versus the pre-milestone
  baseline's 1861 drops across the same 240 runs, while every single run
  across every strategy and candidate size (up to the 32k-token sweep
  against windows as small as 512 tokens) respected its computed budget
  exactly, asserted directly in `tests/unit/test_run_context_eval.py`.
- **Known limitations**: `GraphRAGPipeline`'s own `POST /graph/query`
  endpoint has no `conversation_history` field at all today (only the
  classic-RAG and agent routes do); multi-agent context isolation
  (`multi_agent_experiment/`, its own unmerged branch) was exercised
  only as one synthetic scenario proving the budgeting mechanics
  generalize, specialist agents there do not yet receive their own
  scoped `ContextEngine`; extractive compression and graph-path hop
  parsing both have documented, accepted edge-case gaps (CTX-004).
- **Tests**: ~95 new unit tests (`tests/unit/test_context_*.py`,
  `test_run_context_eval.py`) cover every module plus end-to-end
  overflow-regression invariants matching section 26 of the originating
  design doc (a packed prompt never exceeds its budget; a single giant
  chunk/MCP result/graph path/long history can never overflow it;
  duplicated evidence is bounded; system instructions can never be
  dropped to make room for evidence; prompt injection stays data; a
  redaction marker is never reconstructed). 21 existing test files'
  `FakePipeline`/`RetrievalPipeline` doubles across the agent/API test
  suite needed `build_context_text`/`history_turns` stubs added, the
  same "sibling change, unaware test double" pattern already documented
  for the joint-investigation PRs below. See `KNOWN_ISSUES.md`'s CTX-001
  through CTX-004 and `PROJECT_JOURNAL.md`'s 2026-10-24 session for the
  full implementation narrative, including the compression-threshold and
  graph-relation-parsing bugs found while writing this milestone's own
  tests.
