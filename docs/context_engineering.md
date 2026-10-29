# Context Engineering

Explicit, measurable, token-budget-aware prompt-context assembly
(`rag.context`), layered above the existing retrieval/GraphRAG/agentic
pipelines. Replaces "retrieve top-N chunks, concatenate, hand to the
model" with a shared pipeline every generation call site uses:

```
normalize -> deduplicate -> plan budget -> pack -> order -> render
```

`config.context.enabled` defaults to `false`: every call site falls back
to the original `rag.retrieval.pipeline.build_context()` behavior,
byte-identical, when this subsystem is off.

## 1. Why this exists

Before this milestone, the entire "context management" story was one
fixed number, `retrieval.generation_context_top_n` (default 3): take the
top 3 reranked chunks, join them with `[Source N: ...]` labels, done.
There was no token counting anywhere in the codebase (Ollama only
reports `prompt_eval_count`/`eval_count` *after* a call returns), no
model-context-window concept, no deduplication beyond what fusion
already did implicitly, no compression, and no conversation history at
all (`POST /query`/`POST /agent/query` are, and remain, fully stateless
per request).

That approach works while a request has one evidence source (dense
retrieval) and a model with a huge context window relative to typical
answer length. It stops working the moment evidence volume varies
(GraphRAG facts, MCP business-tool results, relationship-expanded
chunks, long conversation history, oversized documents) or the model's
window is small relative to that volume: a fixed chunk count either
wastes most of a large window or silently overflows a small one.

## 2. Architecture

### 2.1 Before

```
query
  |
  v
retrieve top-N (fixed count, no token awareness)
  |
  v
join chunks ("[Source N: ...]" labels)
  |
  v
LLM
```

### 2.2 After

```
                    User Query
                        |
                Query / evidence composition
                        |
               Context budget planner
                        |
        +---------------+----------------+
        |               |                |
   Vector/Hybrid     GraphRAG        MCP / tool
   evidence          evidence        evidence
        |               |                |
        +---------------+----------------+
                        |
               conversation history
                        |
                normalize (-> ContextItem)
                        |
                   deduplicate
                        |
          (authorization/redaction already applied
           upstream, before normalize ever runs)
                        |
          relevance + authority + freshness ordering
                        |
             token-budget allocation (per source)
                        |
               token-aware packing (+ compression)
                        |
             render-order strategy ("lost in the middle")
                        |
            final prompt/context package
                        |
                       LLM
```

### 2.3 Where this plugs in

Three call sites share one `ContextEngine` instance (built once per
`RetrievalPipeline`, like every other injected singleton in this
codebase's DI pattern):

| Call site | Before | After |
|---|---|---|
| `RetrievalPipeline.answer()` | `build_context(results)` | `self.build_context_text(results, query, history_turns=..., tenant_id=...)` |
| `GraphRAGPipeline._generate_from_results()` | `build_context(results)` | `self._retrieval_pipeline.build_context_text(...)` |
| `agent/graph.py`'s `_synthesize` node | `build_context(ordered_evidence)` | `pipeline.build_context_text(ordered_evidence, ..., history_turns=state.conversation_history, ...)` |

No call site constructs a `ContextEngine` itself; `RetrievalPipeline`
owns the one shared instance and exposes it via `build_context_text()`
and the `context_engine` property, so `GraphRAGPipeline`/the agent graph
never duplicate context-building logic.

## 3. The normalized evidence unit: `ContextItem`

Every evidence source (vector/hybrid chunks, graph facts, MCP business
results, agent tool-fetched chunks, relationship-expanded chunks,
conversation turns) is converted into a `ContextItem`
(`rag.context.types`) before anything else runs. Key fields:

- `content`, `source_type` (`document` / `graph_path` / `mcp_tool` /
  `tool_fetched` / `conversation_history` / `system` / `query`),
  `source_id`, `citation`
- `relevance_score`, `authority_score` (derived from `trust_level`),
  `freshness_status`, `is_current`
- `token_count`, `tenant_id`, `dataset_id`, `sensitive_field_ids`,
  `injection_suspected`
- `document_id`, `chunk_id`, `content_type`, `graph_path` (parsed
  `GraphHop`s, best-effort), `tool_name`, `origin`
- `original_token_count`, `compressed`, `compression_strategy`
- `priority_rank`, `merged_from`, `included`

Deliberately **not** included anywhere in this type or its construction
path: JWTs, secrets, internal service tokens, hidden chain-of-thought,
raw `AuthorizationContext` objects, or unrestricted tool payloads.
`normalize.py` only ever runs on evidence that already passed through
`RetrievalPipeline.sanitize_evidence` (field redaction + injection
flagging); it adds no authorization logic of its own, it repackages
already-safe data.

**Known, documented gap**: `reranker_score` always reads `None`. The
existing `Reranker.rerank()` overwrites `SearchResult.score` in place
rather than keeping pre-rerank and post-rerank scores separately, so
there is no distinct value to carry without changing `Reranker`'s public
contract, which was out of scope for this milestone.

### 3.1 `ConversationTurn`

Caller-supplied, not server-persisted (this system has no session
store): `role` (`"user"` / `"assistant"`, never `"system"`), `content`,
optional `turn_id`. A multi-turn caller re-sends prior turns each
request, the same "caller asserts, server bounds/uses it" pattern this
codebase already applies to `tenant_id`/`roles`/`case_approvals`.

## 4. Model limits and token counting

`rag.context.model_limits.resolve_model_context_limits` reads
`config.context.model_limits.overrides[generation.model_name]`, falling
back to `default_context_window_tokens` (32768) for an unlisted model.
Ollama itself defaults every model to a 2048-token `num_ctx` unless set
explicitly; this table makes the assumed window an explicit, reviewable
config fact rather than a silent guess.

`rag.context.tokens.TokenCounter` has two implementations:

- `HeuristicTokenCounter` (default): `ceil(len(text) / chars_per_token)`,
  `chars_per_token=4.0` (the standard rule of thumb for English prose
  under BPE tokenizers). Always rounds up, never down, since
  under-counting is the failure mode that causes a real overflow.
- `TiktokenTokenCounter` (optional, lazy-imported like `ragas`/
  `datasets`): uses `cl100k_base` as a sub-word tokenizer proxy.
  Explicitly **not** the real tokenizer for any model this codebase
  actually runs (`qwen2.5` via Ollama has its own BPE vocabulary); it is
  offered because it is a tighter estimate for most English text than
  the character heuristic, not because it is exact.

Neither estimate is treated as ground truth: `safety_margin_tokens`
exists specifically to absorb the gap between either estimate and
whatever the real tokenizer would have counted.

## 5. The budget algorithm

```
available_context_tokens =
    model_context_window
    - reserved_output_tokens
    - system_prompt_tokens      (measured, not estimated by proxy)
    - user_query_tokens
    - framework_overhead_tokens (chat-template/role-formatting allowance)
    - safety_margin_tokens
```

Floored at 0. `reserved_output_tokens` mirrors
`generation.max_tokens` by default (overridable). The result is
allocated across four buckets (`document`, `graph_path`, `mcp_tool`,
`conversation_history`) by `SourceBudgetConfig`'s four shares, which are
validated to sum to 1.0 at config-load time (a `model_validator`, fails
fast rather than silently misallocating).

### 5.1 Fixed vs. adaptive allocation

`config.context.adaptive.enabled` (default `true`) classifies each
request's dominant evidence composition *from the evidence actually
gathered*, deterministically, never via a second LLM call
(`rag.context.planner.classify_context_query_type`):

| Classification | Trigger | Effect |
|---|---|---|
| `business` | MCP-tool items dominate | More `mcp_tool` budget share |
| `relationship` | Graph-path items dominate | More `graph_path` budget share |
| `conversational` | History present, no document/graph/tool evidence | More `conversation_history` share |
| `knowledge_only` | Otherwise (the common case) | More `document` share |

Each classification maps to its own `SourceBudgetConfig` in
`AdaptiveBudgetConfig.overrides` (also validated to sum to 1.0).
Disabling adaptive allocation always uses the fixed `source_budgets`.

## 6. Relevance vs. authority vs. freshness

Three different questions, never collapsed into one fabricated
confidence score (`rag.context.prioritize`):

- **Relevance** (`relevance_score`): how well an item matches *this
  query*, as the originating retriever already scored it. Kept for
  observability but **not** used as a cross-source sort key: dense
  cosine similarity, BM25/RRF scores, and the fixed `1.0` constants
  assigned to graph/tool evidence are not on comparable scales (the
  same dense-vs-BM25 incomparability `RetrievalPipeline`'s own
  docstrings already document). Within one source, the originating
  retriever's own rank order already *is* relevance ordering, and is
  preserved as-is.
- **Authority** (`authority_score`): derived only from the existing
  `trust_level` field. `1.0` for `"authoritative"`, `0.0` for
  `"untrusted"`, `0.5` (neutral) for anything else, including unset,
  the three bins this corpus's governance data actually encodes, never
  guessed finer.
- **Freshness** (`is_current`): from `status` alone (`"active"` ->
  `True`, any other value -> `False`, unset -> `None`/unknown). A true
  freshness-family re-resolution (as `retrieval/freshness.py` already
  does for retrieval itself) is not repeated at this layer; `is_current`
  is a direct read of a field already resolved upstream when
  authorization is enabled.

Final ordering is a **stable sort**: freshness tier (current/unknown
before superseded), then authority tier (authoritative before
neutral/unknown before untrusted), then each item's original position.
Deterministic for the same input, every time (`tests/unit/
test_context_prioritize.py::test_ties_break_deterministically_on_original_order`).

## 7. Deduplication

Three passes, in order, over items in priority order (`rag.context.dedup`):

1. **Exact identity**: same `chunk_id`/`source_id` twice.
2. **Exact content hash**: different ids, byte-identical text (sha256,
   never the raw text, stored as the dedup key).
3. **Near-duplicate overlap**: word-shingle (3-gram) Jaccard similarity
   above `overlap_threshold` (default 0.85), scoped per `source_type` so
   a document chunk and a graph fact that happen to restate the same
   sentence are never merged across source types by accident.

Each merge keeps the *first* occurrence (callers hand candidates in
priority order, so "first" already means "best" for ties) and records
the discarded item's `source_id` onto the kept item's `merged_from`,
never silently losing provenance. Two genuinely different facts about
the same topic (`"timeout = 30s"` vs `"timeout = 60s"`) are never merged
by this mechanism; only byte-identical or near-byte-identical text
triggers a merge.

## 8. Compression

Section 10/11 of the design (oversized evidence, tool-output
compression), handled structurally, never by an arbitrary character-
offset cut (`rag.context.compression`):

- **Prose** (`content_type` unset or `"prose"`): sentence-boundary
  trimming. `"extractive"` (default) scores sentences by query-term
  overlap and keeps the highest-scoring ones *in their original order*
  (never reordered, to avoid changing meaning); `"sentence_trim"` keeps
  leading sentences up to budget.
- **Table/code/configuration**: line-boundary trimming, always keeping
  the first line (a table header or an opening key/fence line) so a
  partial table/config block never loses its column meaning.
- **Graph paths**: whole-hop trimming (`rag.context.graph_packing.
  shrink_graph_path`), dropping trailing hops as complete units, never
  cutting a node name in half.

`threshold_tokens` (default 30) gates on the **remaining headroom**, not
the candidate's own size: an item that doesn't fit its allocated space
is always a compression candidate as long as that headroom is large
enough to compress into something usable. This was a real bug found and
fixed during this milestone's own testing (see `KNOWN_ISSUES.md`'s
CTX-001); gating on the item's own size instead caused graceful
degradation to silently return an empty context under tight budgets
with many individually-modest items.

Compression never reintroduces redacted data: it only ever *selects a
subset of already-existing text*, the input to this module already
passed through `sanitize_evidence`, there is no decoding, reconstruction,
or LLM call anywhere in this module.

**MCP/tool-output compression is mostly already solved upstream**:
`rag.agent.mcp_client._business_result_to_search_result` re-validates a
raw MCP result against its Pydantic schema and renders only a handful of
named fields into a short prose string before a `SearchResult` ever
exists, so a verbose raw JSON payload never reaches `ContextItem` in the
first place. This module's handling of `mcp_tool` items is the same
generic oversized-text handling every other bucket gets.

## 9. Token-aware packing: three comparable strategies

`rag.context.packing` implements three strategies, so the design can be
measured against itself (section 9/25):

| Strategy | What it does | Purpose |
|---|---|---|
| **A: `top_n`** | Take the first N items per bucket, no token awareness at all | Reproduces the pre-milestone baseline, for comparison only |
| **B: `naive_token_first`** | Greedy inclusion in existing rank order until the global budget is spent | Token-aware, but ignores authority/freshness/per-bucket allocation |
| **C: (B + dedup)** | `naive_token_first` with deduplication run first | Isolates dedup's own contribution |
| **D: `priority_aware`** (the real algorithm) | Prioritize, respect per-bucket *and* global budgets, compress what doesn't fit, pack | The production default when `context.enabled=true` |

**D's algorithm**: rank via `prioritize_items`; walk the ranked list
once; for each item, compute headroom against both its bucket's
remaining budget and the global remaining budget; include it whole if
it fits; otherwise attempt compression toward that headroom (if enabled
and the headroom is large enough); otherwise exclude it and record why
(`source_budget_exhausted` / `global_budget_exhausted` /
`oversized_after_compression`), continuing to smaller candidates rather
than stopping the whole walk. Never spends past
`available_context_tokens`, which already excludes
reserved-output/system/query/overhead/safety-margin tokens, so
generation's own budget can never be touched by a packing decision.

## 10. Graph-path handling

A multi-hop graph fact already arrives as **one** `GraphFact`/
`SearchResult` (`rag.graph.fusion` never surfaces one `SearchResult` per
hop), so "never split a path into meaningless fragments" is satisfied
structurally one layer below `ContextItem`. `rag.context.graph_packing`
adds a best-effort regex parse of the rendered fact text back into
structured `GraphHop`s (built from the closed `RELATIONSHIP_TYPES`
vocabulary, not a generic uppercase-word pattern, see CTX-002), purely
for display/metadata and for the hop-aware compression shrink that drops
whole trailing hops rather than truncating mid-name.

## 11. Conversation history

`rag.context.history` implements four strategies
(`config.context.history.strategy`):

- `"full"`: every supplied turn (bounded only by the
  `conversation_history` budget bucket).
- `"sliding_window"` (default): keep only the most recent `max_turns`.
- `"summarized"`: summarize everything older than `max_turns` into one
  item, keep recent turns raw. The summary is produced by a
  **deterministic, extractive** sentence-selection pass (query-term
  overlap plus a recency bonus), never an LLM call, so it is testable
  and never asserts a fact the original turns didn't contain.
- `"retrieval"`: keyword-overlap-select the most query-relevant older
  turns instead of a recency window; falls back to the sliding window
  when no turn has any overlap at all.

**Trust boundary**: a `ConversationTurn.role` is only ever `"user"` or
`"assistant"`; nothing in this module ever writes a turn into the fixed
system-instruction string a `PromptTemplate` renders. History turns
become `ContextItem`s and flow through the same budgeting/dedup/
packing/prioritization pipeline as any other evidence, rendered inside
the *user* turn's context block, never the system turn. A poisoned prior
turn is therefore data, not an instruction channel, exactly like a
retrieved document (section 16).

## 12. Render order: "lost in the middle"

Packing decides *what* survives the budget; `rag.context.ordering`
decides what *order* survivors are rendered in, a separate concern.
Four strategies: `priority_first`, `authority_first`,
`begin_end_sandwich` (places the best items at both the start and the
end, the lowest-priority items end up adjacent in the middle),
`grouped_by_source`. Default: `begin_end_sandwich`, motivated by the
widely-cited finding that decoder-only LLMs attend most reliably to the
beginning and end of a long context. **This project did not verify that
finding against its own local model**: no live Ollama instance was
reachable in the sandbox this milestone was built in. The harness to run
that experiment for real exists (`rag.context.ordering`'s four
strategies, directly pluggable via `config.context.ordering_strategy`);
running it against a real model and a real multi-chunk question, with
the answer-correctness signal `run_eval.py`/RAGAS already compute, is
the concrete next step, not yet executed. Do not treat
`begin_end_sandwich`'s default status as a verified conclusion for this
system specifically; it is a reasonable, literature-motivated default
that has not yet been measured here.

## 13. Security analysis

- **No evidence enters `ContextItem` before authorization/redaction.**
  `rag.context.normalize` only ever runs on results that already passed
  through `RetrievalPipeline.sanitize_evidence` or an equivalent
  already-authorized source (MCP business results are authorized inside
  `rag.mcp.business.store` before they are ever converted). This
  subsystem adds no authorization logic of its own anywhere; it is a
  packaging layer over already-safe evidence.
- **Context poisoning / prompt injection**: tested directly
  (`tests/unit/test_context_engine.py::test_prompt_injection_in_a_retrieved_chunk_stays_data_not_instructions`,
  `test_injection_in_conversation_history_also_stays_data_not_instructions`).
  Injected text (from a document, a graph fact, an MCP result, or a
  conversation turn) is rendered only inside a `"[Source N: ...]"`
  evidence block; the fixed system prompt string is measured for its
  token cost but never mutated by, or concatenated with, any evidence
  text at any point in this pipeline.
- **Redaction markers are never reconstructed.** Compression only
  selects existing text (sentences, lines, or hops already present); it
  never decodes, reverses, or regenerates anything, so a
  `[REDACTED:SENSITIVE_FIELD]` marker already in the text can be kept or
  dropped, never un-redacted or reassembled from fragments (tested
  directly).
- **Tenant isolation in caching.** The optional token-count cache
  (`rag.context.caching.TokenCountCache`, off by default) keys every
  entry on `(tenant_id, text_hash, token-counter strategy)`, never the
  raw text itself, so no cache entry can cross a tenant boundary even
  though the cached value (an integer) carries no sensitive information
  on its own today.
- **No cross-tenant deduplication.** Deduplication operates only within
  one request's own candidate set, never against a cache or store shared
  across requests/tenants.
- **System instructions are structurally un-droppable.** They are never
  represented as a `ContextItem` at all; their token cost is subtracted
  from the budget *before* any bucket allocation happens, so there is no
  code path in which a packing decision could exclude them to make room
  for evidence.

## 14. Observability

New Prometheus metrics (`rag.observability.metrics`, bounded labels
only): `rag_context_candidate_tokens`, `rag_context_packed_tokens`,
`rag_context_dropped_items_total{reason}`,
`rag_context_deduplicated_items_total{match_kind}`,
`rag_context_compressed_items_total`,
`rag_context_build_latency_seconds`,
`rag_context_overflow_prevented_total`,
`rag_context_utilization_ratio{strategy}` (unclamped, so values above 1.0
are real overflow), `rag_context_budget_exceeded_total{strategy}`,
`rag_context_budget_infeasible_total`. New OTel spans:
`context_normalize`, `context_deduplication`, `context_packing`, nested
under the same span tree `retrieval`/`agent_query` already populate.
`QueryResponse`/`AgentQueryResponse` both gained an optional
`context_engineering` field (`rag.context.engine.
context_engineering_summary()`'s output): candidate/packed token counts,
utilization, reduction, dedup/compression counts, truncation and
`budget_infeasible` flags, and build latency. Never raw evidence content (already available, redacted,
in `sources`).

## 15. Configuration reference

```yaml
context:
  enabled: false
  reserved_output_tokens: null       # defaults to generation.max_tokens
  framework_overhead_tokens: 128
  safety_margin_tokens: 256
  token_counting:
    strategy: heuristic              # or "tiktoken"
    chars_per_token: 4.0
  model_limits:
    default_context_window_tokens: 32768
    overrides: {qwen2.5:1.5b: 32768, ...}
  source_budgets:
    document_share: 0.55
    graph_path_share: 0.15
    mcp_tool_share: 0.15
    conversation_history_share: 0.15
  adaptive:
    enabled: true
    overrides: {knowledge_only: {...}, relationship: {...}, business: {...}, conversational: {...}}
  deduplication:
    enabled: true
    overlap_threshold: 0.85
  compression:
    enabled: true
    threshold_tokens: 30
    strategy: extractive             # or "sentence_trim"
    max_sentences: 8
  history:
    strategy: sliding_window         # full | sliding_window | summarized | retrieval
    max_turns: 6
    summarize_min_turns_to_trigger: 4
  packing_strategy: priority_aware   # top_n | naive_token_first | priority_aware
  ordering_strategy: begin_end_sandwich
  cache:
    enabled: false
    max_entries: 2048

security:
  dos_limits:
    max_history_turns: 20
    max_history_turn_chars: 4000
```

## 16. Files added / changed

**New package** `src/rag/context/`: `types.py`, `tokens.py`,
`model_limits.py`, `planner.py`, `normalize.py`, `dedup.py`,
`prioritize.py`, `compression.py`, `graph_packing.py`, `packing.py`,
`ordering.py`, `history.py`, `caching.py`, `engine.py`.

**New eval/tooling**: `src/rag/eval/run_context_eval.py`,
`scripts/benchmark_context_engineering.py`.

**Changed**: `src/rag/config.py` (`ContextEngineeringConfig` and its
nested models, `AppConfig.context`, `security.dos_limits.
max_history_turns`/`max_history_turn_chars`), `src/rag/retrieval/
pipeline.py` (`build_context_text()`, `context_engine` property,
`answer()` gained `history_turns`), `src/rag/graph/pipeline.py`
(`_generate_from_results` routed through `build_context_text`),
`src/rag/agent/state.py` (`AgentState.conversation_history`/
`context_engineering`), `src/rag/agent/graph.py` (`_synthesize`/
`_run_classic_rag` thread history through, capture the summary),
`src/rag/api/request_auth.py` (`enforce_history_limits`), `src/rag/api/
routers/query.py`/`agent_query.py`/`agent_stream.py`
(`conversation_history` request field, `context_engineering` response
field), `src/rag/observability/metrics.py` (`rag_context_*` metrics).

**New tests**: `tests/unit/test_context_*.py` (12 files), `tests/unit/
test_run_context_eval.py`. **Patched**: 21 existing test files' fake
`RetrievalPipeline` doubles (`build_context_text`/`history_turns` stubs,
the same "sibling change, unaware test double" pattern this project has
hit before, see `KNOWN_ISSUES.md`'s joint-investigation entry).

## 17. Exact commands to run

```bash
# Unit tests (no external services needed)
PYTHONPATH=src python -m pytest tests/unit/test_context_*.py tests/unit/test_run_context_eval.py -v

# Full regression check
PYTHONPATH=src python -m pytest tests/unit -q

# Synthetic benchmark (no Postgres/Ollama needed)
python -m rag.eval.run_context_eval --report experiments/results/context_engineering/report.json
python scripts/benchmark_context_engineering.py

# Lint/type check
python -m ruff check src/rag/context/ src/rag/eval/run_context_eval.py
python -m mypy src/rag/context/
```

## 18. Measured results (this sandbox, no live LLM)

240-run synthetic sweep (15 scenarios x 4 strategies x 4 candidate-token
targets, `experiments/reports/context_engineering_benchmark.md`):

| Strategy | Mean utilization | Overflowing runs | Mean excess tokens | Mean reduction | Mean build (ms) | Dropped items | Compressed items |
|---|---|---|---|---|---|---|---|
| A: top-N (baseline) | 0.322 | 5/60 | 590.0 | 0.244 | 0.773 | 1861 | 0 |
| B: naive token-first | 0.225 | 0/60 | 0.0 | 0.196 | 0.903 | 1348 | 0 |
| C: token-budget + dedup | 0.043 | 0/60 | 0.0 | 0.342 | 1.372 | 5 | 0 |
| D: full pipeline | 0.031 | 0/60 | 0.0 | 0.357 | 1.758 | 0 | 6 |

Per-item token counts here also include each item's own rendered
`"[Source N: ...]"` label and join-separator overhead, not just its raw
content (CTX-006, see `KNOWN_ISSUES.md`): an earlier version measured
only raw content, under-counting the real serialized prompt cost of
every item.

`context_utilization`/`context_overflow`/`context_excess_tokens` are
reported unclamped (CTX-011, see `KNOWN_ISSUES.md`): an earlier version
of this table clamped utilization to 1.0, which hid real overflow
magnitude and made an overflowing `top_n` run read identically to one
that landed exactly at capacity. Strategies B/C/D (the three that
actually enforce the budget) respect it on every one of their 180 runs
here, across every candidate size up to 32,000 estimated candidate
tokens against windows as small as 512 tokens
(`packed_token_count <= available_context_tokens`, asserted directly in
`tests/unit/test_run_context_eval.py`). `top_n` (baseline A) genuinely
overflows on 5/60 runs, exactly as its own docstring says it will
("ignores token size entirely... even if it alone would overflow the
model's context window downstream"), this is expected baseline
behavior, not a bug, and exists specifically so B/C/D can be measured
against it. D's build latency is roughly 2x A's, both sub-millisecond,
negligible next to this project's own measured ~20-80s LLM generation
latency on local hardware. **Not measured in this sandbox** (no live
Ollama/Postgres/Neo4j reachable): real answer-correctness/citation-
correctness/context-precision-recall against the real corpus, and the
lost-in-the-middle ordering experiment against a real model.

## 19. Known limitations

- `reranker_score` is always `None` (see section 3).
- Extractive/sentence-based compression needs at least one sentence
  boundary; a prose blob with none is kept whole or dropped, never cut
  at an arbitrary offset (CTX-004, accepted).
- Graph-path hop parsing is a best-effort regex over already-rendered
  text, not a read of real structured data; it only affects optional
  display/compression enrichment, never correctness (CTX-004, accepted).
- `GraphRAGPipeline`'s own `POST /graph/query` endpoint has no
  `conversation_history` field at all today; only the classic-RAG and
  agent routes accept one.
- The lost-in-the-middle ordering default (`begin_end_sandwich`) is
  motivated by published literature on decoder-only LLMs generally, not
  verified against this project's own local model; see section 12.
- Multi-agent context isolation (`multi_agent_experiment/`, its own
  unmerged branch) was exercised only as one synthetic scenario proving
  the budgeting mechanics generalize to a scoped-evidence shape;
  specialist agents in that branch do not yet receive their own
  `ContextEngine`-scoped context.
- The token-count cache's safety case (section 13) has not been
  measured under real concurrent load; it is off by default precisely
  because no measurable benefit was demonstrated for this
  single-instance deployment.

## 20. Recommendations for production deployment

1. Enable `context.enabled` behind the same kind of staged rollout this
   project already used for `security.authorization`/`field_redaction`:
   turn it on in a non-default experiment config first
   (`config/experiments/context-engineering-v1.yaml`, not yet created),
   compare `run_eval.py`'s/RAGAS's answer-quality metrics against the
   current baseline on the real corpus before flipping
   `config/default.yaml`.
2. Calibrate `model_limits.overrides` against the actual deployed
   model's real `num_ctx`, not the trained context length; if the
   deployment sets a smaller `num_ctx` than 32768 for memory reasons (a
   real possibility on an 8GB-RAM host), the budget planner will
   otherwise systematically over-allocate.
3. Run the lost-in-the-middle experiment (section 12) against the real
   model and real multi-source questions before trusting
   `begin_end_sandwich` as more than a reasonable default.
4. If `tiktoken` is adopted for tighter estimates, keep
   `safety_margin_tokens` nonzero regardless; it is not a redundant
   knob, it is the only thing standing between "this estimator is
   usually close" and a real overflow on the occasional input it isn't.
5. Treat `context_engineering` response-field exposure as developer/
   debug information, matching this project's existing "production vs.
   developer UI mode" convention; it never carries anything the
   `FeatureFlags`/`RuntimeInfo` endpoints' own leak-guard tests would
   object to, but it is still debug detail, not end-user-facing content.
