# Milestones, history, and experiment branches

An index/timeline of this project's milestones, documentation tooling,
deliberate deviations from the original plan, and the learning-exercise
branches that were never merged into `main`. Each topic's own
day-to-day-relevant behavior lives in its `docs/topics/*.md` page (see
the nav); this page is for "how did we get here" and "what else exists
but isn't part of the production path," not routine reference.

## Documentation site (MkDocs)

A generated docs site (`pip install -e .[docs]`; `make docs-serve` /
`make docs-build`, or `mkdocs serve` / `mkdocs build --strict` without
`make`) sits over the existing Markdown docs and docstrings without
duplicating either. Not published anywhere yet (no `site_url`, no
`gh-deploy` run), `mkdocs build --strict` is a local/CI-detectable gate
only, per the milestone's own explicit instruction not to publish without
being asked.

- **Reuse over rewriting.** `docs/architecture.md` and `docs/metrics.md`
  are unmodified and linked into the nav as-is. `docs/topics/*.md` (one
  per nav topic: ingestion, retrieval, multimodal, security, agentic-rag,
  mcp, evaluation, observability, web-ui, deployment) are short, original
  landing pages that link out to those files rather than restating them, plus
  transclude a handful of README.md sections verbatim via
  `pymdownx.snippets` section markers (`<!-- --8<-- [start:name] -->` /
  `[end:name]`, invisible HTML comments on GitHub), the architecture
  diagram, Prerequisites/Setup, Containerized development, Agentic RAG,
  Observability, and the Benchmarks table. The content lives once, in
  README.md; the docs site only points at it.
- **`docs/reference/*.md`** is 17 pages of `::: rag.module` mkdocstrings
  directives grouped to match this file's own directory map (config,
  schemas, loaders, chunkers, embedders, vectorstore, retrieval,
  rerankers, generation, prompts, vision, agent, mcp, ingestion, eval,
  observability, api), ~400 classes/functions rendered directly from
  the NumPy-style docstrings already required by this file's "Linting,
  typing, and docstrings" section. No hand-written API prose exists
  anywhere in `docs/reference/`; if a rendered page is wrong, the fix is
  in the `.py` docstring, not the `.md` file.
- **Nav-level anchor fragments don't work.** An earlier attempt put
  `architecture.md#some-heading`-style entries directly in `mkdocs.yml`'s
  `nav:` list (to expose architecture.md's internal sections as distinct
  nav items without duplicating the file). MkDocs's own nav-reference
  validation flags these as "not found in the documentation files" even
  though the base file is elsewhere in `nav:`, a real limitation of
  this MkDocs version's nav resolution, not a typo. Anchor links to
  architecture.md's sections work fine as ordinary in-page Markdown links
  (used throughout `docs/topics/*.md`); they just can't be `nav:` entries
  themselves. `architecture.md` is a single flat nav entry as a result.
- **`mkdocs-htmlproofer-plugin`** is what makes broken links/anchors
  detectable at build time (`mkdocs build --strict`, wired as
  `make docs-build`) rather than only visually, on top of MkDocs's own
  nav/relative-link warnings (which, confirmed directly, not assumed,
  also abort `--strict` on their own, independent of htmlproofer).
  `ignore_urls` in `mkdocs.yml` allow-lists three specific, verified-safe
  patterns: two links transcluded from README.md that point at
  README's own repo-root-relative targets (`docs/architecture.md`,
  `#mlflow-tracking`) which can't resolve from inside the built site
  since README.md itself isn't part of it, and `*#rag.*`, mkdocstrings/
  autorefs cross-reference links between `docs/reference/*.md` pages
  (e.g. a "Bases: LLM" link) that this plugin version's resolver can't
  follow, confirmed present and correct by grepping the actual
  rendered HTML (`id="rag.generation.base.LLM"` exists) before
  allow-listing the pattern, not assumed safe.
- **`docstring_options.warn_unknown_params: false`** (mkdocstrings-python
  handler option) suppresses griffe's numpy-docstring parser trying to
  match a `Parameters` section's field names against a real function
  signature, which fails for Pydantic `BaseModel` fields, since
  Pydantic's synthesized `__init__` isn't visible to griffe's static
  analysis. This surfaced a real, pre-existing gap against this file's
  own "Linting, typing, and docstrings" convention (`Parameters` for
  genuine function arguments, `Attributes` for class/model state): most
  config/state Pydantic models in `config.py`, `agent/state.py`,
  `agent/graph.py`, `agent/events.py`, `api/auth.py`, and
  `eval/gold_schema.py` document their fields under `Parameters` instead
  of `Attributes`. Flagged, not fixed, relabeling ~20 docstrings across
  6 files is a separate documentation-quality pass, out of scope for
  standing up the docs site itself.
- **One real docstring fix, not just a suppression**: `AgentConfig`'s
  five `*_prompt_path` fields were documented on one combined numpydoc
  line (`classify_prompt_path, decompose_prompt_path, ... : str`); griffe
  couldn't associate a type with the first three names on that line (a
  real mkdocstrings rendering gap, not a false positive), so they were
  split into five separate `name : str` lines. Docstring-content-only,
  no behavior change.

## Known deviations from the original plan file

- `factory.py` isn't in the originally sketched directory tree. It was
  added during implementation as the single place that turns
  `config.<x>.provider` strings into concrete classes, shared by
  `api/deps.py`, `ingestion/pipeline.py`, `retrieval/pipeline.py`, and
  `eval/run_eval.py`, so the embedding model and DB connection pool are each
  built once (not once per pipeline); important on an 8GB-RAM box.
- `RetrievalPipeline.retrieve()` has three independent, explicitly-named
  cutoffs, not one overloaded parameter (see that method's module
  docstring for the full flow): `candidate_k` (candidate pool depth fetched
  from the vector store, per branch in hybrid mode), `reranker_top_n` (how
  many a REAL reranker keeps after rescoring, `config.reranker.top_n`,
  inert when `reranker.provider == "none"`), and `generation_context_top_n`
  (how many ranked primary chunks reach generation, applied once,
  uniformly, regardless of which reranker ran, `config.retrieval.
  generation_context_top_n`). This replaced an earlier design where a
  single `rerank_top_n` did both jobs at once and `NoOpReranker` silently
  truncated results even with no reranker configured, see
  `PROJECT_JOURNAL.md`'s 2026-08-13 cutoff-semantics entry and
  `KNOWN_ISSUES.md` for the full bug writeup. `answer()` takes only a
  `candidate_k` override (it always reflects true production config for
  reranking/generation-context) but returns `retrieval_ms`/`generation_ms`/
  `total_ms`/`latency_breakdown_ms` (a per-stage embed/dense_search/
  bm25_search/fusion/rerank/expansion/generation split) alongside
  `answer`/`sources`.
- `eval/run_eval.py` calls `retrieve()` at a broad `candidate_k=
  reranker_top_n=generation_context_top_n=10` *and separately* `answer()`
  at production config, per gold question, by design, so retrieval-quality
  metrics (Recall@5/@10) and latency numbers aren't measured under the
  same (mismatched) configuration, and the broad override never depends on
  what production's own `generation_context_top_n`/`reranker.top_n` happen
  to be configured to.
- A controlled A/B (`experiment_016` baseline vs. `experiment_017`
  cross-encoder candidate, both full 84-question `techfusion`, otherwise
  identical config) found the cross-encoder reranker moves Recall/Hit-Rate/
  MRR by less than noise and adds ~1s of retrieval-stage latency that's
  swallowed by qwen2.5:3b's ~23s generation time, but it also demonstrably
  reordered an otherwise-identical 3-chunk context enough to collapse one
  question's `answer_quality` from 0.947 to 0.053 (same evidence, different
  chunk order, a much more hedging answer). Current recommendation:
  `reranker.provider: none` stays the default, see `PROJECT_JOURNAL.md`'s
  2026-08-13 reranker-experiment entry for the full metric table and
  per-question breakdown.
- A controlled context/output-token-budget A/B/C (`experiment_018/019/020`,
  full 84-question `techfusion`, same qwen2.5:3b/prompt-v2/hybrid+RRF/
  relationship-expansion baseline as the reranker experiment, varying only
  `retrieval.generation_context_top_n` (3→2) and `generation.max_tokens`
  (512→256)) found the run-to-run latency deltas were **not attributable
  to either config knob**, `ms/completion_token` swung ~4-5x across the
  three sequential runs despite near-flat completion-token counts (~37/
  answer), pointing to host-level noise (a shared, CPU-only Windows dev
  box) rather than a config effect; see `PROJECT_JOURNAL.md`'s 2026-08-14
  entry for the diagnosis. The one defensible finding: `max_tokens: 256`
  truncated zero observed answers (max completion length seen: 154
  tokens) and is safe to adopt; the `generation_context_top_n: 2`
  `answer_quality` drop (~5%) was mostly a `KeywordOverlapScorer`
  measurement artifact on hand-inspection, not real answer degradation,
  a 15-question RAGAS sample (`experiment_021/022/023`, ~$0.25 actual
  judge cost) didn't contradict that reading either. Production
  `generation_context_top_n`/`max_tokens` were left unchanged pending
  further review, per the milestone's explicit instruction.
- The judge LLM (`config.judge`, used only for RAGAS scoring) is a
  parallel, independent config tree from `config.generation` (used for
  production answers), with its own provider selection defaulting to a
  *hosted* LLM (`openai`) rather than `ollama`, a deliberate asymmetry,
  not an oversight: RAGAS's judge quality matters enough that local models
  are opt-in/experimental, whereas production generation stays
  local-only by design.
- `document_id` identity is scoped to `(source, dataset_id)`, not `source`
  alone, not in the original plan file, added so the same relative path
  can exist in two datasets without one's re-ingestion clobbering the
  other's document row.
- An image's caption is folded into its own `content_type="image"` chunk's
  text (mirroring the pre-existing chart-fence-plus-caption pattern)
  rather than modeled as an independently-retrievable element, a
  deliberate simplification, not an oversight; means "image → caption" is
  implicit (same chunk), not a queryable parent/child link.
  `parent_chunk_id` itself is a one-pass "nearest preceding prose in the
  same section" heuristic, not a real parse tree.
- Experiment 11/12's fixed retrieval baseline (`hybrid`, `rrf_k=60`,
  `reranker=none`) matches `experiment_010`, not `config/default.yaml`'s
  committed `dense`/`none`, a real, confirmed-with-the-user discrepancy
  between the committed default and the last-recorded experiment, not an
  inconsistency to silently resolve.
- `GenerationConfig.seed: int | None` (default `None`) is a real,
  supported passthrough to Ollama's `options.seed` (confirmed directly
  against the installed `ollama` client's `Options.model_fields`, not
  assumed), `OllamaLLM` only includes `"seed"` in `options` when it's
  not `None`, so leaving it unset reproduces prior non-deterministic
  behavior exactly (no `seed` key sent at all, not a literal `0`, which
  is itself a distinct valid seed). Reproducibility is per-host only
  (same model file/quantization/thread count), not a cross-hardware
  guarantee, documented on the config/wrapper docstrings directly.
  `config/experiments/classic-rag-baseline-v1.yaml`
  (`classic_rag_baseline_v1`, recorded as `experiment_024`) is
  `multimodal-v2-relationship-qwen3b.yaml` (`experiment_015`'s config,
  `qwen2.5:3b` + prompt v2 + hybrid+RRF + relationship expansion) with
  only `temperature: 0.2→0.0` and `seed: (unset)→42` changed; every
  retrieval-side metric came back bit-identical to `experiment_015` and
  generation-quality metrics landed within noise, so this is the intended
  fixed reference point for the upcoming safety/freshness milestone, not
  a one-off. `config/default.yaml` stays untouched (`temperature: 0.2`,
  `seed: null`), same "experiment config overlay, not a promoted
  default" pattern as every prior baseline. See `PROJECT_JOURNAL.md`'s
  2026-08-14 `classic_rag_baseline_v1` entry for the full metric table.
- The safety/freshness milestone's `AuthorizationContext` predicate is
  built as a *second, independent* SQL fragment
  (`vectorstore.pgvector.build_authorization_where_clause`), never merged
  into the pre-existing caller-suppliable `filters` dict/`ALLOWED_FILTER_FIELDS`
  mechanism, a deliberate architectural split (not in the original plan
  file), so a caller-controlled filter can never accidentally widen what
  authorization already narrowed. `classification` was added as a stored,
  filterable field but intentionally does **not** independently gate
  retrieval, `allowed_roles` alone is the authoritative per-document ACL
  gate; layering two independently-authored ACL mechanisms that could
  silently drift apart was judged the bigger risk. See
  `docs/architecture.md`'s "Authorization, Freshness, and Trust" section
  for the full design and its documented limitations.
- **`config/production.yaml` exists and is the recommended real-deployment
  profile.** It turns on `security.auth`/`security.authorization`/
  `security.field_redaction`/`security.rate_limit`/`observability.tracing`
  and bumps `generation.prompt` to `v3`, `config/default.yaml` stays off
  on every security toggle and on `generation.prompt: v1` to preserve its
  "zero-setup demo" role. `agent.classify_prompt_path`/
  `agent.synthesize_prompt_path` used to be a real, accidental exception to
  that split (`default.yaml` stayed on the already-known-buggy v1 versions
  with no equivalent deviation note, unlike the rest of this split),
  that gap is now closed: `default.yaml` was bumped to v2 for both fields
  (see `experiment_032`, "baseline now established," in the
  agentic-rag-reeval section above, and `KNOWN_ISSUES.md`'s "A live demo query
  complied with an embedded prompt injection..." entry for the incident
  that originally motivated `production.yaml`). `production.yaml` no
  longer needs to override either field, the two configs now agree on
  agent prompt versions, and differ only on the deliberate security-toggle
  axis described above.

## The joint-investigation backlog: resolved

`.claude-prompts/25.6-Joint-Investigation-Verified-Report.md` reconciled
two independent adversarial audits into 29 confirmed findings. All 29
were implemented, tested, merged into `main`, and are now live in the
code this file otherwise describes, nothing in the rest of this file
should be read as pre-fix legacy behavior on their account. The work was
split into five PRs (#3 through #7 on GitHub), each built by an
independent agent in its own isolated `git worktree` so all five could be
implemented concurrently without colliding on shared files, then merged
in sequence. Headline fix: `PgVectorStore.replace_document_chunks` now
commits a document's checksum, chunk delete, and chunk insert as one
transaction (previously three separate commits with no rollback, the
CRITICAL silent-data-loss bug DATA-1 in `25.6`); `get_or_create_document_id`
resolves via `INSERT ... ON CONFLICT` instead of a racy SELECT-then-INSERT.
The other 27 findings span API reliability/DoS (`/ingest` no longer blocks
the event loop, SSE streams now support cooperative cancellation, DB
connect/statement timeouts and config-driven pool sizes, rate limiting on
`/ingest`/`/feedback`), agent/MCP correctness (a deterministic low-
confidence response path when a run hits its retry bound with insufficient
evidence, a real MCP session-handshake timeout, tightened case-mutation
routing), and a handful of observability/config-cleanup and scalability
items. Merging surfaced one real integration gap the five isolated
worktrees couldn't see individually, two test doubles (a pool-constructor
mock in `test_pgvector_atomicity.py`, a `FakePipeline.resolve_auth` in
`test_agent_graph_cancellation.py`) didn't know about a sibling batch's
new keyword arguments, fixed in a follow-up commit directly on `main`
once the combined suite caught it. See `PROJECT_JOURNAL.md`'s 2026-09-16
sessions 9 and 10 for the full implementation narrative (including a real
cross-worktree `pre-commit` cache race one agent hit and recovered from,
`KNOWN_ISSUES.md`'s "Five parallel AI agents..." entry) and `KNOWN_ISSUES.md`'s
ingestion-atomicity entry for the before/after test numbers.

## LangGraph experiment (learning exercise, not production)

`langgraph_experiment/` is a self-contained, isolated LangGraph agent
workflow built to learn when LangGraph earns its keep over this
codebase's own hand-rolled bounded harness (`rag.agent.graph.run_agent`,
untouched by it). It is not imported by `src/rag/` or `rag.api` anywhere,
and is not a migration path, production still runs the custom harness.
It reuses `rag.mcp.business.store`/`rag.retrieval.pipeline`/
`rag.factory` directly rather than reimplementing RAG or the case-store
business rules. Runs in its own venv (`.venv-langgraph/`, gitignored):
`langgraph>=1.2` pins `langchain-core>=1.4,<2`, incompatible with this
repo's pinned `langchain>=0.3,<0.4` (confirmed via `pip install
--dry-run`, not assumed), see `langgraph_experiment/README.md`'s
"Dependency isolation" section. That README is the full write-up:
architecture, exact run commands, a worked human-in-the-loop
approve/reject/restart-resume demo, an idempotency-guarantee breakdown
(LangGraph vs. its SQLite checkpointer vs. the business-action layer), a
reasoned comparison against the custom harness, and a 15-concept /
20-question interview reference on LangGraph itself. See `KNOWN_ISSUES.md`'s
"A durable LangGraph checkpoint proved nothing about the durability of
what it calls into" entry for a real gap found while building its
human-in-the-loop demo (the synthetic case-store backend's in-memory,
per-process data has no persistence of its own, independent of whether
the LangGraph checkpoint wrapping it is durable).

**Production hardening.** A second, later phase (the same
`.claude-prompts/31.0-LangGraph.txt` file's appended "PRODUCTION WORKFLOW
ADDITIONS" section) added, still with zero changes to `rag.agent.graph`
or to the original phase's own public behavior: a Postgres-backed
checkpointer (`checkpointer.postgres_checkpointer`, preferred over
SQLite, confirmed real/lightweight, a 52 KB `langgraph-checkpoint-
postgres` wheel, by round-tripping a live interrupt/resume/restart
cycle before writing any code); a Postgres-backed idempotency ledger
(`idempotency.py`'s `ActionLedger`, keyed by a UUID `operation_id` minted
once per write request) that *proves*, via a manually-reproduced crash
window (mutation executes, ledger-completion write is skipped, then the
attempt is retried), that a resumed/retried write action cannot
double-mutate, safe only because `update_case_status`'s own
`already_in_status` branch is naturally idempotent, which the ledger
never manufactures on its own; server-side (never CLI-side-only)
approval expiration and a separate per-workflow timeout, plus explicit
cancellation, each its own `ApprovalState`/`termination_reason`; a
`RetryPolicy` on the write node itself, safe only because the ledger
check runs first on every attempt including an automatic retry; explicit
checkpoint retention (`cli/retention_cleanup.py`, terminal-state threads
only, never a still-paused one); an isolated, secrets-free audit trail
(`audit.py`, deliberately not an extension of `rag.audit`'s shared
`AuthEventType`); and an OpenTelemetry span per node plus Prometheus
workflow counters (`observability.py`, its own dedicated registry,
reusing `rag.observability.tracing` unmodified). One real correctness
bug caught by reasoning before running anything: `interrupt()` raises
`langgraph.errors.GraphInterrupt` (a genuine `Exception` subclass) on
every ordinary approval pause, which a naive `except Exception:` around
the new per-node metrics wrapper would have miscounted as a node
*failure*, fixed by excluding `GraphBubbleUp` (confirmed as
`GraphInterrupt`'s parent class) before the generic failure branch. See
`KNOWN_ISSUES.md`'s "A GraphInterrupt would have been silently miscounted as a
node failure in Prometheus metrics" entry, `langgraph_experiment/
README.md`'s "Production hardening" section for the full writeup, and
`PROJECT_JOURNAL.md`'s 2026-09-29 session 18 entry for the implementation
narrative (45/45 tests passing with Postgres up, 40/45 self-skipping
cleanly without it).

## Multi-agent LangGraph experiment (learning exercise, not production)

`multi_agent_experiment/` (branch `experiment/multi-agent`, branched from
`experiment/langgraph-agent`) is a second, isolated LangGraph exercise on
top of `langgraph_experiment/`: a Coordinator/Router node plus two
genuinely distinct specialist agents, a Knowledge Agent (`RetrievalPipeline.
retrieve()` only) and a Business Agent (`rag.mcp.business.store` only),
with parallel `Send`-based fan-out for queries needing both, a bounded
knowledge-retry loop routed back through the Coordinator, and an Evidence
Critic scoped to exactly two mechanical responsibilities (one bounded
retry decision, a no-fabrication guard) rather than added for cosmetic
multi-agent flavor. Like `langgraph_experiment/`, this is not a migration
path: `rag.agent.graph.run_agent` (production) and `langgraph_experiment/`
(the single-agent exercise) are both untouched, and nothing under
`multi_agent_experiment/` is imported by `src/rag/`, `rag.api`, or
`langgraph_experiment/`. Runs in the **same** `.venv-langgraph/` venv
`langgraph_experiment/` already built, no new dependency.

- **The entire business-write branch is reused unmodified.**
  `business_agent.py` rebinds `langgraph_experiment.nodes.
  select_case_tool`/`make_execute_case_read_node`/`validate_write_
  request`/`make_wait_for_approval_node`/`make_execute_write_action_
  node`/`synthesize_write_result` (plus `langgraph_experiment.routing`'s
  two conditional edges) directly, not wrapped, not reimplemented.
  `GraphDeps`/`build_default_deps` (`langgraph_experiment.wiring`) is
  reused wholesale too, so this package adds no new dependency-injection
  type. `rag.mcp.business.store`'s authorization/transition/approval
  enforcement and `RetrievalPipeline`'s authorization/freshness/
  redaction enforcement are both untouched, this graph adds zero
  authorization logic of its own anywhere.
- **Tool isolation is verified statically, not just asserted.**
  `knowledge_agent.py`/`business_agent.py`/`orchestrator.py` are three
  separate modules specifically so a test
  (`tests/test_tool_isolation.py`) can `ast`-parse each one's actual
  `import`/`from` statements and assert the forbidden module never
  appears, a specialist cannot call the other's tool because the code
  that could do so was never written in that file, not because a runtime
  check happens to forbid it.
- **A real LangGraph correctness bug was found and fixed while building
  the "mixed" (parallel) route**, documented in full in `KNOWN_ISSUES.md`: the
  Knowledge Agent branch is one hop from the coordinator's `Send`
  dispatch, the Business Agent's read branch is three
  (`select_tool -> execute_read -> evaluate_read`); with a plain
  `add_edge(..., "merge")` join and no further guard, LangGraph triggers
  the join as soon as *either* branch's edge fires, not once *all* of
  them have, for unequal-length parallel branches this ran `merge`
  (and everything downstream) once early with partial state, then again
  with the complete state, doubling an LLM call and briefly answering
  from half the evidence. Fixed with `graph.add_node("merge", ...,
  defer=True)`, confirmed against `langgraph==1.2.11`'s own docstring
  ("defer: whether to defer execution of the node until the run is
  about to end"), not assumed from the parameter name. Both the buggy
  and fixed behavior are reproduced permanently in `tests/test_probe_
  langgraph_primitives.py` (a minimal two-branch graph, independent of
  this package's own topology) so a future `langgraph` upgrade that
  changes this behavior fails there first.
- **Field-ownership discipline replaces most reducer logic.** Each
  specialist only ever writes its own namespaced `MultiAgentState` keys
  (`knowledge_*` vs. `case_*`/`business_*`), so parallel branches never
  write-race on the vast majority of fields; the two fields that
  genuinely can be written more than once per run (across a
  critic-requested retry) use an explicit `Annotated[list, operator.add]`
  reducer, while `citations`/`tool_call_log` are deliberately plain
  overwrites recomputed fresh by `merge` on every invocation (so a
  retry's fresher evidence replaces stale evidence rather than being
  unioned with it).
- **Deterministic routing, same rationale as the single-agent
  experiment.** `orchestrator.coordinator`'s knowledge/business/mutation
  classification is regex-based (`_looks_like_case_mutation_request`
  reused from `rag.agent.graph`; case-id/target-status extraction reused
  from `langgraph_experiment.nodes`), not LLM-driven, for the same
  reproducibility reason `langgraph_experiment/`'s own `classify` node
  is deterministic. Caught a real routing bug during development: an
  early knowledge-cue regex matched generic interrogative phrases like
  "what is," misrouting a pure business question ("What is the status of
  CASE-1001?") into unnecessarily also selecting the Knowledge Agent,
  fixed by requiring a cue that signals "references the knowledge base,"
  not merely "is a question," caught by `tests/test_coordinator_
  routing.py` before it ever reached a scenario-level test.
- **A dedicated, deterministic evaluation harness**
  (`eval/run_multi_agent_eval.py`, `eval/multi_agent_gold.jsonl`, 12
  rows) runs with **zero external services** (a small synthetic
  knowledge base plus the real, self-contained `rag.mcp.business.
  store`), covering knowledge-only, business-only, mixed, business
  mutation (executed/rejected/invalid-request), authorization denial
  (same-tenant and cross-tenant), an explicit unnecessary-agent check,
  and a specialist-failure/bounded-retry row. Reports `routing_accuracy`,
  `unnecessary_agent_rate`, `answer_correctness`, `citation_grounding`,
  tool/LLM call counts, latency, and `security_failures` (a zero-
  tolerance ceiling, not a quality score, matching the main system's own
  `sensitive_data_false_redaction_rate` convention).
- **Three-way comparison** (`cli/compare_harness.py`) against both the
  production custom harness and the single-agent LangGraph experiment,
  across code complexity, state visibility, branching, retry handling,
  checkpointing, pause/resume, latency, debugging, security surface,
  testability, and (new to this comparison) coordination, concluding
  multi-agent is justified only when a genuine need for two-or-more
  differently-toolable specialists coincides with the single-agent
  experiment's own pause/resume justification, not by default; three of
  this experiment's own five required scenarios (knowledge-only,
  business-only, mutation) show multi-agent adding hops and code for
  zero benefit over the single-agent design, which the eval harness's
  own `unnecessary_agent_rate: 0.000` is the concrete regression guard
  for.
- **Not run against real infra.** `cli/compare_
  harness.py rag`/`mixed` (the timed comparisons needing `make up` +
  native Ollama + an ingested dataset) were written and import-checked
  but not executed against live Postgres (Docker Desktop
  was not running), flagged directly in `README.md`'s "What wasn't
  verified" section rather than presented as measured, matching this
  project's established "flag the gap, don't imply more was checked than
  was" convention.
- See `multi_agent_experiment/README.md` for the full architecture
  diagram, all five required scenarios with real reproducible traces
  (`cli/scenario_walkthrough.py`, zero external services), the complete
  security analysis, the 20-concept and 20-question interview sections,
  and what should NOT be merged into `main` absent a real production
  need for parallel multi-specialist dispatch.

## Kubernetes deployment (learning experiment)

`k8s/` (branch `experiment/kubernetes-deployment`, not merged to `main`
as of this writing) deploys the existing stack onto a local Kubernetes
cluster to practice production Kubernetes concepts hands-on, a learning
exercise, not a production manifest set, and not a redesign of the
application: every manifest targets the same images, `/health` endpoint,
`*_env_var` config conventions, and same-origin frontend/nginx design
already described above, with zero application code changes. `k8s/base/`
holds the core app (a `postgres` StatefulSet with a PVC, `rag-api`
Deployment + Service + HPA, `frontend` Deployment + Service, an `Ingress`,
a `ConfigMap`/`Secret`, and `NetworkPolicy`, composed via `kustomize`),
shared by both cluster options below; `k8s/jobs/` is a one-shot
schema-init `Job`; `k8s/observability/` is an optional Prometheus/
Grafana/Jaeger stack reusing the existing `observability/` config files
unmodified. Ollama stays native on the host exactly as it does for the
Docker Compose deployment, reached from inside the cluster via an
`ExternalName` Service (`ollama-host`), not containerized.

**Two cluster options, sharing `k8s/base/`.** `k8s/kind/` (`kind-cluster.yaml`,
a 3-node topology, plus its own `README.md`/`EXERCISES.md`) is the
original, reference environment, `k8s/base`'s defaults
(`host.docker.internal`, image tag `:kind`) already match its
conventions, so it applies `k8s/base` directly with no overlay.
`k8s/minikube/` (`README.md`/`EXERCISES.md` plus a small `kustomization.yaml`
overlay on top of `../base`) is a working alternative for hosts where
`kind` can't bring up a control plane at all (see the finding below),
its one real manifest-level difference from kind is
`ollama-external-service-patch.yaml`, which repoints the `ollama-host`
`ExternalName` Service from `host.docker.internal` to
`host.minikube.internal` (minikube's own documented, driver-agnostic
mechanism for reaching the host; confirmed both names actually resolve
identically on this project's `--driver=docker` minikube instance, since
it happens to run on Docker Desktop too, but `host.minikube.internal` is
the correct choice rather than depending on that coincidence). `k8s/minikube/EXERCISES.md`
is a delta document, not a duplicate, 17 of 18 exercises in
`k8s/kind/EXERCISES.md` are unmodified `kubectl` commands that work
identically on either cluster; only the node-failure exercise is kind-only
(minikube here is single-node). `k8s/README.md` is the shared setup
walkthrough and runbook (mental model, applying manifests, ingest,
observability, debugging, teardown); `k8s/kind/README.md` and
`k8s/minikube/README.md` hold only what's genuinely cluster-specific
(cluster creation, image loading, ingress, metrics-server, teardown
commands). `k8s/LEARNING.md` is a concept-by-concept reference plus
interview questions grounded in what's actually implemented.

Three real findings surfaced while building this, all in `KNOWN_ISSUES.md`:
`GET /health` always returns HTTP 200 even when a dependency is
unreachable (degradation is reported only in the response body), which
means a Kubernetes readiness probe built on it cannot detect that class
of failure at all, a real, documented gap in the application's own
health contract, not something this deployment could paper over without
changing application code it was explicitly told not to change. Kustomize's
`configMapGenerator` refused to read `scripts/init_db.py`/
`observability/*` because those files sit outside the referencing
kustomization's own directory tree (a real security-boundary limitation
in the installed `kubectl` version, with no `--load-restrictor` escape
hatch available), worked around with plain `kubectl create configmap
--from-file` instead of restructuring the repo to satisfy the tool. And,
separately, `kind create cluster` (and Docker Desktop's own
built-in Kubernetes) turned out to hang indefinitely inside `kubeadm
init` on the primary development host specifically, a `containerd`/
`runc` `CreateContainer` hang tied to that host's WSL2 kernel build, not
a manifest/config problem, diagnosed across fifteen-plus configurations
over several sessions (see KNOWN_ISSUES.md's "kind (and Docker Desktop's own
built-in Kubernetes) cannot bring up a control plane on this host, but
minikube's docker driver can" entry for the full evidence trail).
`minikube start --driver=docker`, on the identical kernel and Docker
Desktop install, does not hit the hang (its node image bundles a
different `containerd` build), verified end to end, not just "the
control plane pods exist": every `kube-system` pod reached `Running`,
and a throwaway `nginx` deployment completed a full image-pull-through-
`Running` lifecycle and was reachable over its `ClusterIP` service by DNS
name from another pod. `k8s/kind/` is kept fully intact as the reference
3-node environment (not modified by this finding, since a host where
`kind` works has no reason to prefer minikube), with `k8s/minikube/`
added alongside it as a separately-documented, separately-tested
alternative, not a replacement.

## Distributed-state experiment (learning exercise, not production)

`distributed_state_experiment/` (branch `experiment/distributed-state`,
branched from `experiment/kubernetes-deployment`, not `main`) is a
self-contained learning exercise reproducing and fixing a horizontal-
scaling failure mode: behavior that is correct with one API replica but
silently wrong with more than one. Same "learning exercise, not
production" status as `langgraph_experiment/`/`multi_agent_experiment/`/
`k8s/`: not merged, not imported by `src/rag/`, and not a redesign of
the existing architecture.

- **The bug, demonstrated before any fix existed.** `api/deps.py:
  get_rate_limiter()`'s `slowapi.Limiter` is an in-memory, process-local
  singleton (already documented as such in `RateLimitConfig`'s own
  docstring). With 3 replicas behind a load balancer, each replica
  enforces its own independent counter, so a tenant limit of 100
  requests/minute becomes 300 in aggregate, reproduced with real
  numbers, not just reasoned through: 3 separate OS processes, each
  running the in-memory limiter, let 300/300 requests through against an
  intended limit of 100, every run.
- **The fix: a config-gated Redis backend, off by default.**
  `RateLimitConfig` gained `backend` (`memory` default | `redis`),
  `redis_url_env_var`, `strategy`, `redis_fail_mode`
  (`fail_open`|`fail_closed`), `redis_key_prefix`; `get_rate_limiter()`
  builds a Redis-backed `slowapi.Limiter` (via the `limits` library's
  `RedisStorage`, an atomic Lua-script increment, not a plain
  `INCR`-then-`EXPIRE`) only when `backend="redis"`. Default behavior
  (`backend="memory"`) is byte-identical to before this milestone,
  same convention as every other feature flag in this codebase. Rerun
  against a genuine `redis:7-alpine` container: 100/100 exact on 3
  consecutive runs (vs. 300/300 for the in-memory 3-replica control).
- **Fail-open vs. fail-closed is a real, explicit, configurable choice**,
  not a default assumed correct: a rate limiter protects availability,
  it is not a security boundary the way `security.field_redaction`'s
  fail-closed-on-missing-identity design is (that comparison is drawn
  out explicitly in the experiment's own README), so the tradeoff is
  argued rather than copied from that precedent.
- **Distributed background processing**: a Redis Streams-backed
  ingestion job queue (`job_queue/`), reusing this repo's real
  `IngestionPipeline`/`Writer`/`factory` wiring for the actual
  chunk/embed/store step rather than a bespoke worker-side
  reimplementation, and reusing the existing `(source, dataset_id)` +
  checksum document-identity scheme as the job's idempotency key rather
  than inventing a parallel concept. Consumer groups
  (`XREADGROUP`/`XACK`/`XCLAIM`/`XPENDING`) give claim/visibility
  semantics and crash recovery for free. Explicitly at-least-once +
  idempotent, not exactly-once, the same practical answer this
  codebase's own `replace_document_chunks` transaction and
  `get_or_create_document_id`'s `INSERT ... ON CONFLICT` already
  represent for a different race, not a new pattern invented for this
  experiment.
- **Two real bugs found only by running real code against a real (if
  simulated) backend, never caught in design review**: `XLEN` does not
  shrink on `XACK` (it counts entries ever added, not remaining work),
  so an early `queue_depth()` built on it read nonzero after every job
  finished, fixed with a separate, explicit `backlog_depth()` counter
  that backpressure and the Prometheus gauge actually read. And
  `IngestionJobQueue`'s `key_prefix` used to default to one global
  constant regardless of `stream_name`, letting two unrelated queues
  silently share one dead-letter counter, fixed by deriving the
  prefix from `stream_name`. See `KNOWN_ISSUES.md` for both writeups.
- **Distributed lock vs. DB-constraint-based idempotency**: a Redis
  `SET key val NX PX ttl` lock (with a Lua-script, token-checked
  release) is demonstrated against a reproduction of this repo's own,
  already-fixed `get_or_create_document_id` race. It works, but the
  experiment's own conclusion (see its README) is that a lock is an
  opt-in convention every caller must remember to use, while
  `INSERT ... ON CONFLICT` is enforced by Postgres itself for every
  caller including a future one nobody remembered to wrap, prefer an
  atomic single-operation primitive on the actual resource over
  layering a separate lock on a non-atomic multi-step operation,
  whenever one exists. A lock earns its place when the protected
  operation spans multiple non-atomic steps no single DB primitive can
  express as one operation (e.g. read-then-call-an-external-API-then-
  write), not the shape of this repo's own document-identity race.
- **What was actually executed vs. reasoned-through, precisely
  documented (not glossed over)**: Experiments 1-12 and the Part 4
  benchmark ran for real as separate OS processes, first against
  `fakeredis.TcpFakeServer` (a real RESP-protocol TCP server, pure
  Python, no Docker needed), then a second time against a genuine
  `redis:7-alpine` container once Docker access was confirmed free of
  concurrent Docker work elsewhere. A single 104/100-allowed
  fakeredis run (1 of 3) was reported honestly rather than hidden, and
  the real-Redis rerun (100/100 on 3/3) is treated as suggestive
  evidence for the leading fixed-window-boundary hypothesis, not proof.
  Kubernetes experiments 11-12 (`k8s/distributed-state/` manifests)
  were checked (`minikube status` confirmed the shared `rag-learning`
  cluster fully stopped) and deliberately not applied, to avoid
  disturbing state the sibling `k8s/`/`experiment-helm` work might
  depend on, documented as "checked and deliberately skipped," the
  same honesty convention every prior experiment in this repo uses.
  Importing `rag.factory` was independently measured at ~19-22 minutes
  in this sandbox (twice, non-flaky), almost certainly first-touch
  antivirus DLL scanning of the ML dependency tree, not a code property, and is the reason the production Redis-rate-limiter wiring tests
  and Part 1's real two-pod HTTP demo were run only once/not run live,
  respectively, with a substitute unit test used for the latter.
- See `distributed_state_experiment/README.md` for the full design (all
  5 parts, all 12 experiments, the benchmark table, the concurrency
  concept explanations) and `distributed_state_experiment/
  INTERVIEW_QUESTIONS.md` for 15 interview Q&A grounded in what was
  actually built.

## Helm chart (learning experiment)

`helm/` (branch `experiment/helm`, branched from
`experiment/kubernetes-deployment`, not merged to `main`) repackages the
hand-written `k8s/base/` manifests as a small, configurable Helm chart,
`helm/rag-system/`, a packaging exercise, not a redesign: same images,
probes, env vars, and security posture as `k8s/base`. Configurable image/
replicas/resources/probes/HPA/ingress/Postgres persistence via
`values.yaml`, plus `values-dev.yaml`/`values-prod.yaml` overlays and the
same production-grade hardening `k8s/` already has (PodDisruptionBudgets,
tightened `securityContext`, RBAC/ServiceAccount, `NetworkPolicy`,
checksum-triggered rollouts). `helm/README.md` is the full writeup:
Helm concepts taught against this chart, live demos (install, values
override, config-only rollout, image upgrade, broken upgrade + rollback,
PDB-blocked eviction, topology spread) with real captured output rather
than a fabricated happy path, CI/CD and GitOps fit, and 20 interview
questions. `helm/rag-system/README.md` is the chart-scoped values-table
reference. Eight real, confirmed issues were found and diagnosed
installing/upgrading this chart on a live minikube cluster, five fixed
in the chart's templates (a postgres init container missing
`DAC_OVERRIDE`, a frontend probe missing `timeoutSeconds`, a schema-init
Job defaulting to disabled that produced a misleading "vectorstore
unreachable" readiness failure, that same Job's hook phase being
`pre-install` when it depends on a Service the chart itself creates, and
a default-deny `NetworkPolicy` silently dropping that Job's own traffic),
three documented as operational/usage lessons rather than chart bugs
(Helm's default hook-wait timeout being too tight for this host, a
zero-downtime rollout strategy deadlocking under this shared cluster's
real memory contention, and a bare `helm rollback` landing on another
failed revision instead of the last known-good one), see `KNOWN_ISSUES.md`
for the condensed version of each.
