# Metrics Inventory

A reference map of every metric this codebase computes, grouped by the
pipeline layer it measures. Sourced directly from `src/rag/eval/` (mainly
`run_eval.py`, `retrieval_attribution.py`, `run_agent_eval.py`,
`ragas_scorer.py`, `corpus_lineage.py`) — this file doesn't introduce any
new metric, it indexes what's already computed and where.

None of this runs automatically. `eval/run_eval.py --gold ... --dataset-id
...` is the main entrypoint; flags/other scripts are noted per section.

## 1. Embedding layer

There is no metric that scores the embedding model in isolation (e.g. no
intrinsic embedding-quality benchmark). Embedding quality is only
observable indirectly, through retrieval-layer metrics (section 2) — a
worse embedding model shows up as lower Recall@k/MRR, not as a
standalone number.

## 2. Retrieval layer

Computed by `eval/run_eval.py:evaluate()` from a broad, fixed
`candidate_k=10` retrieval per gold question (see `eval/metrics.py`):

| Metric | Direction | What it measures |
|---|---|---|
| `recall@5`, `recall@10` | higher better | Fraction of gold `relevant_documents` found in the top-k, path-suffix matched |
| `hit_rate@5`, `hit_rate@10` | higher better | Whether *any* relevant document was found in the top-k (0/1 per question, unlike recall which averages over multiple relevant docs) |
| `mrr` | higher better | Mean reciprocal rank of the first relevant document |
| `content_type_breakdown` | — | Recall@5/@10 + hit_rate@5, bucketed by the gold file's *authored* `content_type` (e.g. `text_only`, `table`, `image_only`, `relationship_aware`) |
| `reference_context_analysis.supporting_context_hit_rate` | higher better | Among questions where the right *document* was retrieved, how often the specific supporting passage (verbatim substring match against `reference_contexts`) was too — buckets A (doc+passage), B (doc only), C (missed) |
| `relevant_image_hit_rate` | higher better | Among gold rows with `relevant_images`, whether a retrieved chunk's resolved image asset matched one |
| `relationship_expansion_contribution_rate` | — | Among `requires_relationship_expansion=true` rows, fraction where an `origin="expanded"` chunk supplied supporting context the pre-expansion set alone didn't. Always 0.0 when `relationship_expansion.enabled=false` |

### Retrieval attribution (`eval/retrieval_attribution.py`, `--attribution` path)

Observability-only — independently fetches dense and BM25 rankings (never
reranked, never expanded) and fuses them via RRF, regardless of
`config.retrieval.provider`. Run via
`python -m rag.eval.retrieval_attribution`.

| Metric | What it measures |
|---|---|
| `metrics_by_retriever` | Recall@5/@10, Hit Rate@5/@10, MRR computed separately for `dense`, `bm25`, and `hybrid` (RRF-fused) rankings, over the identical gold set |
| `contribution_buckets` | Per-question classification: `both_success` / `dense_only_success` / `bm25_only_success` / `neither_success` |
| `rrf_impact` | Per-question classification of what fusion did to the first-relevant-doc rank: `rescued` / `improved` / `unchanged` / `degraded` / `still_missed` / `not_applicable` |
| `reference_context_by_retriever` | Same A/B/C supporting-passage bucket as section 2's table, computed per retriever |
| `breakdowns` | `contribution_buckets`/`rrf_impact`/retriever metrics bucketed by `content_type`/`question_type`/`difficulty`/relationship-awareness |

## 3. Generation layer

Computed by `eval/run_eval.py:evaluate()` when `run_generation=True`
(the default; `--skip-generation` disables this whole section):

| Metric | Direction | What it measures |
|---|---|---|
| `answer_quality.mean_overall` / `mean_answerable` / `mean_unanswerable` | higher better | `KeywordOverlapScorer` — fraction of reference-answer keywords (len>3) that also appear in the generated answer. A cheap placeholder, not a faithfulness judge — see RAGAS (section 3a) for a real semantic scorer |
| `latency_ms.retrieval_mean` / `generation_mean` / `total_mean` | — | Mean per-question latency |
| `latency_breakdown_ms` | — | Per-stage mean latency (embed/dense_search/bm25_search/fusion/rerank/expansion/generation), from `RetrievalPipeline._retrieve_timed` |
| `token_usage.prompt_tokens_mean` / `completion_tokens_mean` | — | From Ollama's `prompt_eval_count`/`eval_count`, when present |
| `refusal_behavior.correct_refusal_rate` | higher better | Among `unanswerable=true` gold rows, fraction whose answer matched a literal refusal-phrase heuristic |
| `relationship_expansion_utilization.answer_appears_to_use_expanded_content_rate` | — | Among questions where an expanded chunk reached generation, whether the answer's wording echoes expanded-only content (keyword-overlap heuristic) — distinct from `relationship_expansion_contribution_rate`, which is retrieval-only and gold-annotation-gated |
| `vision_behavior_breakdown.counts` | — | Heuristic triage of `requires_vision=true` answers in text-only mode: `correct_refusal` / `hallucinated_answer` / `caption_leak_success` / `incorrect_or_missing` |

### 3a. RAGAS (LLM-judge, hosted API, optional — `eval/run_ragas_eval.py`)

Not run by default (costs real money against a hosted judge — `openai`/
`anthropic` by default, see `config.judge`). `ragas_aggregate` reports:

| Metric | What it measures |
|---|---|
| `faithfulness` | Whether the answer is supported by the retrieved context |
| `answer_relevancy` | Whether the answer addresses the question |
| `context_precision` | Whether retrieved context is relevant (judged against `expected_answer`) |
| `context_recall` | Whether retrieved context covers what's needed to answer |
| `answer_correctness` | Whether the answer matches the expected answer |
| `noise_sensitivity` | Robustness to irrelevant retrieved content (`mode=relevant`) |
| `factual_correctness` | Factual overlap with reference (`mode=f1`) |

Not validated against human labels by default — see
`scripts/generate_manual_review.py`/`compare_ragas_manual.py`.
`ragas.cache` in the report also tracks `hits`/`misses`/
`avoided_cost_estimate` for the judge-call cache
(`eval/ragas_cache.py`).

## 4. Safety / security layer

All in `eval/run_eval.py`'s `safety` dict, each shaped
`{"direction", "note", "count", "rate"}` (or `"mean"` for the one
quality-valued entry). Populated only when the gold set has matching
rows (`if <records>:` gates), except `duplicate_sensitive_field_miss_rate`
which always runs.

| Metric | Direction | What it measures |
|---|---|---|
| `document_unauthorized_retrieval_rate` | lower better | A purely-forbidden document (not also in `allowed_documents`) appeared in auth-filtered retrieval |
| `cross_tenant_leakage_rate` | lower better | Same check, restricted to `safety_category=cross_tenant_access` |
| `stale_document_error_rate` | lower better | Retrieved a document version other than `expected_document_version` when `requires_current_document=true` |
| `current_document_retrieval_accuracy` | higher better | Retrieved the *correct* current version when `requires_current_document=true` (retrieval-only) |
| `current_document_answer_quality` | higher better | `answer_quality` restricted to the same `requires_current_document=true` rows (generation-side, kept separate from the retrieval-only metric above) |
| `prompt_injection_success_rate` | lower better | Among `injection_source=user_prompt` rows, whether the answer shows signs the injected instruction worked |
| `retrieved_prompt_injection_success_rate` | lower better | Same, for `injection_source=retrieved_document` |
| `sensitive_data_leakage_rate` | lower better | A synthetic-secret literal pattern leaked into an answer that should have refused |
| `refusal_accuracy` | higher better | Among rows expected to refuse, fraction actually detected as refusals |
| `false_refusal_rate` | lower better | Among rows *not* expected to refuse, fraction refused anyway (over-aggressive safety layer) |
| `poisoned_source_selection_rate` | lower better | Among `safety_category=knowledge_base_poisoning` rows, whether answer quality dropped below threshold (proxy for being swayed by an untrusted claim) |
| `sensitive_data_authorized_disclosure_accuracy` | higher better | An *authorized* caller's answer still discloses a value it's entitled to see (catches over-redaction) |
| `sensitive_data_false_redaction_rate` | lower better | A field was redacted even though the caller's role was actually authorized for it — should be 0 by construction; non-zero is a real bug |
| `encoded_extraction_success_rate` | lower better | A sensitive literal was still recoverable via base64/reversed/split-text extraction attempts |
| `unauthorized_metadata_leakage_rate` | lower better | A forbidden document or sensitive value leaked via *metadata* (`attachment_name`/`section_path`), not just `content` |
| `provider_egress_policy_violation_rate` | lower better | A source with an unredacted restricted sensitive field would violate the egress policy if sent to a hosted provider (checked independent of whether the policy toggle was actually on) |
| `forged_role_acceptance_rate` | lower better | A request body forging a more-privileged tenant/role would have won over a verified JWT identity — should be 0 by construction |
| `duplicate_sensitive_field_miss_rate` | lower better | Corpus-level (not gold-row-driven): a sensitive literal duplicated across chunks, or missing its ingestion-time tag on at least one occurrence |
| `authentication_failure_acceptance_rate`† | lower better | Fraction of adversarial JWTs (missing/expired/bad-signature/malformed/wrong-issuer/wrong-audience) that `POST /query` incorrectly accepted |
| `oversized_request_rejection_accuracy`† | higher better | Fraction of oversized query/top_k/filters payloads correctly rejected with a 4xx |

† Only computed via `evaluate_authentication_boundary_probes()`, opt-in
via `--include-api-probes` (exercises the live FastAPI app through a
`TestClient`, not `RetrievalPipeline` directly) — not part of the default
`evaluate()`/`run()` report.

## 5. Agentic RAG layer (`eval/run_agent_eval.py`)

Deterministic/local only (no RAGAS, no hosted judge). Run via
`python -m rag.eval.run_agent_eval` against
`data/eval/agentic_extension_gold.jsonl`.

| Metric | What it measures |
|---|---|
| `routing_accuracy` | Whether the graph's `classic_rag` vs `agent` routing decision matched the gold-expected route |
| `unnecessary_agent_rate` | Among rows gold-labeled "tool not needed", fraction that still routed to the agent path |
| `tool_selection_accuracy` | Whether the expected tool set is a subset of the tools actually called (not exact-sequence match) |
| `tool_success_rate` | Fraction of individual tool calls that succeeded |
| `average_tool_calls` | Mean tool calls per agent-routed question |
| `evidence_sufficiency_accuracy` | Among rows expecting an insufficient-evidence retry, fraction where `retrieval_attempts >= 2` (proxy — `AgentState` only retains the final sufficiency decision) |
| `retry_success_rate` | Among those retried questions, fraction that terminated via `synthesized` (succeeded after retry) |
| `max_step_termination_rate` | Among rows expecting step-bound termination, fraction that actually hit a bound-termination reason |
| `citation_support_rate` | Fraction of answers where every citation path-suffix-matches a `relevant_documents` entry (grounding, not correctness) |
| `agent_answer_correctness.mean_score` | `KeywordOverlapScorer` against `expected_answer`, for answerable rows |
| `agent_latency_ms` | Mean total latency, split into overall / agent-routed / classic-routed |
| `agent_token_usage` | Mean prompt/completion tokens |
| per-`agentic_category` breakdown | `routing_accuracy` and `average_tool_calls` bucketed by gold `agentic_category` |

## 6. Corpus / experiment lineage (not a quality metric — provenance)

`eval/corpus_lineage.py`, attached to every `run()` report under
`corpus_lineage`: `document_count`, `chunk_count`, `image_count`,
`active_document_count`, `superseded_document_count`, `tenant_count`,
`gold_record_count`, `gold_file_sha256`, `corpus_digest` (sha256 over
sorted `source:checksum` pairs — proves two experiments scored
byte-identical source files). Used to confirm two experiment records
claiming the same dataset actually compared the same corpus, not to
score answer/retrieval quality itself.

## 7. Experiment tracking (MLflow / `experiments/results/*.json`)

`scripts/record_experiment.py` doesn't compute new metrics — it flattens
a subset of the above (config fields + retrieval/hit-rate/MRR/
answer_quality/RAGAS aggregates/every `safety_*` rate/corpus lineage)
into one comparable record per experiment, logged to MLflow and written
to `experiments/results/<id>.json`. `scripts/compare_experiments.py`
renders these as a table (also what backs README's Benchmarks section).

## What's *not* covered

- No intrinsic embedding-quality metric (section 1).
- No cost/latency SLO tracking beyond the means above (no p50/p95/p99).
- No RAGAS metric runs automatically or on a schedule — always a manual,
  cost-aware invocation.
- No production monitoring/alerting on any of these metrics — they're
  all offline-eval, run against a gold set, not live traffic.
