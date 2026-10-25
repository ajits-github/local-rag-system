# Evaluation & RAGAS

`eval/run_eval.py` has no RAGAS/MLflow dependency: it always computes
Recall@5/@10, hit-rate, MRR, a keyword-overlap answer-quality score, and
the deterministic multimodal/relationship/safety metrics against a
mandatory `--dataset-id`-scoped gold set. RAGAS (LLM-judge scoring of
faithfulness, answer relevancy, context precision/recall, answer
correctness, and more) and MLflow experiment tracking are both
independently optional layers on top, gated behind their own extras
(`pip install .[ragas]` / `.[mlflow]`) and never required for the offline
serving path.

See [Metrics reference](../metrics.md) for the full metric-by-metric
breakdown (what each one measures, its known limitations, and which
report section it lives in), and [CI evaluation gates](../ci_eval_gates.md)
for the fast, deterministic subset of this that runs on every PR.

--8<-- "README.md:docs-benchmarks"

API reference: [Evaluation](../reference/eval.md).

## CI evaluation gates (ci-eval-gates milestone)

Adds a fast, deterministic quality/security gate on top of the pre-existing
`code-quality`/`unit-tests`/`integration-tests`/`docker-build` CI jobs.
Reuses existing evaluation and test infrastructure end to end (`rag.eval.
run_eval`, the existing `tests/unit` suite) rather than a parallel
evaluation framework; see `docs/ci_eval_gates.md` for the full design,
this is the condensed version.

- **Two new jobs**, `.github/workflows/ci.yml`: `security-regression-gate`
  (a curated ~230-test subset of `tests/unit`, authorization, field-level
  redaction, DoS/rate limits, MCP identity/business-tool authorization and
  approval logic, agent step/tool-call/retry bounds, insufficient-evidence
  and malformed-tool-argument handling, the case-mutation routing fix,
  citation attribution, no Postgres/Ollama, runs in seconds) and
  `retrieval-eval-gate` (ingests a small tracked sample corpus into an
  ephemeral CI-only Postgres, runs `run_eval --skip-generation`, checks the
  report against `config/eval_gates.yaml` via `scripts/check_eval_gates.py`).
- **The real `techfusion` corpus and every one of its gold files are
  gitignored and unavailable in CI** (already true before this milestone;
  see "Testing conventions"), so `retrieval-eval-gate` cannot reuse
  `experiments/results/*.json`'s recorded baselines (e.g. `experiment_028`'s
  Recall@5 ~= 0.788) at all. It instead ingests the tracked
  `data/sample_docs/` (3 files) against the tracked `data/eval/
  sample_gold.jsonl` (4 questions) through a new, frozen
  `config/experiments/ci_eval.yaml`, a genuinely different, much smaller,
  but fully CI-safe corpus with its own baseline, measured directly (not
  estimated) and recorded in `config/eval_gates.yaml`. Table/code/layout
  content-type diversity (which the milestone's design would prefer) isn't
  present in this sample corpus; that stays part of the full, manual
  `techfusion` evaluation.
- **Most of the milestone's zero-tolerance security invariants aren't
  measurable as a report metric in CI at all**, cross-tenant leakage,
  sensitive-field leakage, forged-role acceptance, unauthorized MCP/
  business actions, approval-required-without-approval mutations, and
  direct-tool-path authorization bypass all need gold rows with tenant/
  `forbidden_documents` fields that only exist in the gitignored
  `security_evaluation` corpus. They're enforced instead as deterministic
  pytest assertions in `security-regression-gate` (each test already
  asserts a denial/redaction/no-mutation outcome directly), not as
  threshold entries in `eval_gates.yaml`. The one security metric that
  *is* measurable from the CI sample corpus's own report,
  `duplicate_sensitive_field_miss_rate`, a corpus-level diagnostic, not
  gold-row-driven, is gated at an exact zero-tolerance ceiling.
- **`scripts/check_eval_gates.py`** reads one `run_eval.py` JSON report,
  checks each gate in `config/eval_gates.yaml` (`metric_path` dotted
  lookup, `higher_is_better`/`lower_is_better` direction, `floor`/
  `ceiling`, `blocking`), and prints a bounded PASS/FAIL/MISSING/MALFORMED
  line per gate, exiting non-zero iff any *blocking* gate did not PASS. A
  small epsilon (1e-9) absorbs float accumulation noise only, never an
  intentional threshold. For a failing recall/hit-rate gate with a
  `--verbose` report (the CI job always passes `--verbose`), it
  recomputes per-example pass/fail from the report's `per_example` list
  (reusing `rag.path_matching.source_matches_relevant`, the same suffix-
  match rule `run_eval.py` itself uses) and prints up to
  `--max-example-failures` (default 5) offending question strings,
  never a full per-example JSON dump. `scripts/` has no `__init__.py` and
  isn't on pytest's configured `pythonpath`, so `tests/unit/
  test_check_eval_gates.py` loads the module dynamically via
  `importlib.util.spec_from_file_location` (registering it in
  `sys.modules` *before* `exec_module`, which the `@dataclass`-decorated
  `GateResult` needs to resolve its owning module) rather than widening
  `pythonpath` for the whole suite just for one script.
- **Baselines are never rewritten automatically.** `config/eval_gates.yaml`
  is a plain, hand-edited YAML file with an explicit `rationale` string per
  gate; `check_eval_gates.py` only reads it. Updating a baseline is a
  reviewed code change (see `docs/ci_eval_gates.md`'s "Updating a
  baseline").
- **`security-regression-gate` deliberately re-runs tests `unit-tests`
  already runs.** Every file in its curated list already executes inside
  the full `tests/unit` suite; the value is a distinctly-named, fast
  status check so a security/guardrail regression is unmistakable at a
  glance rather than needing to dig through a much larger job. An
  acknowledged, documented duplication of test *execution*, not test
  *code*, see `docs/ci_eval_gates.md`.


## Evaluation and experiment tracking

Two independently-optional layers sit on top of the core offline serving
path, neither is installed in the container image (see "Containerized
runtime"):

- **RAGAS** (`eval/ragas_scorer.py`, `ragas_adapters.py`,
  `run_ragas_eval.py`; `pip install .[ragas]`): LLM-judge scoring of
  faithfulness/answer_relevancy/context_precision/context_recall/
  answer_correctness, plus (as of the 2026-08-14 context/token-budget
  milestone) noise_sensitivity/factual_correctness, best-effort loaded
  in `ragas_scorer._load_metrics` (an unsupported/renamed class in a
  future ragas version fails into `metrics_failed` per-metric rather than
  raising). Unlike the original 5 metrics, whose result columns match
  their bare name, these two's ragas-assigned result columns bake in
  their `mode` config (`"noise_sensitivity(mode=relevant)"`, not
  `"noise_sensitivity"`), confirmed directly against the pinned
  `ragas==0.3.9`, not assumed; `ragas_scorer._resolve_result_columns`
  handles this via exact-then-prefix matching, since assuming a metric's
  result column always equals its `.name` silently produced `None`
  scores for both with no exception anywhere (see
  `PROJECT_JOURNAL.md`'s 2026-08-14 entry and `KNOWN_ISSUES.md`). Every
  `ragas`/`datasets` import is lazy (inside
  functions), so importing these modules never requires the package to be
  installed, `RuntimeError` at call time instead, with an actionable
  message. The judge LLM is `config.judge` (openai/anthropic/hosted by
  default; `ollama` is offered for free local experimentation only, using
  a *different* model than `generation.model_name` on purpose, see
  `JudgeConfig`'s docstring in `config.py`). RAGAS scores are **not**
  validated against human labels by default, `CAVEAT` in
  `ragas_scorer.py`, and `scripts/generate_manual_review.py` /
  `compare_ragas_manual.py` are the intended validation workflow before
  trusting them for decisions.
- **RAGAS judge-call caching** (`eval/ragas_cache.py`,
  `config.judge.cache_enabled`/`cache_dir`, default on /
  `.cache/ragas`): `ragas_scorer.score()` passes a
  `NamespacedDiskCache` into `LangchainLLMWrapper(..., cache=...)`,
  which wraps RAGAS's own `ragas.cache.DiskCacheBackend` so an unchanged
  question/answer/context/metric doesn't re-pay for an identical judge
  call across runs. RAGAS's native cache key is a hash of the *rendered
  prompt* only (question/context/answer/metric-instruction text all get
  interpolated into it, so those are covered for free), it never
  includes the *judge LLM instance itself*, since `cacher()` wraps a
  bound method and `self` never enters the hashed args. `ragas_cache.
  judge_fingerprint()` closes that gap by namespacing every key with a
  hash of `(judge provider, judge model, judge temperature, judge
  max_tokens, installed ragas version)`, so switching judge model/
  provider always misses rather than silently replaying another model's
  verdict, see `PROJECT_JOURNAL.md`'s 2026-08-13 entry and
  `KNOWN_ISSUES.md` for why this was checked directly against `ragas.cache`'s
  source rather than assumed safe. `run_ragas_eval.run_ragas()` reports
  hit/miss counts and an `avoided_cost_estimate` (extrapolated from that
  *same run's* uncached-call token usage against a small, verified-only
  `PRICING_USD_PER_1M_TOKENS` table, reports a `reason` instead of a
  guessed number for any unpriced model or a 100%-cache-hit run with
  nothing to extrapolate from) under the report's `ragas.cache` key.
- **MLflow** (`eval/mlflow_logger.py`; `pip install .[mlflow]`,
  `config.mlflow`): local-SQLite-backed by default
  (`sqlite:///mlflow.db`, no server required. MLflow 3.x's plain-
  filesystem `file:./mlruns` backend is deprecated, so this isn't a
  simplification, it's the only non-deprecated zero-server option).
  `scripts/record_experiment.py` is the writer for the classic-RAG path:
  it reads an existing `run_eval.py --verbose` report plus the config
  that produced it, writes `experiments/results/<id>.json` (flat,
  comparable record) and `experiments/config_snapshots/<id>.yaml`
  (config snapshot), and logs the same record as an MLflow run,
  fields with `None` values are skipped (not logged as `0`/`"None"`)
  since MLflow's `log_param`/`log_metric` reject `None` outright and a
  pre-RAGAS record simply has no `ragas_*` values yet.
  `scripts/record_agent_experiment.py` is its agentic-eval sibling (not
  in the original plan file for this section; see "Agentic RAG
  re-evaluation" for its own metric shape): a `run_agent_eval.py`
  report's routing/tool-selection/citation-support metrics don't fit
  the classic schema, so it writes its own
  `experiments/results/agentic/<id>.json`, a subdirectory
  `compare_experiments.py`'s non-recursive glob never sees, and calls
  the same `log_experiment()`, both scripts share one MLflow-logging
  code path, they just build differently-shaped records for it. It
  additionally flattens
  `run_agent_eval.py`'s per-node latency breakdown
  (`agent_node_<name>_latency_ms_mean` for `classify`/`decompose`/
  `tool_select`/`tool_execute`/`evidence_sufficiency`/`synthesize`), a
  termination-reason breakdown (`agent_termination_<reason>_count`/
  `_rate` for `AgentState.termination_reason`'s fixed 5-value Literal,
  plus a combined `agent_guardrail_termination_count`/`_rate` for
  `max_steps`/`max_retrieval_attempts`/`max_tool_calls`), and a
  tool-usage breakdown (`agent_tool_usage_<tool_name>_count`/`_rate` for
  `ToolCallRecord.tool_name`'s fixed 7-value Literal), all three
  bounded, fixed-vocabulary breakdowns computed once per eval run from
  data `AgentState`/`AgentRunResult` already collected (no new
  instrumentation), scoped to `route == "agent"` records only so a
  classic_rag-routed row (which always terminates `"synthesized"` and
  never dispatches a tool) never dilutes the agent loop's own numbers.
  The MLflow `run_name` is a readable slug built by
  `mlflow_logger.build_run_name()` (e.g.
  `experiment_015_qwen2-5-3b_v2_hybrid_rel-exp`) rather than the bare
  `experiment_id`, display-only, the real MLflow run UUID
  (`run.info.run_id`) is unaffected, and every run is additionally
  tagged with `generation_model`/`prompt_version`/`retrieval_provider`/
  `reranker_provider`/`relationship_expansion`/`dataset_id` alongside the
  pre-existing `experiment_id`/`label` tags, all sourced from fields
  already in the record schema (no new record fields, so older
  pre-multimodal records still tag/name themselves cleanly with those
  segments blank rather than raising). `scripts/compare_experiments.py`
  renders those JSON records as a table, writing the *full* history to
  `experiments/reports/comparison.md` and splicing only the most recent
  `README_HIGHLIGHT_COUNT` (7) experiments into README's Benchmarks
  section, the full table grew too long to serve as a readable front
  door, so the README is deliberately kept to a rolling recent-history
  window rather than every record ever recorded; each table's leftmost
  column is the record's real `experiment_id`, not a positional row
  index, so a row number in the README always matches the same
  experiment referenced elsewhere (`PROJECT_JOURNAL.md`, `docs/
  architecture.md`, experiment report filenames). Keep this in mind when
  recording a new experiment: no README edit is needed, `python
  scripts/compare_experiments.py` alone keeps both files correct.
  `scripts/backfill_mlflow.py` re-logs pre-existing records if MLflow
  tracking was enabled after some experiments were already recorded (and
  will now also pick up the readable run names/tags on re-log).

`eval/run_eval.py` itself has no RAGAS/MLflow dependency, it always
computes Recall@5/@10, hit-rate, MRR (`eval/metrics.py`), a
keyword-overlap answer-quality score (`eval/answer_quality.py`), and the
multimodal/relationship-aware deterministic metrics (content-type
breakdown, reference-context A/B/C buckets, image-hit rate,
expansion-contribution rate, vision-behavior triage, see "Multimodal
ingestion" above) with `--dataset-id` mandatory and applied as a hard
filter on every retrieval it makes.
