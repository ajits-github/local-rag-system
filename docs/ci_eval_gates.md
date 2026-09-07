# CI evaluation gates

A fast, deterministic quality/security gate that runs on every PR and push
to `main`, in addition to the pre-existing `unit-tests`/`integration-tests`/
`code-quality`/`docker-build` jobs. It is deliberately small and reuses
existing evaluation infrastructure (`rag.eval.run_eval`, the existing test
suite) rather than a parallel evaluation framework. Full RAGAS scoring,
hosted-judge calls, and broad Ollama-backed agent benchmarks stay manual and
heavy -- see "What stays manual" below.

## What runs on every PR

Two new CI jobs, alongside the four pre-existing ones
(`.github/workflows/ci.yml`):

- **security-regression-gate**: a curated subset of `tests/unit` covering
  document/field-level authorization, DoS limits, rate limiting, MCP
  identity/business-tool authorization and approval logic, agent
  step/tool-call/retry bounds, insufficient-evidence and malformed-
  tool-argument handling, the case-mutation routing fix, and citation
  attribution. No Postgres, no Ollama -- pure unit tests, ~230 tests,
  runs in well under a minute. See "Why a separate job" below for why this
  duplicates part of what `unit-tests` already runs.
- **retrieval-eval-gate**: ingests a small, tracked, deterministic sample
  corpus into a real (ephemeral, CI-only) Postgres+pgvector, runs
  `rag.eval.run_eval --skip-generation` against it, and checks the report
  against explicit thresholds with `scripts/check_eval_gates.py`.

## Blocking metrics

Every gate's baseline, floor/ceiling, and rationale lives in
`config/eval_gates.yaml` (repo root, not part of this docs site), not in
CI YAML. Currently:

| Gate | Metric | Baseline | CI floor/ceiling | Direction |
|---|---|---|---|---|
| `recall_at_5` | `retrieval.recall@5` | 0.625 | >= 0.5 | higher is better |
| `recall_at_10` | `retrieval.recall@10` | 0.75 | >= 0.625 | higher is better |
| `hit_rate_at_5` | `hit_rate.hit_rate@5` | 0.75 | >= 0.5 | higher is better |
| `mrr` | `mrr` | 0.75 | >= 0.625 | higher is better |
| `duplicate_sensitive_field_miss_rate` | `safety.duplicate_sensitive_field_miss_rate.rate` | 0.0 | <= 0.0 (zero-tolerance) | lower is better |

The retrieval floors deliberately leave roughly a one-discrete-step
tolerance at this sample size (4 gold questions) rather than pinning the
exact baseline value, so a single-bit change in chunking/embedding output
doesn't flap the gate -- see the `rationale` field on each gate in
`config/eval_gates.yaml` for the exact reasoning per metric.

The rest of the milestone's zero-tolerance security invariants (cross-tenant
leakage, sensitive-field leakage, forged-role acceptance, unauthorized
MCP/business actions, approval-required-without-approval mutations, direct
tool-path authorization bypass) are **not** measurable as report metrics in
CI: they need gold rows with tenant/`forbidden_documents` fields that only
exist in the real, gitignored `techfusion` `security_evaluation` corpus (see
CLAUDE.md's "Testing conventions"). They are instead enforced as
deterministic pytest assertions in `security-regression-gate` -- each of
those tests already asserts a denial/redaction/no-mutation outcome directly
(pass/fail, not a threshold), so they need no separate entry in
`eval_gates.yaml`.

## Why the retrieval baseline looks nothing like the numbers elsewhere in this repo

`experiments/results/*.json` (e.g. `experiment_028`, Recall@5 ~= 0.788) is
measured against the real, 126-question `techfusion` gold set and its
gitignored corpus. Neither is available in CI. The retrieval-eval-gate job
instead ingests the tracked `data/sample_docs/` (3 Markdown files) against
`data/eval/sample_gold.jsonl` (4 questions) -- a different, much smaller,
but fully CI-safe and deterministic corpus. Its baseline numbers were
measured directly (not estimated) by running the exact commands in "Run the
gate locally" below. Table/code/layout-content-type diversity (which
section 4 of the original design would prefer) isn't present in this
sample corpus; broader content-type coverage stays part of the full,
manual `techfusion` evaluation.

## Why a separate `security-regression-gate` job

Every test file it runs already executes today inside the `unit-tests`
job (which runs the whole of `tests/unit`), so this is a deliberate,
acknowledged duplication of test *execution*, not of test *code*. The
value is a distinctly-named, fast, security/guardrail-focused status
check: a regression here is unmistakable at a glance (a single named job
turning red) rather than needing to dig through a much larger unit-tests
job to notice one security-relevant test failed among hundreds.

## Run the gate locally

```bash
# 1. Bring up Postgres (or use an existing `make up` instance) and init schema
python scripts/init_db.py

# 2. Ingest the tracked CI sample corpus into its own dataset
python -m rag.ingestion.pipeline data/sample_docs \
  --dataset-id ci_eval_sample \
  --config config/experiments/ci_eval.yaml \
  --clear

# 3. Run retrieval-only evaluation (no Ollama needed)
python -m rag.eval.run_eval \
  --gold data/eval/sample_gold.jsonl \
  --dataset-id ci_eval_sample \
  --config config/experiments/ci_eval.yaml \
  --skip-generation --verbose \
  > eval_report.json

# 4. Check it against the gate thresholds
python scripts/check_eval_gates.py --report eval_report.json
```

For the security-regression-gate subset, see the curated file list in
`.github/workflows/ci.yml`'s `security-regression-gate` job, or just run:

```bash
pytest tests/unit -k "authorization or field_policy or dos_limits or rate_limiting or mcp or agent_tool or agent_graph or citation or run_agent_eval"
```

(a superset of the exact CI list -- fine for a local sanity check, but the
CI job uses the explicit file list so it can't silently grow or shrink
with an unrelated future test rename).

## Updating a baseline

A baseline in `config/eval_gates.yaml` is never rewritten automatically.
To update one intentionally (e.g. after a real, reviewed retrieval-quality
improvement to `config/experiments/ci_eval.yaml` or the CI sample corpus):

1. Run the "Run the gate locally" steps above to measure the new numbers.
2. Update the `baseline` value and, if warranted, the `floor`/`ceiling` in
   `config/eval_gates.yaml`, keeping the `rationale` field accurate.
3. Include the measured before/after numbers in the PR description.

## What stays manual/heavy

Never run in PR CI, by design (no paid API, no hosted judge, no broad
Ollama benchmark in the fast path):

- Full RAGAS scoring (`rag.eval.run_ragas_eval`) against `openai`/
  `anthropic`/`ollama` judges.
- The full 84/126-question `techfusion` deterministic evaluation
  (`rag.eval.run_eval` against the real corpus) -- unavailable in CI
  anyway, since the corpus is gitignored.
- The full agentic-RAG benchmark (`rag.eval.run_agent_eval`) against a real
  Ollama model -- covered instead by the deterministic, LLM-free unit tests
  in `security-regression-gate`.
- MLflow experiment logging / `scripts/record_experiment.py` /
  `scripts/record_agent_experiment.py`.

These remain manual, run locally exactly as documented in README's
"Evaluation" and "Continuous Integration" sections (repo root
`README.md`, not part of this docs site).

## Non-blocking / deliberately excluded metrics

None of the current gates are marked `blocking: false` -- every metric in
`config/eval_gates.yaml` today is stable enough (a fixed, tiny, fully
deterministic corpus with no reranker/agent/generation variance) to gate
safely. `blocking: false` exists in the schema for a future metric that
turns out too noisy to gate reliably (see `config/eval_gates.yaml`'s own
comments); none is needed yet.
