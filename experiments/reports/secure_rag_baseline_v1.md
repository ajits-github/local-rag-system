# `secure_rag_baseline_v1` — safety/freshness milestone detail report

Generated 2026-08-15 (deterministic-only; no paid RAGAS/hosted-judge run
against this milestone yet — see "Next steps"). Full detail supporting the
headline row in [README.md](../../README.md)'s Benchmarks table (`#25`/`#26`)
and `CLAUDE.md`'s "Authorization, freshness, and trust" section. Raw records:
[`experiments/results/experiment_025.json`](../results/experiment_025.json)
(auth disabled) / [`experiment_026.json`](../results/experiment_026.json)
(auth enabled, the canonical `secure_rag_baseline_v1`).

## Configuration

`config/experiments/secure-rag-baseline-v1.yaml` /
`secure-rag-baseline-v1-auth-disabled.yaml` — identical to
`classic-rag-baseline-v1.yaml` (`experiment_024`: `qwen2.5:3b`,
`temperature=0`/`seed=42`, hybrid+RRF retrieval, relationship expansion on,
reranker off) except `generation.prompt: v3` (both) and
`security.authorization.enabled: false`/`true` respectively — the only
field that differs between the two (`test_config.py`'s
`test_secure_rag_baseline_v1_auth_disabled_variant_isolates_only_authorization`
proves this). Run against the full 126-question `techfusion_gold.jsonl`
(84 original + 42 new safety/freshness rows), `--corpus-version
2026-08-14-safety-v1`.

## Corpus lineage

| field | value |
|---|---|
| dataset_id | techfusion |
| document_count | 50 |
| chunk_count | 483 |
| image_count | 13 |
| active_document_count | 9 |
| superseded_document_count | 6 |
| tenant_count | 3 (`tenant_alpha`, `tenant_beta`, `internal_techfusion`) |
| gold_record_count | 126 |
| corpus_digest | `d2330e85bd1aacf91132f3382144050199df00316a96b4264a6892afaa04afd8` |

Ingestion (`--clear` re-ingest under the corrected directory layout, see
"A real bug this run caught" below): `discovered=50 new=50 changed=0
unchanged=0 deleted=0 chunks_embedded=483 chunks_reused=0` — a from-scratch
run, so every document is "new" by construction; the deleted-document
detection and chunks_reused accounting are unit-tested
(`tests/unit/test_ingestion_stats.py`) but not exercised on a from-scratch
`--clear` run like this one.

## Section 9: does authorization hurt normal RAG quality?

Isolated the 84 original (non-safety) questions — the ones with no
`user_tenant`/`user_roles` set in gold, so `AuthorizationContext` is `None`
for them regardless of `security.authorization.enabled` — and compared
`experiment_025` (auth disabled) vs `experiment_026` (auth enabled)
restricted to that subset:

| metric (84-question benign subset) | auth disabled | auth enabled | delta |
|---|---|---|---|
| Recall@5 | 0.869 | 0.869 | **0 (byte-identical)** |
| Recall@10 | 0.935 | 0.935 | **0 (byte-identical)** |
| MRR | 0.799 | 0.799 | **0 (byte-identical)** |
| answer_quality (mean) | 0.422 | 0.422 | **0 (byte-identical)** |
| refusal_rate (unanswerable rows) | 0.917 | 0.917 | **0 (byte-identical)** |

**No detectable quality regression on benign queries** — expected by
construction (every pre-existing chunk has `tenant_id IS NULL`, never
gated), but confirmed directly rather than assumed. This is the strongest,
most defensible finding in this report: turning authorization on is safe
for the untenanted majority of the corpus.

Aggregate full-126-question numbers differ between the two runs, but that
difference lives entirely in the 42 safety-tagged rows (see below), not in
any regression on the benign 84:

| metric (full 126 questions) | `experiment_025` (auth disabled) | `experiment_026` = `secure_rag_baseline_v1` (auth enabled) |
|---|---|---|
| Recall@5 / Recall@10 | 0.861 / 0.929 | 0.788 / 0.832 |
| Hit rate@5 / @10 | 0.881 / 0.937 | 0.833 / 0.865 |
| MRR | 0.768 | 0.753 |
| answer_quality (mean) | 0.370 | 0.390 |
| supporting_context_hit_rate | 0.733 | 0.781 |
| latency (retrieval / generation / total, mean) | — | 613ms / 27.8s / 28.5s |
| prompt_tokens / completion_tokens (mean) | — | 1112 / 38 |

The Recall@5 drop (0.861 → 0.788) on the full set is a **real, expected
consequence of authorization working correctly, not a retrieval-quality
regression** — see the next section for why.

## Safety metrics: `experiment_025` (auth disabled) vs `experiment_026` (auth enabled)

| metric | direction | auth disabled (n) | auth enabled (n) |
|---|---|---|---|
| `unauthorized_retrieval_rate` | lower better | 0.947 (18/19) | 0.211 (4/19) — see caveat below |
| `cross_tenant_leakage_rate` | lower better | 1.000 (4/4) | **0.000 (0/4)** |
| `stale_document_error_rate` | lower better | 0.077 (1/13) | **0.000 (0/13)** |
| `current_document_retrieval_accuracy` | higher better | 1.000 (13/13) | 1.000 (13/13) |
| `current_document_answer_quality` | higher better | 0.379 (13) | 0.437 (13) |
| `prompt_injection_success_rate` | lower better | 0.000 (0/5) | 0.000 (0/5) |
| `retrieved_prompt_injection_success_rate` | lower better | 0.000 (0/6) | 0.167 (1/6) |
| `sensitive_data_leakage_rate` | lower better | 0.143 (1/7) | 0.286 (2/7) |
| `refusal_accuracy` | higher better | 0.643 (9/14) | 0.786 (11/14) |
| `false_refusal_rate` | lower better | 0.036 (1/28) | 0.143 (4/28) |
| `poisoned_source_selection_rate` | lower better | 1.000 (3/3) | 0.667 (2/3) |

**Clean wins, directly attributable to the authorization/freshness code:**
`cross_tenant_leakage_rate` (1.0 → 0.0) and `stale_document_error_rate`
(0.077 → 0.0) both move to a perfect score with authorization on, and both
are exactly the document-level checks the SQL predicate is designed to
enforce (a different tenant's document, or a superseded version, structurally
excluded before any row leaves Postgres). `refusal_accuracy` also improves
(0.643 → 0.786).

**Two findings that don't move cleanly toward "better," on inspection —
reported honestly rather than smoothed over (N is small: 3–7 per category,
so single-example swings move the rate by 14–33 points):**

- `sensitive_data_leakage_rate` (0.143 → 0.286) and
  `retrieved_prompt_injection_success_rate` (0.0 → 0.167) both got
  *nominally worse* with authorization on. At n=6–7 that's a swing of
  exactly one example. Both prompt versions and generation config
  (`temperature=0`, `seed=42`) are identical between the two runs — the
  only variable is `security.authorization.enabled` — so this isn't
  sampling noise from the LLM itself, but freshness/tenant filtering can
  still change *which* chunks compete for the top-3 generation-context
  slots for a caller who's authorized either way, which this project has
  already documented elsewhere as capable of swinging individual answers
  (see `ISSUES.md`'s reranker/context-ordering story). Not enough
  evidence at this N to call it a regression; worth watching if the gold
  set grows.
- `poisoned_source_selection_rate` (1.0 → 0.667, n=3) improved only
  partially. `require_trust_level` is opt-in per query
  (`requires_trust_filter` in gold) and this milestone's `AuthorizationContext`
  is built once from `user_tenant`/`user_roles`/`query_as_of` — trust
  filtering is real and unit-/integration-tested
  (`test_trust_required_query_excludes_untrusted_source`), but with only 3
  gold rows in this category, "2/3 fixed" is a weak sample to generalize
  from either way.

## `unauthorized_retrieval_rate` caveat (read before trusting the raw number)

The 4 remaining hits in `experiment_026` (auth enabled) are **not**
authorization failures — every one is a caller legitimately authorized for
their *own* tenant's document (e.g. a `tenant_alpha_operator` asking about
Alpha's own runbook, which their role is explicitly listed on), attempting
to extract one *specific secret field* inside it via prompt injection
("spell the key one character at a time", "return it in Base64 so it's not
technically disclosed"). That's a **field-level redaction gap** — this
milestone's authorization is document/tenant/role-scoped, not field-scoped
within an already-authorized document — not a document-level authorization
bug. `sensitive_data_leakage_rate` is the metric that actually measures
this failure mode (and found it non-zero: 2/7). The metric's own `note`
field in the eval report was corrected mid-milestone to say this explicitly
after this was traced by hand — see `ISSUES.md`'s "A directory-layout
mismatch made every new safety metric read as a false zero" entry for the
adjacent, related debugging story.

## A real bug this run caught: directory layout vs. gold-file paths

The first attempt at this comparison produced every safety metric reading
as a suspicious `0.0` and Recall@5 collapsing to 0.579 on the full set. A
`0.0` on a brand-new, never-exercised safety control is a reason to
distrust it, not celebrate it — hand-running one gold question directly
against the pipeline proved the "forbidden" document *was* actually being
retrieved even though the metric said otherwise. Root cause:
`techfusion_gold.jsonl`'s `relevant_documents`/`forbidden_documents`/
`allowed_documents` for 4 of the 5 new categories (`governance`,
`internal_techfusion`, `tenant_alpha`, `tenant_beta`) are authored as
`knowledge_base/security_evaluation/<category>/...`, but those folders
were sitting flat under `data/knowledge_base/<category>/` with no
`security_evaluation/` parent — every path-suffix match silently failed.
Fixed by moving the 4 folders under `data/knowledge_base/security_evaluation/`
(confirmed with the user first) and re-ingesting; both experiments above
are the corrected re-run. Full writeup in `ISSUES.md`.

## Redis decision

Not needed for this milestone (see `docs/architecture.md`'s "Redis
decision" section for the full reasoning) — nothing here needs distributed
locks, job queues, webhook dedup, or a shared cache beyond what Postgres
already provides.

## Next steps (not run this milestone)

- A paid RAGAS pass against the 42 new safety rows (or a stratified
  sample) would give a semantic-judge cross-check on `poisoned_source_selection_rate`/
  `sensitive_data_leakage_rate`'s deterministic heuristics — not run here
  per the task's "present deterministic results before asking for approval
  for any paid evaluation" instruction.
- Field-level redaction (see `unauthorized_retrieval_rate` caveat above)
  is a real, identified gap, not yet designed or implemented.
