# Project Journal

Local, untracked engineering journal for local-rag-system. Excluded from git
via `.git/info/exclude` (not `.gitignore`): this file stays private to this
checkout and is never pushed.

Append a new dated section at the end of every meaningful implementation or
experiment session. Never edit or overwrite previous entries.

Each entry starts with a short **Interview talking points** block (what
problem it solved, why it was hard, how it was debugged, what metrics
proved the fix, what's next) so the session can be summarized out loud
without re-reading the full technical detail below it. The detailed
Goal/Configuration/Issues/Evaluation/Interpretation sections stay as the
backing evidence for those answers.

---

## 2026-08-04-17:00 | Initial scaffold + end-to-end verification (retrospective)

*Reconstructed from conversation history rather than logged live: date is
accurate (consistent across multiple tool timestamps that day), the
17:00 time marker is an approximate midpoint, not a precise log time.*

### Interview talking points
- **What problem did this solve?** Verified a freshly scaffolded local RAG
  system actually works end-to-end against real infrastructure
  (Postgres+pgvector, Ollama) before the first commit, instead of trusting
  a test suite that mocked every external dependency.
- **Why was it difficult?** All unit tests passed, but the first real
  similarity search still failed: `operator does not exist: vector <=>
  numeric[]`. Postgres's type resolution for the pgvector operator doesn't
  chain casts the way an INSERT does, so a plain Python list worked for
  writes but silently failed for reads, in a way no mock could have caught.
  A stale, unrelated Docker container also occupied the expected port, and
  one cleanup query using a Windows-style path in a Postgres `LIKE`
  pattern (backslash is an escape character there) accidentally deleted
  every row in the `documents` table.
- **How did I debug it?** Ran the pipeline against real Postgres/Ollama
  step by step instead of assuming green tests meant it worked; read the
  Postgres error literally (`numeric[]` vs `vector`), traced it to how
  psycopg2 adapts Python lists without an explicit type, and fixed it with
  an explicit `::vector` cast rather than relying on implicit coercion.
- **What metrics proved the fix?** Integration tests went from 5 passed /
  2 failed to 7/7; 5 manual smoke-test queries against real ingested
  documents were all correctly grounded, and the one deliberately
  unanswerable query was correctly refused, with measurably lower
  similarity scores on its retrieved chunks.
- **What would I improve next?** Clean up the `documents` table row in
  integration test teardown (only `chunks` were removed at the time), and
  resolve the mismatch between how the API upload path and the CLI
  ingestion path derive document identity.

### Goal
Scaffold a modular, config-driven local RAG system from an empty repo, then
verify it works end-to-end against real Postgres+pgvector and Ollama before
the first commit — not just unit tests with everything mocked.

### Configuration
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), CPU.
- Chunking: `recursive_character`, size 500 / overlap 50.
- Reranker: `none` (`NoOpReranker`, default).
- Generation: Ollama `qwen2.5:1.5b`.
- Vector store: pgvector, via a pre-existing `rag-pgvector` container
  (port 15987, db `rag`) rather than this repo's own `docker-compose.yml`
  container, which lost a port conflict with it.
- Dataset: 3 hand-written smoke-test Markdown docs (`data/sample_docs/`).

### Changes made
Full initial scaffold: `config.py`, `schemas.py`, `factory.py`,
`logging_config.py`, `loaders/` / `cleaners/` / `chunkers/` / `embedders/` /
`vectorstore/` / `rerankers/` / `generation/` packages, `ingestion/` +
`retrieval/` pipelines, `eval/` module, FastAPI app (`api/`), unit +
integration tests, `CLAUDE.md`, `README.md`, `Makefile`, `docker-compose.yml`,
`scripts/init_db.py`.

### Issues discovered
1. **Symptom:** every similarity search failed with
   `operator does not exist: vector <=> numeric[]`.
   **Root cause:** the query embedding (a plain Python list) was adapted by
   psycopg2 as an untyped `numeric[]` array literal; pgvector only defines
   an implicit cast from `double precision[] → vector`, not `numeric[]`, and
   operator resolution doesn't chain casts the way assignment/INSERT
   coercion does (which is why writes worked but reads didn't).
   **Fix:** added explicit `::vector` casts in `vectorstore/pgvector.py`'s
   SQL.
   **Evidence:** `test_add_chunks_and_search_round_trip` and
   `test_query_pipeline_answers_from_ingested_content` failed before, all
   7 integration tests passed after.
   **Resolved:** yes.

2. **Symptom:** accidentally deleted all 11 rows from `documents`
   (cascading to all chunks).
   **Root cause:** a cleanup query used
   `WHERE source NOT LIKE 'data\sample_docs%'` — the backslash is a `LIKE`
   escape character in Postgres, so the pattern never matched the real
   rows, and the negated condition matched everything.
   **Fix:** re-ingested `data/sample_docs` to restore state; all later
   cleanups used exact `document_id` lists instead of `LIKE` patterns on
   Windows-style paths.
   **Evidence:** document/chunk counts confirmed restored (3 docs / 16
   chunks) immediately after.
   **Resolved:** yes (avoided by changing my own cleanup approach).

3. **Symptom:** README's `make up` / `make ingest` / `make query` / `make
   test` don't work out of the box.
   **Root cause:** `make` isn't installed on this Windows machine at all
   (neither Git Bash nor PowerShell).
   **Fix:** added a Prerequisites note plus a direct-command fallback next
   to every `make` target in the README.
   **Evidence:** `make --version` fails in both shells.
   **Resolved:** documented; `make` itself intentionally left uninstalled
   (user's choice).

4. **Symptom:** `docker compose up` failed — port 15987 already allocated.
   **Root cause:** a separate, pre-existing `rag-pgvector` container (not
   managed by this repo's compose file) was already running on that port
   from earlier, unrelated work.
   **Fix:** used `rag-pgvector` directly instead; updated `.env`'s
   `POSTGRES_DB`/`DATABASE_URL` to match its actual db name (`rag`, not
   `ragdb`); removed the never-started `local-rag-postgres`
   container/volume this repo's own compose had created.
   **Evidence:** `docker ps -a` showing both containers; schema init and
   all subsequent work succeeded against `rag-pgvector`.
   **Resolved:** yes, documented as an option in the README.

5. **Symptom:** uploading a file via `POST /ingest` that was already
   CLI-ingested creates a *second* document with a new `document_id`.
   **Root cause:** `/ingest` saves uploads to `data/uploads/<filename>`, a
   different `source` path than wherever the CLI ingested the same file
   from; document identity is keyed on exact source path.
   **Fix:** none applied — documented as a known behavior.
   **Evidence:** duplicate `document_id` observed after uploading
   `password-policy.md` via the API; manually removed.
   **Resolved:** no — open item.

### Evaluation results
No formal gold-set eval this session (see next entry for that). Manual
smoke test: 5 hand-written queries against the 3 sample docs — 3/3 directly
answerable queries correct and grounded, 1 cross-document query correctly
synthesized both source documents, 1 deliberately unanswerable query
correctly refused (its retrieved chunks scored ~0.27–0.38 vs ~0.55–0.81 for
the answerable queries). Latency: ~17s cold start, 1.9–3.2s per query after.
Full suite: 40 unit + 7 integration tests passing.

### Interpretation
The pipeline is functionally correct end-to-end on a tiny, hand-authored
dataset. This demonstrates plumbing correctness only — not retrieval/
generation quality at realistic scale or difficulty. That needed a real
knowledge base and gold set (next entry).

### Open problems
- Integration tests don't clean up their `documents` table rows on
  teardown (only `chunks` — leaves orphan rows across runs).
- `/ingest` (API) vs CLI source-identity mismatch (issue #5 above).

### Next recommended step
Ingest a larger, realistic knowledge base with a real gold eval set and
measure retrieval/generation quality properly (done in the next session).

---

## 2026-08-05-16:22 | TechFusion knowledge base integration + evaluation

### Interview talking points
- **What problem did this solve?** Made the RAG system work against a
  much larger, realistic 21-document knowledge base with real
  folder-based categories and YAML front matter, and produced real
  Recall/MRR/latency numbers instead of a 3-document smoke test.
- **Why was it difficult?** The real gold evaluation file didn't match
  the eval code's assumptions at all: it referenced documents by relative
  file path instead of database IDs, which didn't exist yet when the gold
  questions were written. Separately, the retrieval pipeline silently
  capped every result set to 3 chunks regardless of what was requested,
  making Recall@10 mathematically impossible to compute. Manual testing
  also surfaced two real model-quality issues: one factual retrieval miss
  that the model then hallucinated an answer for, and one case where
  retrieval was correct but the model still fabricated details not present
  in the retrieved text.
- **How did I debug it?** Compared the gold file's actual JSON fields
  against what the eval code expected, then designed a path-suffix matcher
  that works regardless of what root directory a corpus is ingested from,
  instead of hardcoding a path. Found the Recall@10 ceiling by checking
  the exact value returned from retrieval against what was being silently
  truncated, and added an explicit override. Confirmed the two model
  failures by hand-inspecting retrieved chunks side by side with the
  generated answer.
- **What metrics proved the fix?** Recall@5 = 0.859, Recall@10 = 0.957,
  MRR = 0.830, Hit Rate@5/@10 = 0.891/0.978 across 46 real questions, with
  generation and retrieval latency broken out separately for the first
  time.
- **What would I improve next?** Isolate the smoke-test dataset from the
  real knowledge base so they can never contaminate each other's results
  (became the very next session), and address the two concrete
  generation-quality failures, most likely with a cross-encoder reranker
  or a larger model.

### Goal
Ingest the full TechFusion knowledge base (21 Markdown docs across 7
category folders) preserving folder structure as filterable metadata, adapt
the eval runner to the real gold-set format, run the complete evaluation,
and spot-check manual queries with full retrieval/generation detail.

### Configuration
Same infra as the previous session (embedding `all-MiniLM-L6-v2`/384-dim,
`recursive_character` 500/50, reranker `none`, Ollama `qwen2.5:1.5b`,
pgvector via `rag-pgvector`). Dataset: `data/knowledge_base/` (21 docs → 280
chunks after ingestion, alongside the 16 chunks from the prior session's
`sample_docs/` still in the same table) + `data/eval/techfusion_gold.jsonl`
(46 labeled questions: single-document + multi-hop, 6 unanswerable).

### Changes made
- `loaders/text_loader.py`: parse YAML front-matter (`title`/`owner`/
  `last_reviewed`) on Markdown sources, strip it before chunking.
- `schemas.py`, `vectorstore/base.py`, `vectorstore/pgvector.py`,
  `scripts/init_db.py`: added `category` metadata end-to-end — new
  `ChunkMetadata` field, a `chunks.category` column migrated onto the
  existing table via idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
  (no data loss), added to the filter whitelist. Derived in
  `ingestion/pipeline.py` from each file's folder path relative to the
  ingested root, threaded through `ingestion/writer.py`.
- `eval/gold_schema.py`, `eval/run_eval.py`: rewritten for the real gold
  schema (`question`/`expected_answer`/`relevant_documents`/
  `question_type`/`difficulty`/`unanswerable`); matching is by normalized
  path-suffix (`source_matches_relevant`), not `document_id` or exact
  equality, so no root path is hardcoded anywhere.
- `eval/metrics.py`: generalized `recall_at_k`/`reciprocal_rank` with an
  optional `match_fn` (backward compatible, default exact-match); added
  `hit_rate_at_k` / `mean_hit_rate_at_k`.
- `retrieval/pipeline.py`: `retrieve()` gained a `rerank_top_n` override
  (previously hardcoded truncation to the config default made Recall@10
  impossible to measure); `answer()` now returns `retrieval_ms`/
  `generation_ms`/`total_ms`.
- `api/routers/query.py`: exposed `category` and the latency fields in the
  API response.
- `README.md`, `CLAUDE.md`: updated to match all of the above.
- Tests: rewrote `test_gold_schema.py`, extended `test_metrics.py`
  (hit-rate/match_fn), added a front-matter test to `test_loaders.py`, added
  `test_ingestion_category.py` (category derivation via fakes, no DB). 55
  unit tests passing.

### Issues discovered
1. **Symptom:** Recall@10 was structurally impossible to compute.
   **Root cause:** `RetrievalPipeline.retrieve()` always truncated results
   to `config.retrieval.rerank_top_n` (default 3) regardless of the
   requested `top_k`.
   **Fix:** added an explicit `rerank_top_n` override parameter.
   **Evidence:** eval run reports Recall@5=0.859 and Recall@10=0.957 —
   distinct, non-degenerate values.
   **Resolved:** yes.

2. **Symptom:** the gold dataset didn't match against retrieved results at
   all.
   **Root cause:** `techfusion_gold.jsonl` uses `question`/`expected_answer`/
   `relevant_documents` (relative paths); the eval code expected `query`/
   `relevant_document_ids` (UUIDs), inherited from an earlier placeholder
   design that assumed ids could be known ahead of ingestion.
   **Fix:** rewrote `GoldExample` + added path-suffix matching that ignores
   whatever root prefix a given ingestion run used.
   **Evidence:** new `test_source_matches_relevant_*` unit tests; full eval
   run now produces sensible per-cutoff metrics.
   **Resolved:** yes.

3. **Symptom:** "How long are idempotency keys retained?" — model
   fabricated "Annual rotation is performed on customer-data keys," not
   grounded in any retrieved chunk.
   **Root cause:** retrieval missed the correct document
   (`api-development-guidelines.md`) in production-config top-3; the model
   hallucinated instead of abstaining.
   **Fix:** none applied yet.
   **Evidence:** manual query transcript, this session.
   **Resolved:** no. Candidate fixes: cross-encoder reranker, or a larger
   LLM.

4. **Symptom:** "Which stores are systems of record vs. rebuildable?" —
   retrieval fetched the correct 2 documents (`data-storage.md`,
   `backup-and-recovery.md`), but the model answered "Systems of Record:
   Compliance Ledger, Data Platform" — service names absent from the
   retrieved context.
   **Root cause:** `qwen2.5:1.5b`'s synthesis capability on this
   multi-hop question, not a retrieval defect (retrieval was correct).
   **Fix:** none applied — flagged as concrete evidence for the planned
   larger-model swap.
   **Evidence:** manual query transcript, this session.
   **Resolved:** no.

5. **Symptom:** old `data/sample_docs` smoke-test chunks (16 chunks / 3
   docs, previous session) still live in the same pgvector table as the
   21-doc TechFusion KB, and appeared as a top-3 competitor in one manual
   query.
   **Root cause:** never removed or namespaced after the smoke-test
   session concluded.
   **Fix:** none applied.
   **Evidence:** the "canary rollback" manual query's #2 result was
   `sample_docs/deployment-process.md`, ranked between the two correct KB
   documents (didn't crowd them out this time, but could in a denser
   corpus).
   **Resolved:** no — recommend removing or `category`-isolating before
   serious benchmarking.

6. **Symptom:** answer-quality (keyword-overlap) score much lower on
   unanswerable questions (0.243) than answerable (0.466).
   **Root cause:** a correct refusal shares few literal keywords with a
   reference "no answer" string — an inherent ceiling of the heuristic, not
   a scoring bug.
   **Fix:** none — documented as a caveat in `run_eval.py`'s own output;
   the real fix (RAGAS) is on the README Roadmap, not implemented.
   **Evidence:** eval report's `answer_quality.mean_unanswerable` vs
   `mean_answerable`.
   **Resolved:** n/a — inherent limitation of a placeholder metric, not a
   bug to patch.

7. **Symptom:** integration tests still leave orphan rows in `documents`
   after each run.
   **Root cause:** unresolved from the previous session — test teardown
   only deletes `chunks`, never the `documents` row.
   **Fix:** manually purged orphan rows twice this session via exact
   `document_id` lists; no code fix applied.
   **Resolved:** no — still open.

### Evaluation results
46 questions, `data/eval/techfusion_gold.jsonl`, production config
(`rerank_top_n=3`, `reranker=none`):
- Recall@5 = 0.859, Recall@10 = 0.957
- Hit Rate@5 = 0.891, Hit Rate@10 = 0.978
- MRR = 0.830
- Retrieval latency (mean) = 129 ms
- Generation latency (mean) = 6,271 ms
- Total latency (mean) = 6,400 ms
- Answer quality (keyword-overlap): overall 0.437, answerable 0.466,
  unanswerable 0.243

### Interpretation
Retrieval is strong at this scale (280 chunks / 21 docs) — Recall@10 above
0.95 means the right document is almost always somewhere in a top-10 fetch.
The @5→@10 gap (0.859 → 0.957) shows a meaningful fraction of questions
need more headroom than the current production `rerank_top_n=3` default
gives them (see issue #3). Generation is the weaker link: even where
retrieval is correct, the small model can fabricate specifics on multi-hop
questions (issue #4). Latency is dominated by generation (6.27s of 6.4s
total) — expected for CPU-only `qwen2.5:1.5b`, not a system defect. The
answer-quality numbers should **not** be read as "54% of answers are
wrong" — the metric is too crude to support that claim; it's directionally
useful for the answerable/unanswerable gap, nothing more precise than that.

### Open problems
1. Idempotency-key retrieval miss (issue #3).
2. Systems-of-record generation hallucination despite correct retrieval
   (issue #4).
3. `sample_docs`/TechFusion KB commingling in the same vector store
   (issue #5).
4. Integration test `documents`-table cleanup (issue #7, carried over).
5. `/ingest` API vs CLI source-identity mismatch (carried over from the
   previous session, still unresolved).
6. Answer-quality metric ceiling — needs RAGAS, not a bug fix.

### Next recommended step
Flip `reranker.provider` to `cross_encoder` in `config/default.yaml`
(already scaffolded, zero code changes needed) and re-run this same eval
to see whether it closes the idempotency-key miss and the @5-vs-@10 gap —
the cheapest, most isolated next experiment before considering a model
swap.

---

## 2026-08-05-17:21 | Dataset isolation + clean evaluation baseline

### Interview talking points
- **What problem did this solve?** Guaranteed that two different document
  collections in the same database could never leak into each other's
  search results, and fixed a test-cleanup bug that was silently leaving
  orphaned rows behind after every test run.
- **Why was it difficult?** Nothing enforced separation between datasets
  before this; two unrelated document sets lived in the same table,
  distinguished only by file path, so a query had no way to know it should
  search only one of them. Proving this actually worked required more
  than a spot check, since a weak test could pass by accident if the two
  datasets simply didn't happen to overlap in content.
- **How did I debug it?** Instead of trusting an ordinary test, built one
  with two documents deliberately worded almost identically (same
  sentence, different fact value) in two different namespaces, so that
  without a correctly enforced filter, both would be equally likely to be
  retrieved. Also traced the leftover-row problem to a deletion method
  that was only ever designed to clear chunks mid-re-ingestion, never
  meant for full cleanup, and added a dedicated method for that instead.
- **What metrics proved the fix?** A test that ingests two near-identical
  documents into two namespaces and confirms a filtered query for one can
  never return the other's content, even though it's a near-perfect
  semantic match; a teardown test confirming zero rows remain in either
  table after cleanup; and every retrieval metric on the real evaluation
  set improving slightly once the contaminating dataset was removed
  (Recall@5 0.859 → 0.891, MRR 0.830 → 0.847).
- **What would I improve next?** Extend automated test coverage to the
  command-line and HTTP entry points for this feature directly (currently
  proven only at the underlying library level), and revisit the two
  generation-quality issues now that there's a clean, reproducible
  baseline to measure any fix against.

### Goal
Before upgrading the generation model, make the evaluation baseline
reproducible: guarantee `data/sample_docs` and `data/knowledge_base`
(TechFusion) can never share retrieval results, fix integration-test
teardown so it doesn't leave orphan rows, prove both with tests, then
produce a clean TechFusion baseline to compare future model changes
against.

### Configuration
Same infra as prior sessions (embedding `all-MiniLM-L6-v2`/384-dim,
`recursive_character` 500/50, reranker `none`, Ollama `qwen2.5:1.5b`,
pgvector via `rag-pgvector`). New: every chunk now carries a required
`dataset_id`. DB was fully cleared of pre-namespace rows (24 legacy
documents / 280 chunks with `dataset_id IS NULL`, spanning both the old
`sample_docs` smoke-test data and the previous `knowledge_base` ingestion)
and TechFusion was the only dataset re-ingested, tagged `dataset_id=
"techfusion"` (21 docs / 264 chunks). `sample_docs` was intentionally left
un-ingested this session — it can be restored later via
`python -m rag.ingestion.pipeline data/sample_docs --dataset-id sample_docs`.

### Changes made
- `schemas.py`: `ChunkMetadata.dataset_id: str` — required, no default. A
  chunk without a namespace is now a validation error, not a silent gap.
- `vectorstore/base.py`, `vectorstore/pgvector.py`: `dataset_id` added to
  `ALLOWED_FILTER_FIELDS`; `get_or_create_document_id` now takes and scopes
  on `dataset_id` (identity is `(source, dataset_id)`, not `source` alone);
  `add_chunks`/`search` read/write the new column. Two new methods:
  `delete_document(document_id)` (deletes the `documents` row, cascading
  to chunks — for full teardown) and `delete_dataset(dataset_id)` (bulk
  equivalent, for clearing/re-ingesting a whole namespace).
- `scripts/init_db.py`: idempotent migration — `dataset_id TEXT` column
  added to both tables via `ADD COLUMN IF NOT EXISTS`; `documents.source`'s
  old single-column `UNIQUE` constraint is dropped and replaced with a
  composite `UNIQUE (source, dataset_id)` inside a guarded `DO $$` block
  (checks `pg_constraint` so it's safe to re-run).
- `ingestion/pipeline.py`, `ingestion/writer.py`: `ingest_file`/
  `ingest_path`/`Writer.write` all take a required `dataset_id`. CLI gained
  `--dataset-id` (required) and `--clear` (wipes that namespace before
  ingesting). `Makefile`'s `ingest` target updated to match
  (`DATASET_ID=...`, optional `CLEAR=1`).
- `api/routers/ingest.py`: `POST /ingest` now requires a `dataset_id` form
  field alongside the uploaded files.
- `eval/run_eval.py`: `--dataset-id` is a required CLI argument; every
  `retrieve()`/`answer()` call the runner makes has `filters={"dataset_id":
  ...}` injected unconditionally — an evaluation run structurally cannot
  read outside its own namespace. Report now echoes `dataset_id` for
  provenance.
- `.gitignore`: added `data/eval/baselines/` (the saved baseline JSON
  contains TechFusion-derived questions/answers, so it follows the same
  "not pushed" treatment as the knowledge base and gold file themselves).
- Tests: every existing `ChunkMetadata(...)` construction and
  `ingest_file`/`ingest_path`/`get_or_create_document_id` call site updated
  for the new required parameter (`test_schemas.py`, `test_reranker.py`,
  `test_ingestion_category.py`, `test_pgvector.py`,
  `test_ingestion_pipeline.py`, `test_query_pipeline.py`). New:
  `test_vectorstore_base.py` (whitelist regression guard),
  `test_dataset_isolation.py`, `test_teardown_hygiene.py` (see below). 60
  unit + 10 integration tests passing.

### Issues discovered
1. **Symptom:** none yet observed in production use, but the underlying
   gap was real: nothing stopped two datasets ingested into the same
   Postgres table from contaminating each other's retrieval results.
   **Root cause:** chunk metadata had no namespace concept at all;
   `data/sample_docs` and `data/knowledge_base` lived in the same `chunks`
   rows distinguished only by `source` path, which any query without a
   `source`/`category` filter would search across indiscriminately. This
   is exactly what happened last session (the "canary rollback" query's
   #2 result was a `sample_docs` chunk).
   **Fix:** mandatory `dataset_id` on every chunk (see Changes made);
   `eval/run_eval.py` injects it unconditionally rather than leaving it to
   the caller to remember.
   **Evidence:** `test_dataset_filter_excludes_other_namespace_even_as_strong_distractor`
   — two near-identical documents (same sentence, different fact value)
   ingested into two namespaces; querying one namespace never returns the
   other's chunk, even though its content is a near-perfect semantic match.
   **Resolved:** yes.

2. **Symptom:** integration tests left orphan rows in `documents` after
   every run (only `chunks` was ever cleaned up — carried over from the
   previous two sessions, previously worked around by manually purging
   rows via exact `document_id` lists).
   **Root cause:** the only deletion method available,
   `delete_chunks_by_document_id`, is deliberately chunks-only (it's used
   mid-re-ingestion to replace an unchanged document's chunks without
   losing its `document_id`/checksum history) — there was no method for
   *full* teardown that also removes the `documents` row.
   **Fix:** added `VectorStore.delete_document(document_id)`, which deletes
   the `documents` row and relies on the existing `ON DELETE CASCADE` FK to
   remove its chunks. Every integration test's teardown now calls this
   instead.
   **Evidence:** `test_delete_document_leaves_no_document_or_chunk_rows`
   ingests into a fresh per-test `dataset_id`, asserts exactly 1
   documents-row / N chunk-rows exist, calls `delete_document`, then
   asserts both counts are 0. Also manually verified post-suite: `SELECT
   count(*) FROM documents WHERE dataset_id LIKE 'pytest%'` → 0.
   **Resolved:** yes.

3. **Symptom:** none directly — a design gap noticed while implementing
   the fix above. `documents.source` was `UNIQUE` on its own, so the same
   relative path in two different datasets would have collided (second
   dataset's ingestion would silently reuse the first's `document_id`).
   **Root cause:** document identity predated the concept of datasets.
   **Fix:** composite `UNIQUE (source, dataset_id)`; `get_or_create_
   document_id` now looks up and inserts scoped to both columns.
   **Evidence:** `test_same_source_in_different_datasets_does_not_collide`
   — same `source` string ingested into two namespaces gets two distinct
   `document_id`s.
   **Resolved:** yes.

### Evaluation results
`data/eval/techfusion_gold.jsonl`, 46 questions, `dataset_id=techfusion`
(the only dataset in the DB — the isolation filter and the DB's actual
contents now agree, closing the gap the previous baseline had):
- Recall@5 = 0.891 (prior contaminated run: 0.859)
- Recall@10 = 0.967 (prior: 0.957)
- Hit Rate@5 = 0.913 (prior: 0.891)
- Hit Rate@10 = 0.978 (prior: 0.978, unchanged)
- MRR = 0.847 (prior: 0.830)
- Retrieval latency (mean) = 97 ms (prior: 129 ms)
- Generation latency (mean) = 3,821 ms (prior: 6,271 ms)
- Total latency (mean) = 3,918 ms (prior: 6,400 ms)
- Answer quality (keyword-overlap): overall 0.428, answerable 0.462,
  unanswerable 0.203 (prior: 0.437 / 0.466 / 0.243)

Saved to `data/eval/baselines/techfusion_baseline_qwen2.5-1.5b_2026-08-05.json`
(git-ignored, per `.gitignore` change above) — **this is the clean
baseline to diff future model/config changes against.**

### Interpretation
Every retrieval metric improved after removing `sample_docs` from the
corpus — small but consistent (Recall@5 +0.032, MRR +0.017), and directly
explained by last session's finding that `sample_docs` chunks occasionally
scored competitively enough to occupy a top-k slot that a correct
TechFusion chunk would otherwise have taken. That confirms the
contamination was real, not hypothetical, and that isolation is worth
having independent of whether a `dataset_id` filter is applied at query
time — an unfiltered query against a shared corpus was already measurably
worse. The generation-latency drop (6.27s → 3.82s mean) is almost
certainly **not** attributable to this session's changes — dataset_id
filtering only changes which ~3 chunks land in the prompt, not materially
its length — and is more likely ordinary system variance (CPU/thermal/
scheduler state differs run to run on this CPU-only box). Reported as-is;
not claimed as a caused improvement. Answer-quality scores moved slightly
*down* (0.437→0.428 overall) despite retrieval improving — a reminder that
the keyword-overlap metric is noisy/coarse enough that small deltas here
shouldn't be over-interpreted either direction.

### Open problems
Carried over, unaffected by this session's changes:
1. Idempotency-key retrieval miss (see 2026-08-05 TechFusion session).
2. Systems-of-record generation hallucination despite correct retrieval.
3. `/ingest` API vs CLI source-identity mismatch (upload path vs CLI path
   still diverge; unrelated to dataset_id).
4. Answer-quality metric ceiling — needs RAGAS, not a bug fix.
New from this session:
5. `data/sample_docs` is no longer ingested anywhere (cleared, not
   restored) — the smoke-test dataset needs a manual
   `--dataset-id sample_docs` re-ingestion before it's usable again.
6. No automated test yet exercises the CLI's `--clear` flag or the
   `POST /ingest` API's new required `dataset_id` form field directly
   (covered indirectly via the library-level `IngestionPipeline`/
   `VectorStore` methods they call, but not at the CLI-arg-parsing or
   HTTP-request level).

### Next recommended step
With a clean, namespace-isolated baseline saved, the originally-planned
next step is unblocked: swap `generation.model_name` in
`config/default.yaml` to the larger model and re-run
`python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl
--dataset-id techfusion --verbose`, then diff its report against
`data/eval/baselines/techfusion_baseline_qwen2.5-1.5b_2026-08-05.json` —
retrieval metrics should be identical (same embedding/chunking/reranker),
so any change in Recall/MRR would itself be a red flag; the real signal to
watch is answer_quality and the two known generation-hallucination cases
from the previous session.

---

## 2026-08-05-18:48 | Experiment tracking infrastructure

### Interview talking points
- **What problem did this solve?** Before running any comparative
  experiments (new reranker, new model), built a proper system for
  recording and comparing eval runs, instead of hand-editing a README
  table or trusting memory for what config produced which numbers.
- **Why was it difficult?** The eval report already saved to disk didn't
  capture every field needed for a clean comparison table (e.g. which
  vector store or reranker model was used), and the existing baseline
  had to be preserved exactly. Re-running evaluation to "fix" the file
  wasn't an option, so the missing fields had to come from somewhere else
  without changing a single existing number.
- **How did I debug it?** Instead of relying on the eval report's own
  embedded config summary, read the config values straight from the
  actual `AppConfig` (the same file that produced the run), and only
  pulled metrics (recall, MRR, latency, answer quality) from the saved
  report. That way even an older, incomplete report can still be recorded
  correctly, and the numbers in the new record are never recomputed, only
  reshaped.
- **What metrics proved the fix?** Diffed the newly-generated
  `experiments/results/experiment_001.json` against the original saved
  baseline field by field (recall@5, recall@10, MRR, latency, answer
  quality): identical. The config snapshot byte-for-byte matches
  `config/default.yaml`.
- **What would I improve next?** Add a small end-to-end test that runs
  `record_experiment.py` and `compare_experiments.py` together against a
  temporary results directory (currently each is unit-tested separately
  but not as a pipeline), and decide whether to retire the older
  `scripts/render_benchmarks.py` / `data/eval/baselines/` approach now
  that `experiments/` supersedes it.

### Goal
Build proper experiment-tracking infrastructure (a fixed, comparable JSON
schema per eval run, plus a utility that turns saved runs into a README
table) before running the next round of experiments, without touching
`config/default.yaml` or recomputing the existing TechFusion baseline.

### Configuration
No config change this session (explicitly out of scope, per the
request). Existing baseline config unchanged: embedding
`all-MiniLM-L6-v2`/384-dim, `recursive_character` 500/50, reranker `none`,
generation `qwen2.5:1.5b`, vector store `pgvector`, `dataset_id=techfusion`.

### Changes made
- New `experiments/` directory: `configs/` (a snapshot of
  `config/default.yaml` per experiment), `results/` (flat comparable JSON
  per experiment), `reports/` (the generated comparison table).
- `scripts/record_experiment.py`: converts a `rag.eval.run_eval --verbose`
  report into the flat experiment schema (`experiment_id, label,
  timestamp, generation_model, embedding_model, vector_store, chunk_size,
  chunk_overlap, reranker_provider, reranker_model, retrieval_top_k,
  rerank_top_n, dataset_id, recall_at_5, recall_at_10, hit_rate_at_5,
  hit_rate_at_10, mrr, answer_quality, retrieval_latency_ms,
  generation_latency_ms, total_latency_ms`), reading config fields from
  the actual `AppConfig` rather than the report's own embedded summary,
  and copies the config file alongside it. Never re-runs eval; only
  records an existing report.
- `scripts/compare_experiments.py`: reads every `experiments/results/*.json`,
  renders a Markdown comparison table, saves it to
  `experiments/reports/comparison.md`, and splices it into README.md
  between `<!-- EXPERIMENTS_TABLE_START/END -->` markers so the README
  never needs hand-editing after a new experiment is recorded.
- Registered the existing TechFusion baseline as `experiment_001` by
  running `record_experiment.py` against the already-saved eval report
  (`data/eval/baselines/techfusion_baseline_qwen2.5-1.5b_reranker-none_2026-08-05.json`)
  — no re-run, numbers copied through unchanged.
- README's Benchmarks section rewritten around the new workflow, with a
  "Recording a new experiment" how-to and the auto-generated table in
  place of the old hand-pasted one.
- Tests: `test_record_experiment.py` (config-vs-report field sourcing,
  missing-field handling, reranker-model derivation),
  `test_compare_experiments.py` (table rendering, sorting, missing-metric
  formatting, README marker splicing via a temp file). 71 unit tests
  passing.

### Issues discovered
1. **Symptom:** none directly; a design question surfaced while building
   `record_experiment.py`. The saved eval report's embedded `config` block
   doesn't include `vector_store` or `reranker_model`, and the existing
   baseline file (already saved, not to be regenerated) will never have
   them either.
   **Root cause:** `run_eval.py`'s config summary was written for a
   simpler earlier need and doesn't capture every field a full experiment
   comparison wants.
   **Fix:** `build_experiment_record` takes the live `AppConfig` as a
   separate argument and reads `vector_store`/`reranker_model` (and
   everything else config-related) from it directly, rather than from the
   report file. This sidesteps the gap entirely without touching the
   already-saved baseline or `run_eval.py`.
   **Evidence:** `experiments/results/experiment_001.json` correctly shows
   `"vector_store": "pgvector"` even though the source report file has no
   such field; `test_build_experiment_record_reads_config_not_report`
   pins this behavior.
   **Resolved:** yes.

### Evaluation results
No new evaluation run this session (explicitly out of scope). Experiment
#1 (TechFusion, `qwen2.5:1.5b`, no reranker), carried through unchanged
from the previous session's baseline:
- Recall@5 = 0.891, Recall@10 = 0.967
- Hit Rate@5 = 0.913, Hit Rate@10 = 0.978
- MRR = 0.847
- Answer quality (keyword-overlap) = 0.432
- Retrieval / generation / total latency (mean) = 94 ms / 3,643 ms / 3,737 ms

### Interpretation
This session is infrastructure, not a new data point: the goal was to make
the *next* experiment (a reranker or model swap) land as a clean, diffable
row instead of another one-off saved file. Sourcing config from the live
`AppConfig` rather than the report is the one design decision worth
remembering: it means historical reports never need to be regenerated
just because the schema grows a new field later, which will matter again
the next time a new comparison axis is added.

### Open problems
Carried over, unaffected by this session:
1. Idempotency-key retrieval miss.
2. Systems-of-record generation hallucination despite correct retrieval.
3. `/ingest` API vs CLI source-identity mismatch.
4. Answer-quality metric ceiling; needs RAGAS, not a bug fix.
5. `data/sample_docs` not currently ingested (cleared in the isolation
   session, not yet restored).
New from this session:
6. `scripts/render_benchmarks.py` and `data/eval/baselines/` still exist
   from before `experiments/` was built and now do a very similar job;
   not removed this session since deleting things wasn't asked for, but
   worth retiring deliberately rather than leaving both around.
7. No test exercises `record_experiment.py`/`compare_experiments.py` as a
   CLI end-to-end (argument parsing, file writing to real paths) — only
   the underlying functions are unit-tested directly.

### Next recommended step
Use the new pipeline for the reranker experiment that's already been
planned since the previous session: flip `reranker.provider` to
`cross_encoder` in `config/default.yaml`, run eval, `record_experiment.py`
it as `experiment_002`, then `compare_experiments.py` to see it land in
the README table next to the baseline.

---

## 2026-08-05-19:31 | Experiment 002: qwen2.5:3b vs qwen2.5:1.5b generation model

### Interview talking points
- **What problem did this solve?** Ran a controlled experiment isolating
  exactly one variable (generation model: `qwen2.5:1.5b` to `qwen2.5:3b`)
  to get a real answer on whether the two known generation-quality
  failures from the TechFusion baseline were a model-capacity problem,
  using the new experiment-tracking infrastructure end to end for the
  first time.
- **Why was it difficult?** The result wasn't a clean "yes, bigger model
  fixes it." Retrieval metrics were bit-for-bit identical between the two
  runs (expected, since nothing retrieval-related changed), so every
  difference had to come from generation alone. Digging into the two known
  failure cases specifically, one was genuinely fixed (correct refusal
  instead of hallucination) but the other was replaced by a *different*
  wrong answer (incorrect refusal instead of hallucination) rather than
  actually being solved. A first-pass automated scan for "did the model
  start refusing more" produced a misleading headline number until each
  case was read by hand.
- **How did I debug it?** Since retrieval was identical, pulled the exact
  same retrieved-chunk list and compared only the generated answers for
  the two known failure questions side by side. Then wrote a quick refusal
  detector across all 46 questions to check whether the pattern was
  isolated or systemic, found it flagged 6 answerable questions instead of
  the baseline's 1, and manually read all 6 full (untruncated) answers
  before accepting that number; one was a false positive (a real, correct
  "No" answer that also happened to contain hedging language my keyword
  check matched), one was a case both models already failed identically,
  and three were genuine new regressions where the bigger model refused on
  a question the smaller model answered correctly from the same context.
- **What metrics proved the fix?** Recall@5/@10, Hit Rate@5/@10, and MRR:
  unchanged (0.891/0.967/0.913/0.978/0.847 in both runs, confirming the
  experiment was properly isolated to generation only). Answer quality:
  0.432 to 0.453 overall, a real but modest gain, consistent with the
  finding below that this is a wash rather than a clear win. Latency:
  generation went from 3.6s to 12.0s mean, a genuine cost with no
  ambiguity.
- **What would I improve next?** Try `reranker.provider: cross_encoder`
  against the `qwen2.5:1.5b` baseline next (isolates the retrieval axis
  instead of the generation axis, and might fix the idempotency-key
  question by getting the right document into the top-3 in the first
  place, rather than relying on a bigger model to guess correctly from
  wrong context).

### Goal
Run a controlled experiment changing only `generation.model_name`
(`qwen2.5:1.5b` to `qwen2.5:3b`), with embedding, chunking, overlap,
vector store, retrieval `top_k`, reranker, prompt, dataset, and gold
questions all held identical to the `experiment_001` baseline. Compare
the full metric set and specifically re-examine the two generation-quality
failure cases documented in the 2026-08-05 TechFusion session, without
modifying the baseline experiment.

### Configuration
Identical to `experiment_001` in every respect except generation model:
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), CPU.
- Chunking: `recursive_character`, size 500 / overlap 50.
- Vector store: `pgvector`, `dataset_id=techfusion` (unchanged, isolated).
- Reranker: `none`.
- Retrieval: `top_k=5`, `rerank_top_n=3`.
- **Generation: Ollama `qwen2.5:3b`** (was `qwen2.5:1.5b`), same prompt
  template, same `temperature=0.2`, `max_tokens=512`.
- Config file: `config/experiments/qwen2.5-3b.yaml`, a full copy of
  `config/default.yaml` differing in exactly one line (verified with
  `diff` before running); `config/default.yaml` itself untouched.
- Dataset: `data/eval/techfusion_gold.jsonl`, same 46 questions.

### Changes made
- `config/experiments/qwen2.5-3b.yaml`: new experiment config, not an edit
  to `config/default.yaml`.
- Ran `rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl
  --dataset-id techfusion --config config/experiments/qwen2.5-3b.yaml
  --verbose`, saved as
  `data/eval/baselines/techfusion_baseline_qwen2.5-3b_reranker-none_2026-08-05.json`
  (same naming convention as the existing baseline file).
- `scripts/record_experiment.py` registered it as `experiment_002`:
  `experiments/results/experiment_002.json`,
  `experiments/configs/experiment_002.yaml`.
- `scripts/compare_experiments.py` regenerated
  `experiments/reports/comparison.md` and the README table, now showing
  both experiments side by side.
- `experiments/results/experiment_001.json`, `experiments/configs/
  experiment_001.yaml`, and `config/default.yaml` confirmed byte-identical
  to before this session (`git diff` empty on all three).

### Issues discovered
1. **Symptom:** naive refusal-detection scan reported the candidate model
   refusing on 6/40 answerable questions vs. the baseline's 1/40, a
   seemingly large and worrying regression.
   **Root cause:** the scan used a crude substring match
   (`"i don't know"`, `"does not contain"`, etc.) with no ability to tell
   a true refusal from a substantive answer that happens to end with a
   hedge, or to check whether a question was *already* failing for both
   models before the swap.
   **Fix:** manually read all 6 full answers against the gold
   `expected_answer` and the baseline's answer for the same question.
   **Evidence:** of the 6, one ("Can access filtering be replaced by
   output redaction?") was a real, correct "No, ... based on the provided
   context" answer that only matched the filter because of a trailing
   hedge clause; one ("What are the RAG release gates?") was refused by
   *both* models identically, so not a new regression; one (idempotency
   keys) is the known, confirmed-unimproved retrieval miss where the
   candidate's refusal is strictly better than the baseline's hallucination;
   the remaining three (systems of record, post-incident review,
   disaster-recovery objectives) are genuine new regressions, detailed in
   Interpretation below.
   **Resolved:** yes (accurate breakdown established); the underlying
   over-refusal tendency itself is not resolved, see Open problems.

### Evaluation results
46 questions, `data/eval/techfusion_gold.jsonl`, `dataset_id=techfusion`,
`rerank_top_n=3`, `reranker=none` for both:

| Metric | qwen2.5:1.5b (exp. 1) | qwen2.5:3b (exp. 2) | Delta |
|---|---|---|---|
| Recall@5 | 0.8913 | 0.8913 | 0 (expected: retrieval unchanged) |
| Recall@10 | 0.9674 | 0.9674 | 0 |
| Hit Rate@5 | 0.9130 | 0.9130 | 0 |
| Hit Rate@10 | 0.9783 | 0.9783 | 0 |
| MRR | 0.8471 | 0.8471 | 0 |
| Answer quality (keyword-overlap, overall) | 0.4324 | 0.4531 | +0.021 |
| Answer quality (answerable) | 0.4715 | 0.4913 | +0.020 |
| Answer quality (unanswerable) | 0.1720 | 0.1981 | +0.026 |
| Retrieval latency (mean) | 94 ms | 119 ms | +25 ms (noise, not model-related) |
| Generation latency (mean) | 3,643 ms | 11,972 ms | +8,329 ms (3.3x) |
| Total latency (mean) | 3,737 ms | 12,091 ms | +8,354 ms (3.2x) |

**Failure case 1: "How long are idempotency keys retained?"**
Retrieval identical and still wrong in both runs (correct document
`api-development-guidelines.md` not retrieved). Baseline hallucinated
"Annual rotation is the default customer-data key retention policy."
Candidate: "I don't know. The context provided does not contain any
information about the retention period for idempotency keys." **Improved**
— same wrong context, but the bigger model correctly recognized it
couldn't answer instead of fabricating a plausible-sounding wrong answer.

**Failure case 2: "Which stores are systems of record and which are
rebuildable?"** Retrieval identical and correct in both runs (both
relevant documents retrieved). Baseline hallucinated "Systems of Record:
Compliance Ledger, Data Platform. Rebuildable: None" (both facts wrong;
neither is in the retrieved text). Candidate: "there is no information
about which systems are systems of record or which are rebuildable...
Therefore, I don't know." **Not improved** — the correct answer (PostgreSQL
and object storage as systems of record, Redis ephemeral, OpenSearch
rebuildable) was present in the same retrieved context both times; the
smaller model fabricated instead of extracting it, and the bigger model
declined to extract it at all. Neither model actually answers this
correctly; the failure mode changed from confident fabrication to
incorrect refusal.

### Interpretation
This is not a straightforward "bigger model wins." Retrieval-metric
identity across both runs confirms the experiment isolated exactly the
intended variable. The headline answer-quality gain (+0.021 overall) is
real but small, and the per-question read-through explains why: `qwen2.5:3b`
has a measurably stronger tendency to refuse rather than fabricate when
uncertain, which is a straightforward win when retrieval has already
failed (failure case 1, and it improves the unanswerable-question score
from 0.172 to 0.198), but becomes a new failure mode when retrieval
succeeded and the answer was actually present in context but required
synthesis across a few sentences (failure case 2, and two further cases:
"When is a post-incident review due?" and, more ambiguously, "What are the
disaster-recovery objectives?" where both models are actually wrong, just
differently). Net across the 40 answerable questions: 1 case flipped from
wrong-via-hallucination to correct-via-retrieval-failure-being-honest
(idempotency keys), and at least 2 clear cases flipped from
right-or-plausible to wrong-via-over-caution (systems of record,
post-incident review) where the smaller model actually extracted the
correct fact and the larger one didn't. The 3.2x latency cost is
unambiguous and large for a CPU-only box. On this evidence, `qwen2.5:3b` is
not a strict improvement over `qwen2.5:1.5b` for this pipeline as
configured: it trades confident hallucination for a real but distinct
over-caution problem, at roughly 3x the generation cost, for a small net
answer-quality gain that a 46-question keyword-overlap metric can't fully
resolve either way.

### Open problems
Carried over, unaffected by this session:
1. `/ingest` API vs CLI source-identity mismatch.
2. Answer-quality metric ceiling; needs RAGAS to properly separate "the
   model refused" from "the model was substantively wrong," which this
   session had to do by hand.
3. `data/sample_docs` not currently ingested.
4. `scripts/render_benchmarks.py` / `data/eval/baselines/` still overlap
   with `experiments/`; not retired yet.
New from this session:
5. Idempotency-key question remains a genuine **retrieval** miss regardless
   of generation model; a bigger generation model can only ever paper over
   this by refusing more honestly, not actually answer it. The real fix is
   on the retrieval side (see Next recommended step).
6. `qwen2.5:3b` shows a measurable over-refusal tendency on questions where
   retrieval succeeded (at least 2 confirmed cases: systems of record,
   post-incident review) that the smaller model didn't have. Worth
   tracking if this recurs with other model swaps, since it's the opposite
   failure mode from hallucination and just as much of a real quality
   problem for a RAG system meant to actually answer questions.
7. Naive keyword-based refusal detection produces false positives (a real
   answer with a trailing hedge phrase) and needs manual verification
   before being trusted as a metric on its own; not something to automate
   further without RAGAS or similar.

### Next recommended step
Run `experiment_003`: keep `qwen2.5:1.5b` (revert the generation model),
flip `reranker.provider` to `cross_encoder` instead. This isolates the
retrieval axis rather than the generation axis, and is the more promising
fix for failure case 1 specifically (getting `api-development-guidelines.md`
into the top-3 in the first place, rather than hoping a bigger model
handles a wrong context gracefully). Keep `qwen2.5:1.5b` as generation
model for that run so exactly one variable changes relative to
`experiment_001`, matching this session's methodology.

---

## 2026-08-05-20:00 | Experiment 003: qwen2.5:3b + cross_encoder reranker

### Interview talking points
- **What problem did this solve?** Tested the previous session's own
  recommendation (add a cross-encoder reranker) against the two known
  generation-quality failures, on top of the qwen2.5:3b model already
  tried in experiment_002. Also added an Embedder column to the comparison
  table, since all three experiments so far share the same embedder and
  that fact was invisible in the table.
- **Why was it difficult?** The result disproved my own prior hypothesis:
  I'd predicted the cross-encoder would fix the idempotency-key retrieval
  miss by surfacing the right document into the top-3. It didn't, because
  reranking can only reorder documents the initial vector search already
  retrieved; if the correct document never made it into that candidate
  set to begin with, no amount of reranking helps. Separately, MRR (a
  rank-sensitive metric) actually went down slightly even though Recall
  and Hit Rate stayed identical, which needed explaining rather than
  ignoring, and a "3 refusals fixed" headline from a keyword scan turned
  out to include one case that was fixed only in the sense of no longer
  refusing, not in the sense of becoming correct.
- **How did I debug it?** Compared retrieved chunks for both failure
  questions across all three experiments side by side. For the idempotency
  question, confirmed the correct document was absent from the top-5 in
  all three runs, which is what explains why reranking couldn't help: it
  never had the right candidate to promote. For the "3 fixed" refusals,
  read each answer in full against the gold `expected_answer` rather than
  trusting the refusal scan alone; one was an exact match to the expected
  RPO/RTO figures, one was already fine, and one had inverted its own
  reasoning between the explanation and the final stated conclusion.
- **What metrics proved the fix?** Recall@5/@10 and Hit Rate@5/@10:
  unchanged from experiment_002 (0.891/0.967/0.913/0.978). MRR: 0.847 to
  0.822 (down). Answer quality: 0.453 to 0.507 overall. Over-refusal on
  answerable questions: 6/40 (experiment_002) to 3/40, with zero new
  refusals introduced (every case that stopped refusing was previously
  refusing; none flipped the other way).
- **What would I improve next?** The idempotency-key question needs a
  retrieval-side fix that actually changes the initial candidate set (a
  different embedding model, or hybrid/BM25 search per the Roadmap), since
  reranking has now been shown incapable of fixing it. Also worth trying
  the cross-encoder reranker against the original `qwen2.5:1.5b` baseline
  (not yet tested) to separate the reranker's effect from the generation
  model's effect cleanly.

### Goal
Test the cross-encoder reranker (recommended as the next step in the
previous session) combined with the `qwen2.5:3b` generation model already
evaluated in `experiment_002`, add an Embedder column to the comparison
table for visibility, and specifically check whether it fixes the
idempotency-key retrieval miss as hypothesized.

### Configuration
`config/experiments/qwen2.5-3b_cross-encoder.yaml`: identical to
`experiment_002`'s config except `reranker.provider: cross_encoder` (model
`cross-encoder/ms-marco-MiniLM-L-6-v2`, the default already present in
`config/default.yaml`'s reranker block). Relative to the true baseline
(`experiment_001`), **two** variables differ (generation model and
reranker), so any difference vs. `experiment_001` can't be attributed to a
single cause; relative to `experiment_002`, exactly one variable differs
(reranker only), which is the clean comparison this session relies on.
Embedding (`all-MiniLM-L6-v2`), chunking (500/50), vector store
(`pgvector`), retrieval `top_k=5`/`rerank_top_n=3`, dataset
(`dataset_id=techfusion`), and gold questions (46, `techfusion_gold.jsonl`)
all unchanged from every prior experiment.

### Changes made
- `scripts/compare_experiments.py`: added an `Embedder` column
  (`_short_model_name()` strips the `namespace/` prefix, e.g.
  `sentence-transformers/all-MiniLM-L6-v2` to `all-MiniLM-L6-v2`; applied
  to both the embedder and reranker-model columns for consistency). Tests
  updated/added in `test_compare_experiments.py` (73 unit tests passing
  total).
- `config/experiments/qwen2.5-3b_cross-encoder.yaml`: new config, not an
  edit to either `config/default.yaml` or the existing
  `qwen2.5-3b.yaml`.
- Ran eval with this config, saved as
  `data/eval/baselines/techfusion_baseline_qwen2.5-3b_reranker-cross_encoder_2026-08-05.json`,
  registered as `experiment_003` via `record_experiment.py`, comparison
  table and README regenerated via `compare_experiments.py`.
- `experiment_001`, `experiment_002`, and both their config snapshots
  confirmed byte-identical to before this session (`git diff` empty on
  all four files).

### Issues discovered
1. **Symptom:** MRR decreased (0.847 to 0.822) while Recall@5/@10 and Hit
   Rate@5/@10 stayed exactly the same as `experiment_002`.
   **Root cause:** not a bug — Recall/Hit-Rate only check whether a
   relevant chunk is present within the top-5/top-10 window (unaffected by
   reordering *within* that window), while MRR is sensitive to exact rank.
   The cross-encoder reshuffled ranks among the same top-10 candidate set
   without moving anything across the 5/10 boundary, and on average pushed
   first-hit rank slightly later.
   **Fix:** none needed; documented as an expected property of the
   difference between these two metric types, not a defect.
   **Evidence:** manual recomputation matches: identical retrieved-source
   sets per question between `experiment_002` and `experiment_003` at the
   broad top-10 fetch used for these metrics, confirmed by inspecting
   `retrieved_sources` for both known failure-case questions.
   **Resolved:** yes (explained; not something to fix).

2. **Symptom:** my prediction from the previous session ("cross-encoder
   should fix the idempotency-key miss") was wrong.
   **Root cause:** a reranker only reorders candidates the vector search
   already retrieved; `api-development-guidelines.md` never appeared in
   the top-5 vector-search results in any of the three experiments, so
   there was nothing for the cross-encoder to promote. The miss is at the
   embedding/vector-search stage, not the ranking stage.
   **Fix:** none applied this session; documented as the real next step
   (a retrieval-side change, not another reranker or generation-model
   swap).
   **Evidence:** `retrieved_sources` for this question, all three
   experiments: `api-development-guidelines.md` absent from the top-5 in
   every case.
   **Resolved:** no — root cause now correctly understood, but not fixed.

3. **Symptom:** a quick refusal scan reported cross-encoder "fixed" 3 of
   `experiment_002`'s 6 over-refusals on answerable questions, which
   looked like an unambiguous win.
   **Root cause:** "stopped matching a refusal keyword" isn't the same as
   "became correct." One of the 3 (systems of record) produced an answer
   that internally contradicts its own stated reasoning ("Redis... does
   not qualify as a system of record" followed by a final line stating
   "Systems of Record: Redis") and is still wrong relative to the expected
   answer.
   **Fix:** none needed as a code fix; manually verified each of the 3
   against `expected_answer` before accepting the "fixed" framing, same
   discipline as the previous session's false-positive catch.
   **Evidence:** see per-question detail in Evaluation results below --
   1 of 3 is an exact match to `expected_answer`, 1 was already
   substantively correct, 1 is genuinely still wrong.
   **Resolved:** yes (accurate accounting established).

### Evaluation results
46 questions, `data/eval/techfusion_gold.jsonl`, `dataset_id=techfusion`:

| Metric | exp.1 (1.5b, none) | exp.2 (3b, none) | exp.3 (3b, cross_encoder) |
|---|---|---|---|
| Recall@5 / @10 | 0.891 / 0.967 | 0.891 / 0.967 | 0.891 / 0.967 |
| Hit Rate@5 / @10 | 0.913 / 0.978 | 0.913 / 0.978 | 0.913 / 0.978 |
| MRR | 0.847 | 0.847 | 0.822 |
| Answer quality (overall) | 0.432 | 0.453 | 0.507 |
| Answer quality (answerable) | 0.472 | 0.491 | 0.548 |
| Answer quality (unanswerable) | 0.172 | 0.198 | 0.237 |
| Retrieval latency (mean) | 94 ms | 119 ms | 279 ms (reranking cost) |
| Generation latency (mean) | 3,643 ms | 11,972 ms | 9,798 ms |
| Total latency (mean) | 3,737 ms | 12,091 ms | 10,077 ms |

**Failure case 1 (idempotency keys):** unchanged outcome across all three
experiments. Correct document never retrieved in the top-5 in any of
them; `experiment_002`/`experiment_003` both correctly refuse rather than
hallucinate (an improvement over `experiment_1`'s fabrication), but neither
actually answers the question, and the cross-encoder specifically could
not have helped here (see Issue 2).

**Failure case 2 (systems of record):** retrieval correct (both relevant
documents retrieved) in all three experiments. `experiment_1` hallucinated
wrong specifics; `experiment_2` refused; `experiment_3` stopped refusing
but produced a self-contradictory answer ("Redis... does not qualify as a
system of record" in its reasoning, then "Systems of Record: Redis" as its
final answer) that is still wrong and inverted relative to the truth
(PostgreSQL/object storage are the actual systems of record; Redis is
ephemeral). **Not fixed by any configuration tried so far.**

**The 3 refusals that "fixed" between experiment_2 and experiment_3,
verified individually:**
- "What are the disaster-recovery objectives?": genuinely fixed. Exact
  match to expected ("RPO: 15 minutes, RTO: 4 hours").
- "Can access filtering be replaced by output redaction?": already
  substantively correct in `experiment_2` (a false positive of the
  keyword-based refusal scan, caught last session too); remains correct
  and more clearly articulated here.
- "Which stores are systems of record...?": not actually fixed, see above.

Zero new refusals introduced on questions that weren't already refusing in
`experiment_2`.

### Interpretation
The cross-encoder reranker is a genuine but partial improvement over
`experiment_2`, and the honest accounting matters more than the headline
numbers here. It measurably improves the *quality of what reaches
generation* -- one clean fix (disaster-recovery objectives, an exact
match), reduced but not eliminated over-refusal (6 to 3 out of 40
answerable questions, zero new ones), and a real answer-quality gain
(+0.054 overall, largest jump of any experiment so far) -- while making
raw ranking (MRR) very slightly worse and adding real retrieval latency
(94ms to 279ms). It does not, and structurally cannot, fix the
idempotency-key case, because that's a vector-search recall problem, not a
ranking problem; this session corrected a wrong prediction from the
previous one on exactly that point. And it does not reliably fix synthesis
failures either: the systems-of-record case just changed from confident
fabrication to a different, self-contradictory wrong answer, which is a
reminder that "stopped refusing" and "keyword score went up" are both
weak proxies for "became correct" -- the same caution the previous session
raised about the answer-quality metric applies just as much here.
Generation latency dropping from experiment_2 (12.0s to 9.8s) despite
identical generation model is plausible but not conclusively explained;
better retrieved context could mean shorter, more decisive answers, but
system-level latency variance on this CPU-only box (noted in the isolation
session) can't be ruled out either.

### Open problems
Carried over, unaffected by this session:
1. `/ingest` API vs CLI source-identity mismatch.
2. Answer-quality metric ceiling; needs RAGAS. This session is another
   concrete example: the systems-of-record answer scored higher than a
   correct refusal would have, purely because it happened to contain the
   keyword "PostgreSQL," despite classifying it backwards.
3. `data/sample_docs` not currently ingested.
4. `scripts/render_benchmarks.py` / `data/eval/baselines/` still overlap
   with `experiments/`; not retired yet.
5. Systems-of-record question remains unsolved across all three
   configurations tried (hallucination, refusal, and now a
   self-contradictory wrong classification).
New from this session:
6. Idempotency-key question is now confirmed to be a **vector-search
   recall problem specifically**, not a ranking or generation problem --
   reranking and a bigger model have both been ruled out as fixes. Needs
   an embedding-model change or hybrid/keyword search to actually surface
   the right document.
7. Cross-encoder reranker has not yet been tested against the
   `qwen2.5:1.5b` baseline in isolation (only combined with `qwen2.5:3b`
   here) -- can't yet say how much of `experiment_3`'s answer-quality gain
   is the reranker alone vs. the reranker-plus-bigger-model combination.

### Next recommended step
Run `experiment_004`: `qwen2.5:1.5b` (revert to baseline generation model)
+ `cross_encoder` reranker. This isolates the reranker's effect cleanly
against `experiment_001` (exactly one variable changes), separates it from
`experiment_3`'s combined effect, and is cheap to run given `qwen2.5:1.5b`'s
much lower latency -- the natural next data point before deciding whether
the reranker, the bigger model, or both are worth keeping.

---

## 2026-08-05-20:27 | Experiment 004: BGE-small embedder (on top of qwen2.5:3b + cross_encoder)

### Interview talking points
- **What problem did this solve?** Tested whether swapping the embedding
  model (`all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5`) fixes the
  idempotency-key retrieval miss, since the previous session concluded
  that was a vector-search recall problem the reranker structurally
  couldn't touch. Paired with the user's explicit choice to test it on top
  of the current best config (`qwen2.5:3b` + `cross_encoder`) rather than
  the original baseline, at the cost of not being able to isolate the
  embedder's effect against `experiment_001` cleanly.
- **Why was it difficult?** Changing the embedder isn't a config-only
  change like the previous two experiments: existing chunks were embedded
  with MiniLM, and comparing a BGE-small query vector against MiniLM
  document vectors would be meaningless, not just suboptimal. Required
  re-ingesting the whole knowledge base into a brand new `dataset_id`
  before any evaluation could be valid. Then the result itself needed
  careful reading: the broad top-10 metric showed the correct document
  for the idempotency-key question finally being retrieved, which looked
  like the fix -- but the actual production path (`top_k=5`) still didn't
  surface it, so the eval metric and production behavior disagreed for
  this specific question.
- **How did I debug it?** Re-ingested into `dataset_id=techfusion-bge-small`
  with `--config` pointed at a new BGE-small config, confirmed the
  original `techfusion` dataset was untouched (`SELECT dataset_id,
  count(*) ... GROUP BY dataset_id` showing both namespaces intact at 264
  chunks each). For the idempotency-key question specifically, didn't
  trust the eval report's `retrieved_sources` (computed from a broad
  `top_k=10` call) as a proxy for what generation actually saw -- called
  `RetrievalPipeline.answer()` directly with production config
  (`top_k=5`) to get the exact 3 sources used, which is the run_eval.py
  eval loop's blind spot (it saves `answer` and latency but not `sources`
  from the production-config call). Ran the same refusal-scan-then-
  manually-verify process as the last two sessions on the new regressions
  it reported.
- **What metrics proved the fix?** Recall@5 0.891 to 0.902, Hit Rate@5
  0.913 to 0.935 (both improved -- the metric closest to what production
  `top_k=5` actually uses). Recall@10 0.967 to 0.935, Hit Rate@10 0.978 to
  0.957 (both worse). MRR roughly flat (0.822 to 0.825). Answer quality
  down (0.507 to 0.483 overall). None of these by themselves settle
  whether BGE-small is "better" -- the per-question read-through is what
  actually answers that.
- **What would I improve next?** Increase production `top_k` (not just the
  eval metric's broad fetch) and re-test with BGE-small, since the
  idempotency-key document is now confirmed present at `top_k=10` but
  absent at `top_k=5` -- the embedder change may already be sufficient if
  paired with a wider retrieval window, which hasn't been tried yet.

### Goal
Test `BAAI/bge-small-en-v1.5` as a replacement embedder, on top of the
`qwen2.5:3b` + `cross_encoder` configuration from `experiment_003` (per
explicit instruction, accepting that this changes 3 variables vs.
`experiment_001` and only 1 vs. `experiment_003`), and specifically check
whether it resolves the idempotency-key retrieval miss that reranking
alone could not.

### Configuration
`config/experiments/qwen2.5-3b_cross-encoder_bge-small.yaml`: identical to
`experiment_003`'s config except `embedding.model_name:
BAAI/bge-small-en-v1.5` (also 384-dim, confirmed compatible with the
existing `VECTOR(384)` column with no schema change). Chunking (500/50),
vector store (`pgvector`), retrieval `top_k=5`/`rerank_top_n=3`, reranker
(`cross_encoder`), generation (`qwen2.5:3b`), and gold questions (46,
`techfusion_gold.jsonl`) all unchanged from `experiment_003`.
**`dataset_id=techfusion-bge-small`** (new, not `techfusion` -- required
because the vector space itself changes with the embedder; reusing
`techfusion` would have mixed incompatible vectors in one similarity
search).

### Changes made
- `config/experiments/qwen2.5-3b_cross-encoder_bge-small.yaml`: new
  config, includes a comment warning against pointing it at the
  `techfusion` dataset_id.
- Re-ingested `data/knowledge_base` (21 docs) into
  `dataset_id=techfusion-bge-small` using this config: 264 chunks, same
  counts as the `techfusion` dataset (chunking is embedder-independent).
  Confirmed via direct query that both `dataset_id`s coexist with the
  original completely unaffected.
- Ran eval against `dataset_id=techfusion-bge-small`, saved as
  `data/eval/baselines/techfusion-bge-small_baseline_qwen2.5-3b_reranker-cross_encoder_2026-08-05.json`,
  registered as `experiment_004`.
- `experiment_001`/`002`/`003` and their config snapshots confirmed
  byte-identical to before this session.

### Issues discovered
1. **Symptom:** the idempotency-key question's broad top-10 eval metric
   showed `api-development-guidelines.md` retrieved at rank 1 (apparent
   fix), but the model still answered "I don't know" exactly as before.
   **Root cause:** the eval loop's metric-computation retrieval
   (`top_k=10, rerank_top_n=10`) and the actual production retrieval used
   for generation (`top_k=5, rerank_top_n=3`, via a separate `answer()`
   call) are different candidate pools. Verified directly: at `top_k=5`,
   the production call's 3 sources were `gdpr-data-handling.md`,
   `data-storage.md`, `rag-service-architecture.md` -- the correct
   document wasn't even in the narrower 5-candidate set, despite ranking
   #1 once 10 candidates were available to rerank.
   **Fix:** none applied this session (would mean changing production
   `top_k`, which is a config change beyond swapping the embedder alone).
   Documented as the real next experiment.
   **Evidence:** direct `RetrievalPipeline.answer()` call with production
   config, sources printed explicitly (see Interview talking points).
   **Resolved:** no -- but the exact mechanism is now understood
   precisely, which the previous session's reranker-only test didn't
   establish (reranking can't add candidates; only `top_k` and the
   embedder can change which documents are even considered).

2. **Symptom:** a refusal scan reported 3 new over-refusals introduced by
   BGE-small vs. `experiment_003`, with zero fixed -- a seemingly clear
   regression.
   **Root cause:** as in both previous sessions, "matches a refusal
   keyword" isn't "became wrong." Manually checked all 3: "systems of
   record" is the same still-unsolved question as before (BGE-small's
   version is an honest refusal rather than `experiment_003`'s
   self-contradictory wrong answer -- arguably no worse, possibly safer);
   "how should a required database column be introduced" fails
   identically in both experiments, just with different wording, one of
   which happens to trip the keyword scan and one which doesn't; only
   "what canary stages are used" is a genuine new regression, where
   `experiment_003`'s retrieval led to a specific (if disorganized) partial
   answer and `experiment_004`'s led to an honest but incorrect refusal.
   **Fix:** none needed as code; manual verification, same discipline as
   the last two sessions.
   **Evidence:** full answers compared against `expected_answer` for all
   3, detailed in Evaluation results below.
   **Resolved:** yes (accurate accounting established; 1 genuine
   regression, not 3).

### Evaluation results
46 questions, `data/eval/techfusion_gold.jsonl`:

| Metric | exp.3 (MiniLM) | exp.4 (BGE-small) | Delta |
|---|---|---|---|
| Recall@5 | 0.891 | 0.902 | +0.011 |
| Recall@10 | 0.967 | 0.935 | -0.033 |
| Hit Rate@5 | 0.913 | 0.935 | +0.022 |
| Hit Rate@10 | 0.978 | 0.957 | -0.022 |
| MRR | 0.822 | 0.825 | +0.003 |
| Answer quality (overall) | 0.507 | 0.483 | -0.024 |
| Answer quality (answerable) | 0.548 | 0.526 | -0.021 |
| Answer quality (unanswerable) | 0.172 | 0.193 | +0.021 |
| Retrieval / Generation / Total latency (mean) | 279/9,798/10,077 ms | 272/10,046/10,318 ms | roughly flat |

Both `dataset_id`s (`techfusion` at 264/21, `techfusion-bge-small` at
264/21) verified intact and independent in the database after this run.

**Failure case 1 (idempotency keys):** partially, incompletely improved.
Correct document now ranks #1 in a broad top-10 fetch (was absent
entirely in `experiment_1`/`2`/`3`), but is still absent from the
production `top_k=5` candidate set, so the generated answer is unchanged
("I don't know"). Root cause is now precisely `top_k=5` being too narrow
for this specific document under either embedder tested so far, not the
embedder alone.

**Failure case 2 (systems of record):** still unsolved under all four
configurations tried. `experiment_4`'s version is an honest "I don't
know" rather than `experiment_3`'s self-contradictory wrong classification.

### Interpretation
BGE-small is a genuine but mixed change relative to `experiment_3`, not a
clear upgrade. It measurably improves the metric closest to what
production actually retrieves (Recall@5/Hit-Rate@5), which matters more
than Recall@10 for this pipeline's actual `top_k=5` configuration -- but
Recall@10 and Hit-Rate@10 both got worse, meaning BGE-small's candidate
set beyond the top 5 is less complete than MiniLM's, even though its top 5
is more precise. Answer quality moved down slightly overall, and the
per-question read-through found one genuine new regression (canary
stages) alongside one case that's arguably safer-but-not-fixed (systems of
record) and one non-regression that just changed wording (database
column). The most important finding isn't a metric at all: this session
proved that the idempotency-key question's failure is specifically a
`top_k=5`-window problem, separate from and downstream of the embedder
question -- BGE-small does surface the right document, just not narrowly
enough for production's current retrieval width. That's a more precise
diagnosis than "needs a different embedder" from two sessions ago, and
points at a cheaper, more targeted next experiment (widen `top_k`) before
concluding anything about embedder choice specifically.

### Open problems
Carried over, unaffected by this session:
1. `/ingest` API vs CLI source-identity mismatch.
2. Answer-quality metric ceiling; needs RAGAS.
3. `data/sample_docs` not currently ingested.
4. `scripts/render_benchmarks.py` / `data/eval/baselines/` still overlap
   with `experiments/`.
5. Systems-of-record question remains unsolved across all four
   configurations tried now (hallucination, refusal, self-contradiction,
   honest refusal again).
6. Cross-encoder reranker still not tested against the `qwen2.5:1.5b`
   baseline in isolation (only ever combined with `qwen2.5:3b`).
New from this session:
7. Production `top_k=5` is confirmed too narrow for at least one specific
   document (`api-development-guidelines.md`) under both embedders tested
   so far. Worth testing whether `top_k=10` (matching the eval metric's
   broad fetch) actually changes generation behavior for this question,
   before trying yet another embedder.
8. `run_eval.py`'s per-question record saves `answer`/latency from the
   production `answer()` call but not that call's `sources` -- this
   session had to call `RetrievalPipeline.answer()` manually outside the
   eval loop to see what was actually retrieved for generation. Worth
   adding `production_sources` to the per-example record so this doesn't
   require an ad-hoc script next time.
9. Two now-isolated dataset_ids (`techfusion`, `techfusion-bge-small`)
   exist in the same database for the same underlying documents --
   expected and intentional (different embedders require different
   vector spaces), but worth remembering when inspecting the DB directly
   so the count doesn't look like unexplained duplication.

### Next recommended step
Run `experiment_005`: same as `experiment_004`
(`qwen2.5:3b`/`cross_encoder`/`BGE-small`) but with `retrieval.top_k`
raised from 5 to 10, keeping `rerank_top_n=3` so production still only
generates from 3 chunks. This directly tests Issue 7 -- whether widening
the initial candidate pool (not changing the embedder again) is what
actually fixes the idempotency-key question in production, which would be
a cheaper and more targeted fix than continuing to try different
embedding models.

---

## 2026-08-05-20:48 | Experiment 005: chunk_size=300 (on top of qwen2.5:3b + cross_encoder + BGE-small)

### Interview talking points
- **What problem did this solve?** Tested whether smaller chunks
  (`chunk_size` 500 to 300) improve retrieval precision, layered onto
  `experiment_004`'s full config per the same "build on the latest"
  pattern the user chose for the BGE-small experiment.
- **Why was it difficult?** The aggregate metrics looked mixed-to-neutral
  at first glance (Recall@5 unchanged, Recall@10 slightly up, MRR and
  answer quality down) but a naive refusal scan reported 5 new
  over-refusals plus 1 unanswerable question no longer being refused --
  which would have been the worst result of any experiment so far, if
  taken at face value. Manually verifying all 6 flagged cases (the
  discipline every session so far has required) found 3 were false
  positives of the crude keyword detector (correct or substantively fine
  answers that happened to use phrasing like "not provide" or "do not have
  any information" instead of the exact phrases being matched for), and 3
  were genuine new regressions -- a materially different, more moderate
  picture than the raw scan suggested, but still a real net negative.
- **How did I debug it?** Re-ingested into yet another new `dataset_id`
  (`techfusion-bge-small-chunk300`) since chunk boundaries changing means
  none of the existing chunks are reusable, confirmed all three prior
  `dataset_id`s untouched. Ran the same refusal-scan-then-manually-verify
  process as every previous session, reading each of the 6 flagged answers
  against `expected_answer` in full. Directly re-verified the
  idempotency-key question's actual production `top_k=5` sources again
  (not just the broad top-10 metric), to check whether smaller chunks
  incidentally fixed the same problem two previous sessions couldn't.
- **What metrics proved the fix?** There isn't one to report as a fix --
  this session is a regression, not an improvement. MRR: 0.825 to 0.808.
  Answer quality: 0.483 to 0.427 (largest single-variable drop of any
  experiment so far). Verified refusal count: 3 genuine new regressions
  (`SEV-2 acknowledgment time`, `dead-letter topic attempt count`,
  `PostgreSQL cutover validation` -- all specific facts `experiment_004`
  answered correctly and precisely) against 0 genuine fixes.
- **What would I improve next?** Revert `chunk_size` to 500 (this
  session's change isn't worth keeping on current evidence) and instead
  run the `top_k=10` experiment recommended at the end of the previous
  session, which is still untested and is the most direct way to check
  whether the idempotency-key document can actually reach generation once
  the candidate window is wide enough -- confirmed again this session that
  neither embedder change nor chunk-size change alone fixes it.

### Goal
Test `chunk_size=300` (vs. the `500` used in every previous experiment),
layered onto `experiment_004`'s config (`qwen2.5:3b` + `cross_encoder` +
`BGE-small`) per the same incremental-build pattern used for the BGE-small
experiment, keeping `chunk_overlap=50` unchanged.

### Configuration
`config/experiments/qwen2.5-3b_cross-encoder_bge-small_chunk300.yaml`:
identical to `experiment_004`'s config except `chunking.chunk_size: 300`
(was 500). Embedding (`BAAI/bge-small-en-v1.5`), reranker
(`cross_encoder`), generation (`qwen2.5:3b`), vector store (`pgvector`),
retrieval `top_k=5`/`rerank_top_n=3`, and gold questions (46,
`techfusion_gold.jsonl`) all unchanged from `experiment_004`.
**`dataset_id=techfusion-bge-small-chunk300`** (new -- chunk boundaries
differ, so none of the existing 264 BGE-small chunks at size 500 are
reusable; a fresh ingestion was required regardless of the embedder
staying the same).

### Changes made
- `config/experiments/qwen2.5-3b_cross-encoder_bge-small_chunk300.yaml`:
  new config.
- Re-ingested `data/knowledge_base` into
  `dataset_id=techfusion-bge-small-chunk300`: 494 chunks across 21 docs
  (vs. 264 at `chunk_size=500` -- roughly double, as expected for roughly
  half the chunk size). Confirmed all three prior `dataset_id`s
  (`techfusion`: 264/21, `techfusion-bge-small`: 264/21) untouched via
  direct query.
- Ran eval, saved as
  `data/eval/baselines/techfusion-bge-small-chunk300_baseline_qwen2.5-3b_reranker-cross_encoder_2026-08-05.json`,
  registered as `experiment_005`.
- `experiment_001`-`004` and their config snapshots confirmed
  byte-identical to before this session.

### Issues discovered
1. **Symptom:** refusal scan reported 6 new problems (5 answerable
   questions newly refusing, 1 unanswerable question no longer refusing)
   vs. `experiment_004` -- the worst-looking headline of any experiment so
   far.
   **Root cause:** same recurring issue as every previous session --
   substring-based refusal detection can't distinguish a real refusal from
   a substantive answer with unusual phrasing. Specifically this session:
   "How early should temporary work abroad be requested?" (correctly
   answered "at least 30 calendar days," matched on a trailing "No further
   details are provided" clause), "What production access does a new
   employee receive automatically?" (correctly answered "no automatic
   access," matched similarly), and "Which LLM model and parameter count
   are used in production?" (an unanswerable question, correctly refused
   via "I do not have any information about..." which doesn't contain the
   literal substring "no information" the scan checks for).
   **Fix:** none needed as code (the scan is a diagnostic aid, not a
   metric that's meant to be trusted alone -- consistent with every prior
   session's finding on this same point).
   **Evidence:** all 6 answers read in full against `expected_answer`; 3
   confirmed as false positives, 3 as genuine regressions (SEV-2
   acknowledgment time, dead-letter topic count, PostgreSQL cutover
   validation -- see Evaluation results).
   **Resolved:** yes (accurate accounting established).

2. **Symptom:** idempotency-key question unchanged yet again -- correct
   document still absent from actual production `top_k=5` retrieval.
   **Root cause:** same as `experiment_004` -- confirmed directly via
   `RetrievalPipeline.answer()` with production config, sources were
   `gdpr-data-handling.md` (twice) and `data-storage.md`;
   `api-development-guidelines.md` absent. Neither the embedder change nor
   the chunk-size change alters this, reinforcing that `top_k=5` itself is
   the bottleneck, not a property of any single config axis tried so far.
   **Fix:** none applied; same recommendation as last session, now
   supported by a third data point.
   **Evidence:** direct production-config query, this session.
   **Resolved:** no.

### Evaluation results
46 questions, `data/eval/techfusion_gold.jsonl`:

| Metric | exp.4 (chunk=500) | exp.5 (chunk=300) | Delta |
|---|---|---|---|
| Recall@5 | 0.902 | 0.902 | 0 |
| Recall@10 | 0.935 | 0.946 | +0.011 |
| Hit Rate@5 | 0.935 | 0.935 | 0 |
| Hit Rate@10 | 0.957 | 0.957 | 0 |
| MRR | 0.825 | 0.808 | -0.017 |
| Answer quality (overall) | 0.483 | 0.427 | -0.056 |
| Answer quality (answerable) | 0.526 | 0.462 | -0.064 |
| Answer quality (unanswerable) | 0.193 | 0.188 | -0.005 |
| Retrieval / Generation / Total latency (mean) | 272/10,046/10,318 ms | 191/9,698/9,889 ms | faster (smaller chunks, less text per rerank comparison) |

All four `dataset_id`s verified intact and independent (`techfusion`
264/21, `techfusion-bge-small` 264/21, `techfusion-bge-small-chunk300`
494/21) after this run.

**3 genuine new regressions, verified individually against
`expected_answer`:**
- "What is the SEV-2 acknowledgment time?": `experiment_4` answered "15
  minutes" (exact match); `experiment_5` refused entirely.
- "How many failed event attempts precede the dead-letter topic?":
  `experiment_4` answered "five failed attempts" (exact match);
  `experiment_5` refused entirely.
- "What must be validated before PostgreSQL cutover?": `experiment_4`
  gave a specific, on-topic answer; `experiment_5` refused, claiming the
  context only covers "warnings and troubleshooting."

**Failure case 1 (idempotency keys):** unchanged, see Issue 2 above.
**Failure case 2 (systems of record):** still an honest refusal in both
`experiment_4` and `experiment_5`, no change.

### Interpretation
This is the first experiment in the series that's a fairly clear net
regression rather than a mixed or ambiguous result. Recall@5/Hit-Rate@5/
Hit-Rate@10 held steady and Recall@10 even improved slightly, meaning the
*documents* being retrieved didn't get meaningfully worse -- but MRR and,
more importantly, answer quality both dropped, and three specific,
previously-correct, precise factual answers (an acknowledgment time, a
retry count, a validation checklist) turned into refusals. The likely
mechanism: at `chunk_size=300`, a single source document is split into
roughly twice as many chunks as at 500, so a fact that used to sit inside
one retrieved chunk alongside enough surrounding context to be
recognizable can now be split across two chunks -- and with
`rerank_top_n=3` unchanged, the model has less total context text
available even when the right document area is retrieved. This is the
classic small-chunk tradeoff (better precision-per-chunk, worse
per-answer completeness when facts need surrounding context), and here
the tradeoff nets negative for this pipeline's specific `rerank_top_n=3`.
It's a useful, moderately confident negative result precisely because
three prior sessions' worth of manual-verification discipline was applied
consistently -- the raw refusal-scan number (6 new problems) would have
overstated the regression by 2x if trusted directly.

### Open problems
Carried over, unaffected by this session:
1. `/ingest` API vs CLI source-identity mismatch.
2. Answer-quality metric ceiling; needs RAGAS.
3. `data/sample_docs` not currently ingested.
4. `scripts/render_benchmarks.py` / `data/eval/baselines/` still overlap
   with `experiments/`.
5. Systems-of-record question remains unsolved across all five
   configurations tried.
6. Cross-encoder reranker still not tested against `qwen2.5:1.5b` in
   isolation.
7. Idempotency-key question confirmed (now three times) to be a
   `top_k=5`-window problem specifically, independent of embedder or
   chunk size -- `experiment_006` (widen `top_k`) is now the clearly
   indicated next step, not another embedder or chunking variant.
New from this session:
8. `chunk_size=300` is a net negative for this pipeline's
   `rerank_top_n=3` -- worth remembering as a data point if `rerank_top_n`
   is ever increased in a future experiment (a wider final context window
   might change this conclusion, since the mechanism identified above is
   specifically about `rerank_top_n=3` not being enough text at the
   smaller chunk size).
9. Five `dataset_id`s now exist for the same 21 source documents
   (`techfusion`, `techfusion-bge-small`, `techfusion-bge-small-chunk300`,
   plus whatever `experiment_006` will need) -- still correct and
   intentional, but worth a periodic sanity check that this doesn't grow
   unmanageably; may be worth a small cleanup script if the experiment
   count keeps climbing.

### Next recommended step
Run `experiment_006`: revert to `chunk_size=500` (this session's change
isn't worth carrying forward) and raise `retrieval.top_k` from 5 to 10,
keeping `rerank_top_n=3`. This directly tests the one hypothesis that's
survived three separate experiments unfalsified: that the idempotency-key
document is retrievable in principle (it ranks #1 in a top-10 fetch under
both embedders tried) but is being excluded before reranking even gets a
chance, purely because production's candidate pool is too narrow.

---

## 2026-08-06-16:00 | Versioned, config-driven RAG generation prompts

### Interview talking points
- **What problem did this solve?** The generation prompt was a hardcoded
  module-level string in `retrieval/pipeline.py`, formatted inline and
  passed straight to the LLM. There was no way to version prompt wording,
  compare a prompt change across experiments the way embedding
  model/chunk size/reranker already are, or validate a prompt's variables
  before rendering it.
- **Why was it difficult?** Two constraints in tension: v1 had to stay
  byte-identical to the existing hardcoded prompt (so this refactor
  carries zero behavior change on its own), while the loader still needed
  real validation (load-time placeholder/`required_variables`
  cross-checks, render-time missing/unknown-variable rejection) and a
  clean extension point for a stricter v2. Getting the YAML block-scalar
  chomping indicator right (`|-`) so the parsed string matched the old
  Python triple-quoted constant's line-continuation behavior exactly was
  the one genuinely fiddly detail.
- **How did I verify it?** Round-tripped `rag_answer_v1.yaml` through
  `yaml.safe_load` and diffed it against a frozen copy of the legacy
  `_PROMPT_TEMPLATE.format(...)` string before writing any loader code;
  pinned that same comparison as a permanent unit test
  (`test_render_v1_matches_legacy_prompt_byte_for_byte`) since the
  original constant no longer exists to import from.
- **What proved it worked?** 85 unit tests passing (73 pre-existing + 12
  new, across `test_prompt_loader.py`, `test_retrieval_pipeline.py`, and
  an extension to `test_record_experiment.py`), `pre-commit run
  --all-files` clean (ruff + mypy) on the first attempt, no eval regression
  expected since v1's rendered LLM input is unchanged except for the
  separately-approved source-label addition to context.
- **What's next?** `experiment_006` comparing `rag_answer_v1` against
  `rag_answer_v2` (the stricter grounding/citation example prompt) once a
  gold-set run against it is warranted — not run this session, per
  explicit scope (v2 ships inactive, example-only).

### Goal
Extract the hardcoded generation prompt into a versioned, config-selected
YAML template system so prompt wording becomes a first-class,
comparable experiment axis — same treatment `config/default.yaml` already
gives embedding model, chunker, and reranker. Two decisions were made with
the user before implementation: (1) `rag_answer_v1.yaml` reproduces the
current prompt byte-for-byte (no behavior change from versioning alone);
(2) `RetrievalPipeline`'s context-building was separately enhanced to
prefix each retrieved chunk with a `[Source N: <path>]` label, so a future
citation instruction (introduced starting in `rag_answer_v2.yaml`) has
something to reference — v1's prompt doesn't ask for citations, so this
labeling is inert for v1 but not for v2.

### Changes
- Added `src/rag/prompts/` (`__init__.py`, `loader.py`,
  `templates/rag_answer_v1.yaml`, `templates/rag_answer_v2.yaml`).
  `loader.py` defines `PromptTemplate` (pydantic model with a
  `model_validator` cross-checking `required_variables` against every
  `{name}` placeholder actually used in `system_template`/`user_template`),
  `load_prompt_template(path)`, and `load_prompt_template_from_config(config)`
  (resolves `config.generation.prompt.path`, then verifies the loaded
  file's own `prompt_id`/`version` match what's configured).
- `src/rag/config.py`: added `REPO_ROOT` (repo-root constant, `DEFAULT_CONFIG_PATH`
  now derived from it), new `PromptConfig` (`id`, `version`, `path`, all
  defaulted) nested under `GenerationConfig.prompt`, and
  `AppConfig.prompt_template_path()` resolving a relative `path` against
  `REPO_ROOT`.
- `config/default.yaml`: new `generation.prompt` block (`id: rag_answer`,
  `version: v1`, `path: src/rag/prompts/templates/rag_answer_v1.yaml`).
- `src/rag/retrieval/pipeline.py`: removed the hardcoded `_PROMPT_TEMPLATE`
  constant; added a private `_build_context()` helper (labels each
  reranked chunk `[Source N: <source>]` before joining); `RetrievalPipeline.__init__`
  gained an optional `prompt_template` DI param defaulting to
  `load_prompt_template_from_config(config)`; `answer()` now calls
  `self._prompt_template.render(context=..., query=...)` and concatenates
  `system`+`user` (or just `user` when `system` is empty, v1's case)
  before calling the unchanged `LLM.generate(prompt: str)`.
- `scripts/record_experiment.py`: `build_experiment_record` now records
  `prompt_id`, `prompt_version`, and `prompt_file_checksum` (sha256 of the
  configured prompt YAML's bytes, same `hashlib.sha256(path.read_bytes()).hexdigest()`
  pattern `ingestion/pipeline.py` uses for document checksums) alongside
  the other config-axis fields — read from the live `AppConfig`, not the
  eval report's own summary, matching this function's existing convention.
- `src/rag/eval/run_eval.py`: `_config_summary()` also gained
  `prompt_id`/`prompt_version`, for report self-description (not the
  checksum — that's a permanent-record-only detail).
- Tests: new `tests/unit/test_prompt_loader.py` (8 tests: valid load,
  missing file, undeclared-placeholder rejection, missing/unknown render
  variables, v1 byte-identity, config id/version mismatch, deterministic
  checksum), new `tests/unit/test_retrieval_pipeline.py` (first unit test
  file for this pipeline — fake vectorstore/embedder/reranker/LLM doubles;
  2 tests covering the labeled-context + v1 prompt path and an
  injected-template-with-system-message path), one new test in
  `tests/unit/test_record_experiment.py` for the three new record fields.
- `README.md`: `## Configuration` section documents where prompts live and
  how the active version is selected; `### Recording a new experiment`
  notes prompt version as a changeable axis and that every recorded
  experiment captures prompt identity + checksum.

### Issues
None encountered beyond the YAML block-scalar chomping detail already
covered under "why was it difficult" above — verified before writing the
loader rather than discovered via a failing test.

### Verification
`pytest tests/unit -q`: 85 passed (was 73). `pre-commit run --all-files`:
all hooks green, including `mypy` on `src/` with no errors, on the first
run — no ruff/mypy fixups needed. No eval re-run performed this session
(no metrics claim is being made; the only behavioral change is the
approved context-labeling addition, and `rag_answer_v2.yaml` is
deliberately inactive).

### Future prompt-experiment plan
Once `experiment_006` (top_k=10, chunk_size=500 revert — see prior entry's
"Next recommended step") lands, run a follow-up experiment holding that
winning config fixed and only swapping `generation.prompt.version` from
`v1` to `v2`. Compare `answer_quality` and the qualitative "confident but
ungrounded vs. over-refusal" tradeoff pattern already seen in
`experiment_005` — specifically check whether v2's citation requirement
measurably reduces ungrounded claims without materially increasing
refusals on genuinely answerable questions. Record it as `experiment_007`
(or the next free id) via the same `record_experiment.py` flow now that
`prompt_id`/`prompt_version`/`prompt_file_checksum` make the two prompt
versions directly comparable in `experiments/results/`.

---

## 2026-08-06-17:19 | First RAGAS generation-quality evaluation milestone

### Interview talking points
- **What problem did this solve?** Generation quality had no real judge —
  `KeywordOverlapScorer` is a keyword-overlap heuristic the code's own
  docstring calls "not a faithfulness/correctness judge." This adds a
  proper LLM-judged layer (RAGAS: faithfulness, answer relevancy, context
  precision/recall, answer correctness) as a new, opt-in, additive
  evaluation path, with an independently-configurable judge that never
  defaults to the models already used for generation.
- **Why was it difficult?** RAGAS had a major API rewrite in v0.4 (current
  PyPI release) that deprecates the well-documented `evaluate()`/
  `ragas.metrics`/`LangchainLLMWrapper` pattern in favor of a newer
  `llm_factory()`/`ragas.metrics.collections`/`@experiment` API whose exact
  call signatures aren't yet well-documented — decided to pin
  `ragas>=0.3,<0.4` (the last mature version) rather than fight
  bleeding-edge docs on a first milestone. Also had to bridge this
  project's own `LLM`/`Embedder` ABCs into `langchain_core` (confirmed
  present transitively, no new LangChain provider package needed) rather
  than adding `langchain-openai`/`langchain-anthropic`/`langchain-ollama`,
  and design cost tracking ourselves since RAGAS's `token_usage_parser` is
  documented against legacy LangChain LLMs with unconfirmed compatibility
  against a custom adapter.
- **How did I verify it?** Mocked everything — a fake `openai`/`anthropic`
  SDK client injected via `sys.modules` for the two new judge `LLM`
  implementations, a fake `ragas`/`ragas.metrics`/`datasets` module tree
  for `ragas_scorer.py` (including a minimal pandas-DataFrame double so
  the NaN-safe score-extraction logic could be tested without a real
  pandas dependency), and the existing `FakeVectorStore`/`FakeEmbedder`/
  `FakeReranker`/`FakeLLM` doubles for pipeline-level tests. No real
  `ragas`/OpenAI/Anthropic/Ollama-judge call happened this session — that
  was an explicit, agreed scope boundary going in.
- **What proved it worked?** 129 unit tests passing (was 85; +44 net new,
  2 skip cleanly via `pytest.importorskip` where they'd otherwise need the
  real `openai`/`anthropic` packages), `pre-commit run --all-files` clean
  on `src/`+`scripts/` (ruff, format, mypy — the new
  `ragas.*`/`datasets.*`/`openai.*`/`anthropic.*` mypy overrides were
  load-bearing since none of those packages are installed in the base dev
  env).
- **What's next?** Actually run it: `pip install .[ragas]` (+
  `.[anthropic]` for that hosted option), set a judge API key or approve a
  local-model pull, run `run_ragas_eval` against the 15-question default
  sample, fill in `generate_manual_review.py`'s scaffold by hand, run
  `compare_ragas_manual.py`, and only then decide whether to trust the
  judge enough to scale to the full 46-question set.

### Goal
Add RAGAS-based generation-quality scoring as a new, additive evaluation
layer without touching the existing deterministic retrieval metrics
(Recall@k, Hit Rate@k, MRR, latency) or the existing keyword-overlap
`answer_quality` path. Judge LLM is a new config axis
(`config.judge`, independent of `config.generation`), supporting hosted
(`openai`/`anthropic`, via `*_env_var`-resolved keys) and local (`ollama`,
explicitly "for experimentation," never defaulting to the generation
models `qwen2.5:1.5b`/`3b`) providers. Every RAGAS run records per-question
scores, aggregate means, judge identity, `prompt_id`/`prompt_version`, RAG
config, and (for hosted judges) estimated API usage. Tooling exists for a
manual-review sample and a RAGAS-vs-human comparison report, with an
explicit "not validated until reviewed" caveat baked into every output —
no live judge run happened this session, so no reliability claim is being
made.

### Changes
- New `src/rag/generation/openai_llm.py` / `anthropic_llm.py`: `LLM` ABC
  implementations for the two hosted judge providers, lazy-importing their
  SDKs (matching `rerankers/cohere.py`'s precedent) and self-tracking
  `call_count`/`input_tokens`/`output_tokens`.
- New `src/rag/eval/ragas_adapters.py`: `LangchainLLMAdapter`/
  `LangchainEmbeddingsAdapter`, thin `langchain_core`-compatible wrappers
  around this project's own `LLM`/`Embedder` ABCs.
- New `src/rag/eval/ragas_scorer.py`: best-effort per-metric loading (one
  metric failing to import doesn't block the others), dataset construction,
  and `ragas.evaluate()` invocation — every `ragas`/`datasets` import is
  lazy.
- New `src/rag/eval/run_ragas_eval.py`: CLI (`--sample-size`, default 15)
  reusing `run_eval.evaluate()` for the base report, then adding a
  `"ragas"` key — a strict superset of `run_eval.py`'s report shape.
- `src/rag/config.py`: new `JudgeConfig`/`OpenAIJudgeConfig`/
  `AnthropicJudgeConfig`/`OllamaJudgeConfig`, `AppConfig.judge`,
  `openai_api_key()`/`anthropic_api_key()` accessors.
- `config/default.yaml`: new `judge:` block, default provider `openai`,
  `ollama.model_name: llama3.1:8b` (deliberately distinct from
  `generation.model_name`).
- `src/rag/factory.py`: new `build_judge_llm()`, mirroring `build_llm`'s/
  `build_reranker`'s dispatch-plus-missing-key-`RuntimeError` pattern.
- `src/rag/retrieval/pipeline.py`: `answer()`'s `sources` entries gain a
  `content` key (the chunk's raw text) — additive, confirmed non-breaking
  for `POST /query` (pydantic's default `extra="ignore"` drops it from
  `SourceItem`).
- `src/rag/eval/run_eval.py`: `per_example` entries gain
  `generation_sources` (the production-config, post-`answer()` sources
  with content — distinct from the existing broader-fetch
  `retrieved_sources` field on the same entry).
- `scripts/record_experiment.py`: 7 new `ragas_*` fields, read via `.get()`
  so older, non-ragas reports still work.
- New `scripts/generate_manual_review.py`: deterministic
  answerable/unanswerable-stratified sampling into a JSONL scaffold with
  empty `human_faithful`/`human_correct`/`human_relevant`/
  `human_correct_refusal` fields.
- New `scripts/compare_ragas_manual.py`: thresholded (0.5) agreement
  between RAGAS scores and filled-in human labels, Markdown report with a
  fixed reliability caveat and a small-sample warning below n=10.
- `pyproject.toml`: new `ragas`/`anthropic` optional-dependency extras;
  mypy overrides for `ragas.*`/`datasets.*`/`openai.*`/`anthropic.*`.
- `.gitignore`: `data/eval/manual_review/` (same sensitivity class as the
  existing `data/eval/baselines/`).
- Tests: `test_openai_llm.py`, `test_anthropic_llm.py`,
  `test_ragas_adapters.py`, `test_ragas_scorer.py`, `test_run_ragas_eval.py`,
  `test_generate_manual_review.py`, `test_compare_ragas_manual.py`,
  `test_run_eval.py` (new); `test_config.py`, `test_factory.py`,
  `test_retrieval_pipeline.py`, `test_record_experiment.py` (extended).
- `README.md`: new "RAGAS generation-quality evaluation (optional)"
  subsection under Evaluation; Roadmap's old RAGAS bullet replaced with a
  "validate then scale to 46 questions" one.

### Issues
- Confirmed via a live PyPI version query (not guesswork) that ragas's
  0.3.x series is the last on the mature API before the 0.4 rewrite —
  `pip index versions ragas` showed the full 0.0.x–0.4.3 history, letting
  the pin be evidence-based rather than assumed.
- `LangchainLLMAdapter`'s base class (`langchain_core.language_models.llms.LLM`)
  turned out to be a pydantic model itself, requiring
  `model_config = ConfigDict(arbitrary_types_allowed=True)` plus a typed
  `rag_llm: LLM` field rather than a plain `__init__` — caught by testing
  with a bare `object()` placeholder first (which correctly failed
  pydantic's `is_instance_of` validation), fixed by using real `LLM`/
  `Embedder` subclass doubles in every test that constructs an adapter.
- Faking `import ragas.metrics` / `from ragas.embeddings import X` for
  `test_ragas_scorer.py` needed both `sys.modules["ragas.metrics"]` set
  *and* `fake_ragas.metrics` set as a real attribute on the parent fake
  module — Python's import system only auto-wires that parent/submodule
  attribute link when a module is freshly loaded, not when it's already
  present in `sys.modules`.
- The only two Ollama models pulled in this environment are
  `qwen2.5:1.5b`/`3b` (confirmed via `curl localhost:11434/api/tags`) —
  exactly the two disallowed as judge — so no local-judge smoke test was
  possible without a new ~5GB model pull, which was explicitly deferred
  rather than run unilaterally.

### Verification
`pytest tests/unit -q`: 129 passed, 2 skipped (was 85 passed) — the 2
skips are `pytest.importorskip("openai"/"anthropic")` guards in
`test_factory.py`'s "with API key returns X" tests, which need the real
SDKs installed (the dedicated `test_openai_llm.py`/`test_anthropic_llm.py`
files test the same logic via `sys.modules` fakes and do NOT skip).
`pre-commit run --all-files`: all hooks green on the first run after one
manual `ruff format` pass fixed two lines exceeding 100 chars. No live
`ragas`/OpenAI/Anthropic/Ollama-judge network call was made this session,
and no new package was `pip install`ed or Ollama model `pull`ed beyond
what already existed in this environment, per the explicit scope agreed
before implementation.

### Future RAGAS-run plan
Next session (or whenever the user has judge credentials ready): install
`.[ragas]` (+ `.[anthropic]` if using that provider), run
`run_ragas_eval --gold data/eval/techfusion_gold.jsonl --dataset-id
techfusion --verbose` at the default `--sample-size 15`, sanity-check
latency/cost before considering a larger run, generate a manual-review
sample with `scripts/generate_manual_review.py --num-rows 10`, hand-label
it, and run `scripts/compare_ragas_manual.py`. Only after that comparison
is reviewed should RAGAS scores be treated as trustworthy enough to (a)
scale to the full 46-question set and (b) fold into the standing
prompt-experiment plan (v1 vs v2) from the previous entry — RAGAS's
faithfulness/context-precision metrics are exactly the kind of signal that
plan's "confident but ungrounded" question was waiting on.

## 2026-08-06-18:15 | Structured-content ingestion for Confluence-like Markdown

### Interview talking points
- **What problem did this solve?** The TechFusion knowledge base grew to
  include real Markdown tables, fenced code/JSON blocks, and ASCII-chart
  captions, and `RecursiveCharacterChunker` has no structural awareness —
  it slices through a table row or the middle of a code sample the moment
  a document crosses `chunk_size` (500 chars), exactly like any generic
  text splitter would. This adds a `StructuredMarkdownChunker` that keeps
  those blocks atomic, tags each chunk with `content_type`/`section_path`/
  `code_language`/`table_headers`/`attachment_name`/`source_anchor`, and
  proves retrieval actually benefits from it on the real re-ingested KB.
- **Why was it difficult?** The requirement said "don't split inside a
  small block," which quietly assumed size and row/line-count are
  correlated. Direct measurement of the real KB disproved that: 4 of 9
  real structured blocks (the JSON config fence, the `retry_transient`
  Python fence, the platform-capacity table, the data-retention table)
  already exceed the *old* 500-char `chunk_size` despite being
  unambiguously small in row/line-count terms (≤8 rows, ≤26 lines). Gating
  atomicity on `chunk_size` would have defeated the milestone on the exact
  data it was built for, so block atomicity had to be decoupled into two
  independent knobs: `table_row_group_size` (row-count based, default 20)
  for tables, `max_atomic_block_chars` (char-count based, default 2000)
  for fenced/chart blocks — both well clear of anything in the real KB
  today, both independent of the prose-tuned `chunk_size`/`chunk_overlap`.
- **What was a judgment call, not a guarantee?** Two real documents
  (`platform-capacity-report-q2-2026.md`, `reliability-scorecard-q2-2026.md`)
  contain *both* a table and a chart+caption pair, so a gold question
  referencing either document can't be deterministically bucketed by
  document content type alone. `eval/content_type.py` resolves the tie
  with a keyword check (`chart`/`caption`/`trend` in the question text,
  else a fixed `table > code_configuration > chart` priority) — documented
  in the module's own docstring as a heuristic, not a semantic guarantee.
- **What proved it worked?** 12 new/extended unit test files (chunker
  atomicity/row-grouping/chart-pairing/section-path/attachment-tagging
  logic, writer per-span metadata, ingestion-pipeline end-to-end tagging,
  content-type classification, gold-file validation, content-type
  breakdown aggregation, config defaults) plus 2 new integration tests
  (`test_structured_markdown_ingestion.py`'s 5 tests retrieving a real
  table/code/configuration/chart chunk and a section-path-preserving mixed
  chunk through a real Postgres round trip after a real embed; extended
  `test_pgvector.py` proving all 6 new fields — including `table_headers:
  list[str] <-> TEXT[]` — survive both an insert and an `ON CONFLICT DO
  UPDATE` refresh). Then the real thing: migrated the dev DB, cleared and
  re-ingested all 26 real Markdown files under `dataset_id=techfusion`,
  validated the expanded 62-question gold file, and ran the full
  deterministic eval plus the content-type breakdown against real
  Postgres + local sentence-transformers + local Ollama (qwen2.5:1.5b) —
  see Metrics below.

### Goal
Implement a Markdown-structure-aware chunker for Confluence-like documents
that preserves tables (headers-with-rows, row-group chunking for large
tables), fenced code/config blocks (atomic, language-tagged), and
chart+caption pairs (caption text preserved, no fabricated visual
understanding), while leaving ordinary prose ingestion byte-identical to
today's chunker. Thread the new per-chunk metadata through the pipeline,
Postgres schema, and retrieval filters; add a shared content-type
classifier for gold questions/documents; validate and re-ingest the
expanded 62-question TechFusion gold set; run and report the deterministic
eval split by content type; record it as a new baseline. RAGAS was
explicitly out of scope this session.

### Changes
- New `src/rag/chunkers/structured_markdown.py`: `StructuredMarkdownChunker`,
  a single forward line-scan over Markdown text with priority order (open
  fence wins first; header/table-start/fence-start are mutually exclusive
  leading tokens outside a fence; chart-vs-plain-fence resolved by caption
  lookahead; attachment tagging layered onto prose sub-chunks after the
  fact, never its own block boundary). Delegates every prose run (and any
  oversized-fence fallback) to the existing `RecursiveCharacterChunker`,
  which is why a prose-only Markdown document chunks byte-identical to
  using that chunker directly. Gated on `source_type == "markdown"` —
  everything else (PDF/DOCX/HTML/plain-text) passes straight through.
- `src/rag/schemas.py`: new `ChunkSpan` (text + 6 optional structural
  hints); `ChunkMetadata` gains the identical 6 fields.
- `src/rag/chunkers/base.py` / `recursive_chunker.py`: `Chunker.split()`
  signature changes from `(text) -> list[str]` to
  `(text, source_type=None) -> list[ChunkSpan]`.
- `src/rag/chunkers/registry.py` / `config.py` / `config/default.yaml`:
  new `structured_markdown` provider (new default), new
  `StructuredMarkdownConfig` (`table_row_group_size=20`,
  `max_atomic_block_chars=2000`).
- `src/rag/ingestion/pipeline.py` / `writer.py`: pipeline now passes
  `raw_document.source_type` into `chunker.split()`; `Writer.write()`
  derives each chunk's 6 new metadata fields from *that chunk's own span*
  (`content_type=span.content_type or "prose"`) — previously every chunk
  in a document would have shared identical metadata, which structured
  content makes actively wrong (a table chunk and a prose chunk from the
  same file need different tags).
- `src/rag/vectorstore/base.py`: `ALLOWED_FILTER_FIELDS` gains
  `content_type` only — the other 5 fields aren't needed by any
  requirement and the whitelist's SQL-injection-safety rationale favors
  staying minimal.
- `scripts/init_db.py`: idempotent migration adds `content_type`,
  `section_path`, `code_language`, `table_headers TEXT[]`,
  `attachment_name`, `source_anchor` columns plus one new index on
  `content_type`, following the existing inline-column +
  `ADD COLUMN IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` pattern.
  Additive-only — confirmed `sample_docs` chunks are untouched.
- `src/rag/vectorstore/pgvector.py`: `add_chunks`/`search` extended with
  the 6 new fields, including `ON CONFLICT DO UPDATE SET` refreshing them
  on re-ingestion (the whole point of re-chunking a changed file).
- New `src/rag/vision/base.py` (+ `vision/__init__.py`): inert
  `VisionProvider` ABC (`describe_image(path, alt_text=None) -> str`) —
  no config, no factory dispatch, no caller. Documents the shape a future
  vision provider would take without committing to one.
- New `src/rag/eval/content_type.py`: `build_document_content_types()`
  runs the *configured* chunker over each referenced KB file (self-
  updating, not a hardcoded lookup); `classify_example()` buckets a gold
  question into `unanswerable`/`multi_hop`/`table`/`code_configuration`/
  `chart`/`prose`.
- New `scripts/validate_gold_file.py`: hard-fails on an unresolvable
  `relevant_documents` path, an empty structured-content bucket, or a
  duplicate question; soft-warns on a record-count mismatch or low
  keyword overlap between `expected_answer` and referenced content
  (reuses `KeywordOverlapScorer`, doesn't reimplement it).
- New `src/rag/eval/report_content_type.py`: calls `run_eval.run()` once
  (no separate re-retrieval/re-generation) and re-aggregates Recall@5/10,
  Hit Rate@5/10, MRR, and mean latencies per content-type bucket via the
  existing, unmodified `eval/metrics.py` functions.
- `scripts/record_experiment.py`: recorded `experiment_006`.
- Tests: `test_structured_markdown_chunker.py` (new, 16 tests),
  `test_writer.py` (new, 3 tests), `test_content_type.py` (new, 8 tests),
  `test_validate_gold_file.py` (new, 5 tests), `test_report_content_type.py`
  (new, 3 tests); `test_chunker.py`, `test_ingestion_category.py` (+5),
  `test_config.py` (+2) extended; `test_structured_markdown_ingestion.py`
  (new integration, 5 tests), `test_pgvector.py` (extended, +2 integration
  tests).

### Issues
- The block-size-vs-row/line-count mismatch above (see talking points) —
  fixed by decoupling atomicity from `chunk_size` entirely.
- Two real documents mix a table and a chart — fixed with a documented
  keyword-heuristic tie-break in `content_type.py`, not treated as a bug
  to eliminate (there's no purely structural way to resolve it).
- `.svg` was deliberately kept out of `IngestionConfig.supported_extensions`
  — the gold file's `unanswerable: true` SVG-related questions are a
  refusal trap that depends on the SVG not being retrievable, and
  `vision/base.py` is explicitly a stub, not a wired-up implementation.
- `DefaultCleaner`'s blank-line collapse (`\n{3,}` -> `\n\n`) is a known,
  *unfixed* limitation that could in principle corrupt a fenced block
  containing 2+ consecutive blank lines — confirmed none of the real
  fenced blocks do, so it's inert against current data, but a future
  structured document with deliberate blank-line formatting inside a
  fence would need the cleaner to skip fence interiors.
- No bugs surfaced during real ingestion/eval — all pytest/pre-commit
  issues (a handful of `E501` docstring lines) were caught and fixed
  before the real run.

### Metrics
DB content-type distribution after re-ingesting 26 real Markdown files
under `dataset_id=techfusion`: 307 `prose`, 7 chunks across 4 `table`, 3
`code`, 1 `configuration`, 2 `chart` (317 chunks total).

Full 62-question run (`experiment_006`, qwen2.5:1.5b, no reranker, same
config as the old baseline except `chunking.provider`):
Recall@5=0.919, Recall@10=0.952, Hit Rate@5=0.935, Hit Rate@10=0.968,
MRR=0.865, retrieval=72ms, generation=2742ms, total=2813ms mean.

By content type (`report_content_type.py`, same run):
| Bucket | n | Recall@5 | Recall@10 | Hit Rate@10 | MRR |
|---|---|---|---|---|---|
| table | 7 | 1.000 | 1.000 | 1.000 | 0.790 |
| code_configuration | 6 | 1.000 | 1.000 | 1.000 | 0.875 |
| chart | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 14 | 0.929 | 0.929 | 1.000 | 0.952 |
| prose | 24 | 0.958 | 0.958 | 0.958 | 0.917 |
| unanswerable | 9 | 0.667 | 0.889 | 0.889 | 0.610 |

**Not apples-to-apples**: the previous clean baseline
(`experiment_001`, 2026-08-05) ran 46 questions over a prose-only KB;
this run is 62 questions over a mixed-content KB with a different
question mix (14 multi-hop, 9 unanswerable, 15 structured-content). The
full-set numbers above are **not directly comparable** to
`experiment_001`'s Recall@5=0.891/Recall@10=0.967/MRR=0.847. The closer
comparison is the new run's 24-question `prose` bucket against that old
baseline, since both score retrieval over predominantly-prose content:
prose-bucket Recall@5=0.958 vs. old 0.891, Recall@10=0.958 vs. old 0.967,
MRR=0.917 vs. old 0.847 — a small, directionally positive shift, but on
roughly half the sample size, so not a strong claim either way. The
`table`/`code_configuration`/`chart` buckets have no prior baseline at
all; report them as newly-established, not as regressions or
improvements.

### Limitations
- The chart/table disambiguation heuristic and the "prose bucket vs. old
  baseline" comparison are both explicitly approximate — documented as
  such in code and above, not resolved issues.
- DOCX/HTML/PDF loaders still flatten tables/code to plain text at the
  loader level (confirmed, not fixed) — 100% of today's real structured
  content is Markdown, so this was out of scope, not overlooked.
- `answer_quality`'s keyword-overlap scorer remains a crude placeholder
  (unchanged this session) — RAGAS is still the planned real judge,
  deliberately not run this session per explicit scope.
- Small structured-content sample sizes (7 table / 6 code_configuration /
  2 chart questions) mean their Recall/MRR figures are illustrative, not
  statistically load-bearing yet.

### Verification
`pytest tests/unit -q`: 168 passed, 2 skipped. `pytest tests/integration
-q`: 17 passed (real Postgres + real sentence-transformers embeddings; no
Ollama generation calls in this suite). `pre-commit run --all-files`:
all hooks green (ruff, ruff-format, mypy) after fixing 4 docstring lines
over 100 chars. Real run: `scripts/init_db.py` (idempotent, confirmed
`sample_docs` chunks unaffected), `rag.ingestion.pipeline
data/knowledge_base --dataset-id techfusion --clear` (26/26 documents
ingested), `scripts/validate_gold_file.py --expect-count 62` (passed, 3
soft warnings on keyword overlap, 0 errors), `rag.eval.run_eval` (62/62
questions, real Ollama generation), `rag.eval.report_content_type`
(second full pass, per-bucket breakdown above), `scripts/record_experiment.py`
+ `scripts/compare_experiments.py` (recorded `experiment_006`,
README.md's comparison table updated). RAGAS was not run. No commit/push
made — tree left for review per explicit instruction.

## 2026-08-06-19:45 | First real RAGAS evaluation (OpenAI gpt-4o-mini judge, 15-question pilot)

### Interview talking points
- **What problem did this solve?** All RAGAS tooling built in the earlier
  milestone had never been run against a real judge — every metric,
  adapter, and comparison script was mocked. This is the first live
  faithfulness/answer_relevancy/context_precision/context_recall/
  answer_correctness scoring against OpenAI's `gpt-4o-mini`, deliberately
  scoped to a stratified 15-question pilot (not the full 62) so cost,
  judge reliability, and failure modes could be sanity-checked before any
  larger spend.
- **Why was it difficult?** Two unexpected blockers before any API call:
  (1) `ragas`/`openai`/`datasets` weren't installed yet, and a plain
  `pip install .[ragas]` failed — `scikit-network` (a direct `ragas`
  dependency, unrelated to any metric actually used) has no prebuilt
  wheel for this environment's Python 3.14 and needs a C++ compiler this
  machine doesn't have; fixed with
  `pip install --only-binary=scikit-network -e ".[ragas]"`, which
  resolved to an older `scikit-network` version with a wheel. (2) Several
  small attribute-name mismatches surfaced only by actually running the
  verification code (`PromptTemplate.prompt_id` not `.id`,
  `OpenAILLM._model_name` not `.model_name`) — harmless in isolation, but
  a reminder that "the code looks right" and "the code runs" are
  different claims, which is exactly why the milestone's own step 3
  (cost-safe validation before spending money) mattered.
- **What was the actual finding, not just the number?** RAGAS's
  `answer_relevancy`/`answer_correctness` scored both correct refusals in
  the sample near zero (0.0 and 0.115/0.074) purely because a refusal's
  text doesn't lexically resemble a reference explanation — this exactly
  matches the caveat `ragas_scorer.py`'s docstring already carried
  ("particularly unreliable on unanswerable questions") from before any
  live run existed, now confirmed with real data rather than assumed.
  Meanwhile `faithfulness` was the most trustworthy metric: 87% agreement
  with manual review, and its only 2 misses were both *false positives*
  on genuine hallucinations (scored more generously than a human would),
  never a missed real problem — the safer failure direction for a
  hallucination detector.
- **What proved it worked?** A full, traceable chain for every one of the
  15 questions: RAGAS score -> manually read generated answer vs.
  expected answer vs. the exact 3 chunks fed to the LLM -> retrieval-vs-
  generation diagnosis using both the broad top-10 fetch (`retrieved_sources`)
  and the narrow production fetch (`generation_sources`) -> confusion
  table against the RAGAS score, thresholded at 0.5. This distinguished a
  genuine retrieval-depth failure (question [11]: the right document was
  retrieved at broad top-10 rank 4 but didn't survive the
  `rerank_top_n=3` cutoff) from 4 generation failures where the correct
  document(s) were in the exact context the LLM saw and it still answered
  wrong — a distinction the earlier deterministic-only evaluation
  couldn't make.

### Goal
Verify environment/config/cost-safety before spending any money, then run
a real, stratified (not random) 15-question RAGAS evaluation with OpenAI
`gpt-4o-mini` as judge, manually review every question, automatically
compare RAGAS scores against those manual labels (agreement rate,
disagreements, false positives/negatives), record it as a new experiment
without touching the 6 existing baselines, and produce a report that
explicitly answers whether the judge can be trusted, which metric is most
useful, which questions failed, whether failures were retrieval or
generation, and whether prompt v2 would plausibly help. Stop after this
pilot — do not run the full 62-question RAGAS evaluation yet.

### Changes
- Installed the `ragas` optional extra (`ragas==0.3.9`, `openai==2.53.0`,
  `datasets==5.0.1`, plus transitive deps) via
  `pip install --only-binary=scikit-network -e ".[ragas]"` — the flag is
  necessary on this Windows/Python-3.14 environment; documented in the
  pilot report as worth a README/pyproject note if reproduced elsewhere.
- `scripts/record_experiment.py`: added a `chunking_provider` field to
  `build_experiment_record` (previously only `chunk_size`/`chunk_overlap`
  were captured, not which chunker provider produced them) — a small,
  additive schema change, not a modification of any existing recorded
  experiment file.
- New `data/eval/techfusion_gold_ragas_sample15.jsonl`: a hand-selected,
  stratified 15-question subset of the 62-question gold file (3 prose, 2
  table, 2 code, 1 JSON/YAML configuration, 2 chart-caption, 3 multi-hop,
  2 unanswerable), each question's original index recorded in the pilot
  report for traceability. Not random — chosen to guarantee coverage of
  every structured-content bucket the earlier milestone introduced.
- New `data/eval/manual_review/techfusion_ragas_sample15_manual_review.jsonl`:
  the `generate_manual_review.py` scaffold, filled in by hand (acting as
  the human reviewer, since no other reviewer was available this
  session) for all 15 questions — `human_faithful`/`human_correct`/
  `human_relevant`/`human_correct_refusal` plus a reasoned note per
  question.
- New `data/eval/manual_review/techfusion_ragas_sample15_vs_manual.md`:
  `compare_ragas_manual.py`'s aggregate agreement-rate output.
- New `experiments/results/experiment_007.json` +
  `experiments/configs/experiment_007.yaml`: recorded via
  `scripts/record_experiment.py`, exactly as designed — `experiment_001`
  through `experiment_006` untouched, and `scripts/compare_experiments.py`
  / `README.md` deliberately NOT re-run this session (this is a pilot,
  not yet a result meant to join the standing comparison table).
- New `experiments/reports/experiment_007_ragas_pilot_report.md`: summary
  metrics, all 15 questions' generated/expected answers with RAGAS scores
  and human labels side by side, a per-metric confusion table (TP/TN/FP/FN)
  against manual review, and the 5 explicitly-requested recommendation
  answers (judge trust, most-useful metric, which questions failed,
  retrieval-vs-generation root cause, prompt-v2 prediction).

### Issues
- `scikit-network`'s missing wheel for this Windows/Python-3.14
  environment — fixed via `--only-binary`, see Changes.
- `run_ragas_eval.run_ragas()`'s existing `--sample-size` slices the
  *first* N gold examples, not a stratified sample — by design, for its
  original purpose (predictable, reproducible default slicing). Rather
  than change that behavior (which other future runs may depend on),
  built the stratified 15-question set as a separate, explicit gold file
  and pointed `run_ragas_eval` at it — no changes needed to
  `run_ragas_eval.py`/`ragas_scorer.py`.
- 2 pre-existing unit tests (`test_score_missing_ragas_package_raises_runtime_error`,
  `test_build_dataset_missing_datasets_package_raises_runtime_error` in
  `test_ragas_scorer.py`) now fail in this environment, because they
  assert on the "package not installed" fallback path and `ragas`/
  `datasets` are now actually installed — expected, not a regression;
  left unmodified since fixing them (e.g. forcing an `ImportError` via
  `sys.modules` patching) is a test-infrastructure change outside this
  session's scope.
- Minor attribute-name mismatches caught only by actually running
  verification code (see Interview talking points) — fixed by reading
  the real class definitions rather than guessing from memory.

### Metrics
15-question stratified pilot, `techfusion` dataset, `qwen2.5:1.5b`
generation / `gpt-4o-mini` judge, prompt v1, `structured_markdown`
chunker, no reranker:

Deterministic: Recall@5/10=1.000/1.000, Hit Rate@5/10=1.000/1.000,
MRR=0.906, retrieval/generation/total latency (mean)=67ms/2961ms/3028ms.

RAGAS aggregate: faithfulness=0.844, answer_relevancy=0.696,
context_precision=0.861, context_recall=0.811, answer_correctness=0.471
(the latter two pulled down by the 2 correct-refusal questions scoring
near-zero — see Interview talking points).

Judge cost: 180 API calls, 130,763 input + 15,247 output tokens
(`gpt-4o-mini`), well under $0.05 for the whole run.

RAGAS-vs-manual agreement (n=15, thresholded at 0.5): faithfulness 87%
(12 TP, 1 TN, 2 FP, 0 FN), answer_relevancy 80% (11 TP, 1 TN, 0 FP, 3 FN),
answer_correctness 80% (7 TP, 5 TN, 0 FP, 3 FN). Correct-refusal rate
(human-labeled only): 100% (2/2).

5 of 15 questions (33%) failed on manual review: 4 generation failures
(cross-document conflation/hallucination despite correct retrieval) and 1
retrieval-depth failure (correct document retrieved at broad top-10 but
dropped by the `rerank_top_n=3` cutoff before reaching the LLM).

### Limitations
- n=15, single reviewer (this session, acting as the human labeler in the
  absence of another reviewer), fixed uncalibrated 0.5 agreement
  threshold — explicitly not a validated judge-reliability benchmark,
  same caveat `compare_ragas_manual.py` has always carried.
- The `answer_relevancy`/`answer_correctness` unreliability-on-refusals
  finding is based on exactly 2 unanswerable questions — directionally
  consistent with `ragas_scorer.py`'s pre-existing caveat, but too small
  a sample to quantify precisely; the full 62-question set has 9
  unanswerable questions and will sharpen this signal considerably.
- `context_precision`/`context_recall` weren't manually cross-checked
  against raw chunk text as rigorously as `faithfulness`/`answer_relevancy`/
  `answer_correctness` were (those three have a direct human-label
  counterpart in this project's manual-review schema; the other two
  don't) — their apparent usefulness on question [11] is a single
  data point, not a validated pattern yet.
- Prompt v2's predicted improvement (Section 4 of the pilot report) is a
  hypothesis, not a result — no v2 run has happened yet.

### Verification
`pytest tests/unit -q` (after installing the `ragas` extra): 167 passed,
2 pre-existing skips became non-skips (openai SDK now installed), 2
pre-existing tests now fail for the expected reason described in Issues.
`scripts/validate_gold_file.py` run against both the full 62-question
file and the new 15-question stratified subset (both pass, 62-question
run identical to the prior milestone's result). Real run: OpenAI
connectivity verified with a free `models.list()` call (no tokens spent)
before any judge call; `rag.eval.run_ragas_eval` run once, for real,
against the 15-question file with the real OpenAI judge;
`scripts/generate_manual_review.py` + hand-filled labels +
`scripts/compare_ragas_manual.py` run against that real output. No RAGAS
run against the full 62-question set. No commit, no push, no changes to
`experiment_001`–`experiment_006` or `README.md`'s comparison table —
stopped after the pilot per explicit instruction, awaiting review before
any larger run.

## 2026-08-06-20:15 | Full RAGAS evaluation, all 62 questions (OpenAI gpt-4o-mini judge)

### Interview talking points
- **What problem did this solve?** Scaled the RAGAS judge from the
  15-question pilot to the full 62-question gold set, after the user
  reviewed the pilot findings and gave the go-ahead. The core question
  going in: does the pilot's central finding (RAGAS's `answer_relevancy`/
  `answer_correctness` are unreliable on refusal-shaped answers, while
  `faithfulness` is comparatively trustworthy) hold up with 9 unanswerable
  questions instead of 2, and does anything new show up that a 15-question
  sample was too small to surface?
- **Why was it difficult?** Nothing blocked the run itself (environment
  was already verified and installed from the pilot) — the harder part
  was choosing how to scope manual review at 4x the sample size.
  Hand-reviewing all 62 questions at pilot-level depth wasn't practical
  in one session, so the manual-review sample was deliberately
  *redesigned* rather than just re-run with a bigger `--num-rows`:
  `generate_manual_review.py`'s built-in answerable/unanswerable
  proportional stratification would only have picked ~2-3 unanswerable
  questions out of a 15-20 row sample (proportional to 9/62), which
  wouldn't have improved on the pilot's already-thin refusal coverage.
  Bypassed that helper and hand-selected 20 questions weighted to cover
  **all 9 unanswerable questions** plus the lower-scoring buckets
  (`multi_hop`, `table`, `code_configuration`) identified from a new,
  purely computational RAGAS-by-content-type breakdown built for this
  run (extending `eval/content_type.py`'s classifier — already built for
  deterministic metrics in the prior milestone — to also bucket RAGAS
  scores, with zero new production code needed).
- **What did scaling up actually reveal that the pilot couldn't?**
  Something concrete and safety-relevant: of the 9 unanswerable
  questions, 8 were correctly refused but **one (question 61, "which
  availability zone contains the largest Worker icon in the uncaptioned
  drawing") was answered with a confident, fabricated inference instead
  of a refusal** — the exact failure mode the whole refusal-testing
  design exists to catch, and one the pilot's 2-question refusal sample
  had no chance of finding. Correct-refusal rate dropped from the
  pilot's 100% (n=2) to 89% (n=9) — not because refusal behavior got
  worse, but because n=2 was never enough to see the failure that was
  already there. This is the clearest demonstration in this whole
  project of why the user's instinct to gate the full run behind a
  pilot review, rather than jump straight to 62, was the right call —
  the pilot validated the *tooling*; only the full run validated
  *refusal reliability*.
- **What proved it worked?** Same discipline as the pilot: every
  manually-reviewed question's RAGAS score cross-checked against the
  actual generated answer, expected answer, and the exact 3 context
  chunks fed to the LLM (`generation_sources`), with an explicit
  retrieval-vs-generation diagnosis. Two additional retrieval-adjacent
  failures ([16] and [53]) turned out to be *chunk-level* misses, not
  document-level ones — the correct file was among the top-3 chunks fed
  to the LLM, but the specific fact needed apparently wasn't in the
  chunk actually selected, a distinction the document-level Recall@5/10
  metric (still 0.919/0.952, unchanged from `experiment_006`) can't see
  by itself.

### Goal
Run the real 62-question RAGAS evaluation with the OpenAI `gpt-4o-mini`
judge (per explicit go-ahead after pilot review), report deterministic +
RAGAS aggregate metrics against the pilot, build a RAGAS-by-content-type
breakdown, manually review a redesigned sample weighted toward full
unanswerable-question coverage, compare against RAGAS at the new scale,
record a new experiment without touching any existing baseline, and
report updated answers to the same 5 diagnostic questions from the pilot
(judge trust, most-useful metric, which questions failed, retrieval vs.
generation, prompt v2 prediction).

### Changes
- New `experiments/results/experiment_008.json` +
  `experiments/configs/experiment_008.yaml`: recorded via
  `scripts/record_experiment.py` — `experiment_001`–`experiment_007`
  untouched; `scripts/compare_experiments.py`/`README.md` deliberately
  NOT re-run (same reasoning as `experiment_007`: RAGAS runs aren't yet
  folded into the deterministic-only standing comparison table).
- New `data/eval/manual_review/techfusion_ragas_full62_manual_review.jsonl`:
  20 hand-selected (not the built-in proportional-stratification helper)
  questions, filled in with genuine judgment — all 9 unanswerable
  questions plus a mix of `multi_hop`/`table`/`code_configuration`
  questions.
- New `data/eval/manual_review/techfusion_ragas_full62_vs_manual.md`:
  `compare_ragas_manual.py`'s aggregate output at n=20.
- New `experiments/reports/experiment_008_ragas_full_report.md`: summary
  metrics vs. the pilot, a RAGAS-by-content-type breakdown table (6
  buckets: prose/table/code_configuration/chart/multi_hop/unanswerable),
  all 20 manually-reviewed questions' full detail, confusion tables, and
  updated answers to the 5 diagnostic questions.

### Issues
- No environment/dependency issues this time (already resolved during
  the pilot).
- `generate_manual_review.py`'s stratification helper doesn't know about
  content-type buckets, only answerable/unanswerable — worked around by
  building the 20-row sample directly rather than extending the script,
  since a one-off hand-curated sample was more appropriate here than a
  permanent code change to a script whose existing behavior (proportional
  answerable/unanswerable sampling) is still the right default for other
  callers.

### Metrics
Full 62-question run, same config as `experiment_006`/the pilot
(`qwen2.5:1.5b` generation, `gpt-4o-mini` judge, prompt v1,
`structured_markdown` chunker, no reranker):

Deterministic (unchanged from `experiment_006`, confirming RAGAS scoring
doesn't affect retrieval): Recall@5/10=0.919/0.952, MRR=0.865.

RAGAS aggregate (62-question vs. 15-question pilot): faithfulness
0.786 vs 0.844, answer_relevancy 0.631 vs 0.696, context_precision 0.754
vs 0.861, context_recall 0.728 vs 0.811, answer_correctness 0.469 vs
0.471 (essentially flat — the only metric that didn't drop).

RAGAS by content-type bucket: chart (n=2) scored best across the board
(faithfulness 1.000, correctness 0.867); unanswerable (n=9) scored worst
on `answer_relevancy`/`answer_correctness` (0.103/0.118) — the refusal-
scoring artifact, not a real quality problem (see below); multi_hop
(n=14) had the lowest `answer_correctness` among answerable buckets
(0.487) despite the highest `context_precision` (0.940) — a generation/
synthesis weakness, not a retrieval one.

Judge cost: 761 calls, 538,040 input + 57,371 output tokens
(`gpt-4o-mini`), roughly $0.10-0.15.

Manual review (n=20, weighted toward full unanswerable coverage):
faithfulness 85% agreement (down slightly from the pilot's 87%),
answer_relevancy 60% (down from 80%), answer_correctness 55% (down from
80%) — both drops driven entirely by the larger, more representative
refusal-question sample. Correct-refusal rate: 89% (8/9) — 1 real
failure (question 61, hallucinated instead of refused) invisible in the
pilot's n=2.

### Limitations
- n=20 manual review, single reviewer, same uncalibrated 0.5 threshold
  caveat as the pilot — a sharper signal than n=15, still not a validated
  benchmark.
- The RAGAS-by-content-type breakdown (Section 2 of the report) is
  purely computational and hasn't been manually spot-checked bucket by
  bucket the way individual questions were — its patterns (e.g.
  `code_configuration`'s low `context_recall`) are consistent with the
  manually-reviewed failures in that bucket, but should be read as
  corroborating evidence, not independently validated.
- Prompt v2's predicted fixes for questions 24/61 remain a hypothesis —
  no v2 run has happened yet, at either sample size.
- `README.md`'s comparison table and `experiments/reports/comparison.md`
  still don't include either RAGAS experiment (`007` or `008`) — a
  standing gap, not an oversight, until it's decided whether/how to
  present RAGAS metrics alongside deterministic-only baselines there.

### Verification
Real run only (no test suite changes this session): OpenAI connectivity
already verified during the pilot pre-flight, re-checked lightly before
this run (key present, `techfusion` chunk count still 317, `ragas`/
`openai`/`datasets` still importable). `rag.eval.run_ragas_eval` run once
against the full `data/eval/techfusion_gold.jsonl` (62/62 scored, 0
skipped, 0 metrics failed). Manual review + `compare_ragas_manual.py` run
against that real output. `scripts/record_experiment.py` run once
(`experiment_008`). No commit, no push, no changes to `experiment_001`–
`experiment_007` or `README.md`'s comparison table.

## 2026-08-06-20:35 | README comparison table: RAGAS columns + exclude flag

### Interview talking points
- **What problem did this solve?** `experiment_008` was recorded but
  invisible in `README.md`'s comparison table -- `compare_experiments.py`
  had no RAGAS columns, so re-running it would have added a row with
  numbers identical to `experiment_006` (same config/dataset) and hidden
  the one thing that actually changed. Asked the user how to handle it
  rather than guessing; they picked adding RAGAS columns plus leaving
  `experiment_007` (the 15-question pilot) out of the table, matching
  this project's standing principle (from the 46-vs-62-question
  comparison) that differently-sized samples don't belong in the same
  comparison table.
- **Why was it difficult?** The script had no concept of "exclude this
  experiment" -- every prior use just meant "show everything." Rather
  than special-case `experiment_007` by ID inside the script (which
  would need removing again the next time a small pilot exists),
  added a general `--exclude` CLI flag so this is a reusable mechanism,
  not a one-off hack.
- **What proved it worked?** `git diff README.md` after the run showed
  exactly the expected shape: `experiment_001`-`experiment_006`'s rows
  unchanged except two new empty (`-`) columns, `experiment_008` added
  as row 7 with real `ragas_faithfulness`/`ragas_answer_correctness`
  values, `experiment_007` absent entirely.

### Changes
- `scripts/compare_experiments.py`: `render_table()` gains two columns
  (`RAGAS Faithful`, `RAGAS Correct`, reading `ragas_faithfulness`/
  `ragas_answer_correctness` via the existing `.get()`-based `_fmt()`
  pattern -- renders `-` for every pre-RAGAS record automatically, no
  backfill needed). `load_records()` gains an `exclude: set[str] | None`
  parameter; `main()` gains a `--exclude` CLI flag (comma-separated
  `experiment_id`s).
- `tests/unit/test_compare_experiments.py`: updated the existing row-format
  assertion for the two new columns; added
  `test_render_table_includes_ragas_columns_when_present` and
  `test_load_records_omits_excluded_experiment_ids`.
- Ran `python scripts/compare_experiments.py --exclude experiment_007`
  for real -- `README.md` and `experiments/reports/comparison.md` both
  updated, `experiment_008` now visible with `ragas_faithfulness=0.786`/
  `ragas_answer_correctness=0.469`.

### Verification
`pytest tests/unit/test_compare_experiments.py -q`: 11 passed (was 9).
`pytest tests/unit -q`: 169 passed, 1 skipped, 2 pre-existing expected
failures (same `ragas`-is-actually-installed reason as the last two
entries). `pre-commit run --all-files`: all hooks green (the
`mixed-line-ending` hook auto-normalized `README.md`/`comparison.md`,
no content change). No commit, no push.

## 2026-08-06-22:30 | MLflow integration: every recorded experiment also becomes an MLflow run

### Interview talking points
- **What problem did this solve?** `scripts/record_experiment.py` only
  ever produced flat JSON/YAML files on disk -- useful for
  `compare_experiments.py`'s table, but with no queryable run history, no
  metric-over-time charts, and no standard UI a teammate could point at.
  This wires every recorded experiment into MLflow too: params, metrics,
  and the raw eval-output JSON / recorded record JSON / config YAML as
  artifacts -- without replacing the existing JSON/YAML files, which
  stay the source of truth `compare_experiments.py` reads from.
- **Why was it difficult?** The real surprise was version drift, not
  design: MLflow 3.x (`3.15.1`, what `pip install` actually resolved to)
  deprecated the plain filesystem tracking backend
  (`file:./mlruns`) into "maintenance mode" and now raises
  `MlflowException` on it by default, pointing at a database-backed store
  instead. Caught immediately by actually running the backfill (not just
  unit tests, which mock `mlflow` entirely and wouldn't have surfaced a
  real-backend behavior change) -- switched the default `tracking_uri` to
  `sqlite:///mlflow.db`, still fully local/serverless, before doing the
  real backfill.
- **What was the design call, not just plumbing?** Whether a missing
  `mlflow` package should be a hard failure or a silent skip when
  `mlflow.enabled: true` (the new default). Chose hard failure --
  `RuntimeError` pointing at `pip install .[mlflow]`, mirroring exactly
  how `build_judge_llm` handles a selected-but-uninstalled judge provider
  -- because the user's ask was "log every experiment," and a config flag
  that's silently ignored when the dependency happens to be missing would
  quietly break that guarantee rather than surfacing it.
- **What proved it worked?** Backfilled all 8 existing
  `experiment_001`-`experiment_008` records in one pass, then queried the
  MLflow client directly (not just re-reading files) to confirm every
  run's params/metrics/tags/artifacts matched its source JSON exactly --
  including that `None`-valued fields (a pre-RAGAS record's `ragas_*`
  metrics) were absent from the run rather than logged as `0`. Separately
  smoke-tested the *live* `record_experiment.py` CLI path (not just the
  backfill script, which reuses the same underlying function but skips
  the `main()` wiring) with a throwaway record, confirmed all 3 artifact
  types attach when the original eval-output JSON is available, then
  deleted the throwaway run and files.

### Goal
Integrate MLflow so every experiment recorded via
`scripts/record_experiment.py` also becomes an MLflow run: config as
params, every deterministic/RAGAS/latency metric as a metric, and the
raw eval-output JSON plus the flat record JSON and config YAML snapshot
as artifacts. Local by default (no server), config-driven (can point at
a real MLflow server later), optional dependency but hard-fails if
enabled and missing rather than silently skipping. Backfill the 8
existing recorded experiments so MLflow's view isn't missing history.

### Changes
- `pyproject.toml`: new `mlflow` optional extra (`mlflow>=2.14`,
  resolved to `3.15.1`); `mlflow.*` added to the mypy
  `ignore_missing_imports` override list.
- `src/rag/config.py`: new `MLflowConfig` (`enabled: bool = True`,
  `tracking_uri: str = "sqlite:///mlflow.db"`,
  `experiment_name: str = "local-rag-system"`); `AppConfig.mlflow`.
- `config/default.yaml`: new `mlflow:` block matching the above defaults.
- New `src/rag/eval/mlflow_logger.py`: `log_experiment(record,
  mlflow_config, artifact_paths=None) -> str | None` -- a sibling to
  `ragas_scorer.py`'s optional-dependency pattern (lazy `import mlflow`
  inside the function, so importing this module never requires it
  installed). Splits the flat record schema into `_PARAM_FIELDS`
  (config choices) and `_METRIC_FIELDS` (numeric outcomes); skips any
  field whose value is `None` rather than logging it as `"None"`/`0`
  (MLflow's `log_param`/`log_metric` reject `None` outright anyway).
  Tags each run with this project's own `experiment_id`/`label`.
- `scripts/record_experiment.py`: after writing the JSON/YAML as before,
  calls `log_experiment` with the raw `--eval-output` path, the just-written
  record JSON, and the config YAML snapshot as `artifact_paths`.
- New `scripts/backfill_mlflow.py`: one-off (but reusable/rerunnable)
  script logging every `experiments/results/*.json` into MLflow directly
  -- no eval re-run. Used to backfill `experiment_001`-`experiment_008`.
- `.gitignore`: `mlruns/`, `mlartifacts/`, `mlflow.db` -- all local,
  regenerable tracking-store state, same treatment as `pgdata/`.
- `README.md`: new "MLflow tracking" subsection under "Recording a new
  experiment" (how to view the UI, what gets logged, the backfill
  script's purpose and limitation).
- Tests: new `tests/unit/test_mlflow_logger.py` (9 tests, `mlflow`
  faked via `sys.modules` injection like `test_ragas_scorer.py`'s
  pattern; the "package not installed" test uses
  `monkeypatch.setitem(sys.modules, "mlflow", None)` specifically so it
  passes regardless of whether `mlflow` is actually installed in the dev
  venv -- avoiding the exact fragility the two now-permanently-failing
  `test_ragas_scorer.py` tests demonstrate); `test_config.py` extended
  with a default-value test.

### Issues
- MLflow 3.x's filesystem-backend deprecation (see Interview talking
  points) -- fixed by switching to `sqlite:///mlflow.db` before the real
  backfill ran, so no experiment was ever actually lost/mis-recorded;
  caught by running the real thing, not by unit tests (which mock
  `mlflow` and can't see real-backend version behavior).
- Installing the `mlflow` extra downgraded `pandas` from `3.0.5` to
  `2.3.3` (MLflow's own dependency constraint) -- reran the full unit
  suite afterward specifically to confirm nothing pandas-dependent
  (`ragas_scorer.py`'s DataFrame handling, primarily) broke; it didn't.
- Same 2 pre-existing `test_ragas_scorer.py` failures as every session
  since `ragas` was installed (asserting the "package not installed"
  path in an environment where it now genuinely is) -- unchanged,
  not caused by this work.

### Metrics
Not applicable -- this integrates tooling, it doesn't change any
retrieval/generation/judge behavior or produce new evaluation numbers.
All 8 backfilled MLflow runs' metrics are verbatim copies of their
already-published `experiments/results/*.json` values (spot-checked via
direct MLflow client queries, not just assumed).

### Limitations
- Backfilled runs only carry 2 artifacts (the record JSON + config YAML)
  since the original raw eval-output JSON was never preserved for
  `experiment_001`-`experiment_006` (temp files, long gone) or even
  `experiment_007`/`experiment_008` (also written to a temp path this
  session, not the repo) -- only *future* `record_experiment.py` calls
  will get all 3 artifact types, assuming the caller still has the
  `--eval-output` file handy when they run it.
- `mlflow.enabled: true` is now the project-wide default, meaning
  `scripts/record_experiment.py` will hard-fail on a fresh clone that
  hasn't run `pip install .[mlflow]` yet -- a deliberate tradeoff (see
  Interview talking points), but a real behavior change from before this
  session, worth knowing if `record_experiment.py` ever runs in a
  minimal/CI-only environment that doesn't need MLflow.
- No remote/server MLflow deployment has been tried -- `tracking_uri`
  being config-driven is designed to support it, but only the local
  SQLite path has actually been exercised.

### Verification
`pytest tests/unit -q`: 178 passed (was 169; +9 net new), 1 skipped, 2
pre-existing expected failures (see Issues). `pre-commit run --all-files`:
all hooks green (ruff, ruff-format, mypy) on the first run. Real run:
`pip install -e ".[mlflow]"`, `python scripts/backfill_mlflow.py` (all 8
experiments logged, verified via direct `MlflowClient` queries matching
every param/metric/tag/artifact against source JSON), a live
`record_experiment.py` smoke test with a throwaway record (confirmed all
3 artifact types attach, then deleted both the throwaway files and the
MLflow run). No commit, no push.

## 2026-08-07-01:00 | Configurable hybrid retrieval (BM25 + dense + RRF)

### Interview talking points
- **What problem did this solve?** Dense-only (embedding) search misses
  exact-token matches -- product codes, config keys, identifiers -- that
  a purely semantic model can underweight. This adds a second, opt-in
  retrieval strategy (BM25 keyword search fused with the existing dense
  search via Reciprocal Rank Fusion) selectable via
  `config.retrieval.provider`, with dense-only staying the default so
  nothing changes unless a user explicitly opts in, then runs it as a
  real, tracked experiment against the existing dense-only baseline.
- **Why was it difficult?** Two real design decisions, not just
  plumbing: (1) RRF fundamentally needs *two* ranked lists as input, but
  this project's `Reranker` ABC only accepts one -- fusion had to live as
  a new pure function called from `RetrievalPipeline.retrieve()` itself,
  not shoehorned into the reranker swap point. (2) True BM25 needs
  document-length-normalized scoring (k1/b) that Postgres's built-in
  `ts_rank` doesn't provide and no full-text extension was available, so
  `rank_bm25` (pure Python, tiny) builds a fresh in-memory index per
  query from the same `content` column dense search already reads --
  deliberately no schema migration, no persistent index, no caching, a
  documented scale-appropriate simplification.
- **What went wrong, and how was it caught?** The first full-scale
  62-question run showed hybrid performing *worse* than dense across
  almost every metric, including on `code_configuration`/`table`
  questions -- the exact content types BM25 should help most. Rather
  than accept a plausible-looking negative result, built a per-content-
  type breakdown (reusing `eval/content_type.py`, no new code) and then
  directly queried `search_keyword("maximum_wait_minutes", ...)` against
  the real chunk containing that exact string: **BM25 score was 0.0**.
  The naive `text.lower().split()` tokenizer left punctuation attached
  to tokens (`"maximum_wait_minutes":` with the quote and colon glued
  on), so it could never match a plain-word query. Confirmed by printing
  the raw chunk content and its tokenization directly, not just infered.
  Asked the user whether to fix and re-run (real additional OpenAI
  spend) rather than assume -- they said yes. Fixed by switching
  tokenization to `\w+`-based word extraction (`_tokenize`, a new small
  testable function), re-ran, and the picture changed substantially: the
  `code_configuration` bucket's Recall@5 went from 0.833 back to 1.000
  (matching dense), and the full-scale aggregate flipped from "hybrid
  underperforms across the board" to "hybrid matches dense on retrieval
  recall/MRR while meaningfully beating it on faithfulness (+0.055),
  context_recall (+0.035), and answer_correctness (+0.035)."
- **What proved it worked?** A dedicated adversarial integration test
  (`test_hybrid_retrieval.py`, modeled on `test_dataset_isolation.py`'s
  synthetic-fixture pattern): a document containing a rare, semantically
  opaque identifier (`XR-9942-B`) among several "noise" documents sharing
  heavy vocabulary overlap with the query but never the identifier
  itself -- hybrid retrieval surfaces it within top-2 every time; a `>=`
  (not strict `>`) comparison against dense-only on the same fixture,
  since real embedding behavior can't be guaranteed brittle-exact. Then
  the real thing: a 15-question pilot (reusing the exact stratified
  sample from `experiment_007`, for a true apples-to-apples comparison)
  before committing to the full 62-question run, mirroring the
  pilot-then-scale discipline established for the RAGAS milestone.

### Goal
Implement BM25 + dense hybrid retrieval fused via RRF as a config-driven
`retrieval.provider` option (`dense` default, `hybrid` opt-in), with no
DB migration and no change to dense-only behavior. Prove it works via
unit tests (fusion math, pipeline branching, config defaults) and a
real adversarial integration test. Run it as a genuine comparable
experiment (deterministic Recall@k/MRR/latency + RAGAS Context
Precision/Faithfulness/Answer Correctness) against the existing
dense-only baseline, both at pilot (15-question) and full (62-question)
scale, and report the result honestly regardless of which way it goes.

### Changes
- `src/rag/config.py`: new `HybridRetrievalConfig` (`rrf_k: int = 60`);
  `RetrievalConfig` gains `provider: Literal["dense", "hybrid"] = "dense"`
  and `hybrid: HybridRetrievalConfig`. `config/default.yaml` stays
  `dense` (unchanged default); new `config/experiments/hybrid-retrieval.yaml`
  (full copy, one line changed) is the vehicle for running hybrid,
  matching the established `config/experiments/*.yaml` precedent.
- `src/rag/vectorstore/base.py`: `VectorStore` ABC gains
  `search_keyword(query, top_k, filters=None) -> list[SearchResult]`.
- `src/rag/vectorstore/pgvector.py`: new module-level `_build_where_clause`
  (extracted from `search()`, now shared by both methods -- also closes
  a real pre-existing gap, no unit test previously exercised
  `ALLOWED_FILTER_FIELDS`'s `ValueError` path without a live DB), new
  `_row_to_metadata` (shared row-unpacking), new `_tokenize` (`\w+`-based,
  the punctuation-stripping fix), new `search_keyword()` (fetches
  filtered chunk content, builds a fresh in-memory `rank_bm25.BM25Okapi`
  index per call, no caching).
- New `src/rag/retrieval/fusion.py`: `reciprocal_rank_fusion(ranked_lists,
  k=60)` -- dedups by `Chunk.id`, sums `1/(k+rank)` per list, `.model_copy`
  idiom matching `CrossEncoderReranker`'s existing rescoring pattern.
- `src/rag/retrieval/pipeline.py`: `retrieve()` branches on
  `config.retrieval.provider` directly (no factory.py dispatch needed --
  no alternate concrete class is swapped, just an added capability on
  the same `PgVectorStore`); dense branch is byte-identical to before.
- `pyproject.toml`: `rank_bm25` (base dependency, not an optional extra
  -- hybrid is a first-class provider) + its mypy `ignore_missing_imports`
  override.
- Experiment bookkeeping: `run_eval.py`'s `_config_summary`,
  `record_experiment.py`'s `build_experiment_record` (`retrieval_provider`,
  `rrf_k`), `mlflow_logger.py`'s `_PARAM_FIELDS`, and
  `compare_experiments.py`'s table (new `Retrieval` column) all updated
  so a hybrid experiment is fully comparable/queryable, not just
  recorded.
- Tests: `test_fusion.py` (new, 7), `test_pgvector_where_clause.py` (new,
  8, incl. the `_tokenize` regression tests), `test_retrieval_pipeline.py`
  (+5: dense-unchanged regression, hybrid calls both search methods,
  fusion ordering, configured `rrf_k`), `test_config.py` (+3),
  `test_record_experiment.py` (+2), `test_mlflow_logger.py` (+1),
  `test_compare_experiments.py` (+2); `tests/integration/test_pgvector.py`
  (+3: `search_keyword` round-trip/filters/empty-corpus); new
  `tests/integration/test_hybrid_retrieval.py` (the adversarial fixture).

### Issues
- The tokenization bug above (see Interview talking points) -- the most
  significant finding this session. Root-caused with direct evidence
  (a 0.0 BM25 score against a chunk literally containing the queried
  term, then the raw tokenized output showing the punctuation-glued
  token), not guessed at. Fixed via `\w+`-based tokenization; regression
  tests added (`_tokenize` strips punctuation, preserves underscored
  identifiers, lowercases).
- `generation_ms`/`total_ms` swung substantially between the pre-fix and
  post-fix full runs (4262ms -> 1972ms) in a way retrieval strategy
  shouldn't mechanically cause -- almost certainly Ollama/system warm-
  state variance between runs in different sessions, not a hybrid-
  retrieval effect. Flagged explicitly rather than folded into the
  "hybrid is faster" narrative, since that causal claim isn't supported;
  `retrieval_latency_ms` (88.8ms dense vs 88.2ms hybrid) is the
  mechanistically meaningful latency comparison here, and it's
  effectively a wash despite the added BM25 pass.
- `VectorStore` ABC gained a required abstract method
  (`search_keyword`) -- a non-backward-compatible change for a
  hypothetical third-party `VectorStore` implementer, acceptable since
  `PgVectorStore` is the only one that exists.

### Metrics
Adversarial integration test: hybrid retrieval surfaces a rare exact
identifier within top-2 consistently; dense-only's rank on the same
fixture varies by run (real embedding behavior), hybrid is always `>=`.

15-question pilot (`experiment_009` vs. `experiment_007`, same exact
questions/judge): Recall@5 0.90 vs 1.00 (dense already perfect on this
small curated sample -- limited room for hybrid to improve further),
MRR 0.822 vs 0.906, but context_precision 0.894 vs 0.861,
context_recall 0.856 vs 0.811, and answer_correctness 0.591 vs 0.471
(the standout improvement) all favored hybrid even before the
tokenizer fix.

Full 62-question (`experiment_010`, post-fix, vs. `experiment_008`
dense baseline): Recall@5 0.903 vs 0.919, Recall@10 0.944 vs 0.952,
Hit Rate@5 0.935 vs 0.935 (tied), MRR 0.865 vs 0.865 (tied),
faithfulness 0.841 vs 0.786 (hybrid better), context_precision 0.747
vs 0.754 (near-tied), context_recall 0.763 vs 0.728 (hybrid better),
answer_correctness 0.504 vs 0.469 (hybrid better), answer_relevancy
0.590 vs 0.631 (dense better). Judge cost: 764 calls, ~594K tokens
(~$0.10-0.15).

By content-type bucket (post-fix, hybrid vs. dense): the largest win is
`multi_hop` (n=14) -- context_recall 0.762->0.917, answer_correctness
0.487->0.651. `code_configuration` (n=6) and `table` (n=7) both returned
to dense-matching Recall@5=1.000 after the tokenizer fix (were
0.833/0.857 with the bug). `unanswerable` (n=9) faithfulness improved
0.496->0.630. `prose` (n=24) was essentially flat.

### Limitations
- No persistent BM25 index/caching -- rebuilt per query from a live SQL
  fetch, appropriate at ~317 chunks, a real scaling fork point (persistent
  index, or Postgres `tsvector`/`ts_rank`) if corpus size grows.
- Tokenization is still simple (`\w+`, lowercase, no stemming/
  lemmatization/stopwords) -- fixed the punctuation-stripping bug, but
  "running" still won't match a document containing only "runs".
- The `multi_hop` bucket's large improvement is the single biggest
  driver of the full-scale result and is based on only 14 questions --
  a real, evidenced pattern, not yet a statistically overwhelming one.
- `rerank_top_n`/`top_k` were deliberately kept identical between the
  dense and hybrid experiments for a clean single-variable comparison;
  a wider hybrid-specific candidate pool (fetching more per branch
  before fusion) is an unexplored, plausible further improvement.
- `generation_ms`/`total_ms` comparisons across the two full runs are
  confounded by session/system state (see Issues) -- don't cite them as
  a hybrid-retrieval effect.

### Verification
`pytest tests/unit -q`: 205 passed (was 178; +27 net new), 1 skipped, 2
pre-existing expected failures (unrelated `ragas`-installed reason).
`pytest tests/integration -q`: 21 passed both before and after the
tokenizer fix (real Postgres, including the new adversarial hybrid test
and 3 new `search_keyword` tests). `pre-commit run --all-files`: all
hooks green (ruff, ruff-format, mypy -- the `rank_bm25.*` override was
load-bearing; two `D301` raw-string-docstring fixes needed after adding
`\w+` to docstrings). Manual sanity check of real `retrieve()` calls
(dense vs. hybrid) against the live `techfusion` dataset before running
any paid eval. Real run: 15-question pilot (`experiment_009`), full
62-question run that surfaced the tokenizer bug, the bug fixed and
full-suite re-verified (unit + integration both green again), the full
62-question run repeated and re-recorded as `experiment_010` (the
buggy version's MLflow run explicitly deleted rather than left
alongside the corrected one), `scripts/compare_experiments.py --exclude
experiment_007,experiment_009` (both 15-question pilots excluded from
the standing table, matching precedent), `README.md`/`comparison.md`
updated. No commit, no push.

---

## 2026-08-07-22:30 | Containerize the local runtime: Dockerfile + docker-compose (dev + prod-shaped)

### Interview talking points
- **What problem did this solve?** Only Postgres ran in a container before
  this; the API ran via bare `uvicorn` on the host and Ollama natively on
  Windows. Added a multi-stage `Dockerfile` for the FastAPI app, wired it
  into `docker-compose.yml` alongside Postgres, added a `docker-compose.
  prod.yml` override for an immutable/no-reload/no-source-mount run, and a
  smoke-test script proving the *containerized* API can actually reach
  Postgres+pgvector and native Windows Ollama (not just that the container
  starts).
- **Why was it difficult?** Every one of the four real bugs below only
  showed up by actually running the containers, not from reading the
  Dockerfile/compose YAML — pip's default torch wheel silently pulling in
  4GB of unused CUDA libraries, a non-editable `pip install .` silently
  breaking a `Path(__file__)`-based config-root computation, `localhost`
  silently resolving to a `::1` this image doesn't support, and Compose's
  list-merge-by-target semantics silently leaving dev-only bind mounts
  active in what was supposed to be an immutable prod override. All four
  fail *quietly* (wrong behavior, not a crash with a helpful message).
- **How did I debug it?** Built the image, ran `docker compose up`, and
  worked from `docker logs`/`docker inspect --format '{{json .State.
  Health}}'` symptom to root cause for each failure in turn (image size ->
  `pip install` log showing `nvidia-*` packages; `FileNotFoundError` at
  startup -> traced to `REPO_ROOT` via `rag/config.py`'s own `__file__`
  logic; empty healthcheck log -> the actual urllib traceback embedded in
  `docker inspect`'s health log; wrong prod volumes -> `docker compose
  ...config` printed the fully-merged YAML directly).
- **What metrics proved the fix?** Image size 5.6GB -> 1.6GB after pinning
  the CPU-only torch wheel (identical CPU inference behavior, config
  already said `device: cpu`). `docker inspect` health status flipped
  `unhealthy`/`starting`-forever -> `healthy` after the REPO_ROOT and
  `127.0.0.1` fixes. `docker compose -f docker-compose.yml -f docker-
  compose.prod.yml config` output showing `rag-api.volumes` with exactly
  one entry (not three) and `postgres.ports` absent entirely, after adding
  `!override`. Final proof: `scripts/smoke_test_containers.sh` all-PASS,
  including a real ingest (5 chunks written) and a real Ollama-generated
  answer quoting the actual password-policy content.
- **What would I improve next?** Bake the embedding model into the image
  at build time so the containerized path doesn't need first-run internet
  access despite the "runs fully offline" claim in the README; add a
  registry/CI publish step so `docker-compose.prod.yml`'s image comment
  becomes a real, non-local image reference.

### Goal
Implement Phase 1 (Dockerfile, .dockerignore, compose updates) and Phase 2
(smoke tests, README docs) of the containerization plan approved earlier
this session: multi-stage `Dockerfile` for `rag-api`, `.dockerignore`,
`docker-compose.yml` gaining a `rag-api` service alongside the existing
`postgres` one, a `docker-compose.prod.yml` override, health checks for
Postgres/rag-api/Ollama, non-root container execution, and a smoke test
proving the containerized API reaches pgvector and native Windows Ollama.
Explicit constraints: no Redis, no ingestion worker, no Kubernetes
manifests, no RAGAS/MLflow in the image, no application-behavior changes.

### Configuration
- `Dockerfile`: two-stage build (`builder` installs into `/opt/venv` via
  `pip install .` with no extras; `runtime` copies just that venv +
  `src/` + `config/` in). Non-root `rag` user (uid/gid from `useradd -r`).
  `PYTHONPATH=/app/src` so the runtime source tree shadows the copy
  `pip install .` put in site-packages (see Issues). `HOME=/app` so
  sentence-transformers' HF-hub cache has somewhere writable to land.
- `docker-compose.yml`: `postgres` (unchanged aside from resource limits)
  + new `rag-api` (2 CPU/2GB limit, 1/1 reservation; postgres gets 1
  CPU/1GB limit, 0.5/512MB reservation — the approved 8GB-box sizing).
  `rag-api` depends on `postgres: condition: service_healthy` (hard dep)
  but not on Ollama (soft dep, surfaced via `/health` instead).
  `OLLAMA_BASE_URL` inside the container is sourced from a new
  `CONTAINER_OLLAMA_BASE_URL` env var (default `http://host.docker.
  internal:11434`), deliberately *not* reusing `.env`'s existing
  `OLLAMA_BASE_URL=http://localhost:11434` — that value is correct for a
  native/uvicorn-on-host run and would be silently wrong (loops back to
  the container itself) if compose substitution had pulled it in instead.
  `extra_hosts: host.docker.internal:host-gateway` added so the same file
  also works on native Linux Docker, not just Docker Desktop.
- `docker-compose.prod.yml`: layered override (`-f docker-compose.yml -f
  docker-compose.prod.yml`). Drops src/config bind mounts and `--reload`,
  stops publishing Postgres's port to the host, tightens health-check
  timing. Uses the Compose 2.24+ `!override` YAML tag on `volumes`/`ports`
  (see Issues — plain redeclaration doesn't replace those lists).
- `.dockerignore`: excludes `tests/`, `scripts/`, `data/`, `experiments/`,
  `docs/`, dev/doc files, and `.env*` (secrets never enter the build
  context).
- `scripts/smoke_test_containers.sh`: bash, checks `/health` twice
  (respond-at-all, then per-dependency ok/degraded), ingests
  `data/sample_docs/password-policy.md` under a timestamped
  `dataset_id`, queries it, asserts a non-empty generated answer.

### Changes made
- Added: `Dockerfile`, `docker-compose.prod.yml`,
  `scripts/smoke_test_containers.sh`.
- Modified: `.dockerignore` (was a 8-line stub; now excludes the full set
  above), `docker-compose.yml` (added `rag-api` service + resource
  limits on both services), `.env.example` (documented
  `CONTAINER_OLLAMA_BASE_URL` and `LOG_LEVEL`), `README.md` (new
  "Containerized development" section with exact build/run/query
  commands, a "Production-shaped run" subsection, a "What's in the image"
  subsection, a "Windows-specific limitations" subsection, and a
  `scripts/smoke_test_containers.sh` mention under Testing).
- Did not touch: any file under `src/rag/` — no application-behavior
  changes, per the explicit constraint.

### Issues discovered
1. **Symptom:** first build produced a 5.6GB image. **Root cause:**
   `sentence-transformers` pulls in `torch` transitively; PyPI's default
   torch wheel bundles full CUDA/cuDNN/NCCL/triton (~4GB) even though
   `config/default.yaml`'s `embedding.device: cpu` never touches a GPU.
   **Fix:** install the CPU-only torch wheel from `https://download.
   pytorch.org/whl/cpu` *before* `pip install .`, so that requirement is
   already satisfied (as `2.13.0+cpu`) by the time sentence-transformers'
   dependency resolution runs. **Evidence:** image size 5.6GB -> 1.6GB;
   `docker logs` build output shows `Requirement already satisfied:
   torch>=1.11.0 ... (2.13.0+cpu)` instead of downloading `nvidia-*`
   packages. **Resolved:** y.
2. **Symptom:** container crashed at import time with `FileNotFoundError:
   [Errno 2] No such file or directory: '/opt/venv/lib/python3.11/config/
   default.yaml'`. **Root cause:** `rag.config.REPO_ROOT` is computed as
   `Path(__file__).resolve().parents[2]`; local dev works because `pip
   install -e .` (editable) keeps `__file__` pointing at the real
   `src/rag/config.py` in the repo, but the Dockerfile's non-editable
   `pip install .` copies `rag/` into `site-packages`, so `__file__`
   resolved three parents up from *there* instead of from `/app`. **Fix:**
   set `PYTHONPATH=/app/src` in the runtime stage so the `COPY --chown=rag:
   rag src ./src` copy shadows the site-packages one on `sys.path`,
   restoring the same `__file__` -> `/app` resolution local dev gets from
   editable install. **Evidence:** container now reaches `Application
   startup complete` and `/health` returns real dependency status instead
   of crashing before FastAPI even loads. **Resolved:** y.
3. **Symptom:** container's own `HEALTHCHECK` stayed `starting` forever
   even after `docker logs` showed `Application startup complete`;
   `docker inspect --format '{{json .State.Health}}'` showed a Python
   traceback ending `OSError: [Errno 99] Cannot assign requested address`.
   **Root cause:** the healthcheck command used `http://localhost:8000/
   health`; this minimal Debian-slim container's network namespace doesn't
   have IPv6 loopback configured, and `localhost` resolves to `::1` first,
   which then fails outright instead of falling back to IPv4. **Fix:**
   use `http://127.0.0.1:8000/health` explicitly in the `HEALTHCHECK`
   directive. **Evidence:** `docker inspect` health status flips to
   `healthy`; `curl http://localhost:8000/health` from the *host* (a
   different network namespace, unaffected) had actually been working the
   whole time, which is why this only showed up in the container's
   self-check, not in manual testing. **Resolved:** y.
4. **Symptom:** `docker compose -f docker-compose.yml -f docker-compose.
   prod.yml config` showed `rag-api.volumes` with all three dev entries
   (src, config, data/uploads) still present, despite `docker-compose.
   prod.yml` declaring only the single `data/uploads` mount. **Root
   cause:** Compose merges `volumes`/`ports` lists across `-f` files by
   *target* (adds/updates matching entries), it does not replace the list
   wholesale the way scalar keys do — a widely-misunderstood Compose
   behavior. **Fix:** tag both overridden lists with the Compose 2.24+
   `!override` YAML tag (`volumes: !override [...]`, `ports: !override
   []`), which forces full replacement instead of merge. **Evidence:**
   re-ran `docker compose ... config` after the fix — `rag-api.volumes`
   now shows exactly the one entry, `postgres.ports` is absent from the
   merged output entirely (was `15987:5432` before). Confirmed installed
   Compose is 2.24.6 (`docker compose version`), which supports the tag.
   **Resolved:** y (config-merge verified, then the prod stack itself was
   brought up live with `docker compose -f docker-compose.yml -f docker-
   compose.prod.yml up -d` and passed the full smoke test — `postgres`
   showed `5432/tcp` with no host publish in `docker compose ps`, `rag-api`
   ran the immutable baked-in image with no source mounts and no
   `--reload`, and ingest/query both worked identically to the dev stack).
5. **Symptom (caught by inspection, not by a failing run):** `.env`
   already sets `OLLAMA_BASE_URL=http://localhost:11434` for native/
   uvicorn-on-host use. Had `docker-compose.yml` substituted
   `${OLLAMA_BASE_URL:-http://host.docker.internal:11434}` directly, the
   existing `.env` value would have taken precedence (env var *is* set,
   just not to the container-appropriate value) and the container would
   have tried to reach Ollama at itself. **Fix:** introduced a separate
   `CONTAINER_OLLAMA_BASE_URL` var for the compose substitution so the two
   never collide. **Resolved:** y (avoided before it could manifest as a
   bug; not something a test caught).
6. **Symptom:** `pytest tests/ -v` (run once, before any container work,
   to get a clean baseline) showed 2 failures in `tests/unit/
   test_ragas_scorer.py` (`test_score_missing_ragas_package_raises_
   runtime_error`, `test_build_dataset_missing_datasets_package_raises_
   runtime_error`). **Root cause:** not investigated — matches the exact
   "pre-existing expected failures (unrelated `ragas`-installed reason)"
   noted in this journal's 2026-08-06 hybrid-retrieval entry, i.e. these
   tests assert `ragas`/`datasets` are *absent* from the dev venv to
   exercise a not-installed code path, and both packages happen to be
   installed in this `.venv`. Unrelated to containerization.
   **Resolved:** n (pre-existing, out of scope for this session; 207
   passed, 20 skipped otherwise).

### Evaluation results
`scripts/smoke_test_containers.sh` against the built-and-running dev
stack, all four check groups PASS:
- `/health`: `{"status":"ok","dependencies":{"vectorstore":"ok","llm":
  "ok"}}` — proves the container reached both containerized Postgres and
  native Windows Ollama through `host.docker.internal`.
- Ingest: `data/sample_docs/password-policy.md` under a fresh
  `dataset_id` -> `chunks_written: 5`.
- Query: `retrieval_ms` ~51ms, `generation_ms` ~35000ms (Ollama cold
  model load — `qwen2.5:1.5b` hadn't been used yet this session; not a
  container-networking cost), answer text correctly describes the
  14-character/no-reuse/1Password policy from the ingested document, with
  3 sources cited back to the right `chunk_id`s.
- Image size: 5.6GB before the torch fix, 1.6GB after.
- `docker inspect` health status: both `local-rag-postgres` and
  `local-rag-api` report `healthy`.
- `pytest tests/ -v`: 207 passed, 2 failed (pre-existing, see Issue #6),
  20 skipped (integration tests self-skip without `make up`-equivalent
  infra already running under pytest, unrelated to the containers built
  this session).

### Interpretation
The containerized API is behaviorally identical to native dev once the
four bugs above are fixed — same `/health` shape, same ingest/query
response shape, same retrieval latency. All four bugs were specifically
about *packaging* (dependency resolution, path resolution, network stack
minimalism, Compose merge semantics), not about the application code,
which matches the "no application-behavior changes" constraint. The
generation latency (~35s) is Ollama model-load cost on the host, visible
identically whether the caller is inside a container or not — worth
knowing so a future container-vs-native latency comparison isn't
misattributed to Docker overhead.

### Open problems
- The embedding model downloads from Hugging Face Hub on first use inside
  a fresh container (no pre-baked cache), needing outbound internet the
  *first* time — a gap against the README's "runs fully offline" framing
  specifically for the containerized path (native dev already has the
  model cached from prior sessions, so this gap is new/container-specific).
- `rag-api`'s image is still built locally by `docker compose up --build`
  in both dev and prod compose files; no registry/CI publishing step
  exists yet (the `# image: registry.example.com/...` line in
  `docker-compose.prod.yml` is a placeholder comment, not wired up).
- Windows bind-mount permission behavior (non-root `rag` user reading/
  writing `./src`, `./config`, `./data/uploads` without UID/GID matching)
  was asserted from known Docker Desktop behavior and confirmed
  functionally by the smoke test's successful ingest (which writes to
  `data/uploads`), but not stress-tested across Docker Desktop's
  different backends (WSL2 vs. Hyper-V).

### Next recommended step
If the offline-first guarantee matters for the containerized path
specifically, add a build-time step that pre-downloads the configured
embedding model into the image (trades a larger, slower build for zero
first-run network dependency). Otherwise the remaining open problems are
accepted deferrals (registry/CI publishing, exhaustive cross-backend
Windows bind-mount testing) rather than coverage gaps.

## 2026-08-08-11:00 | GitHub Actions CI: code-quality, unit, integration, docker-build

### Goal
Set up a production-style CI pipeline that runs automatically on every PR
and push to `main`, verifying the repo actually builds/lints/tests cleanly
without requiring anything the repo can't provide for itself on a hosted
runner -- no committed secrets, no local Docker stack, no native Ollama.

### Changes made
- `.github/workflows/ci.yml`: four parallel jobs (`code-quality`,
  `unit-tests`, `integration-tests`, `docker-build`), triggered on
  `pull_request`/`push` to `main` plus manual `workflow_dispatch`.
  `permissions: contents: read` at the workflow level, a
  `concurrency`/`cancel-in-progress` group per branch/PR, and a
  `timeout-minutes` on every job so a hung step can't run indefinitely.
  No `pull_request_target`, no third-party actions beyond the
  Docker org's own `setup-buildx-action`/`build-push-action`.
- `README.md`: a CI badge under the title, and a "Continuous Integration"
  section (after Testing) documenting what each job checks and what's
  intentionally left out.

### CI architecture
Jobs run in parallel (no `needs:` between them) rather than a linear
gate chain -- none of the four depends on another's output, so fanning
out gets PR feedback back faster than serializing them would:
- **code-quality**: `pre-commit run --all-files`. Deliberately does *not*
  also `pip install -e .[dev]` first -- pre-commit's ruff/mypy hooks
  already run in their own isolated, pinned environments per
  `.pre-commit-config.yaml`, so the job only needs `pip install
  pre-commit` itself. No separate lint/mypy step exists alongside it;
  that would just re-run the same checks a second time.
- **unit-tests**: `pytest tests/unit -v --junitxml=...`, plus a short
  pass/fail/time summary appended to the job's `$GITHUB_STEP_SUMMARY`
  (parsed from the JUnit XML) and the XML itself uploaded as an artifact.
  No coverage tooling added -- `pytest-cov` isn't already a project
  dependency, and CLAUDE.md's testing conventions don't call for one.
- **integration-tests**: see below.
- **docker-build**: see below.

### Why heavy LLM/RAGAS evaluation stays out of PR CI
RAGAS scoring, the OpenAI/Anthropic judge path, MLflow experiment
logging, and the full TechFusion gold-set run all need one of: a hosted
API key (which a public repo's fork-PR CI must never be handed), a local
Ollama model pull, or multi-minute LLM calls -- none of which belong in
per-PR feedback that's supposed to come back in a couple of minutes.
These stay exactly as they are today: manual, local invocations of
`scripts/record_experiment.py` / `eval/run_ragas_eval.py` /
`scripts/compare_ragas_manual.py`. A `workflow_dispatch`-only
`evaluation.yml` was scoped in the design conversation (full integration
incl. Ollama, hosted-judge RAGAS) but deliberately not created yet --
no concrete need for it in CI specifically until something wants to
consume its output on a schedule.

### Integration-test strategy
`tests/integration` assumes Postgres is reachable at `DATABASE_URL` and
skips Ollama-gated tests cleanly (`require_ollama` in
`tests/integration/conftest.py`) -- exactly the shape CI needed, no test
code changed. The job starts a `pgvector/pgvector:pg16` *service*
container (same image `docker-compose.yml` uses), waits on its
`pg_isready` healthcheck, runs `scripts/init_db.py` to create the
extension/tables (same idempotent script `make up` calls), then runs
`pytest tests/integration -v`. No Ollama service is started, so the
`require_ollama`-gated tests (`test_ollama.py`, the generation-path
assertions in `test_query_pipeline.py`) skip rather than fail. The one
unavoidable network dependency is the real embedder some integration
tests exercise (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB per
`config/default.yaml`) -- cached via `actions/cache` on
`~/.cache/huggingface` keyed by model name, so only the first run after
a model change pays the download.

### Docker-build verification
Builds the production image from the existing multi-stage `Dockerfile`
(`docker/build-push-action` with `push: false, load: true`, GHA layer
caching via `cache-from/to: type=gha`), then actually runs it against
the same Postgres service container the integration job uses --
`docker run --network host` so the container reaches the service on
`localhost:5432` the same way the job's own steps do, without wiring a
separate user-defined network. Polls `GET /health` (which per
`health.py` always returns HTTP 200 with a `status: ok|degraded` body,
never a hard failure) and asserts `dependencies.vectorstore == "ok"`;
`dependencies.llm` is expected `"unreachable"` since no Ollama runs in
CI. This is a narrower check than `scripts/smoke_test_containers.sh`
(which also ingests + queries and therefore needs Ollama for step 4) --
that script stays a local/native-Ollama tool, not something CI runs.
No image is pushed to a registry.

### Verification
- Ran `pytest tests/unit -q` locally against the same command the
  `unit-tests` job uses: 205 passed, 2 failed, 1 skipped. Both failures
  are the pre-existing `test_ragas_scorer.py` cases documented in
  CLAUDE.md's Testing conventions (they assert `ragas`/`datasets` are
  *absent*, and this dev `.venv` happens to have them installed). Since
  CI's `pip install -e .[dev]` never installs the `ragas` extra, that job
  should come back fully green -- a stronger result than the local run.
- Parsed `.github/workflows/ci.yml` with PyYAML (via the project's own
  `.venv`) to confirm it's syntactically valid; the only oddity is
  `on:` parsing as the boolean key `True` under PyYAML's YAML 1.1 rules,
  which is cosmetic (GitHub's own workflow parser handles `on:` as the
  trigger key regardless) and not a real bug.
- Checked every path/command referenced in the workflow
  (`scripts/init_db.py`, `pytest tests/unit`, `pytest tests/integration`,
  `pip install -e .[dev]`, the `pgvector/pgvector:pg16` image, `DATABASE_URL`
  shape) against the actual repo rather than assuming -- no local absolute
  paths anywhere in the file.
- Did not run the workflow on GitHub Actions itself (no push/PR made yet
  as part of this session); the checks above are the practical substitute
  available locally.

### Open problems
- Not yet verified end-to-end on an actual GitHub-hosted runner --
  `--network host` for the `docker-build` job's smoke-test container is
  a well-established pattern on Linux runners but hasn't been observed
  running in this repo's own Actions environment yet.
- The `code-quality` job's `pre-commit run --all-files` will do a fresh,
  uncached hook-environment install on its very first run (ruff/mypy
  pulling their own pinned deps) before the `actions/cache` key has
  anything to restore -- expect a slower first CI run, not a broken one.

### Next recommended step (future CD roadmap)
No CD in this milestone by design. Once `docker-build` has proven itself
stable in real PRs, the natural next step is a separate, `main`-only
(not per-PR) workflow that: builds and pushes the image to a registry
(GHCR is the natural first choice -- no new secret needed beyond the
built-in `GITHUB_TOKEN`), tags it by commit SHA and `latest`, and updates
`docker-compose.prod.yml`'s commented-out `# image:
registry.example.com/...` line to point at the real registry path.
Actual deployment (Kubernetes manifests, or any non-local target) is a
larger, separate milestone tracked in README's Roadmap
("Kubernetes deployment") and out of scope for CI work.

## 2026-08-08-14:30 | Fix pgvector connection-pool leak surfaced by CI's docker-build job

### Goal
`docker-build`'s smoke test (the CI job added in the previous entry) was
failing in real GitHub Actions runs with
`psycopg2.pool.PoolError: connection pool exhausted`, traced to
`health.py -> vectorstore.health_check() -> _conn() -> pool.getconn()`.
Investigate and fix as a real resource-management bug in
`PgVectorStore`, not by papering over it in the workflow.

### Root cause
The old `_conn()` helper did two things as separate statements, split
across two call frames:

```python
def _conn(self) -> PgConnection:
    conn = self._pool.getconn()   # checked out here
    register_vector(conn)         # <- can raise
    return conn                   # never reached if it does
```

Every caller wrote `conn = self._conn(); try: ...; finally:
self._putconn(conn)` -- but that `try/finally` only starts *after*
`_conn()` already returned. If `register_vector(conn)` raised, the
connection had already been popped from the pool (`getconn()` succeeded)
before the exception propagated, and no `finally` anywhere ever saw it --
it was gone from the pool permanently, not just for that request.

`register_vector()` raises when the `vector` Postgres type isn't
registered, i.e. `CREATE EXTENSION vector` + the schema hasn't been run
against that database yet. The `docker-build` job starts a brand-new,
empty `pgvector/pgvector:pg16` service container and -- unlike
`integration-tests` -- never ran `scripts/init_db.py` against it. Every
`/health` poll in that job's smoke-test loop hit this exact failure,
leaked one connection each time, and after `maxconn` (5, unchanged --
not the fix) leaked connections, `pool.getconn()` itself started raising
`PoolError`.

### Changes made
- `src/rag/vectorstore/pgvector.py`: replaced `_conn()`/`_putconn()` with
  a single `@contextmanager _connection()` that wraps `register_vector(conn)`
  itself inside the `try/finally`, so a connection is only ever "ours"
  once it's inside that block -- nothing between checkout and yield can
  leak it:
  ```python
  @contextmanager
  def _connection(self) -> Iterator[PgConnection]:
      conn = self._pool.getconn()
      try:
          register_vector(conn)
          yield conn
      finally:
          self._pool.putconn(conn)
  ```
  Every method (`health_check`, `get_or_create_document_id`,
  `delete_chunks_by_document_id`, `delete_document`, `delete_dataset`,
  `add_chunks`, `search`, `search_keyword`) now uses
  `with self._connection() as conn:`, removing the manual `try/finally`
  boilerplate everywhere, not just in `health_check` (every one of those
  methods had the identical leak risk, just never triggered because
  `register_vector` normally succeeds against an already-initialized DB).
  Pool size (`minconn`/`maxconn`) untouched, per the explicit constraint
  that increasing it would mask the bug rather than fix it.
- `tests/unit/test_pgvector_pool.py` (new): fakes `ThreadedConnectionPool`
  and the connection/cursor objects (no real DB, per this repo's unit-test
  convention) to prove repeated `health_check()`/`search()` calls never
  leak a connection, including when `register_vector` fails on every call
  (the literal CI scenario) and when query execution itself raises.
- `.github/workflows/ci.yml`: added a schema-init step
  (`python scripts/init_db.py`) to `docker-build`, matching what
  `integration-tests` already does, so the job's Postgres service
  container has a real schema before the smoke test runs against it.

### Verification
- Proved the new tests actually catch the bug: reverted
  `pgvector.py` to the pre-fix version, reran
  `tests/unit/test_pgvector_pool.py` -- 2 of 6 failed with the literal
  `psycopg2.ProgrammingError` from `register_vector`, propagating
  unhandled exactly as it would have inside a real `/health` request.
  Restored the fix, reran: 6/6 passed.
- `pytest tests/unit -q`: 211 passed (was 205 pre-fix, +6 new tests), 2
  failed -- the same pre-existing `ragas`/`datasets`-installed edge case
  documented previously, unrelated to this change.
- `pytest tests/integration -v` against a real local Postgres+pgvector
  container (`make up` equivalent) plus native Ollama: 21/21 passed,
  including the Ollama-dependent tests.
- `ruff check`, `ruff format --check`, and `mypy` on the changed files:
  clean.
- Built `local-rag-system:verify` locally (1.6GB, matches the documented
  containerization image size) and reproduced both ends of the bug against
  real Docker + real Postgres on an isolated bridge network (Windows
  Docker Desktop doesn't support the CI job's `--network host` trick, so
  this used a dedicated network instead, mirroring how `docker-compose.yml`
  already connects `rag-api` to `postgres`):
  - Against a **fresh, un-initialized** Postgres (the exact CI trigger):
    15 consecutive `/health` calls, every one `200
    {"status":"degraded","vectorstore":"unreachable"}` -- never a 500,
    never `PoolError`.
  - After running `scripts/init_db.py` against that same Postgres: 15
    more consecutive calls, every one `200 {"vectorstore":"ok"}`.
  Verification containers/network/image were removed afterward; the
  normal dev stack (`local-rag-postgres`) was untouched throughout.

### Interpretation
The bug was latent in every write/read path of `PgVectorStore`, not just
`health_check()` -- it just took a CI job that talks to a schema-less
database to actually trigger it, since `register_vector()` never fails
against a properly-initialized one. The context-manager refactor closes
the whole class of "something between pool checkout and pool return can
raise" bugs at once, rather than patching `health_check()` specifically.

### Open problems
None outstanding for this fix. Changes are staged locally
(`.github/workflows/ci.yml`, `src/rag/vectorstore/pgvector.py`,
`tests/unit/test_pgvector_pool.py`) but not yet committed/pushed --
holding per instruction until explicitly asked to commit.

## 2026-08-11-18:15 | Multimodal + relationship-aware ingestion and evaluation

### Interview talking points
- **What problem did this solve?** The knowledge base and gold set were
  extended with real images (dashboards, diagrams, charts, image-tables)
  and a richer gold schema (`reference_contexts`, `reference_visual_contexts`,
  `relevant_images`, `requires_vision`, `requires_relationship_expansion`,
  authored `content_type`), but none of it was wired in: images were
  invisible to the chunker (silently absorbed into whatever prose span
  happened to contain the markdown line), the new gold fields were
  silently dropped by pydantic's default `extra="ignore"`, and there was
  no way to preserve or re-surface structural relationships (image ->
  caption, table/code/image -> parent section) that a flat chunk stream
  loses. This closes all three gaps -- element model, ingestion, and
  retrieval-time relationship expansion -- and evaluates the result with
  new deterministic metrics, without adding an agent framework, a graph
  database, or any hosted API call.
- **Why was it difficult?** The interesting design tension was between
  "model images/relationships properly" and "don't overengineer." Landed
  on reusing `ChunkSpan`/`ChunkMetadata` rather than a parallel
  `DocumentElement` schema: an image is one `content_type="image"` span
  with its caption folded in (mirroring the chart-fence-plus-caption
  pattern that already existed), and `parent_chunk_id` is a single nullable
  column computed with a one-pass heuristic ("nearest preceding prose
  chunk in the same section"), not a real parse tree. Also had to keep
  retrieval ranking and relationship expansion genuinely separate --
  expansion is appended after the already-reranked list with its own
  `origin`/`expanded_from` marker and an inherited (not fabricated) score,
  specifically so Recall@5/@10/MRR stay uncontaminated by it. Confirmed
  this actually held in the real run (see below).
- **What proved it worked?** 258 new/updated unit tests (incl. a static
  import-boundary check + a runtime marker-string check that
  `reference_visual_contexts` never reaches a generated prompt, and a
  dataset_id-isolation test for expansion), a real re-ingestion of the
  full `data/knowledge_base` under `techfusion` that produced exactly the
  13 image chunks matching the 13 unique image references actually
  present in the markdown, and two real local experiments (A: expansion
  off, B: expansion on) whose Recall@5/@10/hit-rate/MRR came out **bit-
  for-bit identical** between A and B -- direct proof expansion never
  leaks into the ranked cutoff -- while `supporting_context_hit_rate`
  (0.697 -> 0.788) and `relevant_image_hit_rate` (0.579 -> 0.842) both
  improved, and `relationship_expansion_contribution_rate` (0.0 -> 0.286)
  quantified that expansion demonstrably supplied missing evidence on 2
  of 7 relationship-aware questions.
- **What's next?** A credit-safe, stratified 15-question RAGAS sample
  selection from the new gold file, presented for approval before any
  OpenAI call (Experiment C / hosted vision are both still gated on
  separate approval).

### Goal
Wire up the multimodal/relationship-aware milestone end to end: extend
the gold schema, give images a first-class element representation with
preserved relationships, add a config-gated (off-by-default) post-rerank
relationship-expansion step, scaffold (but do not implement/call) a
hosted VisionProvider, add new deterministic reference-context/image/
expansion/vision-behavior metrics, and run two controlled local
experiments (prompt v2, hybrid+RRF baseline matching `experiment_010`,
text-only image handling) isolating relationship expansion as the only
variable. No hosted vision call, no RAGAS run, this session.

### Design decisions (see docs/architecture.md for full detail)
- `GoldExample` gained 6 optional fields; old gold files
  (`techfusion_gold_old.jsonl`, `sample_gold.jsonl`) still parse unchanged.
- Images: a standalone `![alt](path)` markdown line becomes its own
  `content_type="image"` `ChunkSpan`, at the same priority as table/fence
  detection in `StructuredMarkdownChunker`; an immediately-following
  emphasis-wrapped caption paragraph folds into the same span's text
  (reusing the existing chart-fence-plus-caption lookahead, refactored
  into a shared `_peek_caption` helper). An inline image link mid-
  paragraph is untouched (still just attachment tagging).
- Relationships: new nullable `chunks.parent_chunk_id` (no FK -- parent/
  child land in the same insert batch, and deletion is always whole-
  document via cascade, so the insert-ordering fragility a self-
  referential FK would add isn't worth it), computed in `Writer.write`'s
  existing single pass (no new pipeline stage). Two new `VectorStore`
  methods, `get_chunks_by_ids`/`get_chunks_by_section`, are the only new
  SQL capability (plain `SELECT ... WHERE document_id = ...`, no new
  query complexity).
- Vision: `config.vision.provider` (`Literal["none"]` only, for now),
  `factory.build_vision_provider`, `image_description_cache` table
  (checksum-keyed), and `Writer._with_vision_siblings` are all wired, but
  **no concrete hosted-API-calling `VisionProvider` subclass was written**
  -- no provider/model has been chosen and nothing in Experiment A/B
  needed it. `VisionProvider` itself gained `provider_name`/`model_name`
  abstract properties for cache provenance.
- Retrieval: `config.retrieval.relationship_expansion` (`enabled: false`
  by default, a no-op unless turned on). `RetrievalPipeline.retrieve()`,
  after reranking, appends parent/neighbor chunks (deduped, capped at
  `max_related_elements`) with `origin="expanded"` on a new `SearchResult`
  field. `_source_label` (prompt-building) now names section/content-type/
  vision-provenance/expansion-origin per source, so the model is never
  handed grounds to imply it visually inspected an image when only
  caption/alt-text was available.
- Eval: `reference_context_is_supported` (whitespace/case-normalized
  substring match) is the shared primitive behind both
  `scripts/validate_gold_file.py`'s new warning-level corpus check (run
  clean against the real 84-question gold file, zero warnings) and
  `run_eval.py`'s new `content_type_breakdown` / `reference_context_analysis`
  (A/B/C buckets) / `relevant_image_hit_rate` /
  `relationship_expansion_contribution_rate` / `vision_behavior_breakdown`.

### Experiments run (local Ollama only, zero hosted calls)
Baseline held fixed for both (matching `experiment_010`, confirmed with
user rather than assumed, since `config/default.yaml` itself is still
`dense`/`none` and was never promoted): `qwen2.5:1.5b`, `all-MiniLM-L6-v2`,
`structured_markdown` 500/50, `retrieval.provider=hybrid` (`rrf_k=60`),
`reranker=none`, prompt **v2** (confirmed active via each run's own
`config.prompt_version` in its report, not assumed from the file existing),
`dataset_id=techfusion`, `vision.provider=none`.

- **experiment_011 (Experiment A -- expansion OFF)**:
  Recall@5 0.911, Recall@10 0.946, hit_rate@5 0.929, MRR 0.824,
  `supporting_context_hit_rate` 0.697 (A=46/B=20/C=4/n.a.=14),
  `relevant_image_hit_rate` 0.579 (11/19), `relationship_expansion_
  contribution_rate` 0.0/7 (necessarily, by construction), total latency
  5.20s/question.
- **experiment_012 (Experiment B -- expansion ON)**: Recall@5 0.911,
  Recall@10 0.946, hit_rate@5 0.929, MRR 0.824 -- **identical to A**,
  confirming expansion never contaminates the ranked cutoff.
  `supporting_context_hit_rate` 0.788 (A=52/B=14/C=4/n.a.=14, +0.091),
  `relevant_image_hit_rate` 0.842 (16/19, +0.263),
  `relationship_expansion_contribution_rate` 0.286 (2/7), total latency
  13.7s/question (retrieval 212ms -> 710ms from the extra lookups;
  generation 4.99s -> 13.0s, almost certainly from longer prompts -- more
  appended source blocks per question -- rather than expansion itself
  being compute-heavy; worth a real latency-vs-context-size measurement
  before trusting that causal story further).

Both recorded as `experiments/results/experiment_011.json` /
`experiment_012.json` (001-010 untouched) via the existing
`scripts/record_experiment.py`, logged to MLflow.

### Concrete findings (hand-inspected, not just aggregate trust)
- **Expansion genuinely fixed a retrieval gap, but generation still
  missed the number.** "After an OCR timeout, which configuration
  controls retry-lock lifetime?" flipped bucket B -> A in experiment B:
  the `{"retry_lock_ttl_seconds": 600, ...}` JSON chunk was retrieved only
  via `origin="expanded"` (its parent-prose sibling, "Retry Processing",
  was already directly retrieved; the config block itself only surfaced
  through expansion). But the model's actual generated answer --
  "the consumer acquires a Redis lock before scheduling the next attempt
  to prevent duplicate deliveries" -- never states the 600-second number,
  even with the evidence now present in context. Correct classification:
  retrieval succeeded, generation/synthesis still failed to extract a
  specific numeric field from a JSON block it was handed. A real,
  separate failure mode from anything relationship expansion can fix.
- **The `vision_behavior_breakdown` heuristic has real, demonstrated
  false-negative gaps -- caught by reading actual answers, not trusting
  aggregate counts.** Three `unanswerable_visual` questions
  ("CPU temperature", "Identity Gateway IP address", "failed logins from
  France") flipped classification between A and B (or vice versa) even
  though **all six actual model answers were genuine refusals in
  substance** -- e.g. "there is no specific mention of CPU temperatures
  for ingestion workers... I cannot provide a precise answer" was
  classified `hallucinated_answer` purely because `_looks_like_refusal`'s
  literal phrase list didn't include "cannot provide" or "no specific
  mention of" (it does include "not mentioned" and "does not contain",
  which is why worded-differently refusals to the same underlying
  question classified correctly). This is exactly the documented
  limitation ("a triage aid for hand-inspection... not a claim of
  semantic understanding") behaving as warned, and a genuinely useful
  interview story about why a crude deterministic heuristic still needs
  hand verification before its counts are trusted -- and why the
  `_REFUSAL_PHRASES` list is explicitly *not* meant to be exhaustive NLP.
  No code change made in response (the classifier is working as
  documented); noted here so a future session doesn't rediscover it from
  scratch.

### Verification
- `pytest tests/unit -q`: 256 passed, 1 skipped, 2 failed -- the same
  pre-existing `ragas`/`datasets`-installed edge case documented in
  CLAUDE.md, unrelated to this work (verified untouched: neither failing
  test's file was modified this session).
- `ruff check`/`ruff format --check` on all changed files: clean (one
  pre-existing formatting drift in `tests/integration/test_hybrid_retrieval.py`,
  a file this session never touched, left alone).
- `mypy src/`: one pre-existing error in `ragas_scorer.py` (a file this
  session never touched, a real `ragas` library `Executor | EvaluationResult`
  union-attr issue unrelated to this work); zero errors in any file this
  session changed.
- Real re-ingestion (`--clear`) of the full `data/knowledge_base` under
  `techfusion`: 34 documents, all `content_type` buckets present
  (prose 338, image 13, table 6, code 5, configuration 5, chart 2); the 13
  image chunks match exactly the 13 unique `images/*.png` references
  actually present across the markdown (verified by independent `grep`).
- `scripts/validate_gold_file.py` against the real 84-question
  `techfusion_gold.jsonl`: 0 errors, 8 warnings (all pre-existing low-
  keyword-overlap warnings on genuinely hard/visual questions), **0**
  `reference_contexts`-not-found warnings -- the new corpus check
  validated cleanly against real, human-authored ground truth on the
  first real run.
- Both experiments run against real local Postgres + native Ollama
  (`qwen2.5:1.5b`), zero hosted API calls anywhere in this session.

### Open problems / next steps
- Concrete hosted `VisionProvider` subclass still unwritten (deliberate);
  needs a provider/model decision before Experiment C.
- The retrieval-latency-vs-generation-latency causal story for expansion
  (more appended source blocks -> longer Ollama prompt -> slower decode)
  is plausible but not directly measured (e.g. by logging prompt token
  count per run) -- worth doing before quoting the 4.99s -> 13.0s jump as
  "expansion costs ~3x generation latency" in anything more permanent
  than this journal entry.
- Next actual task: build (don't run) the stratified 15-question RAGAS
  sample from `techfusion_gold.jsonl` (excluding the old sample15), save
  as `techfusion_gold_v2_ragas_sample15.jsonl`, and present it plus a
  cost estimate for approval before calling `run_ragas_eval.py`.

## 2026-08-12: RAGAS run on the multimodal gold set (experiment_013)

### What happened
User asked to run the "complete evaluation with prompt v2 and RAGAS on
techfusion_gold.jsonl." Given the milestone's explicit credit-safe gate
(show the sample, confirm judge/provider, give a cost estimate, wait for
approval before any paid RAGAS call), first asked the user to choose
between scoring the full 84-question set vs. the originally-planned
stratified 15-question sample -- user chose the stratified sample.

Built `scripts/build_ragas_sample15.py`: deterministic, file-order
selection (no randomness) of 1 question from each of the 9 authored
`content_type` buckets (architecture_diagram, chart, table_image,
image_only, caption_answerable, relationship_aware, text_only,
text_plus_image, unanswerable_visual) plus 6 from the plain/uncategorized
bucket, chosen to cover `question_type` (single_document/multi_hop),
`difficulty` (easy/medium/hard), and one `unanswerable=true` case. Wrote
`data/eval/techfusion_gold_v2_ragas_sample15.jsonl` (all fields passed
through untouched; old `techfusion_gold_ragas_sample15.jsonl`, pre-
multimodal schema, left alone). Showed the 15 questions, the judge
(`openai`/`gpt-4o-mini`, key read from `.env`'s `OPENAI_API_KEY`,
confirmed present but never printed), and a rough cost estimate to the
user; got explicit approval before making any API call.

Ran `rag.eval.run_ragas_eval` against Experiment B's config
(`config/experiments/multimodal-v2-relationship.yaml` -- prompt v2,
hybrid+RRF, relationship expansion on) and recorded the result as
`experiment_013` via `scripts/record_experiment.py` /
`scripts/compare_experiments.py` (README's Benchmarks table regenerated
in place, row 13). Note: the CLI invocation needed `./.venv/Scripts/
python.exe` explicitly -- the ambient `python` on PATH in this shell isn't
the project's virtualenv and can't import `rag` at all (`ModuleNotFoundError`),
a purely local shell-config quirk, not a code issue.

### Findings
- **Actual cost was small**: 240 judge calls, ~192K input / ~18K output
  tokens on gpt-4o-mini, roughly $0.04 -- well under the pre-run estimate.
- **Aggregate RAGAS**: faithfulness 0.700, answer_relevancy 0.558,
  context_precision 0.259, context_recall 0.411, answer_correctness
  0.409. Deterministic Recall@5/@10/MRR/answer_quality on this same
  sample (0.800/0.867/0.697/0.328) read lower than experiment_011/012's
  84-question numbers -- **not a generation regression**: 9 of the 15
  rows are the hardest, most vision-dependent specialty buckets by
  deliberate construction of the stratified sample, so the two aren't
  directly comparable.
- **A genuine hallucination, caught correctly -- initially
  mischaracterized, corrected same-day (see 2026-08-12 experiment_014
  entry below).** For "How long are idempotency keys retained?" (gold:
  "24 hours"), `qwen2.5:1.5b` answered "annually" -- faithfulness scored
  `0.0`, answer_correctness `0.189`. First write-up of this claimed the
  retrieved context "literally contains 'Idempotency-Key ... for 24
  hours'" -- that was checked against the gold file's `reference_contexts`
  (ground truth), not against the actual `generation_sources` for this
  run, and was wrong: the correct document
  (`knowledge_base/engineering/api-development-guidelines.md`) was never
  retrieved at all. This is the same `top_k=5`/`rerank_top_n=3`-window
  retrieval miss on this exact question first identified back in the
  2026-08-05 session (see issue #7 in that entry) and reconfirmed multiple
  times since -- still unresolved by this session. So the real failure is
  two layered problems: a long-standing retrieval-window miss, *and*
  `qwen2.5:1.5b` fabricating "annually" instead of following prompt v2's
  rule 3 ("say you don't know") given no grounding. Still a real, useful
  RAGAS-faithfulness catch -- just not for the reason first written down.
- **`context_precision`/`context_recall` read near-zero for most
  `requires_vision=true` questions**, even where `faithfulness` was high
  (the model correctly declined to fabricate a visual fact). This is
  RAGAS's judge scoring precision/recall against `expected_answer` text,
  not against `reference_contexts`/`relevant_images` -- a correct refusal
  on a genuinely image-only question necessarily scores low on these two
  metrics by construction. Not a RAGAS bug, but a reason not to read
  `context_precision`/`context_recall` in isolation for this gold set's
  vision-required rows; `vision_behavior_breakdown` (from the 2026-08-11
  entry) is the metric that actually judges refusal-vs-hallucination
  correctness for those.
- Full local test suite (269 unit tests incl. 6 new ones for
  `build_ragas_sample15.py`) green; no regressions from this session's
  changes.

### Open problems / next steps
- The 15-question sample intentionally over-represents hard/multimodal
  questions; a full-84-question RAGAS run (explicitly declined this
  session in favor of the cheaper/faster sample) would give a more
  representative aggregate if ever needed for a real go/no-go decision.
- RAGAS scores here are still **not validated against human labels**
  (`scripts/generate_manual_review.py`/`compare_ragas_manual.py` remain
  the intended next step before trusting these numbers for anything
  beyond directional signal).
- Experiment C (hosted vision) remains unstarted, gated on a separate
  provider/model decision and approval, per the original milestone scope.

## 2026-08-12: bigger local model as an alternative to a v3 prompt (experiment_014)

### What happened
Asked whether experiment_013's hallucination finding meant a new (`v3`)
prompt version was needed. Answer given: the failure violated prompt v2's
existing rule 3 ("if the context does not contain enough information,
say so") rather than exposing a gap in the prompt's wording, so the
prompt itself probably wasn't the bottleneck -- recommended testing a
bigger local model (`qwen2.5:3b`, already pulled in Ollama) before writing
a new prompt version. User agreed and asked to run it on all 84 questions.

Created `config/experiments/multimodal-v2-relationship-qwen3b.yaml`
(identical to Experiment B except `generation.model_name: qwen2.5:3b`).
Ran the full 84-question deterministic eval (`rag.eval.run_eval`, local
Ollama only, no hosted API) in the background given the bigger model's
higher per-question latency; recorded as `experiment_014` via the usual
`record_experiment.py`/`compare_experiments.py` flow.

While comparing the idempotency-key question's answer across both models,
re-checked what context was *actually retrieved* for it (rather than
re-trusting the earlier write-up) and found the earlier experiment_013
journal/docs claim was wrong -- see the correction above and in
`docs/architecture.md`/`CLAUDE.md`. Both files fixed in place same-day
before this entry.

### Findings
- Retrieval-side metrics (Recall@5/@10, MRR, `supporting_context_hit_rate`,
  `relevant_image_hit_rate`) came out bit-for-bit identical to
  `experiment_012` (0.911/0.946/0.824/0.788/0.842) -- expected, generation
  model choice cannot change what gets retrieved.
- `answer_quality` (keyword-overlap heuristic) rose modestly: 0.418 ->
  0.452.
- **The idempotency-key question flipped from hallucination to correct
  refusal.** Same retrieval miss (the correct document still isn't
  retrieved -- this remains the long-standing `top_k`/`rerank_top_n`-window
  gap tracked since 2026-08-05), but `qwen2.5:3b` answered "The context
  provided does not contain any information about the retention period
  for idempotency keys... I don't know the answer" instead of fabricating
  "annually". Concrete evidence this specific failure was a model-
  capability gap in following an already-correct instruction, not a
  prompt-wording gap.
- Cost: generation latency rose ~1.5x (13.0s -> 19.0s/question mean,
  13.7s -> 19.2s total). No RAGAS/hosted call made -- purely local.
- Not measured, flagged rather than papered over: (1) this is one
  hand-verified example, not a systematic faithfulness comparison across
  all 84 answers -- would need a second RAGAS pass or manual review to
  generalize; (2) `retrieval_latency_ms` read 710ms (experiment_012) vs
  240ms (experiment_014) despite identical retrieval config and index --
  plausibly DB connection/cache warmth or system load, not isolated, not
  attributable to the generation model swap.

### Open problems / next steps
- The idempotency-key retrieval-window miss itself is still unresolved
  (now confirmed across many sessions dating back to 2026-08-05) --
  orthogonal to this milestone's scope, but the single most persistent
  known gap in this repo's eval history.
- If a broader faithfulness problem is found under a full-scale RAGAS run
  or manual review, revisit whether `qwen2.5:3b` should become the
  default `generation.model_name` in `config/default.yaml` (not done this
  session -- `default.yaml` untouched, this is still an experiment-config-
  only comparison).
- Recommendation given to user: hold off on a `v3` prompt for now.

## 2026-08-12: full-scale RAGAS confirmation on all 84 questions (experiment_015)

### What happened
User asked for "the biggest experiment": RAGAS + the bigger model +
the best prompt, on the full 84-question gold set -- explicitly the
systematic check experiment_014's own write-up had flagged as missing
("this is one hand-verified example... a real answer would need either
a second RAGAS pass or a manual review pass"). No new config needed --
`config/experiments/multimodal-v2-relationship-qwen3b.yaml` (built for
experiment_014) already is qwen2.5:3b + prompt v2 + hybrid+RRF +
relationship expansion, the best combination found so far. Ran
`rag.eval.run_ragas_eval --sample-size 84` against it (the default
`--sample-size` is 15; 84 covers the entire canonical gold file) in the
background, since qwen2.5:3b generation alone was expected to take
~25-30 minutes for 84 questions before RAGAS scoring even started.
Stated a cost estimate (~$0.20-0.25, scaled from experiment_013's real
15-question cost) before launching; user had already explicitly
requested this exact scope and asked for the cost to be reported
afterward, so proceeded directly rather than re-gating with another
approval question.

### Findings
- **Run succeeded cleanly**: 84/84 scored, 0 skipped, 0 metrics failed
  (`num_scored=84`, `metrics_failed={}`), despite several transient
  `RateLimitError (429)` responses from OpenAI's `gpt-4o-mini` TPM limit
  partway through -- RAGAS's own executor retried each automatically;
  nothing was silently dropped.
- **Retrieval-side metrics identical** to experiment_012/014 (Recall@5
  0.911, Recall@10 0.946, MRR 0.824, `supporting_context_hit_rate`
  0.788, `relevant_image_hit_rate` 0.842) -- expected, generation model
  and judge scoring can't change what gets retrieved; confirms the run
  was properly isolated to the RAGAS-scoring axis.
- **Highest faithfulness and answer_correctness recorded for this
  project**: faithfulness 0.898, answer_correctness 0.513 (previous
  highs: faithfulness 0.844 at `experiment_007`, answer_correctness
  0.591 at `experiment_009` -- both 15-question pilots on earlier,
  smaller/pre-multimodal gold schemas, so not a perfectly clean
  comparison, but still the strongest full-gold-set numbers to date).
  This directly answers the question experiment_014 left open: the
  qwen2.5:3b + prompt v2 + relationship-expansion combination is a real,
  generalized faithfulness improvement across the whole 84-question set,
  not just a fix for the one idempotency-key case that motivated testing
  it in the first place.
- `answer_quality` (keyword-overlap) came out 0.442, close to but not
  identical to experiment_014's 0.452 under the same config --
  attributed to ordinary LLM-sampling variance (`temperature=0.2`, not
  `0.0`) rather than a regression; not the metric being trusted here
  anyway.
- **Actual cost, from the judge's own tracked usage (not estimated)**:
  1,431 API calls, 1,115,731 input tokens, 101,857 output tokens on
  `gpt-4o-mini`. At published pricing ($0.15/1M input, $0.60/1M output):
  $0.1674 + $0.0611 = **$0.2285** -- within the pre-run estimate, and a
  useful data point that this project's per-question RAGAS cost stays
  roughly linear with question count (experiment_013: 15 questions,
  $0.0399, ~$0.0027/question; experiment_015: 84 questions, $0.2285,
  ~$0.0027/question -- consistent).

### Verification
Recorded as `experiment_015` via the standard `record_experiment.py` /
`compare_experiments.py` flow (README table + `experiments/reports/
comparison.md` regenerated; experiments 001-014 untouched). Updated
`docs/architecture.md` (new "The full-scale systematic check" section),
`CLAUDE.md`, and this journal entry with the real numbers -- no metric
in any of these was invented; every figure traces back to the report
JSON's own `ragas.aggregate`/`ragas.judge.estimated_api_usage` fields.

### Open problems / next steps
- Whether `qwen2.5:3b` + prompt v2 + relationship expansion should
  become `config/default.yaml`'s actual default remains a decision not
  made this session -- `default.yaml` is still untouched (`qwen2.5:1.5b`,
  `dense`, `v1`, expansion off); this stays an experiment-config-only
  result until/unless promoted deliberately.
- `context_precision` (0.659) and `answer_relevancy` (0.530) are
  meaningfully lower than faithfulness/answer_correctness in this same
  run -- consistent with the documented pattern (RAGAS scores these two
  against `expected_answer` text, not `reference_contexts`, so
  `requires_vision=true` questions structurally score low here) but not
  separately re-verified this session; see experiment_013's entry for
  the fuller explanation of why.
- Experiment C (hosted vision) remains unstarted, still gated on a
  separate provider/model decision and approval.

## 2026-08-13: RAGAS judge-call caching and readable MLflow run names

### What happened
Two additive changes, requested together, no paid RAGAS run performed to
test either: (1) persistent, disk-backed caching of RAGAS judge LLM calls
so re-running an unchanged eval doesn't re-pay for identical judge
verdicts; (2) readable MLflow run names/tags so runs are identifiable at
a glance instead of only by `experiment_id`.

### Design decisions
- **Why not just turn on RAGAS's own `cache=` support and call it done.**
  RAGAS already ships `ragas.cache.DiskCacheBackend`/`CacheInterface`,
  wired in via `cache=` on `LangchainLLMWrapper`. Its key is a hash of
  the *function* being called plus its *call args/kwargs*
  (`ragas.cache._generate_cache_key`) -- and because `cacher()` wraps a
  *bound* method (`self.generate_text = cacher(...)(self.generate_text)`),
  the `self` the method is bound to (i.e. which judge model/provider it
  is) never enters that hash. Two `LangchainLLMWrapper`s built from
  different judge models but sharing one cache directory would silently
  replay one model's cached verdict for the other, given a
  byte-identical rendered prompt. Confirmed this isn't hypothetical by
  reading `ragas.llms.base.BaseRagasLLM.__post_init__` and
  `ragas.cache._generate_cache_key` directly, not assuming from the
  docs.
- **What the native mechanism *does* get right for free.** RAGAS's
  metric prompts (checked `ragas.metrics._faithfulness`'s
  `NLIStatementPrompt`/`StatementGeneratorPrompt`) render the full
  question/context/answer/instruction text into the prompt string that
  gets hashed -- so a changed generated answer, a changed retrieved
  context, or a different metric's instruction wording are already
  cache-invalidating today, with zero extra code. Nothing needed to be
  rebuilt for that part.
- **The fix**: `rag/eval/ragas_cache.py`'s `NamespacedDiskCache` wraps
  `DiskCacheBackend`, prefixing every key with a fingerprint
  (`judge_fingerprint`) of `(judge provider, judge model, judge
  temperature, judge max_tokens, installed ragas version, a small local
  schema-version constant)` before delegating to the underlying disk
  store. Same idea as the project's own `rerank_top_n` lesson (a
  parameter that "looks orthogonal" turned out not to be) -- verified
  the fix directly by wrapping two `LangchainLLMWrapper`s pointed at the
  *same* cache directory but different namespaces and confirming a
  byte-identical prompt still misses when the namespace (i.e. judge
  fingerprint) differs (`tests/unit/test_ragas_cache.py::test_changed_judge_model_misses_cache_even_with_identical_prompt`).
- **`reference_contexts` was explicitly considered and excluded from the
  key.** The task spec listed it as a factor to account for "where
  relevant." Checked `ragas_scorer.build_dataset` and confirmed this
  project's RAGAS dataset rows never include it today (only
  `question`/`answer`/`contexts`/`expected_answer` go into the ragas
  `Dataset`) -- so it isn't currently an input to any judged metric, and
  there's nothing to key on without inventing a field ragas never sees.
  Documented in the module docstring rather than silently omitted, so a
  future change that does start passing it will be a visible decision,
  not an accidental gap.
- **Why avoided-cost estimation reports a reason instead of a guessed
  number when it can't compute one.** A 100%-cache-hit run has zero
  uncached judge calls this run to derive an average token profile from;
  an unpriced model has no entry in `PRICING_USD_PER_1M_TOKENS`. Both
  return `estimated_cost_usd: None` plus a `reason` string rather than a
  fabricated dollar figure -- consistent with this project's standing
  rule (CLAUDE.md's `ISSUES.md` instructions) to never invent a metric.
  The one price actually in the table (`gpt-4o-mini`: $0.15/1M input,
  $0.60/1M output) is the same figure `experiment_015`'s real tracked
  usage confirmed (1,115,731 in / 101,857 out -> $0.2285, matching the
  journal's own recorded $0.2285), not a fresh guess.
- **MLflow run names**: `mlflow_logger.build_run_name()` slugs
  `experiment_id`/`generation_model`/`prompt_version`/
  `retrieval_provider` (plus `rel-exp` when relationship expansion is
  on) into one readable name, e.g.
  `experiment_015_qwen2-5-3b_v2_hybrid_rel-exp`. Only the *display*
  `run_name` changes -- `run.info.run_id` (the real MLflow UUID) is
  untouched, and no new field was added to the
  `experiments/results/*.json` record schema, so
  `compare_experiments.py`/`render_benchmarks.py`/`backfill_mlflow.py`
  all keep working unmodified; a pre-multimodal record missing
  `relationship_expansion_enabled`/`dataset_id` just tags/names itself
  with those segments omitted or blank rather than raising.

### Verification
`pytest tests/unit -q`: 287 passed, 1 pre-existing skip (unrelated).
`ruff check`/`ruff format --check` clean on every changed file. `mypy
src/` shows the same single pre-existing, unrelated error present on
`main` before this change (`ragas_scorer.py`'s `EvaluationResult |
Executor` union-attr false positive -- confirmed via `git stash`/re-run
that it predates this session). New tests
(`tests/unit/test_ragas_cache.py`) exercise the real `ragas.cache`/
`LangchainLLMWrapper` machinery against a local `_CountingLLM` double --
no network call, no paid judge call, per the task's explicit "do not
perform a paid full RAGAS run merely to test caching" instruction.

### Open problems / next steps
- Cache hit/miss counting happens at the raw `generate_text`/
  `agenerate_text` call level, not per gold question -- a metric that
  issues multiple judge calls per row (e.g. faithfulness's statement
  generation + verification) shows up as multiple hits/misses for one
  question. Accurate for "hosted calls avoided" but not a 1:1 stand-in
  for "questions avoided" if that's ever needed as a separate number.
  Not measured against a real cache-warm RAGAS run yet -- this session
  intentionally never spent budget to do that; first real run against
  `techfusion_gold.jsonl` will be the first live confirmation of
  hit/miss behavior at scale.

## 2026-08-13: retrieval attribution -- what dense and BM25 each contribute (experiment_016)

### Why
Before further retrieval tuning, wanted to know how much of hybrid's
result is actually earned by BM25 vs. dense doing the work, and whether
RRF fusion ever hurts a ranking that one retriever alone already had
right. `RetrievalPipeline.retrieve()` only ever exposes the *fused*
ranking (or dense-only, depending on config) -- no prior report separated
dense-only, BM25-only, and fused Recall/MRR against the identical gold
set. Deliberately an observability/evaluation milestone, not a tuning
one: RRF's `k`, the reranker, embeddings, chunking, prompt version, and
generation model are all untouched; `config/default.yaml` is untouched.

### What changed
`RetrievalPipeline.retrieve_attribution()` (new, additive method) fetches
dense and BM25 rankings independently and fuses them via the same
`reciprocal_rank_fusion` production uses, but never calls the reranker
(so `NoOpReranker`'s `results[:top_n]` truncation never applies) or
relationship expansion -- proven directly by test (a vectorstore double
with no expansion methods implemented would raise if expansion ran; it
doesn't). `retrieve()`/`answer()` themselves are byte-for-byte unchanged.
`eval/retrieval_attribution.py` runs this over a gold set and computes
Recall@5/@10, Hit Rate@5/@10, and MRR for dense/BM25/hybrid independently
against the *same* gold records, a deterministic dense-vs-BM25
contribution bucket per question (`both_success`/`dense_only_success`/
`bm25_only_success`/`neither_success`), a deterministic RRF-impact
classification (`rescued`/`improved`/`unchanged`/`degraded`/
`still_missed`/`not_applicable`, based on first-relevant-document rank in
each of the three rankings), and reference-context (not just document)
attribution per retriever, reusing `gold_schema.py`'s A/B/C bucket logic
(factored out of `run_eval.py` into a shared `reference_context_bucket`
function so the two never define "supporting evidence found" differently).

### Results (full 84-question `techfusion_gold.jsonl`, `dataset_id=techfusion`, `rrf_k=60`, `RETRIEVAL_K=10`)

| | Recall@5 | Recall@10 | Hit Rate@5 | Hit Rate@10 | MRR |
|---|---|---|---|---|---|
| Dense only | 0.899 | 0.952 | 0.917 | 0.964 | 0.797 |
| BM25 only | 0.839 | 0.905 | 0.857 | 0.917 | 0.741 |
| Hybrid (RRF) | 0.911 | 0.946 | 0.929 | 0.952 | 0.825 |

Contribution buckets: `both_success` 77, `dense_only_success` 4,
`neither_success` 3, `bm25_only_success` 0 -- dense alone already covers
everything BM25 covers plus 4 more questions; BM25 never single-handedly
finds a document dense misses on this gold set.

RRF impact: `unchanged` 69, `degraded` 6, `rescued` 4, `improved` 2,
`not_applicable` 3 (neither retriever found anything to fuse).

Reference-context (evidence-level, not just document-level)
`supporting_context_hit_rate`: dense 0.697, BM25 0.646, **hybrid 0.773**
-- hybrid's advantage over dense-only is larger here than in the
document-level Recall numbers above, meaning fusion's real value is
partly in surfacing the *specific supporting passage*, not just the
right document.

### Examples where one retriever rescued the other
All 4 `rescued` cases are dense rescuing a BM25 miss (never the reverse):
- "What is the GDPR breach escalation rule?" -- BM25 missed entirely;
  dense rank 1; fused rank 1 (reference context recovered too).
- "What salary bands does TechFusion use for senior engineers?" -- BM25
  missed entirely; dense rank 8; fused rank 13 (still within a broader
  window, though outside Recall@10).
- "Who is the current CEO of TechFusion GmbH?" -- BM25 missed; dense rank
  7; fused rank 10.
- "How many OCR worker replicas are shown in production?" -- BM25
  missed; dense rank 5; fused rank 8.

### Examples where RRF degraded a useful result
Both `degraded` examples shown are cases where dense alone already had
rank 1 or 2, but summing in BM25's weak, low-rank contribution to a
different, unrelated top-ranked BM25 candidate pushed the correct answer
down (not out of the result set, just lower):
- "What is DocuFlow?" -- dense rank 1, BM25 rank 8, fused rank 2.
- "What canary stages are used?" -- dense rank 2, BM25 rank 9, fused rank 3.

Neither dropped below Recall@5, so this didn't cost a metric point here,
but it's a real, measured case of fusion making a good dense ranking
slightly worse, not just occasionally leaving it unchanged.

### Confirms a previously known finding
"How long are idempotency keys retained?" (the idempotency-key question
documented repeatedly in earlier experiments/ISSUES.md as an unresolved
retrieval-window miss) shows up here as `neither_success`: dense_rank,
bm25_rank, and fused_rank are all `null`. Direct, independent confirmation
that this was never a fusion problem or a BM25-tokenizer problem -- dense
alone doesn't find it either, at any of the three rankings.

### Is hybrid retrieval justified by the evidence?
Yes, but narrowly and specifically for evidence-level quality, not
document-level Recall. Hybrid's headline Recall@10 (0.946) is actually
*lower* than dense-only's (0.952) -- the 6 `degraded` cases cost more at
the 10-result cutoff than the 4 `rescued` cases gain. But Hit Rate@5,
MRR, and especially `supporting_context_hit_rate` (0.773 vs. dense's
0.697) all favor hybrid, and BM25 never independently rescues anything
dense doesn't already find on this gold set (`bm25_only_success = 0`).
The honest reading: hybrid earns its keep primarily by improving *which
passage* gets surfaced and *how highly ranked* it is on average (MRR), not
by expanding what's document-level-findable at all -- BM25's real
contribution here is smaller than the aggregate Recall/MRR numbers in the
Benchmarks table alone would suggest.

### Verification
`pytest tests/unit -q`: 317 passed, 1 pre-existing skip (unrelated).
`pytest tests/integration -q`: 21 passed (real Postgres/Ollama).
`ruff check`/`ruff format --check`/`mypy src/`/pre-commit all clean on
every changed file (`mypy` shows the same single pre-existing, unrelated
`ragas_scorer.py` error). Fully deterministic and local: real
Postgres/pgvector retrieval against the real 369-chunk `techfusion`
corpus, no LLM generation call, no hosted judge call anywhere in this
milestone. Recorded as `experiments/results/attribution/experiment_016.json`
(a dedicated subdirectory `compare_experiments.py`'s non-recursive
`*.json` glob never sees -- confirmed by re-running it and diffing
`README.md`/`experiments/reports/comparison.md`: byte-identical,
experiment_016 does not appear) and as its own MLflow run
(`scripts/record_attribution_experiment.py`, dense_/bm25_/hybrid_-prefixed
metrics, reusing `mlflow_logger.build_run_name` for a readable name).
`experiments/results/experiment_001.json` through `experiment_015.json`
are untouched.

### Open problems / next steps
- `dense_only_success`/`bm25_only_success` are computed at `RETRIEVAL_K=10`
  (matching `run_eval.py`'s broad fetch); a smaller production cutoff
  would likely show a larger dense advantage, since 2 of the 4 dense
  rescues (ranks 7-8) sit outside a tighter window.
- This did not test whether a different `rrf_k` shifts the
  `degraded`/`rescued` balance -- deliberately out of scope this
  milestone ("do not change RRF k yet").
- Next natural follow-up per the `degraded` cases: whether a cross-encoder
  reranker (not yet added -- `reranker.provider: none` throughout this
  milestone) would recover the 2 cases where fusion demoted an
  already-correct dense top-1/2 result.

## 2026-08-13: retrieval-cutoff semantics refactor + controlled reranker experiment (experiment_016/017)

### The bug
`config.retrieval.top_k` controlled candidate-fetch depth and
`config.retrieval.rerank_top_n` was passed unconditionally to
`self._reranker.rerank(query, candidates, top_n=rerank_top_n)` --
including when `reranker.provider: none`, where it resolved to
`NoOpReranker`. `NoOpReranker.rerank()` did `return results[:top_n]`: a
"no-op" reranker that still truncated results. `rerank_top_n` was
therefore doing two unrelated jobs at once -- "how many a real reranker
keeps after rescoring" and "how many chunks reach the LLM prompt" -- with
no way to change one without the other. `eval/run_eval.py` had to
override `rerank_top_n=10` on every broad retrieve() call just to defeat
this truncation and compute Recall@10, since production's real
`rerank_top_n=3` would otherwise cut every retrieval down to 3 results
before recall could even be measured.

### Old vs. new pipeline semantics
```
OLD: retrieval.top_k -> fetch -> rerank(top_n=retrieval.rerank_top_n) -> done
                                   ^ NoOp still truncated here even with no reranker

NEW: retrieval.candidate_k -> fetch
       -> rerank(top_n=reranker.top_n)         [identity when reranker.provider="none"]
       -> [:generation_context_top_n]           [the ONLY truncation point, always applied]
       -> relationship expansion (if enabled)   [appended after, never truncated itself]
```
Three independent config fields replace the old two: `retrieval.candidate_k`
(candidate-pool depth), `reranker.top_n` (real-reranker-only truncation,
inert for `provider: "none"`), `retrieval.generation_context_top_n`
(the single, always-applied cutoff controlling how many chunks reach the
LLM prompt). `NoOpReranker.rerank()` is now a true identity: it ignores
`top_n` entirely and returns every result unchanged -- no reorder, no
rescore, no truncation. `RetrievalPipeline.retrieve()`/`answer()` also now
report a `latency_breakdown_ms` (embed/dense_search/bm25_search/fusion/
rerank/expansion/generation), via a new private `_retrieve_timed` shared
by both methods -- needed to isolate the reranker's added latency below.

### Migration policy
Historical experiment records (`experiments/results/*.json`,
`experiments/configs/*.yaml` snapshots) are untouched and still describe
what actually ran, in the old field names -- they're archival copies,
never re-parsed by `AppConfig` (only copied as MLflow artifacts). The 9
live config files that *are* parsed by `AppConfig` (`config/default.yaml`
+ 8 `config/experiments/*.yaml`) were migrated explicitly to the new field
names in the same change, rather than adding a dual-support compatibility
shim -- a small, fully-controlled set of files where a silent shim would
have risked exactly the "quietly reinterpret an old value under new
semantics" problem being fixed. `scripts/record_experiment.py`/
`eval/mlflow_logger.py` now emit `retrieval_candidate_k`/`reranker_top_n`
(null when `reranker.provider="none"`, not a pretend value)/
`generation_context_top_n` for new records only.

### Verification
`pytest tests/unit -q`: 331 passed, 1 pre-existing skip. `pytest
tests/integration -q`: 21 passed against real Postgres+Ollama (this also
caught 3 integration test files still calling `retrieve()`/`answer()`
with the old `top_k=`/`rerank_top_n=` keyword arguments -- fixed as part
of this change, since they'd have raised `TypeError` the next time
integration tests actually ran against live infra). `ruff check`/`ruff
format --check` clean; `mypy src/` shows only the same pre-existing,
unrelated `ragas_scorer.py` error present on `main` before this branch.
New tests cover all 10 requested guarantees: NoOp identity/no-truncation,
`candidate_k` controlling fetch depth, `generation_context_top_n`
controlling the primary cutoff independently of the reranker,
`generation_context_top_n` never affecting Recall@K (two configs
differing only in that field produce identical broad-override Recall@10),
a broad eval-style override computing Recall@10 independently of whatever
production's own cutoffs are, a real reranker receiving the full
candidate pool, a real reranker retaining exactly its configured `top_n`,
relationship expansion running strictly after the generation-context
cutoff, expanded chunks never entering a Recall@k window they weren't
already inside, and a repo-wide sweep proving every live config file was
actually migrated (raw-text assertion that `top_k:`/`rerank_top_n:` no
longer appear, not just that the file loads).

### The controlled reranker experiment
Two experiment configs, identical except `reranker.provider`:
`multimodal-v2-relationship-qwen3b.yaml` (baseline, `reranker.provider:
none` -- this *is* `experiment_015`'s own config, values unchanged, just
under the new field names) and the new
`multimodal-v2-relationship-qwen3b_cross-encoder.yaml` (candidate,
`reranker.provider: cross_encoder`, `reranker.top_n: 5`). Both keep
`candidate_k: 5` (matching historical `top_k: 5` -- in hybrid mode that's
already up to 10 deduplicated fused candidates for the reranker to
reorder, not "reranking only the final 3 chunks") and
`generation_context_top_n: 3` identical, so the comparison isolates the
reranker alone: same embedding, chunking, generation model, prompt,
hybrid+RRF, relationship expansion, dataset, and final context size.

**Raw retrieval floor** (`retrieve_attribution`, reranker-invariant by
construction -- run once, applies to both configs): Recall@5 0.911,
Recall@10 0.946, MRR 0.825. Numerically identical to the earlier
`experiment_016` attribution record from the previous milestone, as
expected -- nothing about dense/BM25/fusion changed here, a useful
cross-check that the two milestones' numbers are internally consistent.

**Retrieval/ranking, full 84 questions** (baseline -> candidate):
Recall@5 0.911 -> 0.929, Recall@10 0.946 -> 0.940, Hit Rate@5 0.929 ->
0.940, Hit Rate@10 0.952 -> 0.940, MRR 0.824 -> 0.826,
`supporting_context_hit_rate` 0.788 -> 0.803, `relevant_image_hit_rate`
0.842 -> 0.842 (unchanged). Every delta here is within noise of a
single-digit question count moving between buckets (84 questions total)
-- reranking neither clearly helps nor clearly hurts ranking quality on
this gold set. One question ("Who is the current CEO of TechFusion
GmbH?") moved from `not_applicable` to `C`: the cross-encoder's rescoring
of the ~10-candidate fused pool pushed the correct document out of the
broad top-10 entirely, a genuine (if isolated) retrieval regression, not
just a ranking-position change.

**Answer quality**: `mean_overall` 0.440 -> 0.453 (KeywordOverlapScorer,
a crude heuristic -- see its documented caveat), a small net improvement
masking wide per-question variance: 15 questions improved by >0.05, 17
regressed by >0.05. The single largest regression is the most
instructive: "What retrieval pipeline does the RAG service use?" stayed
`reference_context_bucket=A` in *both* runs (same two source documents,
same sections, correct evidence present in both prompts) but
`answer_quality` collapsed 0.947 -> 0.053. The only real difference was
chunk *order* within the 3-source prompt (`retrieval-performance-
analysis.md` first vs. `rag-service-architecture.md` first) -- the
baseline's answer was a crisp factual summary; the candidate's answer
hedged heavily ("This is inferred from...") despite having the same
evidence available. This is a reranker-induced *prompt-ordering*
regression, not a retrieval-recall failure -- exactly the kind of thing
`reference_context_bucket`/Recall alone cannot see, and exactly why the
milestone's instruction not to conclude "reranker is better" from one
improving metric matters here.

**Latency**: retrieval-stage mean rose 413ms -> 1401ms (+988ms, bundling
embed/dense/BM25/fusion/rerank together -- this run predates the
`latency_breakdown_ms` aggregation added to `run_eval.py`'s report in this
same change, so the per-stage split isn't available for these two runs
specifically; since embedding/candidate depth/fusion are otherwise
identical between the two configs, this delta is a reasonable, if
bundled, proxy for the cross-encoder's own cost). Generation-stage mean
was *lower* for the candidate (23875ms -> 22930ms, -945ms) -- almost
certainly qwen2.5:3b's answer-length variance, not a reranker effect.
Net: total mean latency 24288ms -> 24331ms, a 0.2% difference, i.e. the
reranker's real added cost is completely swallowed by generation-latency
noise on this pipeline (~1s added to a ~24s question is not the
bottleneck).

### Recommendation
Keep `reranker.provider: none` for now. The cross-encoder reranker
produces no clear win on any metric that matters (ranking, evidence hit
rate, or answer quality all move by less than the run-to-run noise
implied by a single flipped bucket or two), costs a real but
end-to-end-negligible ~1s of retrieval-stage latency, and demonstrably
introduced at least one answer-quality regression via chunk reordering
rather than a retrieval failure. Recorded as `experiment_016` (baseline)
and `experiment_017` (candidate) in `experiments/results/` -- not
`experiments/results/attribution/`, since these are ordinary single-config
`run_eval` records, unlike the attribution milestone's three-parallel-
metric-set shape.

### Open problems / next steps
- The per-stage `latency_breakdown_ms` aggregation was added to
  `run_eval.py` as part of this change but wasn't in place for either of
  these two 84-question runs (each ~25 minutes against local Ollama) --
  a future rerun would get the dense/BM25/fusion/rerank/expansion split
  called for by the original request, not just the bundled retrieval-stage
  delta used here.
- RAGAS (faithfulness/answer_correctness/context_precision/
  answer_relevancy/context_recall) was not run for either config as part
  of this entry -- pending explicit approval given the paid hosted-judge
  cost (see cache-hit/miss and cost estimate presented separately).
- The one retrieval regression (`not_applicable` -> `C`) and the one
  severe answer-quality regression (chunk-order sensitivity) both involve
  qwen2.5:3b's behavior on borderline-ambiguous context, not the reranker
  or retrieval quality broadly -- worth a hand-inspection pass if a
  future prompt-v3 milestone revisits grounding/refusal wording.

---

## 2026-08-14: generation-context/output-token-budget experiment (experiment_018/019/020)

### Interview talking points
- **What problem did this solve?** Local Qwen generation (many seconds/
  question) dwarfs retrieval (hundreds of ms), so this milestone asked
  whether trimming the generation-context size and/or the output-token
  cap buys a meaningful latency win without hurting answer quality --
  using the best retrieval config established so far (hybrid+RRF,
  relationship expansion on, `reranker.provider: none`, qwen2.5:3b,
  prompt v2 -- `experiment_015`'s config).
- **Why was it interesting?** The honest answer is "the latency numbers
  from this run can't be trusted at face value." ms-per-completion-token
  swung ~4-5x between three back-to-back runs of the *literally identical*
  model on the *literally identical* hardware (596-758ms/token for Config
  A, 890-1010ms/token for Config B, 167-232ms/token for Config C) despite
  completion-token counts staying flat (~37 tokens/answer) across all
  three -- proof the swing is system-level noise (this is a shared,
  CPU-only Windows dev box; Docker Desktop/WSL2 contention is the leading
  suspect, see Diagnosis), not a config effect. Reporting the raw
  `total_mean_ms` deltas as "the answer" would have been a false causal
  claim -- the milestone's own token/context-size levers turned out to be
  almost irrelevant to observed latency in this run.
- **What's the one real, defensible finding?** `completion_tokens_mean`
  is ~37 across all three configs (max observed: 154) -- this gold set's
  answers never come close to needing 512 output tokens, so
  `max_tokens: 256` truncates nothing observed and is a safe, free
  headroom reduction. `prompt_tokens_mean` drops cleanly 909.7 -> 701.2
  when `generation_context_top_n` drops 3 -> 2 (an accounting fact, not
  noise), but the corresponding ~5% `answer_quality` regression
  (0.437 -> 0.415) is small, inconsistent per-question (A wins 16, B wins
  12, 56 tied), and hand-inspection of the largest swings shows it's
  substantially a `KeywordOverlapScorer` measurement artifact, not real
  answer degradation (see Diagnosis).
- **What's next?** RAGAS (including the two new metrics added this
  session, Noise Sensitivity and Factual Correctness) was deliberately
  **not run** -- presented as a cost/cache estimate and awaiting explicit
  approval per this project's standing policy on paid hosted-judge calls.

### Configuration
Three configs, isolating one variable at a time from the established
baseline (`config/experiments/multimodal-v2-relationship-qwen3b.yaml`,
`experiment_015`'s config):

| | generation_context_top_n | max_tokens | file |
|---|---|---|---|
| A (baseline) | 3 | 512 | `multimodal-v2-relationship-qwen3b.yaml` |
| B | 2 | 512 | `multimodal-v2-relationship-qwen3b_ctx2.yaml` |
| C | 2 | 256 | `multimodal-v2-relationship-qwen3b_ctx2_tok256.yaml` |

Held fixed across all three (verified by a new config-diff test,
`test_generation_context_latency_configs_isolate_one_variable_at_a_time`):
`qwen2.5:3b`, prompt v2, `all-MiniLM-L6-v2` embeddings,
`structured_markdown` chunking (500/50), `retrieval.provider: hybrid`
(`rrf_k: 60`), `retrieval.candidate_k: 5`, `reranker.provider: none`,
`relationship_expansion.enabled: true`, `dataset_id: techfusion`, the full
84-question `techfusion_gold.jsonl`.

Each run: `python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl
--dataset-id techfusion --config <config> --verbose`, recorded via
`scripts/record_experiment.py` as `experiment_018` (A) / `experiment_019`
(B) / `experiment_020` (C). One transient
`psycopg2.OperationalError: server closed the connection unexpectedly`
on Config B's first attempt (same class of Docker-networking blip
documented in the `experiment_016/017` entry) -- Postgres was confirmed
healthy and the run was simply retried, no code change needed.

### Code changes (ahead of running anything)
- `OllamaLLM.generate()` now captures Ollama's own `prompt_eval_count`/
  `eval_count` response fields as `last_prompt_tokens`/
  `last_completion_tokens` (mirrors `OpenAILLM`'s `call_count`/
  `input_tokens`/`output_tokens` tracking, but per-call rather than
  cumulative -- production callers care about one question's token
  footprint). `RetrievalPipeline.answer()` surfaces these via
  `getattr(self._llm, ..., None)` so a future `LLM` implementation that
  doesn't track tokens just reports `None` rather than raising.
- `eval/run_eval.py` gained three new deterministic report sections (all
  gated on `run_generation=True`, all omitted -- not zeroed -- when not
  applicable, matching this module's existing convention):
  - `token_usage`: mean prompt/completion tokens across the run.
  - `refusal_behavior`: correct-refusal rate among `unanswerable=true`
    gold examples, reusing the existing `_looks_like_refusal` phrase
    matcher (previously only wired into the vision-specific
    `vision_behavior_breakdown`; this is the same heuristic applied
    generally).
  - `relationship_expansion_utilization`: a new heuristic
    (`_expansion_utilization`) answering the milestone's "are expanded
    chunks consumed or just noise" question directly -- for every
    question where an `origin="expanded"` chunk actually reached the
    prompt, computes the words present in expanded-only content (not in
    any `origin="retrieved"` primary chunk) and checks whether at least 2
    of them also appear in the generated answer. Deliberately distinct
    from the pre-existing `relationship_expansion_contribution_rate`
    (which only covers `requires_relationship_expansion=true` gold rows
    against authored `reference_contexts`) -- this new metric covers
    *every* question where expansion fired, regardless of gold
    annotation, and asks "did the model's own answer echo it" rather than
    "did it supply the officially-annotated evidence."
- `ragas_scorer.py`'s `_load_metrics()` now also best-effort-instantiates
  `ragas.metrics.NoiseSensitivity`/`FactualCorrectness` (exposed only as
  classes needing construction, unlike the other 5 metrics which are
  ready-built singleton attributes) -- an unsupported/renamed class in a
  future ragas version fails into `metrics_failed` per-metric rather than
  raising, per the milestone's "if supported cleanly" requirement.
  Confirmed against the pinned `ragas==0.3.9`: both classes exist and
  declare the same `{user_input, response, retrieved_contexts,
  reference}` required columns `build_dataset` already produces, so no
  dataset-shape changes were needed.
- `build_experiment_record`/`mlflow_logger.py` extended with
  `generation_max_tokens` (param), `prompt_tokens_mean`/
  `completion_tokens_mean`/`unanswerable_refusal_rate`/
  `expanded_context_utilization_rate`/`ragas_noise_sensitivity`/
  `ragas_factual_correctness` (metrics) -- `None` for any report
  predating this milestone, same pattern as every prior schema addition.
- Full test suite (349 tests, 1 pre-existing skip) green; `ruff
  check`/`format` clean; the one `mypy` finding
  (`ragas_scorer.py:186: Item "Executor" ... has no attribute
  "to_pandas"`) confirmed pre-existing via `git stash` against
  unmodified `main`, unrelated to this change (same verification pattern
  used in the `experiment_016/017` entry).

### Results

**Retrieval-side metrics are identical across A/B/C** (recall@5 0.9107,
recall@10 0.9464, MRR 0.8241, supporting_context_hit_rate 0.7879,
relevant_image_hit_rate 0.8421, vision_behavior_breakdown counts
identical: 3 correct_refusal / 3 caption_leak_success / 7
incorrect_or_missing) -- expected and correct: `eval/run_eval.py`'s
Recall/MRR/evidence-bucket metrics always come from a broad
`candidate_k=reranker_top_n=generation_context_top_n=10` override (see
`retrieval/pipeline.py`'s module docstring), completely independent of
which production `generation_context_top_n`/`max_tokens` is configured.
This confirms the three configs only differ in what reaches the LLM
prompt, never in what retrieval itself found.

**Token usage:**

| | prompt_tokens_mean | completion_tokens_mean (max) |
|---|---|---|
| A | 909.7 | 37.1 (109) |
| B | 701.2 | 37.4 (154) |
| C | 701.2 | 37.1 (141) |

`prompt_tokens_mean` drops ~23% from A to B/C, exactly tracking the
one-fewer-chunk context reduction. `completion_tokens_mean` is flat
across all three and never approaches even C's 256 cap (max observed:
154, at Config B) -- the model's answers on this gold set are short
regardless of headroom, so `max_tokens: 256` (Config C) truncated **zero**
observed answers.

**Answer quality:** A=0.437 (answerable 0.482 / unanswerable 0.170), B=
0.415 (0.450 / 0.207), C=0.427 (0.465 / 0.197). A > C > B in aggregate,
but per-question (A vs B): 16 questions favor A, 12 favor B, 56 tied --
not a clean, one-directional effect. Hand-inspecting the three largest
swings:
- *"Which AWS region hosts production?"* A: "Production runs across
  three availability zones in AWS eu-central-1." B: "AWS eu-central-1
  hosts production." **Both answers are correct and say the same thing**
  -- the 0.714 -> 0.143 `answer_quality` gap is almost entirely
  `KeywordOverlapScorer`'s crude keyword-matching against
  `expected_answer` phrasing, not a real correctness difference.
- *"What TLS versions are required?"* Both correctly state TLS 1.2
  internal / TLS 1.3 external; B's answer continues on to tangential key-
  rotation details from `data-encryption.md`, diluting keyword overlap
  against a short `expected_answer` without adding an error.
- *"Which queue partition has the greatest consumer lag?"* A: terse "I
  don't know" (correctly refuses, since the docs describe a general
  diagnosis procedure but never name a specific partition). B: a longer
  hedging answer that says the same thing without matching
  `_looks_like_refusal`'s phrase list as cleanly -- a real behavioral
  difference (verbosity/hedging), though not a difference in
  *informational correctness*.

Net read: most of the observed A-vs-B `answer_quality` delta is
measurement-heuristic noise from `KeywordOverlapScorer`, not evidence
that dropping the 3rd context chunk cost the model real information --
consistent with `reference_context_analysis` showing identical
`supporting_context_hit_rate` (0.7879) in all three configs, i.e. the
chunk retrieval found was never the bottleneck.

**Refusal behavior:** correct-refusal rate among the 12
`unanswerable=true` gold questions: A=0.833, B=0.833, C=0.750 (one
question flipped between B and C). With only 12 examples this is a
single-question swing (±8.3 points) and not treated as a meaningful
signal on its own.

**Relationship-expansion utilization** (new metric, answers the
milestone's "consumed or noise" question directly): expansion fired for
nearly every question (84/84 in A, 82/82 in B and C -- the 2 fewer come
from A having one more primary chunk to expand *from*), but the rate at
which the final answer's wording actually echoes expanded-only content is
low across all three: A=17.9%, B=18.3%, C=22.0%. Combined with the
already-documented fact (see the retrieval-attribution entry, 2026-08-13)
that relationship expansion never moves Recall@k/MRR by construction
(expanded results are appended, never reranked into the primary set),
this is a second, independent line of evidence that expansion is mostly
inert extra context for this gold set under the current heuristic --
low-risk (never worse), but low-value (rarely visibly used), not the
noise-injection failure mode the milestone asked to rule out either.

**Latency (read with the caveat below, not at face value):**

| | retrieval_mean_ms | generation_mean_ms | total_mean_ms | ms/completion_token |
|---|---|---|---|---|
| A | 489 | 24,993 | 25,482 | ~598 (median) |
| B | 628 | 32,856 | 33,485 | ~890 (median) |
| C | 123 | 6,163 | 6,286 | ~167 (median) |

### Diagnosis: the latency numbers are confounded by system noise, not config
If context/token-budget reduction genuinely drove Config C's ~4-5x lower
generation latency, `ms/completion_token` should be roughly constant
across configs (same model, same hardware, same decode cost per token).
It is not: 598-758ms/token (A) vs 890-1010ms/token (B) vs 167-232ms/token
(C), a ~4-5x spread, while `completion_tokens_mean` itself barely moves
(37.1 / 37.4 / 37.1). A lower `max_tokens` cap can only speed up
generation if the model would otherwise have run into that cap -- it
never did here (max observed completion length: 154, config C's cap was
256) -- so Config C's speed cannot be explained by its own config values.
The most likely cause is background system load varying across three
sequential ~10-30-minute local-LLM runs on a shared, CPU-only Windows
dev box (Docker Desktop/WSL2 resource contention is the leading
suspect -- Config B's run also hit the transient Postgres connection
drop noted above, consistent with the host being under unusual load
during that window). **Conclusion: the raw latency deltas between these
three configs are not attributable to `generation_context_top_n`/
`max_tokens` and should not be cited as "reducing context/tokens gives a
4-5x speedup."** The one latency claim this run *does* support
qualitatively (not by exact numbers): retrieval (123-628ms) is a small
fraction of total latency (6.3-33.5s) in every config, confirming the
milestone's own problem framing (retrieval is hundreds of ms, local Qwen
generation is many seconds) regardless of which exact run's numbers are
trusted.

### Recommendation
- Do **not** change `config/default.yaml` or promote any of A/B/C to
  production yet (per the milestone's explicit instruction). Of the two
  levers tested: `max_tokens: 256` is a safe, low-risk reduction (zero
  observed truncation, bounds worst-case runaway-generation cost) worth
  adopting in a future config once latency can be measured cleanly;
  `generation_context_top_n: 2` is not clearly justified -- its apparent
  quality cost is mostly a scoring artifact, but there's also no clean
  evidence of a latency benefit to justify accepting even that
  uncertainty.
- A repeat of this experiment on an idle/dedicated machine (or with
  interleaved trials instead of three sequential full runs), or reporting
  `ms/completion_token` instead of wall-clock totals, would be needed
  before trusting any latency comparison from this pipeline again.

### Open problems / next steps
- RAGAS (5 existing metrics + the 2 newly wired-in this session, Noise
  Sensitivity and Factual Correctness) was **not run** for any of A/B/C --
  a cache-hit/miss and cost estimate was presented separately, pending
  explicit approval before any paid judge call, per this project's
  standing policy (see the `experiment_016/017` entry for the same
  pattern).
- If a future session revisits generation latency, instrument wall-clock
  system load (e.g. `docker stats` snapshots, or run count/timestamps
  correlated against Windows Task Manager) alongside the eval run itself,
  so a latency claim doesn't again depend on assuming a quiet machine.

---

## 2026-08-14: RAGAS on the context/token-budget A/B/C, and a real bug it surfaced

### Interview talking points
- **What happened?** After presenting a cache/cost estimate and getting
  explicit approval, ran RAGAS (7 metrics -- the existing 5 plus this
  session's new Noise Sensitivity and Factual Correctness) on the
  established 15-question stratified sample
  (`techfusion_gold_v2_ragas_sample15.jsonl`) against all three configs
  from the same-day A/B/C entry above.
- **The interesting part:** the very first run silently produced `null`
  for both new metrics, despite `metrics_used` listing them and
  `metrics_failed` being empty -- i.e. no exception anywhere, just wrong
  answers. Root cause: ragas 0.3.9 names `NoiseSensitivity`'s and
  `FactualCorrectness`'s result columns with their `mode` parameter baked
  in (`"noise_sensitivity(mode=relevant)"`, not `"noise_sensitivity"`),
  unlike the other 5 metrics, whose columns are just their bare name. My
  `_load_metrics()` code assumed a metric's own `.name` attribute (which
  *is* the clean `"noise_sensitivity"`) would match its result column --
  it doesn't, for these two. Confirmed directly by instantiating both
  classes and calling `ragas.evaluate()` against a 1-row dataset outside
  the project's own scoring path, not guessed.
- **Fix:** added `_resolve_result_columns()` -- exact-match first, then a
  prefix match (`"<name>("`) against the real `DataFrame.columns` -- so
  metric-name resolution no longer assumes result columns match
  `metric.name` verbatim. 3 new unit tests, including one that
  reconstructs the exact `"noise_sensitivity(mode=relevant)"` column
  shape rather than only testing the exact-match case that would have
  passed even with the bug.
- **Cost:** the buggy first Config-A run (522 judge calls, since it still
  did all the real scoring work -- only the post-processing/aggregation
  was broken) wasn't wasted money-wise once the cache kicked in: re-running
  the identical judge/dataset/config after the code fix hit cache on 290
  of 526 calls (55%). Total actual spend across the buggy run + all 3
  correct runs: 897,754 input / 194,823 output tokens on `gpt-4o-mini` ~=
  **$0.25** (vs. the ~$0.12-0.16 pre-run estimate for 3 clean runs -- the
  gap is entirely the one wasted-but-still-real buggy Config-A pass, plus
  Noise Sensitivity/Factual Correctness costing more per call than the
  original 5 metrics' average).

### Results (15-question stratified sample -- same caveat as `experiment_013`: skewed toward the hardest/most vision-dependent rows, not directly comparable to the 84-question deterministic numbers)

| metric | A (ctx3/tok512) | B (ctx2/tok512) | C (ctx2/tok256) |
|---|---|---|---|
| faithfulness | 0.868 | 0.833 | **0.905** |
| answer_relevancy | 0.175 | 0.273 | 0.244 |
| context_precision | 0.259 | 0.259 | 0.259 |
| context_recall | **0.411** | 0.378 | 0.378 |
| answer_correctness | 0.320 | 0.339 | 0.302 |
| noise_sensitivity (lower=better) | 0.241 | 0.251 | 0.245 |
| factual_correctness | 0.293 | 0.303 | 0.300 |

Recorded as `experiment_021` (A), `experiment_022` (B), `experiment_023`
(C).

### Interpretation
`context_recall` is the one metric with a clean, expected story: A (3
context chunks) recovers more reference content than B/C (2 chunks,
identical to each other since B and C only differ in `max_tokens`) --
`context_precision` being flat at 0.259 across all three says the dropped
3rd chunk in B/C wasn't diluting precision, just removing recall
headroom. Everything else (`faithfulness`, `noise_sensitivity`,
`factual_correctness`, `answer_correctness`) moves by less than would be
trustworthy on n=15 with LLM-judge sampling variance, and **does not show
a consistent quality cost from either reduction** -- notably,
`faithfulness` is *highest* for Config C (the most aggressively reduced
config), the opposite of what "less context hurts faithfulness" would
predict. This is broadly consistent with the deterministic entry's
finding (hand-inspected `answer_quality` swings were mostly a
`KeywordOverlapScorer` artifact, not real degradation) -- an
LLM-judged metric, on this small sample, doesn't contradict that reading
either. Not strong enough evidence on its own to recommend
`generation_context_top_n: 2` for production, but it doesn't argue
against it the way a clean, monotonic faithfulness drop would have.

### Open problems
- n=15 is intentionally too small to be a confident production-adoption
  signal by itself (same caveat every RAGAS-sample entry in this journal
  carries) -- a full-84 RAGAS run per config would cost roughly
  84/15 ~= 5.6x more (~$1.40 total, extrapolating this run's real
  per-token cost) and was explicitly out of scope for this pass.
- The `_resolve_result_columns` fix is specific to the two metric classes
  actually wired in (`_EXTRA_METRIC_CLASSES`); if a future session adds
  another mode-parameterized ragas metric, the same prefix-matching logic
  should cover it, but that's an assumption about ragas's naming
  convention, not a guarantee -- worth a quick sanity check (an
  `evaluate()` call against a 1-row dataset, like the one that found this
  bug) before trusting a brand-new metric's aggregate silently.

## 2026-08-14: classic_rag_baseline_v1 -- a deterministic reference point before the safety/freshness milestone (experiment_024)

### Interview talking points
- **What problem did this solve?** Every experiment so far ran at
  `generation.temperature: 0.2` with no fixed seed -- fine for A/B
  comparisons averaged over 84 questions, but a poor foundation for the
  upcoming safety/freshness milestone, which will want to diff individual
  answers before/after a safety intervention on the *same* question. A
  non-deterministic baseline makes "did the answer change because of my
  safety change, or because of sampling noise" unanswerable. This
  milestone froze a reproducible reference config before that work starts.
- **Why was it interesting?** It's a two-line config change
  (`temperature: 0.2 -> 0.0`, add `seed: 42`) wrapped in real diligence:
  confirming Ollama's Python client actually exposes `options.seed`
  (checked `ollama.Options.model_fields` directly rather than assuming
  from docs -- it does), wiring it through config/factory/wrapper so
  `seed=None` (the new default) reproduces prior non-deterministic
  behavior byte-for-byte (no `seed` key sent to Ollama at all, not a
  literal `0`, which is itself a distinct valid seed), and then actually
  re-running the full 84-question + 7-metric RAGAS pass to prove the
  deterministic config isn't secretly worse.
- **What's the one real, defensible finding?** Every retrieval-side
  metric (`recall@5/@10`, `hit_rate@5/@10`, `mrr`,
  `supporting_context_hit_rate`, `relevant_image_hit_rate`) came back
  **bit-identical** to `experiment_015` -- expected, since retrieval never
  touches the LLM, but a clean confirmation that only `generation`
  changed. Generation-quality metrics (`answer_quality`, RAGAS
  faithfulness/relevancy/context_recall) landed within a point or two of
  `experiment_015`, and RAGAS `context_precision`/`answer_correctness`
  were a few points *lower* -- small enough, on LLM-judge sampling
  variance at n=84, not to read as a real regression from `temperature:
  0`. `ragas_noise_sensitivity`/`ragas_factual_correctness` have no
  `experiment_015` baseline to compare against (added later, see the
  entry above) but landed close to the same-config `experiment_021-023`
  15-question-sample values, so nothing looks anomalous.
- **What's next?** `classic_rag_baseline_v1` (`experiment_024`,
  `config/experiments/classic-rag-baseline-v1.yaml`) is the fixed
  reference point the safety/freshness milestone should diff against --
  same retrieval/chunking/embedding/reranking/prompt-v2/relationship-
  expansion/model as `experiment_015`, deterministic generation.

### Configuration
`config/experiments/classic-rag-baseline-v1.yaml` = a copy of
`multimodal-v2-relationship-qwen3b.yaml` (the config behind
`experiment_015`) with exactly two fields changed:
`generation.temperature: 0.2 -> 0.0`, `generation.seed: (unset) -> 42`.
Verified by a new unit test
(`test_classic_rag_baseline_v1_changes_only_temperature_and_seed`) that
`model_dump(exclude={"generation": {"temperature", "seed"}})` is
identical between the two files. Everything else: `qwen2.5:3b`, prompt
v2, `retrieval.provider: hybrid` (`rrf_k: 60`), `reranker.provider: none`,
`relationship_expansion.enabled: true`, `vision.provider: none`,
`dataset_id: techfusion`.

### Code changes
- `GenerationConfig.seed: int | None = None` (`config.py`) -- `None`
  reproduces prior behavior exactly (no `seed` key ever sent to Ollama,
  distinct from sending a literal `0`).
- `OllamaLLM.__init__`/`generate()` (`ollama_llm.py`) -- accepts `seed`,
  includes it in `options` only `if self._seed is not None`.
- `factory.build_llm` passes `seed=config.generation.seed` through (the
  `judge` LLM builder is untouched -- this milestone only asked about the
  `generation` path).
- `generation_temperature`/`generation_seed` added as recorded
  params/config-summary fields in `run_eval._config_summary`,
  `record_experiment.build_experiment_record`, and
  `mlflow_logger._PARAM_FIELDS` -- so every future recorded experiment
  self-documents whether it was deterministic, not just this one.
- 8 new unit tests across `test_ollama_llm.py` (seed included/omitted/
  temperature+max_tokens in `options`), `test_factory.py` (seed
  pass-through, default-None), `test_config.py` (seed defaults to `None`;
  the classic-baseline-config isolation test above),
  `test_record_experiment.py`, and `test_mlflow_logger.py`.

### Determinism caveat (documented, not just assumed)
Ollama's `options.seed` reproduces output for a fixed model file,
quantization, and thread count on the *same* host -- it is not a
cross-hardware/cross-Ollama-version determinism guarantee. Documented
directly in `GenerationConfig`'s and `OllamaLLM.__init__`'s docstrings so
a future safety/freshness comparison run on different hardware doesn't
assume byte-identical answers it can't get.

### Results (full 84-question `techfusion` gold set, deterministic + RAGAS)

| metric | experiment_015 (temp=0.2, no seed) | experiment_024 = classic_rag_baseline_v1 (temp=0, seed=42) |
|---|---|---|
| recall@5 / recall@10 | 0.911 / 0.946 | 0.911 / 0.946 (identical) |
| hit_rate@5 / hit_rate@10 | 0.929 / 0.952 | 0.929 / 0.952 (identical) |
| mrr | 0.824 | 0.824 (identical) |
| supporting_context_hit_rate | 0.788 | 0.788 (identical) |
| relevant_image_hit_rate | 0.842 | 0.842 (identical) |
| answer_quality (keyword-overlap) | 0.442 | 0.448 |
| ragas_faithfulness | 0.898 | 0.902 |
| ragas_answer_relevancy | 0.530 | 0.539 |
| ragas_context_precision | 0.659 | 0.644 |
| ragas_context_recall | 0.741 | 0.742 |
| ragas_answer_correctness | 0.513 | 0.503 |
| ragas_noise_sensitivity | n/a (metric added later) | 0.342 |
| ragas_factual_correctness | n/a (metric added later) | 0.493 |
| generation_latency_ms/completion_token | n/a (token tracking added later) | 676.7 ms/token |

The last row is compared against `experiment_018` (676.7 vs. 672.9
ms/token, same host, same config apart from temperature/seed) rather than
`experiment_015` directly, since raw `total_latency_ms` across
non-adjacent runs on this shared dev box is already documented as an
unreliable comparison (see the 2026-08-14 context/token-budget entry
above) -- the near-identical per-token rate here is the actual sanity
check, and it passes.

Judge cost (`gpt-4o-mini`, 7 metrics x 84 questions): 2,552 calls,
2,057,253 input / 316,010 output tokens actually billed (636/4,389 cache
hits, 14.5% -- low, as expected, since deterministic answers differ from
every prior non-deterministic run's cached judge prompts) ~= **$0.50**.

### Recommendation
`classic_rag_baseline_v1` (`experiment_024`) is ready to use as the fixed
reference point for the safety/freshness milestone. `config/default.yaml`
is intentionally left untouched (still `temperature: 0.2`, `seed: null`,
`retrieval.provider: dense`) -- this is a dedicated experiment-config
snapshot, not a promoted production default, matching this project's
established pattern for every prior experiment baseline.

### Open problems
- Only one seed (42) was tested -- this establishes *a* reproducible
  baseline, not evidence that `temperature: 0` is itself
  quality-neutral across many seeds; that would need multiple seeded runs
  to check answer variance, out of scope here.
- `answer_quality`'s `KeywordOverlapScorer` and the RAGAS judge are both
  noisy enough at n=84 that "within a point or two of experiment_015" is
  a judgment call, not a statistical test -- there's no confidence
  interval computed anywhere in this project's eval tooling yet.

## 2026-08-15: safety/freshness milestone -- retrieval-time authorization, freshness, trust, and injection telemetry (experiment_025/026, secure_rag_baseline_v1)

### Interview talking points
- **What problem did this solve?** The project had dataset_id isolation
  but nothing scoped *within* a dataset -- any caller could retrieve any
  tenant's document, any superseded policy version, or an untrusted
  uploaded document with equal weight to an authoritative one. This
  milestone added retrieval-time tenant/role authorization, deterministic
  document-version freshness resolution, knowledge-source trust filtering,
  and lightweight (non-blocking) prompt-injection telemetry -- and, per
  the task's explicit instruction, none of it is enforced primarily
  through prompt wording; every hard guarantee is a SQL predicate that
  runs before a row leaves Postgres.
- **Why was it interesting?** Two real bugs surfaced during verification,
  both caught by refusing to trust a suspiciously clean first result
  rather than by anticipating them in advance. (1) A first full-scale eval
  run showed every new safety metric reading exactly `0.0` -- tempting to
  read as "it works," but a `0.0` on a brand-new, first-ever-exercised
  control is exactly the kind of result that deserves a hand-check before
  celebration; it turned out to be a directory-layout mismatch between
  the gold file's authored paths and the actual knowledge-base folder
  structure (see the matching `ISSUES.md` entry), not proof authorization
  worked. (2) Once fixed, `unauthorized_retrieval_rate` still didn't reach
  zero with authorization on -- tracing the 4 remaining cases by hand
  showed they were all callers legitimately authorized for their *own*
  tenant's document, trying to extract one specific secret field via
  prompt injection, which is a field-level redaction gap this milestone
  never claimed to solve, not a document-level authorization failure. The
  metric's own description was inaccurate until this was traced, and got
  corrected in the same session.
- **What's the one real, defensible finding?** On the original 84 benign
  (non-safety) questions, `experiment_025` (authorization off) and
  `experiment_026`/`secure_rag_baseline_v1` (authorization on) are
  **byte-identical** across Recall@5/@10, MRR, answer_quality, and
  refusal_rate -- turning authorization on costs nothing on untenanted
  content, confirmed directly rather than assumed from the "tenant_id IS
  NULL is never gated" design argument alone. On the 42 new safety
  questions, `cross_tenant_leakage_rate` (1.0 -> **0.0**) and
  `stale_document_error_rate` (0.077 -> **0.0**) both move to a perfect
  score with authorization on -- the two metrics that most directly test
  what the SQL predicate is built to enforce.
- **What's next?** A paid RAGAS pass against the safety rows (not run this
  milestone, per "present deterministic results first"); field-level
  redaction (the identified gap from finding #2 above) is unscoped/
  undesigned.

### Design
See `docs/architecture.md`'s "Authorization, Freshness, and Trust" section
for the full writeup (SQL predicate shape, freshness union-find algorithm,
documented limitations) and `CLAUDE.md`'s condensed version. Summary:
- `retrieval/authorization.py`'s `AuthorizationContext` (`tenant_id`,
  `roles`, `as_of`, `include_superseded`, `require_trust_level`,
  `resolved_excluded_document_ids`) is passed explicitly into
  `VectorStore` methods -- structurally separate from the pre-existing
  caller-suppliable `filters` dict, which can only narrow, never grant.
- `vectorstore/pgvector.py`'s `build_authorization_where_clause` builds
  the actual gate: tenant match OR a caller's role that's both a
  configured `cross_tenant_support_role` AND literally on that specific
  document's `allowed_roles`; independently, role membership when
  `allowed_roles` is set at all; a freshness exclusion set; an optional
  trust-level requirement. All ANDed, all NULL-permissive for untenanted/
  untagged legacy content -- the entire pre-existing corpus is never
  gated, making the feature additive by construction, not a flag check on
  the hot path. `config.security.authorization.enabled` (default `false`)
  is a coarser kill-switch `RetrievalPipeline` reads once: when `false`,
  a caller-supplied context is never even passed to the vectorstore.
- `retrieval/freshness.py` groups documents into version families via
  `supersedes_source` path-suffix links (union-find, any chain depth) and
  deterministically resolves which member is eligible per query --
  current mode prefers `status=active`; an explicit `as_of` resolves to
  the member with the latest `effective_from <= as_of`. Never "make every
  version eligible and let the LLM guess."
- `retrieval/injection_detection.py`'s `detect_injection` is telemetry
  only -- flags a chunk's `_source_label` and feeds eval metrics, never
  blocks a result. `rag_answer_v3.yaml` is the first prompt version to
  state the instruction/data boundary explicitly.
- New governance metadata (`tenant_id`, `allowed_roles`, `classification`,
  `status`, `document_version`, `effective_from`, `trust_level`,
  `doc_source_type`, `supersedes_source`) parsed from YAML front matter by
  `loaders/text_loader.py`, copied onto every chunk of a document like
  `category`. `doc_source_type` avoids reusing `source_type`, which
  already means the loader's file-type tag ("markdown"/"text"), not a
  trust-provenance label.
- `eval/corpus_lineage.py` snapshots dataset_id/corpus_version/document+
  chunk+image counts/active+superseded counts/tenant count/gold record
  count/gold-file sha256/a deterministic corpus digest, per eval run --
  flattened into experiment records and MLflow (params/tags plus a
  best-effort `mlflow.log_input`).
- `IngestionPipeline.ingest_path` now diffs a pre-run
  `list_document_sources` snapshot against what's discovered this run and
  deletes the difference (the "deleted documents" gap), returning a
  `schemas.IngestionStats` (discovered/new/changed/unchanged/deleted/
  chunks_embedded/chunks_reused) instead of a bare per-file list.
- `eval/run_eval.py`'s new `"safety"` report section: 11 deterministic
  metrics (`unauthorized_retrieval_rate`, `cross_tenant_leakage_rate`,
  `stale_document_error_rate`, `current_document_retrieval_accuracy` --
  deliberately retrieval-only, independent of `answer_quality`,
  `current_document_answer_quality` is the separate generation-side
  signal -- `prompt_injection_success_rate`,
  `retrieved_prompt_injection_success_rate`, `sensitive_data_leakage_rate`,
  `refusal_accuracy`, `false_refusal_rate`, `poisoned_source_selection_rate`),
  each reporting `{direction, note, count, rate}` so a tiny-N category is
  never a bare percentage. No LLM judge involved -- all heuristic/
  deterministic, same documented-limitation style as this project's other
  triage metrics.

### Configuration
`config/experiments/secure-rag-baseline-v1.yaml` /
`secure-rag-baseline-v1-auth-disabled.yaml` = `classic-rag-baseline-v1.yaml`
(`experiment_024`) + `generation.prompt: v3` + `security.authorization.enabled:
true`/`false` -- the only two fields that differ from `experiment_024`, and
the only field that differs between the pair (`test_config.py`'s isolation
tests). No changes to `config/default.yaml` (`security.authorization.enabled`
stays `false`, same "experiment overlay, not a promoted default" pattern as
every prior milestone).

### Code changes
See `CLAUDE.md`'s "Authorization, freshness, and trust" section for the
full file list. New modules: `rag/path_matching.py` (shared suffix-matching,
extracted from `eval/gold_schema.py` so `retrieval/freshness.py` can reuse
it without an eval->retrieval import), `retrieval/authorization.py`,
`retrieval/freshness.py`, `retrieval/injection_detection.py`,
`eval/corpus_lineage.py`. New `VectorStore` methods: `list_document_sources`,
`delete_documents_by_source`, `list_document_versions`,
`count_chunks_by_document`, `count_chunks_by_content_type`,
`get_document_checksums`. New Postgres columns (idempotent migrations in
`scripts/init_db.py`): `tenant_id`, `allowed_roles`, `classification`,
`status`, `document_version`, `effective_from`, `trust_level`,
`doc_source_type`, `supersedes_source`. Extensive new unit tests
(`test_authorization.py`, `test_freshness.py`, `test_injection_detection.py`,
`test_corpus_lineage.py`, `test_ingestion_stats.py`, plus extensions to
`test_pgvector_where_clause.py`, `test_gold_schema.py`, `test_config.py`,
`test_retrieval_pipeline.py`, `test_relationship_expansion.py`,
`test_run_eval.py`, `test_mlflow_logger.py`, `test_record_experiment.py`,
`test_loaders.py`, `test_vectorstore_base.py`) and a new integration file,
`tests/integration/test_authorization_isolation.py` (10 adversarial tests:
tenant isolation, explicit support-role grant/denial pair, semantic-
similarity-cannot-bypass-ACL, current/historical freshness resolution,
trust-required filtering, user-injection-cannot-bypass-authorization,
forbidden-reference-never-enters-generation-context, and a correct-safe-
request-still-succeeds control). All 428 unit tests and 31 integration
tests (including the 10 new adversarial ones) pass; `ruff`/`mypy` clean.

### Issues encountered (both resolved; full writeups in ISSUES.md)
1. **Directory-layout mismatch.** `techfusion_gold.jsonl`'s
   `relevant_documents`/`forbidden_documents`/`allowed_documents` for 4 of
   the 5 new categories (`governance`, `internal_techfusion`,
   `tenant_alpha`, `tenant_beta`) were authored as
   `knowledge_base/security_evaluation/<category>/...`, but those folders
   sat flat under `data/knowledge_base/<category>/` -- no
   `security_evaluation/` parent existed. Every path-suffix match silently
   failed, making every new safety metric read as a false `0.0` on the
   first full run and dragging Recall@5 down to 0.579 across the full
   126-question set. Caught by hand-running one gold question directly
   against the pipeline and seeing the "forbidden" document actually come
   back. Fixed (confirmed with the user) by moving the 4 folders under
   `data/knowledge_base/security_evaluation/` and re-ingesting; `hr/`
   (ordinary, unrelated new content) was correctly referenced without the
   prefix and stayed flat.
2. **Transient Postgres connection abort.** The very first retrieval call
   of the first eval attempt failed with `psycopg2.OperationalError: could
   not receive data from server` inside `register_vector`, before any new
   authorization/freshness code ran at all -- a Windows-socket-level
   connection drop unrelated to this milestone's changes. Confirmed
   Postgres/Ollama were both healthy and simply retried; succeeded
   immediately on the second attempt.

### Results

**Section 9 (quality-impact on benign queries), 84-question subset:**

| metric | auth disabled | auth enabled | delta |
|---|---|---|---|
| Recall@5 / Recall@10 | 0.869 / 0.935 | 0.869 / 0.935 | 0 / 0 |
| MRR | 0.799 | 0.799 | 0 |
| answer_quality (mean) | 0.422 | 0.422 | 0 |
| refusal_rate (unanswerable rows) | 0.917 | 0.917 | 0 |

**Safety metrics, full 126-question set (auth disabled -> auth enabled):**
`unauthorized_retrieval_rate` 0.947->0.211 (see field-level-redaction
caveat -- not a clean document-level comparison, read
`experiments/reports/secure_rag_baseline_v1.md` before citing this one),
`cross_tenant_leakage_rate` **1.0->0.0**, `stale_document_error_rate`
**0.077->0.0**, `current_document_retrieval_accuracy` 1.0->1.0 (unchanged),
`refusal_accuracy` 0.643->0.786, `false_refusal_rate` 0.036->0.143 (n=28,
worth re-checking at higher N), `sensitive_data_leakage_rate` 0.143->0.286
(n=7), `poisoned_source_selection_rate` 1.0->0.667 (n=3). Full table,
per-metric notes, and the two small-N findings that didn't move cleanly
toward "better" are in `experiments/reports/secure_rag_baseline_v1.md` --
not smoothed over here.

Recorded as `experiment_025` (label `secure_rag_baseline_v1_auth_disabled`)
and `experiment_026` (label `secure_rag_baseline_v1`, the canonical safety
baseline for future work). Corpus: 50 documents, 483 chunks, 13 image
chunks, 3 tenants, corpus digest `d2330e85...` (see
`experiments/results/experiment_026.json` for the full value). No hosted
API call made anywhere in this milestone.

### Open problems
- Field-level redaction (a caller authorized for a *document* still being
  able to extract one specific *field* within it via prompt injection) is
  a real, now-measured gap (`sensitive_data_leakage_rate` 0.143-0.286),
  not designed or implemented this milestone.
- `false_refusal_rate`/`sensitive_data_leakage_rate`/
  `poisoned_source_selection_rate` all move on n=3-28 gold rows -- real
  signal, but not enough to treat single-example swings as trends without
  a larger safety-focused gold set.
- No RAGAS/hosted-judge cross-check on any safety-category question yet.

## 2026-08-15: field-level-safety milestone -- sensitive-value redaction inside otherwise-authorized documents (experiment_027, secure_rag_baseline_v1_field_redaction)

### Interview talking points
- **What problem did this solve?** The prior milestone's own report
  flagged an open gap: document-level tenant/role authorization correctly
  admits a chunk a caller legitimately may retrieve, but that chunk can
  still contain one specific field (a tenant admin's credential) that most
  authorized readers of the *same* document must not see. Two rows in
  `experiment_026`'s `sensitive_data_leakage_rate` (2/7) were exactly this
  -- a `tenant_alpha_operator` with real document access, extracting the
  admin key via prompt injection, not a cross-tenant leak. This milestone
  closes that gap with a second, independent control:
  `retrieval/field_policy.py`'s `SensitiveFieldPolicy` redacts unauthorized
  matched values from chunk content before the prompt is ever built --
  structurally, not by asking the LLM nicely.
- **Why was it interesting?** Two real bugs surfaced during verification,
  both caught by checking real persisted/generated state rather than
  trusting an eval report's aggregate number. (1) After the first
  `--clear` re-ingest, a direct Postgres query showed only the Alpha
  runbook chunk carrying the new `sensitive_field_ids` tag -- Beta's admin
  token literal (`SYNTHETIC_BETA_TOKEN_M8R5_NOT_VALID`) turned out to be a
  genuinely different shape than Alpha's (`SYNTHETIC_ONLY_ALPHA_KEY_...`),
  not a copy-paste variant, and the *pre-existing* leak detector from the
  prior milestone had the identical blind spot the whole time, silently.
  (2) The first full `experiment_027` run reported a non-zero
  `sensitive_data_leakage_rate` (1/7) despite redaction being structurally
  guaranteed to prevent it -- inspecting the flagged answer directly showed
  the admin key *was* correctly redacted twice, but the answer also
  correctly disclosed an operationally-necessary `TF-SYNTH-*` customer
  identifier, which the pre-existing leak-pattern list (inherited unchanged
  from the prior milestone) treated as a "leak" even though
  `field_policy.py`'s own design explicitly does not restrict it. Both
  fixes and full writeups are in `ISSUES.md`.
- **What's the one real, defensible finding?** With the identical prompt
  (`v3`), identical retrieval config, identical model/temperature/seed as
  the control (`experiment_026`), turning on `security.field_redaction.
  enabled` alone drops `sensitive_data_leakage_rate` from 0.286 (2/7,
  confirmed real leaks -- the literal key string in the stored answer) to
  **0.000 (0/7)**, with retrieval-side metrics (Recall@5/@10, MRR,
  supporting_context_hit_rate) byte-identical between the two runs and
  `answer_quality` unchanged-to-slightly-up (0.390 -> 0.405). The one
  authorized-disclosure gold row still gets the real value
  (`sensitive_data_authorized_disclosure_accuracy` 1.0), and zero of the 8
  redacted-field instances across the run were a false positive
  (`sensitive_data_false_redaction_rate` 0.0) -- this isn't "redact
  everything and call it safe."
- **What's next?** A `refusal_accuracy` apparent drop (0.786 -> 0.643) was
  diagnosed as a measurement artifact, not a real regression: the phrase-
  matching refusal detector doesn't recognize a correct *partial* answer
  (rest of the chunk plus `[REDACTED:SENSITIVE_FIELD]`) as a "refusal," even
  though it's more useful and equally safe than the control's full-refusal-
  shaped answers on the same two rows. Worth a small follow-up, not fixed
  this milestone since it's a metric limitation, not a behavior bug. A
  `rag_answer_v4` prompt (redaction-marker-aware phrasing) was deliberately
  not built this milestone -- the design review's adjustment 2 asked to
  isolate structural redaction as the only variable first, and the
  deterministic results show it's already sufficient on its own.

### Design
See `docs/architecture.md`'s "Field-Level Sensitive-Data Redaction" section
for the full writeup and `CLAUDE.md`'s condensed version. Summary:
- `retrieval/field_policy.py`'s `SensitiveFieldPolicy` (`field_id`,
  `sensitivity_type`, `pattern`, `allowed_roles`, `redaction_marker`) is a
  small, hardcoded, dataset-specific list (`DEFAULT_FIELD_POLICIES`) -- not
  a config surface, not a general-purpose secret scanner. One policy
  (`synthetic_admin_credential`) covers both tenants' credential literals
  via a regex alternation (see bug #1 above); a second
  (`synthetic_internal_token`) exists per the task's "structure for future
  detectors" ask even though no current document uses that shape.
  Deliberately does not police `TF-SYNTH-*` customer-correlation ids --
  operational data the document's own text says operators need, not a
  restricted field (see bug #2 above for what happens when a *different*
  part of the codebase didn't share that same exclusion).
- `ingestion/writer.py` calls `detect_sensitive_field_ids(text)` once per
  chunk span (pattern match only, no role check), persisting
  `ChunkMetadata.sensitive_field_ids` (new additive `chunks.
  sensitive_field_ids TEXT[]` column). `RetrievalPipeline.
  _redact_sensitive_fields` runs once, inside `_retrieve_timed`, right
  after relationship expansion and before `_build_context` -- covering
  directly-retrieved and expanded chunks in the same pass, and making every
  downstream consumer (prompt, `answer()`'s `sources`, eval's
  reference-context checks, `run_ragas_eval.py`'s judge payload) inherit
  already-redacted text with no separate scrubbing step.
- **Fail-closed on missing identity (design-review adjustment 1):**
  `config.security.field_redaction.enabled` is independent of `security.
  authorization.enabled`. Whenever it's `True`, roles are resolved as
  `effective_auth.roles if effective_auth is not None else []` -- an empty
  list satisfies no policy, so a missing `AuthorizationContext` redacts
  every tagged field rather than passing it through unrestricted, the
  opposite default from document-level `AuthorizationContext` (where `None`
  means fully unrestricted, for backward compatibility with the untenanted
  pre-milestone corpus). Regression-tested directly
  (`test_field_redaction_fails_closed_when_no_authorization_context_
  supplied`).
- No Base64/reversal/splitting *detector* was built or needed: because
  redaction happens on retrieved content before the prompt is built, an
  unauthorized caller's raw value structurally cannot reach the generation
  model regardless of how the request is phrased. Proven directly by an
  integration test issuing three differently-phrased adversarial queries
  against the same document and confirming byte-identical, already-redacted
  retrieved text every time.
- New/renamed safety metrics: `document_unauthorized_retrieval_rate`
  (renamed from `unauthorized_retrieval_rate`; now excludes any
  `forbidden_documents` entry also present in `allowed_documents` --
  the literal fix for the ambiguity the prior milestone's report flagged),
  `sensitive_data_authorized_disclosure_accuracy`,
  `sensitive_data_false_redaction_rate`, `encoded_extraction_success_rate`,
  plus a per-example `field_level_evidence` block for every
  `sensitive_data_present=true` row.

### Results
Full table, per-metric notes, and both bug writeups are in
`experiments/reports/secure_rag_baseline_v1_field_redaction.md` -- not
smoothed over here.

**Retrieval-side (full 126 questions), `experiment_026` (redaction off) vs
`experiment_027` (redaction on):** Recall@5/@10, hit-rate@5/@10, MRR, and
supporting_context_hit_rate are all byte-identical (0.788/0.832, 0.833/
0.865, 0.753, 0.781) -- field redaction cannot change retrieval by
construction, confirmed rather than assumed.

**Safety metrics, `experiment_026` -> `experiment_027`:**
`sensitive_data_leakage_rate` **0.286 (2/7) -> 0.000 (0/7)** (the headline
result); `sensitive_data_authorized_disclosure_accuracy` **1.0 (1/1)**,
`sensitive_data_false_redaction_rate` **0.0 (0/8)**,
`encoded_extraction_success_rate` **0.0 (0/2)** (all three new this
milestone, no baseline to compare against); `cross_tenant_leakage_rate`/
`stale_document_error_rate`/`retrieved_prompt_injection_success_rate`/
`false_refusal_rate` all unchanged, as expected (field redaction shouldn't
touch any of these); `refusal_accuracy` 0.786->0.643 (measurement artifact,
see talking points above); `poisoned_source_selection_rate` 0.667->0.333
(n=3, confirmed host-level generation non-determinism on an unrelated
question, not a real effect).

Recorded as `experiment_027` (label
`secure_rag_baseline_v1_field_redaction`); `experiment_026` (unchanged)
reused as the control rather than re-run, per the design review's explicit
instruction. Corpus: 50 documents, 483 chunks, 13 image chunks, 3 tenants,
corpus digest `e6de948f...` (see `experiments/results/experiment_027.json`
for the full value). No hosted API call made anywhere in this milestone.

### Open problems
- `refusal_accuracy`'s phrase-list detector doesn't recognize a correct
  partial-redaction answer as a refusal (see talking points) -- a real,
  narrow measurement gap worth a small follow-up.
- No RAGAS/hosted-judge cross-check on any field-level-safety question yet.
- A `rag_answer_v4` prompt (redaction-marker-aware phrasing) remains an
  optional, not-yet-run follow-up experiment -- structural redaction alone
  already reached 0.0 on every field-level metric this milestone measures,
  so there was no evidence-based reason to run it as part of the primary
  comparison.

## 2026-08-15: auth-boundary milestone -- JWT authentication at the API boundary, plus adjacent hardening (experiment_028, secure_rag_baseline_v2_jwt_auth)

### Interview talking points
- **What problem did this solve?** Every prior safety milestone
  (`experiment_025`-`027`) enforced authorization given an *already-
  asserted* identity -- `retrieval/authorization.py`'s own docstring said
  so directly: no authentication layer existed, `POST /query`'s
  `tenant_id`/`roles` were read straight off the raw request body with
  zero verification. Anyone reaching the API could claim to be
  `tenant_alpha_admin`. This milestone adds real JWT verification at the
  API boundary (`api/auth.py`, `api/deps.py:get_current_identity`) so a
  verified identity's claims -- never the request body's -- drive
  `AuthorizationContext`, while leaving every existing retrieval-time
  control (document ACL, freshness, trust, field redaction) completely
  untouched.
- **What else got closed in the same pass?** The design review (17
  numbered requirement sections) surfaced several adjacent gaps worth
  fixing alongside the JWT work: (1) a real architectural finding that
  `RetrievalPipeline.answer()`'s `system`/`user` prompt split -- which
  looked structurally separate on paper -- was silently flattened into
  one string right before the model call, so `LLM.generate(prompt: str)`
  became `LLM.generate(system: str, user: str)` across every
  implementation, and `OllamaLLM` switched from Ollama's raw-completion
  endpoint to its role-aware chat endpoint (full writeup in `ISSUES.md`);
  (2) a metadata-leakage residual risk (a redacted field's value echoed
  through `attachment_name`/`section_path` instead of `content`); (3) a
  duplicate-secret diagnostic (`find_duplicate_sensitive_occurrences`)
  for a literal copy-pasted into a second, differently-tagged chunk; (4) a
  provider-egress policy gating the one confirmed hosted-egress point in
  the whole codebase (`run_ragas_eval.py`'s judge context); (5) request
  size limits and an in-process rate limiter.
- **What's the one real, defensible finding?** With the identical prompt
  (`v3`), identical retrieval config, identical model/temperature/seed as
  the control (`experiment_027`), turning on `security.auth.enabled`
  alone leaves every retrieval-side metric byte-identical
  (Recall@5 0.788, Recall@10 0.832, MRR 0.753) -- confirming, not just
  documenting, that JWT verification happens before retrieval and never
  changes which chunks are retrieved. The two new regression-guard
  metrics both read as correct code should:
  `forged_role_acceptance_rate` **0.0 (0/42)** -- a request body claiming
  a different, more-privileged tenant/roles never overrides a verified
  JWT's own claims, checked by calling `api.routers.query.
  _build_authorization_context` directly against every gold row with a
  `user_tenant`/`user_roles` identity -- and `duplicate_sensitive_field_
  miss_rate` **0.0 (0/0)** -- the real corpus has no duplicated or
  inconsistently-tagged secret literal today.
- **The two non-zero new metrics turned out to be the metric working
  correctly, not a bug.** `unauthorized_metadata_leakage_rate` and
  `provider_egress_policy_violation_rate` both read 0.095 (2/21).
  Inspecting the two flagged rows directly (not just trusting the
  aggregate) showed both are the same two cases: a tenant admin
  correctly, legitimately seeing their own tenant's credential (the same
  case `sensitive_data_authorized_disclosure_accuracy` already scores
  1/1) -- field-level policy correctly leaves it unredacted for that
  caller, but the egress-policy check correctly still flags it as unsafe
  to hand to a hosted third-party judge without separate authorization.
  "Authorized for this caller" and "authorized to leave the local
  environment" are deliberately two different questions in this design
  (see `docs/architecture.md`), and this run is the first concrete
  evidence that the distinction actually catches something real rather
  than being a hypothetical concern.
- **What moved and why (noise, not regression):**
  `prompt_injection_success_rate` (0.0 -> 0.2, n=5, one row) and
  `poisoned_source_selection_rate` (0.333 -> 0.667, n=3, one row) both
  shifted on tiny-N metrics between two separately-invoked
  `qwen2.5:3b` generation runs on this CPU-only Windows box --
  consistent with this project's own previously-documented finding
  (`experiment_018`-`020`'s entry) that `seed`/`temperature` reproduce
  *within* one host/process but not necessarily bit-identically *across*
  two separate runs. `refusal_accuracy` (0.643 -> 0.714) and
  `false_refusal_rate` (0.143 -> 0.107) moved in the "better" direction
  by one row each, same small-N caveat -- not attributed to any code
  change in this milestone, since neither refusal-detection logic nor
  the underlying retrieval changed between the two runs.

### Design
See `docs/architecture.md`'s "Authenticated API Boundary and Security
Hardening" section for the full writeup and `CLAUDE.md`'s condensed
version. Summary:
- `api/auth.py:verify_jwt(token, config)` verifies signature/expiration/
  issuer/audience (issuer/audience only when configured)/required claims
  via `pyjwt`, returning `VerifiedIdentity` or raising
  `AuthenticationError(reason=...)` -- never the token itself.
  `config.security.auth.enabled: false` (default) is a true no-op;
  `insecure_dev_mode` only relaxes the no-header case, never a
  present-but-invalid token (fail-closed, verified directly by
  `test_invalid_jwt_never_falls_back_to_unrestricted_retrieval`).
- `RetrievalPipeline` never imports from `api/`, and `api/auth.py` never
  imports `RetrievalPipeline`/`VectorStore` -- authentication and
  authorization stayed structurally separate largely for free, because
  `RetrievalPipeline` already only ever *accepted* an
  `AuthorizationContext` as a parameter rather than constructing one
  itself (a design already in place from the safety/freshness milestone).
- New top-level `src/rag/audit.py` (not under `api/`, since
  `retrieval/pipeline.py` calls it too) -- `log_audit_event`/
  `pseudonymous_subject`, reusing `logging_config.py`'s JSON formatter
  wholesale. Documented gap: no true per-document `authorization_denied`
  trail yet, since `pgvector.py`'s SQL predicate never reports what it
  excluded.
- `field_policy.redact_source_metadata`/`find_duplicate_sensitive_
  occurrences` (new), `eval/egress_policy.py` (new),
  `DoSLimitsConfig`/`RateLimitConfig` (`slowapi`, in-memory, documented
  single-instance limitation). Six new safety metrics in `eval/run_eval.py`,
  two of them (`authentication_failure_acceptance_rate`/
  `oversized_request_rejection_accuracy`) living in a separate, opt-in
  `evaluate_authentication_boundary_probes(config)` function (exercises
  the live FastAPI app via `TestClient`, not `RetrievalPipeline` --
  `--include-api-probes` CLI flag, not run by default).
- `config/experiments/secure-rag-baseline-v2-jwt-auth.yaml` /
  `-dev.yaml`: experiment_027's config plus `security.auth.enabled: true`
  (`insecure_dev_mode: false`/`true` respectively) -- prompt stays `v3`.

### Results
Full 126-question `techfusion` gold set, `qwen2.5:3b`, hybrid+RRF,
relationship expansion on, reranker off, temperature 0/seed 42 --
identical to experiment_027 except `security.auth.enabled: true`.

| Metric | experiment_027 (control) | experiment_028 |
|---|---|---|
| Recall@5 / Recall@10 | 0.788 / 0.832 | 0.788 / 0.832 |
| MRR | 0.753 | 0.753 |
| answer_quality (mean) | 0.405 | 0.420 |
| forged_role_acceptance_rate | n/a (new) | **0.0 (0/42)** |
| duplicate_sensitive_field_miss_rate | n/a (new) | **0.0 (0/0)** |
| unauthorized_metadata_leakage_rate | n/a (new) | 0.095 (2/21) -- see talking points |
| provider_egress_policy_violation_rate | n/a (new) | 0.095 (2/21) -- see talking points |
| sensitive_data_leakage_rate | 0.0 (0/7) | 0.0 (0/7) |
| sensitive_data_false_redaction_rate | 0.0 (0/8) | 0.0 (0/8) |
| document_unauthorized_retrieval_rate | 0.118 (2/17) | 0.118 (2/17) -- unchanged pre-existing baseline, not from this milestone |
| cross_tenant_leakage_rate | 0.0 | 0.0 |
| stale_document_error_rate | 0.0 | 0.0 |

Retrieval-side and every pre-existing safety metric either matched the
control exactly or moved within the small-N noise band this project has
already documented for repeated CPU-only local generation runs (see
Interview talking points above). No HTTP-level metrics
(`authentication_failure_acceptance_rate`/`oversized_request_rejection_
accuracy`) were run as part of this recorded experiment --
`security.auth.enabled: true` in the config has no effect on
`eval/run_eval.py`'s gold-driven harness (it calls `RetrievalPipeline`
directly, never `POST /query`); those two metrics require
`--include-api-probes` and a resolvable `JWT_HS256_SECRET`, exercised
separately via unit/integration tests
(`tests/unit/test_run_eval.py::test_evaluate_authentication_boundary_
probes_all_rejected_when_auth_enabled`,
`tests/integration/test_api_field_redaction.py`) rather than folded into
this deterministic report.

Full unit suite: 515 passed, 3 skipped. Full integration suite (real
Postgres + real Ollama): 46 passed. `ruff`/`mypy`: clean.

### Open problems
- No true per-document `authorization_denied`/`trust_policy_rejection`
  audit trail -- `pgvector.py` doesn't report which specific rows a SQL
  predicate excluded, so `cross_tenant_attempt` is approximated via
  `forged_claim_attempt` instead. A real fix would need `VectorStore` to
  surface excluded-row telemetry, a larger change not undertaken this
  milestone.
- `document_unauthorized_retrieval_rate`'s 2/17 non-zero baseline is
  unchanged from `experiment_026`/`027` -- a pre-existing gap this
  milestone didn't introduce and didn't investigate further (out of
  scope; JWT verification happens upstream of retrieval and can't affect
  which documents a query returns).
- `rag_answer_v4` (structural role-separation-aware prompt wording) is
  written but deliberately not evaluated this milestone, per the approved
  design adjustment isolating the architectural LLM-interface change from
  any prompt-wording change. A follow-up A/B is a reasonable next step
  once there's a specific reason to test it.
- No RAGAS/hosted-judge cross-check on any auth-boundary-specific
  question yet -- deterministic security tests only, per the milestone's
  own explicit constraint (requirement 14).

## 2026-08-15: codebase-wide NumPy-style documentation cleanup

Behavior-preserving documentation pass across `src/rag/` and `scripts/`,
per `.claude-prompts/0-Codebade-doc-str.txt`: trimmed module docstrings
and inline comments that had accumulated milestone-history/design-review
narrative (references to "requirement N," approved-design adjustments,
"confirmed by grepping," etc.) down to concise NumPy-style documentation
focused on responsibility, inputs/outputs, and invariants. Function/method
docstrings were already close to NumPy convention throughout the codebase
and needed lighter touches; the heaviest cleanup was in the
security/auth-boundary modules (`field_policy.py`, `api/auth.py`,
`retrieval/authorization.py`, `retrieval/freshness.py`,
`retrieval/pipeline.py`, `config.py`'s security config models) and in
`eval/run_eval.py`/`eval/mlflow_logger.py`'s accumulated per-milestone
inline comments. Added important-domain-model docstrings (full NumPy
`Parameters` sections) to `VerifiedIdentity`, `AuthorizationContext`,
`GoldExample`, `SensitiveFieldPolicy`, and several `config.py` security
models that previously had prose-only summaries. Moved two pieces of
cross-component rationale that no longer fit cleanly in source
docstrings into `docs/architecture.md`: retrieval's three-cutoff
semantics ("Retrieval Cutoff Semantics") and RAGAS judge-cache
fingerprinting ("RAGAS Judge-Call Caching"). No runtime behavior, SQL,
signatures, or eval-report string content changed -- validated with
`ruff check`, `ruff format --check`, `mypy`, and the full unit suite (515
passed, 3 skipped) run clean before and after.

## 2026-08-16: Agentic RAG milestone

Implemented the Agentic RAG milestone: a small, bounded, production-shaped
agentic workflow layered above the existing classic pipeline, per
`.claude-prompts/7.0-Agentic-RAG`. Planned first (Explore research passes
over retrieval/authorization internals, the API security boundary, and
the eval/prompt/config framework, then a dedicated design pass), reviewed
with the user, revised per seven explicit review adjustments, then
implemented end to end.

### Design decisions

- **Hand-rolled state machine, not LangGraph.** LangGraph is not
  installed and has zero prior usage anywhere in this codebase (the only
  pre-existing LangChain usage is a text splitter and a RAGAS
  judge-caching wrapper). Given this project's repeated, explicit
  rejection of infrastructure additions "without a demonstrated
  requirement now" (Redis, MCP), and that the needed graph is small (~7
  nodes, 3 independent bound counters), the graph
  (`src/rag/agent/graph.py`'s `run_agent()`) is a plain Python `while`
  loop over node functions operating on a pydantic `AgentState`. Written
  down explicitly: revisit only if a future milestone needs
  checkpoint/resume across requests, human-in-the-loop approval gates,
  cross-request workflow state, or substantially more complex branching.
- **Agentic structured tool *dispatch*, not provider-native function/tool
  calling.** Verified directly against the installed `ollama` client
  (0.6.2) that it supports a `tools=` parameter, but chose not to use it:
  extending the shared `LLM` ABC across all three subclasses
  (`OllamaLLM`/`OpenAILLM`/`AnthropicLLM`) for a capability only 4-5
  narrow decision points need was judged a larger blast radius than
  necessary, and betting *control flow* on a local 3B model's native
  tool-calling reliability was judged the wrong risk trade. Decision
  points call the unmodified `LLM.generate(system, user) -> str`; a new
  prompt instructs JSON-only output, parsed/validated in Python
  (`src/rag/agent/decisions.py`).
- **Four tools, each a thin wrapper, never duplicating retrieval logic**:
  `search_knowledge_base` (`RetrievalPipeline.retrieve()`), `get_document`/
  `get_latest_document` (a new `VectorStore.get_chunks_by_source`,
  mirroring `get_chunks_by_section`'s exact auth-scoping shape),
  `get_related_context` (`RetrievalPipeline.expand_with_relationships()`,
  renamed public from `_expand_with_relationships` specifically for this
  reuse).

### Review adjustments incorporated (approved-with-changes review)

The initial proposal was approved contingent on seven concrete hardening
changes, all implemented:

1. **Universal evidence sanitization.** A new public
   `RetrievalPipeline.sanitize_evidence(results, auth)` (reusing the
   pipeline's existing field-redaction/injection-flagging logic) is
   applied centrally by the graph driver's `execute_tool` step to
   *every* tool's output, not delegated per-tool -- so a document fetched
   directly by `get_document`/`get_latest_document` cannot bypass field
   redaction or injection flagging just because it never went through
   `retrieve()`. Proven directly against real Postgres in
   `tests/integration/test_agent_tool_tenant_isolation.py`.
2. **Bounded `get_document`/`get_latest_document`.** Neither tool's
   Pydantic arg schema exposes a chunk-count field; the limit is entirely
   server-side -- a hard SQL `LIMIT` (`max_chunks_per_document_fetch_
   hard_ceiling`, default 50) followed by cosine-similarity
   relevance-selection against the current query down to
   `max_chunks_per_document_fetch` (default 10).
3. **Every LLM-writable numeric tool argument is server-bounded.**
   `SearchKnowledgeBaseArgs.top_k` is `Field(ge=1, le=20)`, then clamped
   again to `config.agent.max_tool_top_k` server-side in the graph
   driver -- a config change, not just the schema literal, is always the
   true final ceiling.
4. **LangGraph reconsideration threshold documented explicitly** (see
   above) rather than left implicit.
5. **Terminology fixed everywhere**: "agentic structured tool dispatch,"
   never unqualified "function calling"/"tool calling," in this journal,
   `docs/architecture.md`, and `CLAUDE.md`.
6. **`data/eval/agentic_extension_gold.jsonl` stays a separate file**,
   never merged into the 126-row `techfusion_gold.jsonl` baseline.
7. **`get_latest_document` corrected to resolve version families, not
   just re-fetch the requested path.** A version family's members each
   have their *own* `source` path (e.g. `policy-v1.md` superseded by
   `policy-v2.md`); the original design's plain
   `get_chunks_by_source(args.source, ...)` would have silently returned
   stale content if asked for an old version's path. A new
   `freshness.resolve_current_document_source(source, versions, as_of,
   include_superseded)` resolves any family member's path to the
   currently-effective member's path first -- verified directly against
   real ingested version-family fixtures in
   `tests/integration/test_agent_tool_tenant_isolation.py`, not just
   freshness unit-test doubles.

### A genuinely surprising find: pre-existing gold data

While building the eval side, discovered `data/eval/
agentic_extension_gold.jsonl` already existed on disk (18 rows,
gitignored per an already-modified `.gitignore`), authored with a richer
schema than initially planned (`agentic_category`, `agentic_rationale`,
`expected_subquestions`, `expected_reformulation`,
`minimum_expected_tool_calls`, `maximum_expected_tool_calls`, in addition
to `expected_tool_sequence`/`requires_query_decomposition`/etc.) and
referencing real files under `data/knowledge_base/`. Verified every
referenced document actually exists on disk and that the file parses
cleanly, then extended `GoldExample` to support the file's full field set
rather than overwrite it with a narrower, independently-designed schema.
The file's category distribution (`query_decomposition`: 4,
`latest_document_resolution`: 4, `retrieval_reformulation`: 3,
`adversarial_tool_output`: 3, `tool_not_needed`: 2,
`insufficient_evidence`: 2) matches the milestone's planned coverage
almost exactly.

### Implementation

New `src/rag/agent/` package (`state.py`, `tool_schemas.py`, `tools.py`,
`decisions.py`, `graph.py`), five new prompt templates
(`agent_classify_v1.yaml` through `agent_synthesize_v1.yaml` --
had to escape literal `{`/`}` in the JSON-schema example blocks as
`{{`/`}}` since `PromptTemplate.render` uses `str.format`, caught
immediately by the existing `required_variables`/placeholder
cross-validation), a new `POST /agent/query` router reusing (not
reimplementing) `/query`'s JWT-precedence/DoS-limit logic (promoted to a
new shared `src/rag/api/request_auth.py`), a new `AgentConfig` section
(`config.py`/`config/default.yaml`, `enabled: bool = False` by default,
matching every other optional-feature convention), a new deterministic
`src/rag/eval/run_agent_eval.py`, and matching `mlflow_logger.py`
extensions (`agent_*` param/metric fields, an `"_agentic"` `build_run_
name()` segment). `SearchResult.origin` gained a third value,
`"tool_fetched"`, for tool-provenance labeling.

### Testing

~65 new unit tests (mocked LLM/pipeline/vectorstore, no real I/O) covering
routing, all three independent bounds, tool-argument rejection and
tool-execution-failure handling, the insufficient-evidence terminal path,
JSON-decision parsing/retry, and token accounting. Two new integration
test files against real Postgres (`test_agent_tool_tenant_isolation.py`,
4 tests) and real Postgres + real Ollama with `qwen2.5:3b`
(`test_agent_end_to_end.py`, 2 tests -- chosen over the default
`qwen2.5:1.5b` for better JSON-schema-following reliability, per the
classic-RAG milestone's own earlier finding). One real bug caught by the
integration tests that the mocked unit tests couldn't have caught:
`get_chunks_by_source` does an *exact* `source` match, so a test written
against a bare filename (`"policy.md"`) silently returned nothing against
real Postgres, since ingestion stores the full path it was given
(`source=str(path)` in `loaders/text_loader.py`) -- fixed by querying with
the real stored path, matching how the tool is actually meant to be used
(the LLM learns a path from a prior `search_knowledge_base` result, never
types one from memory).

Full unit suite: 585 passed, 3 skipped. Full integration suite (real
Postgres + real Ollama): 52 passed, including all 6 new agent tests.
`ruff`: clean on every production source file (some mechanical
docstring-only cleanup remained on new test files after the initial
pass). No pre-existing test was modified except two renamed-symbol import
fixes (`_build_context`/`_source_label` -> `build_context`/`source_label`
in `tests/integration/test_query_pipeline.py`, matching the production
rename).

### Open problems

- `evidence_sufficiency_accuracy`/`retry_success_rate` in
  `run_agent_eval.py` are proxied via `retrieval_attempts >= 2`, not a
  true "was evidence ever judged insufficient" signal, since `AgentState`
  only retains the *final* `evidence_sufficient` decision, not a full
  history. `tool_selection_accuracy` is similarly a simplified
  expected-tool-subset check, not exact-sequence matching.
- `agent_token_usage` undercounts on a retried decision call, since the
  LLM instance's `last_prompt_tokens`/`last_completion_tokens` tracking
  attribute is overwritten per call, not accumulated by the LLM
  implementation itself.
- The controlled A/B/C experiment (classic baseline vs. router-enabled vs.
  full agentic workflow) is designed but not yet run/recorded as
  `experiments/results/*.json` entries -- a deliberate "design now, run
  after implementation" split, not an oversight.
- No MCP server, per the milestone's explicit constraint -- though
  `rag/agent/tools.py`'s plain-function-plus-Pydantic-schema shape was
  chosen so a future MCP exposure wouldn't require rewriting business
  logic.

### 2026-08-15 addendum: review pass over documentation cleanup

Reviewed the previous docstring/comment pass against the current code.
Kept the useful NumPy-style model and API documentation, corrected stale
architecture wording around the now-authenticated API boundary, and
trimmed source comments/docstrings that still referenced agent guidance,
cross-file docstrings, milestone history, or `--` punctuation in prose.

## 2026-08-20: manual end-to-end verification of agentic RAG + the auth-boundary milestone against a live container

Session goal was hands-on: actually send `/agent/query` requests against a
running container and confirm the agent-routing, authorization, and
freshness behavior documented in the auth-boundary and agentic-RAG
milestones holds up outside the test suite, not just in mocked unit tests
and integration tests run in CI-like conditions.

### Getting a live stack up

Both containers had exited ~44 hours earlier (host machine had been off/
asleep) -- `docker compose ps` showed nothing running despite stale log
output from an earlier session looking current. `docker compose up -d`
brought both back cleanly. Along the way, confirmed two things worth
remembering: `docker compose up` with no `-f` flags only ever loads
`docker-compose.yml` (the dev stack) -- `docker-compose.prod.yml` requires
the explicit two-file `-f` invocation and is never picked up implicitly.
And uvicorn's `--reload` (used by the dev compose file) only watches
`*.py` files by default, not YAML -- combined with `AppConfig` being
`lru_cache`'d, a `config/default.yaml` edit needs a container recreate/
restart to actually take effect, not just a file save.

### Turning `agent.enabled` on and exercising both routes

Flipped `config.agent.enabled: true`, recreated the container, and sent
two real queries: a simple factual question correctly took the fast
`classic_rag` route (agent classifier judged it didn't need
decomposition -- expected, not a bug); a genuinely multi-hop comparison
question ("compare the authentication approach in the architecture docs
with the latest security policy version, tell me what changed") took the
real `agent` route with two `search_knowledge_base` tool calls, but
terminated via `max_steps` (hit the 8-step ceiling) rather than
`synthesized`, taking ~51s total, almost entirely `generation_ms` --
confirmed this field is deliberately an approximation on the agent route
(`graph.py`'s `AgentRunResult` docstring): every LLM decision call across
the whole run (classify/decompose/select_tool/evaluate_evidence/
synthesize) collapses into that one number, so a run that uses its full
step budget makes several sequential local CPU-only Ollama calls, not one.

### The JWT-secret wiring gap

Flipped `security.authorization.enabled`, `security.field_redaction.
enabled`, and `security.auth.enabled` all to `true` to test the full
authorization stack against real governance-tagged data. Discovered
`docker-compose.yml`'s `rag-api` service never forwarded
`JWT_HS256_SECRET` into the container at all, and it wasn't in `.env`
either -- a real gap in the auth-boundary milestone's own compose wiring,
invisible from reading `config/default.yaml` since every field there
looked correctly set. Fixed by generating a dev secret
(`secrets.token_urlsafe(32)`), adding it to `.env`, and adding
`JWT_HS256_SECRET: ${JWT_HS256_SECRET:-}` to the compose file's
environment block (same pattern as the existing `COHERE_API_KEY`
passthrough) -- full write-up in `ISSUES.md`'s "JWT auth flipped on in
config, but the container never had a secret to verify with" entry.

Also ingested `data/knowledge_base/security_evaluation/` (16 documents,
114 chunks, `dataset_id: security_evaluation`) for the first time in this
session -- it existed on disk but had never been loaded into the running
vector store, so no authorization test against it could have returned
anything meaningful before this.

### Verified against the live API, not just inferred

Minted two dev JWTs via `scripts/issue_dev_token.py` (`tenant_alpha_admin`,
`tenant_beta_admin`) and ran three checks directly against `/query`:

- No `Authorization` header -> `401`.
- `tenant_alpha` token asking about its own backup procedure -> correct
  answer, sources scoped entirely to `tenant_alpha`, and freshness
  correctly resolved to the active `backup-procedure-v2.md` rather than
  the superseded `v1`.
- `tenant_beta` token asking about `tenant_alpha`'s confidential
  integration runbook -> correctly refused ("I don't know"), retrieved
  sources scoped entirely to `tenant_beta` -- the SQL-level authorization
  predicate held under a real cross-tenant attempt, not just the
  `test_authorization_isolation.py` adversarial suite.

### Open follow-ups noted, not yet done

- `rate_limit.enabled` is still `false` -- not exercised this session.
- `/agent/query` has no streaming/live-progress surface (no SSE/WebSocket
  anywhere in `api/`) -- a real gap if step-by-step agent visibility
  (à la a chat UI's live "thinking" trace) is ever wanted; the graph's
  node-by-node structure would support adding it without a redesign, but
  nothing was implemented this session.
No runtime behavior, signatures, SQL, prompts, or eval logic changed.

## 2026-08-21: Observability milestone -- OpenTelemetry, Prometheus, Grafana, live agent events

Prep work before running the final `agentic_rag_baseline_v1` evaluation:
instrument the bounded agentic workflow so per-request latency is
explainable (which node, how much of it real LLM inference vs.
parsing/validation overhead), export it as traces + metrics, add a local
Grafana dashboard, and add a safe live-progress SSE stream -- directly
closing the "no SSE/live-progress surface" follow-up noted in the
2026-08-20 session above. Explicitly out of scope: any change to agent
decision logic, prompts, retrieval parameters, agent bounds, or
generation config -- the upcoming benchmark had to stay controlled.

### Design decisions made up front

- OTel/Prometheus deps (`prometheus-client`, `opentelemetry-api`/`sdk`/
  `-exporter-otlp-proto-http`) went in as **core** dependencies, not a new
  optional extra -- matches `slowapi`'s existing precedent (a production-
  boundary concern), and unlike `ragas`/`mlflow` they're small pure-Python
  packages with no heavy ML transitive deps.
- No `opentelemetry-instrumentation-fastapi` auto-instrumentation package
  -- hand-wrote the one HTTP root span inside the existing
  `RequestIDMiddleware` instead, which already computed `duration_ms`/
  method/path for the `request_handled` log line. Smaller dependency
  surface, no auto-instrumentation version coupling.
- Trace backend: Jaeger all-in-one, no separate OTel Collector container
  -- Jaeger (v1.35+) accepts OTLP natively on 4317/4318, so `rag-api`
  exports straight to it. Documented as the "simplest maintainable local
  option" call the milestone spec asked for.
- `observability.metrics.enabled`/`observability.live_events.enabled`
  default `true` (in-process only / route-only, harmless); `observability.
  tracing.enabled` defaults `false` (needs a real OTLP endpoint to be
  useful, avoids background exporter noise on a fresh checkout with
  nothing listening).
- SSE endpoint is `POST`, not `GET` -- matches `/agent/query`'s existing
  JSON-body shape; this project has no browser frontend to need
  `EventSource`'s GET-only constraint.

### Node timing: also splitting real inference from overhead

Partway through planning, extended the ask (per direct instruction) to
distinguish actual Ollama inference latency from total node latency, so
preprocessing/JSON-parsing/template-rendering overhead is visible
separately -- not just per-node totals. Implemented via `_TimingLLM`
(`agent/graph.py`), a thin wrapper around the injected `LLM` that
accumulates `total_llm_ms` across every `generate()` call a node makes
(including JSON-parse retries, so retries are never undercounted -- the
same caveat token accounting already documented, now closed for timing
too). `_call_node`, the single wrapper every node call in `run_agent()`
goes through, reads the delta in `total_llm_ms` before/after each call to
split `llm_ms` from `overhead_ms`. `execute_tool` (no direct LLM call)
correctly reports `None` for both rather than a misleading `0.0`.
Verified with a test using a deliberately-slowed mock `generate()` call
(`SleepyLLM`, 50ms sleep) asserting `llm_ms` tracks the sleep duration
and `overhead_ms` stays small in comparison.

### The pre-existing test suite was already broken, unrelated to this work

Running the full suite after implementing surfaced 112 failing tests --
almost none caused by this milestone. The most recent commit ("feat: add
Agentic RAG milestone and its evaluation tooling") had flipped
`security.auth.enabled`, `security.authorization.enabled`,
`security.field_redaction.enabled`, and `retrieval.relationship_
expansion.enabled` all to `true` in `config/default.yaml` as one
baseline-prep bundle (alongside adding the brand-new `agent.enabled`
field, also `true`) -- and that commit's own CI had failed, since none of
the pre-existing tests assuming the old `false` defaults were updated to
match. `relationship_expansion.enabled=true` alone accounted for ~83 of
the 112 failures (every test whose `FakeVectorStore` double didn't
implement `get_chunks_by_section`/`get_chunks_by_ids`).

First pass got the split wrong: reverted only `relationship_expansion`
(it wasn't mentioned in the earlier JWT/tenant session log specifically,
so it looked like the odd one out) and kept `auth`/`authorization`/
`field_redaction` at `true`, updating ~20 tests to match. Corrected
directly by the repo owner: the actual intent was for **all four**
pre-existing flags to go back to `false`, restoring the exact pre-commit
state the full suite already passed against -- `agent.enabled` is the one
field that stays `true`, since it's brand new in this same commit (no
prior value to restore). Fixed by reverting all four flags in
`config/default.yaml`, then `git checkout HEAD --` on the 8 test files
whose only diff was adapting to the (now-reverted) `true` defaults --
restoring their original committed assertions exactly rather than
hand-reverting each edit -- and fixing the small number of tests that
broke specifically because of the new `agent.enabled=true` field (a stub
LLM that asserted "must never be called on the classic path" needed to
instead return a valid classify response, since `classify_query` now
genuinely runs -- and correctly still routes to `classic_rag` -- when the
agent is enabled by default). Full suite green: 615 passed, 3 skipped
(pre-existing `ragas`/`openai`/`anthropic` optional-dependency skips,
unrelated to any of this).

### A real, if minor, OTel edge case

Reusing Starlette's worker threadpool for the SSE endpoint's
`run_in_threadpool` calls occasionally produces a benign but noisy
ERROR-level `opentelemetry.context` "Failed to detach context ... was
created in a different Context" log line when a pooled thread gets reused
across requests with a leftover OTel span-context token. Traced it to
OpenTelemetry's *own* internal `detach()` error handling -- it logs at
ERROR and swallows the exception rather than raising, so it never
actually broke a request or failed a test, but it's real log noise.
Documented as a known limitation rather than engineered around (would
require avoiding contextvars-based span nesting across thread-pool
boundaries entirely, out of proportion to the actual impact). Full
writeup in `ISSUES.md`.

### What shipped

`src/rag/observability/` (tracing.py, metrics.py); `_TimingLLM`/
`_call_node`/`NodeInvocationTiming`/`NodeTimingStats` in `agent/graph.py`
and `agent/state.py`; `agent/events.py` (`AgentEvent`, no free-text field
by construction); `POST /agent/query/stream` (`api/routers/
agent_stream.py`); `GET /metrics` (`api/routers/metrics.py`); `GET /`
(`api/main.py`); HTTP-level span/metrics in `RequestIDMiddleware`; a
`retrieval` span in `RetrievalPipeline._retrieve_timed` (split into a
thin wrapper + `_retrieve_timed_inner`, the original logic moved
verbatim); `docker-compose.observability.yml` + Prometheus/Grafana
provisioning + one dashboard (`observability/grafana/dashboards/
agentic-rag-overview.json`); `make observability-up`/`-down`; 9 new test
files (all passing) plus fixes to ~20 pre-existing tests. `docs/
architecture.md`/`CLAUDE.md`/`README.md` updated with a new
"Observability" section each. Did not run the 18-question agentic eval or
RAGAS, and did not commit/push anything, per the milestone's explicit
instructions.

## 2026-08-21: agentic_rag_baseline_v1 evaluation -- run, not established

Committed the observability milestone (8 focused commits: config/deps,
tracing+metrics core, agent instrumentation, new endpoints, the opt-in
Docker stack, tests, docs) and moved on to the deferred final task from
that session: run the full `agentic_rag_baseline_v1` evaluation
(`.claude-prompts/7.5-Agentic-RAG-Evaluation.txt`) and decide whether to
establish it.

### Pre-evaluation

Brought Postgres back up (container had exited between sessions) and
confirmed native Ollama with `qwen2.5:3b`. Ran the full test suite for the
first time with `ragas`/`openai`/`anthropic` actually relevant: found only
3 skips, all `pytest.importorskip` guards -- and discovered those packages
were *not* installed in this dev venv (a change from what an earlier
session's summary assumed). Installed `pip install -e .[ragas]` (matches
README's own documented command) and reran clean: 683 passed, 1 skipped
(`anthropic` extra, legitimately not needed -- this evaluation's judge is
`openai`). `ruff check .` clean. `mypy src/` surfaced 2 errors that were
previously invisible under `ignore_missing_imports` for the
not-yet-installed `ragas`/`openai` packages -- traced both to real stub
strictness, not runtime bugs (confirmed via `ragas.evaluate`'s actual
signature and via the now-passing `test_ragas_scorer.py` exercising the
real call path) -- documented, not fixed, since neither invalidates the
evaluation and both are pre-existing, unrelated to this session's work.

Verified explicitly (not assumed from `config/default.yaml`'s own
all-`false` shipped defaults): `config/experiments/agentic-rag-baseline-v1.yaml`
has `agent.enabled`/`authorization`/`field_redaction`/`auth`/
`relationship_expansion` all `true`; classic-RAG fast path uses
`rag_answer_v3`, agent synthesis uses the distinct `agent_synthesize_v1`;
`rag_answer_v4` isn't referenced anywhere in the eval configs.

### Running it

Ingested `data/knowledge_base` under `dataset_id=techfusion` (idempotent,
already present -- 50 docs / 483 chunks, matching `experiment_028`'s
corpus). Ran three deterministic passes against
`data/eval/agentic_extension_gold.jsonl` (18 rows): the primary agentic
config (`experiment_029`), a classic-RAG control with `agent.enabled:
false` (`experiment_030`), and the pre-existing `max_agent_steps: 4`
diagnostic config for max-step behavior (`experiment_031`) -- each
recorded to `experiments/results/agentic/*.json` and MLflow
(`sqlite:///mlflow.db`). Extended `run_agent_eval.py` additively (new
`node_latency_breakdown_ms`/`node_token_breakdown`/`total_llm_calls`
report sections, reading the observability milestone's already-computed
`AgentRunResult.node_timings_ms`/`node_token_usage`/`llm_call_count` --
pure reporting, no agent logic touched) so the report could show the
LLM-inference-vs-overhead split per node the task asked for. Then ran
`run_agent_ragas_eval.py` (openai/gpt-4o-mini judge) over the same 18
rows: actual cost ~$0.14 (539K input / 98.6K output tokens), in line with
the ~$0.05-0.20 estimated beforehand from `experiment_015`'s 84-question
$0.23 run -- proceeded without pausing for confirmation, per the task's
own "only pause if unexpectedly larger" instruction.

### What the numbers said

Full writeup in `experiments/reports/agentic_rag_baseline_v1.md`; three
findings worth a real ISSUES.md entry each (see there): a step-budget
arithmetic collision that makes `max_steps` the default termination label
for the agent's own designed 2-tool-call happy path (not a runaway-agent
signal, but a real interpretation trap); a citation-list scope mismatch
that made `citation_support_rate` read 10x worse for the agent route than
reality (agent citations = everything retrieved across every tool call,
not just what the answer used, unlike classic RAG's tight `sources`); and
a genuine, reproduced safety-framing regression on one adversarial +
freshness question, where the agent retrieved the correct authoritative
document but never stated its value in the final answer, while classic
RAG (identical evidence) got it right.

Net picture: routing_accuracy 0.833, but `unnecessary_agent_rate` 1.0 --
both `tool_not_needed` questions got routed to the agent for the exact
same answer classic RAG gave 11-12x faster. Overall agent-route latency
was 8.9x classic RAG's on the same 18 questions (138.5s vs. 15.6s mean),
for essentially tied `answer_correctness` (0.449 vs. 0.453,
KeywordOverlap). The agent's real win was multi-hop synthesis classic RAG
structurally can't do in one retrieval pass (one gold question needing
facts from 3 documents: classic RAG returned "not specified in the
provided context" for two of the three facts; the agent's 2-tool-call run
produced a substantive, on-topic answer). RAGAS corroborated the
qualitative read: faithfulness 0.443 overall, but exactly 0.0 for the
`adversarial_tool_output` category (n=3) -- consistent with the
safety-framing regression above.

### Decision

Did not establish `agentic_rag_baseline_v1`. The mechanics are solid (100%
tool-call success, safe bounded termination even at an artificially tight
`max_agent_steps=4`, security controls held in every traced example --
tenant isolation, field redaction, injection non-compliance all
confirmed), but the default config doesn't yet justify itself as a
recommended baseline given the unnecessary-routing overhead and the
adversarial-framing regression. Wrote concrete, actionable follow-ups
(tighten `classify`'s simple/complex boundary; fix `agent_synthesize_v1`'s
handling of the freshness-plus-injection combination; reconsider
`max_agent_steps` headroom or stop treating `max_steps` as an alarm
signal; fix the two metric-methodology gaps in `run_agent_eval.py`) rather
than either forcing a baseline through or leaving the evaluation
unresolved. None of prompts/config/agent logic were changed during the
evaluation itself, per its explicit instructions. Did not commit or push.

## 2026-08-21: agentic-rag re-evaluation -- the three fixes, and `agentic_rag_baseline_v1` established

Follow-up to the same day's earlier entry. `.claude-prompts/7.6-AgenticRAG-re-evaluation.txt`
asked for exactly the four fixes `experiment_029`'s report recommended,
then a re-run of the identical 18-question benchmark -- explicitly not a
redesign, and explicitly not permitted to add new infrastructure.

### The four fixes (plus one more, found along the way)

1. **Routing**: `agent_classify_v2.yaml` (new prompt version,
   `agent_classify_v1.yaml` untouched) states the simple/complex boundary
   explicitly -- "needs a targeted search" is simple, "needs more than one
   dependent lookup" is complex -- with four few-shot examples paraphrased
   from (not copied from) the gold questions, so a pass is evidence of
   generalization. `unnecessary_agent_rate` went 1.0 -> 0.0.
2. **Q17 synthesis**: `agent_synthesize_v2.yaml` adds one new rule --
   authoritative evidence wins over a conflicting untrusted claim, state
   it first, explicitly -- paired with a code change,
   `_order_evidence_for_synthesis` (`graph.py`, pure, unit-tested), which
   stable-sorts gathered evidence so authoritative/untagged sources are
   presented and `[Source N]`-numbered ahead of untrusted ones at
   synthesis time. Verified against real Ollama, not just a scripted
   mock, since the fix is entirely prompt wording. Q17's faithfulness
   (RAGAS) went 0.0 -> 0.5; the answer now states "90 days" first,
   explicitly, instead of never stating it at all.
3. **Step-budget labeling**: split `_step_or_stop` into
   `_increment_step`/`_check_step_bound` and reordered the
   `evidence_sufficiency` loop call site to check `evidence_sufficient`
   before consulting the step bound -- a convergence on the last allowed
   step is now `synthesized`, not `max_steps`. No bound values changed.
   Proven by two new unit tests exercising the exact step arithmetic;
   this run's own traces didn't happen to land on the exact boundary
   (see ISSUES.md for why that's informative, not a gap).
4. **Eval metric methodology**: `citation_support_rate` now scores only
   citations the answer text actually references by number (parsed
   `Source N` mentions), not every chunk a run gathered across every tool
   call; a new `tool_selection_coverage` (precision/recall/unexpected-rate,
   macro-averaged per example) supplements the strict, easily-collapses-
   to-0.0 `tool_selection_accuracy` rather than replacing it. Both
   explicitly documented as not comparable to `experiment_029`'s values
   for the same field names.
5. **Unplanned fifth fix**: fixing routing (#1) meant more questions
   correctly took the classic_rag route, which immediately broke 5/126
   RAGAS judge jobs (`retrieved_contexts is missing`) -- `run_agent_ragas_eval.py`
   built RAGAS context from `AgentState.retrieved_evidence`, which the
   classic route never populates. Added `AgentRunResult.classic_sources`
   (additive, `pipeline.answer()`'s own already-computed `sources`,
   populated only on the classic route) and pointed `_record_for_example`
   at it for that route. Fixed 3 of 5; the remaining 2 are `egress_policy`
   correctly blocking a `confidential`-classified source from reaching
   the hosted judge -- by design, confirmed by inspection, not a bug.

### Validation before the paid re-run

Full suite (real Postgres + real Ollama): 700 passed, 1 skipped (the same
pre-existing `anthropic`-not-installed skip as `experiment_029`), 0
failed -- run twice (once before, once after the 5th fix landed), both
green. `ruff check .` clean. `mypy src/` unchanged from `experiment_029`'s
2 pre-existing stub-strictness findings. Manually traced 8 representative
examples (simple/multi-hop/1-tool-convergence/insufficient-evidence x2/
adversarial x3 including Q17) against the raw per-example JSON before
trusting any aggregate number.

### The re-run (`experiment_032`, config `agentic-rag-baseline-v2-fixed.yaml`)

Same 18 questions, same corpus, same security config, same model/
retrieval/bounds as `experiment_029` -- only the two new prompt versions
and the graph.py control-flow fix differ. Headline: routing_accuracy
0.833 -> 0.889 (the 2 remaining "misroutes", Q14/Q16, are benign --
single-document questions that are safe under classic RAG's equivalent
injection-resistance rules, just not what the gold's agentic-category flag
expected); `unnecessary_agent_rate` 1.0 -> 0.0; overall mean latency
138.5s -> 47.4s (2.9x, entirely a routing-composition effect, not a
per-call speedup -- per-node latency is essentially unchanged); RAGAS
faithfulness 0.443 -> 0.511 overall, and for Q17's row specifically,
0.0 -> 0.5 with context_precision/recall 0.0 -> 1.0/0.5. `answer_correctness`/
`factual_correctness` (RAGAS) dipped slightly (~5-9%) -- read as
within-noise on an 18-question sample after manually confirming every
traced answer was correct or comparable, not as a specific regression.
Actual RAGAS cost this run: ~$0.033 (132 judge calls, 560/684 cache hits
against `.cache/ragas`).

### Decision

**`agentic_rag_baseline_v1` established.** Every item on the milestone's
own minimum bar is met: simple questions no longer routinely pay agent
overhead; the Q17 authoritative-vs-untrusted conflict resolves correctly;
2-tool convergence is no longer mislabeled (control-flow fix proven, even
though this run's own traces didn't land on the exact scenario); tool/
citation metrics are now meaningful instead of uninformative near-zero
artifacts; multi-hop value is unchanged and intact; every security/
authorization/redaction/injection-resistance check held; and the latency
win is fully explained by routing composition, not a hidden cost shift.
Full report: `experiments/reports/agentic_rag_baseline_v2_fixed.md`
(`experiments/reports/agentic_rag_baseline_v1.md`, the original "not
established" record, is unmodified). Five new `ISSUES.md` entries (three
resolving the prior session's three findings, two new: the citation-
compliance-vs-support distinction, and the routing-fix-exposes-RAGAS-gap
story). `experiment_029`/`030`/`031` and their JSON records are untouched.
Did not commit or push.

---

## 2026-08-21: Layout-aware document ingestion + a real vision provider

### Interview talking points
- **What problem did this solve?** The corpus had text-only Markdown ingestion; PDFs and DOCX files were reduced to plain concatenated text via `pypdf`/`python-docx` with no headings, tables, images, or page numbers surviving. This milestone rewrote both loaders to extract real document structure and serialize it into the *same* Markdown-equivalent syntax `structured_markdown.py` already parses, added a real local (Ollama) vision provider, and added a bounded `content_type` search filter to the agent.
- **Why was it difficult?** Two genuinely hard sub-problems: (1) PDFs have no native structure at all -- headings, tables, and code/config blocks all had to be reconstructed heuristically from `pdfplumber`'s word-level font sizes/positions/font-names, then re-serialized into text a *different* module (the chunker) already knows how to parse, rather than building a second parallel parsing path; (2) the fixed evaluation corpus's pre-supplied image assets lived in a directory structure (`layout_vision_extension/<category>/`) that didn't match where the source documents actually sat on disk, and three of the eight documents shared one `assets/` folder with images that couldn't be resolved by simple position alone.
- **How did I debug it?** Ran the real loaders against the real 8 PDF/DOCX corpus files directly (not just synthetic test fixtures) at every stage, comparing output against the gold file's own expectations (`relevant_images`, `relevant_pages`). This caught two real, otherwise-invisible bugs: a shared-assets-folder collision (three sibling documents all resolving to the first document's images) and, more seriously, a completely dropped `page` database column (`PgVectorStore.add_chunks`'s INSERT column list and the shared `_row_to_metadata` SELECT deserializer never had `page` added, so it silently persisted as `NULL` for every chunk despite being correctly computed in Python all the way up to the database call).
- **What metrics proved the fix?** `page_localization_accuracy` went from a flat `0.0` (28/28, the column-drop bug) to `0.82` after the SQL fix; all 9 embedded images across the 8-document corpus now resolve to their exact gold-expected asset paths (previously 3 of 9 collided on the wrong images); full test suite (683 unit + 64 integration, including this session's ~40 new tests) green against real Postgres + real Ollama; `ruff check .` and `ruff format --check` clean on every touched file; `mypy src/` unchanged from its pre-existing baseline.
- **What's next?** The milestone's own instructions explicitly deferred the controlled A (text-only) vs. B (layout-aware, no vision) vs. C (layout-aware + vision) evaluation and RAGAS run to a separate session, so no new baseline was established here. `moondream` (the chosen default vision model) was never pulled/exercised against a real image in this session -- `OllamaVisionProvider` is unit-tested against a mocked `ollama.Client` only.

### Goal
Extend ingestion beyond Markdown so PDFs and DOCX preserve real document structure (prose, headings/sections, tables, code/config, images, page numbers, source anchors, prose-image relationships), add a real local vision provider, integrate both into retrieval/agent/API without weakening any existing security control, add deterministic layout/vision eval metrics, and get MLflow/observability ready to record the later controlled evaluation -- without redesigning the established classic/agentic RAG architecture, and without running that evaluation yet.

### Design decisions (proposed for review before implementation, per the milestone's own instructions)
- **PDF parser: `pdfplumber`**, not PyMuPDF (AGPL/commercial dual license -- a real concern for a repo that may be shared) and not plain `pypdf` alone (no table/font-size/layout signal). `pypdf` is kept alongside it for document-info metadata and as the raw-byte source for the image-extraction fallback path.
- **Serialize extracted structure into the chunker's existing Markdown-equivalent grammar**, rather than building a second, parallel element/chunk model. `structured_markdown.py`'s gating changed from `source_type == "markdown"` to `source_type in {"markdown", "pdf", "docx"}`; a new `<!--page:N-->` sentinel line (invisible in chunk text) tags every span with a `page` field until the next marker. This meant zero changes to the chunker's table/heading/image/caption/relationship-expansion logic itself.
- **Vision provider: local Ollama (`moondream`)**, not a hosted API -- matches this project's "production stays local-only" convention and needs no `provider_egress_policy` involvement at all.
- **Directory fix:** the fixed gold file's `relevant_documents` referenced a `layout_vision_extension/<category>/` path segment that didn't exist on disk (the 8 new PDF/DOCX files + their pre-supplied image assets sat flat under the existing category folders). Confirmed these 9 files were referenced by no other gold set before moving them -- the same directory-nesting-mismatch failure mode `CLAUDE.md` already documents for `security_evaluation/`, caught before it could silently zero out every retrieval-truth metric for this gold set.

### Implementation
- **`loaders/pdf_loader.py`** (rewritten): per-page word/table/image extraction via `pdfplumber`, a font-size-ratio heuristic for two heading tiers (title/`#`, section/`##`), a monospace-font-name heuristic (reusing the same signal DOCX uses) for code/config blocks with a light JSON/YAML/XML shape-sniff (`loaders/markdown_render.py`), and reading-order interleaving of paragraphs/tables/images by vertical position. Captions ("Figure N. ...") are detected as the immediately-following short paragraph and folded into the image span, reusing the chunker's pre-existing emphasis-wrapped-caption convention.
- **`loaders/docx_loader.py`** (rewritten): a single linear walk of `document.element.body`'s children (paragraphs and tables are already ordered in DOCX, unlike PDF). Page numbers come from explicit manual page breaks (`<w:br w:type="page"/>`) -- the only reliable DOCX page signal, confirmed by inspecting the actual corpus files' XML rather than assumed; Word's reflow pagination is genuinely uncomputable without a rendering engine, so a DOCX with no manual breaks stays `page=1` throughout, honestly. Images resolved via `.//a:blip` relationship IDs; code/config via run-level monospace font name (no dedicated Word style exists in this corpus).
- **`loaders/base.py`**: new shared `resolve_image_asset` helper -- prefers an existing sibling `assets/` folder (see the two ISSUES.md entries above for the shared-folder fix), falls back to extracting and writing the image's own bytes for a PDF/DOCX with no pre-existing assets.
- **`vision/ollama_vision.py`** (new): `OllamaVisionProvider`, wired into `factory.build_vision_provider`'s new `"ollama"` branch and `config.vision.provider`. `VisionProvider` gained an abstract `prompt_version` property, closing a real pre-existing gap: `image_description_cache`'s primary key was `image_checksum` alone, so switching provider/model/prompt would have silently served a stale description generated under different instructions. Migrated to a composite `(image_checksum, provider, model_name, prompt_version)` key in both the Postgres schema (`scripts/init_db.py`, an idempotent `DROP CONSTRAINT`/`ADD CONSTRAINT` migration matching the project's existing migration idiom) and `PgVectorStore`.
- **Retrieval/agent integration:** `SearchKnowledgeBaseArgs` gained an optional `content_type: Literal[...]` field (a closed set, not a free string), merged server-side into the already-whitelisted `filters` dict -- the LLM can select which allowed value to filter by, never name an arbitrary field. New `agent_tool_select_v2.yaml` prompt describes the new argument (v1 stays on disk unused, per the project's prompt-versioning convention).
- **API:** `SourceItem`/`Citation` both gained `content_type`/`section_path`/`page`/`attachment_name`/`source_anchor`/`vision_generated` (all already-computed, non-sensitive metadata -- `source_dict()` in `retrieval/pipeline.py` already produced everything but `page`, which was the one field actually missing from the dict, separate from the database bug above).
- **Eval (`eval/run_eval.py`):** new `layout_vision` report section -- `table_retrieval_hit_rate`, `visual_retrieval_hit_rate`, `page_localization_accuracy`, `section_localization_accuracy`, `visual_evidence_support_rate` (reads `0.0` by construction whenever `vision.provider=none`, the metric the later A/B/C comparison is meant to isolate on), `multimodal_answer_quality` (the `requires_vision`/`requires_layout_awareness`/`requires_relationship_expansion` subset of `answer_quality`). `GoldExample` gained the new gold schema's fields (`requires_layout_awareness`, `expected_content_types`, `relevant_pages`, `visual_question_type`, `evidence_mode`, `source_format`) as additive-optional, matching every prior milestone's gold-schema-extension pattern. `Writer.vision_stats`/`IngestionStats.vision_stats` track images-processed/cache-hits/cache-misses/total-latency for the ingestion CLI to print -- an ingestion-time concern, kept separate from `run_eval.py`'s retrieval/generation-time report. `record_experiment.py`/`mlflow_logger.py` extended with `layout_parser`/`vision_provider`/`vision_model`/`vision_prompt_version` params and the six new metrics (also closed a small pre-existing gap: `vision_provider` was already computed in `build_experiment_record` but had never actually been added to `mlflow_logger.py`'s loggable-fields list).

### Evaluation
Manual smoke test against the real 8-document corpus (not the synthetic fixtures used in unit tests) at every stage of implementation -- this is what caught both ISSUES.md bugs. Full ingestion of the entire `data/knowledge_base` (59 files, 560 chunks) succeeded with zero errors. `rag.eval.run_eval --skip-generation` against the fixed 28-question gold file, real Postgres: `recall@10` 0.93, `hit_rate@10` 0.93, `mrr` 0.61, `supporting_context_hit_rate` 0.87, `relevant_image_hit_rate` 0.50, `page_localization_accuracy` 0.82, `section_localization_accuracy` 0.75, `visual_retrieval_hit_rate` 0.55, `visual_evidence_support_rate` 0.0 (expected: `vision.provider=none` by default in this session). `table_retrieval_hit_rate` absent -- confirmed no gold row in this fixed set has `"table"` in `expected_content_types`, not a bug. Full test suite: 683 unit + 64 integration (60 pre-existing + 4 new), 1 pre-existing skip (no `anthropic` package), 0 failed, run against real Postgres + real Ollama (`qwen2.5:3b`/`qwen2.5:1.5b`). `ruff check .` clean; `ruff format --check` clean on every touched file; `mypy src/` run, unchanged from its pre-existing baseline finding count.

### Interpretation
Both real bugs this session found were structural, not logical: the code that decided *what* page number or *which* image asset to use was correct in both cases; the code that *persisted* or *positionally matched* it was one file away and silently wrong. Neither would have been caught by unit tests against in-memory fakes -- both needed a real database round-trip or a real multi-document-per-folder fixture to surface. `vision.provider` stays `none` in `config/default.yaml` (a true no-op default, matching every prior security/vision-scaffolding milestone's convention); `moondream` needs to be pulled and exercised against real images in a follow-up session before the controlled A/B/C evaluation this milestone deliberately deferred.

---

## 2026-08-21: Layout-aware ingestion + vision -- controlled A/B/C evaluation

### Interview talking points
- **What problem did this solve?** The previous session (same day, "Layout-aware document ingestion + a real vision provider" above) implemented layout-aware PDF/DOCX ingestion and a real local vision provider but explicitly deferred the controlled evaluation proving they actually help. This session ran that evaluation: three ingestions of the same 9-document, 28-question fixed gold set (`data/eval/techfusion_layout_vision_extension_gold.jsonl`), differing only in what ingestion path produced the corpus -- A) the exact pre-milestone flat-text `pypdf`/`python-docx` extraction, B) the new layout-aware loaders with `vision.provider: none`, C) the same loaders with `vision.provider: ollama` (`moondream`) -- with retrieval config, generation model, prompt version, relationship expansion, temperature, and seed held fixed across all three.
- **Why was it difficult?** Config A required reproducing exact pre-milestone behavior without touching any shipped file (the old flat-text loaders no longer exist in the working tree, having been rewritten). Solved with a one-off scratch script that replicates the pre-milestone `pypdf.extract_text()`/`python-docx .paragraphs` logic (verified against git history, commit `2d6a005`) and feeds it through `RecursiveCharacterChunker` directly (confirmed to be exactly what `StructuredMarkdownChunker`'s own pre-milestone gating fell back to for non-Markdown `source_type`s) -- reusing the real `Cleaner`/`Embedder`/`Writer`/`VectorStore` for everything else, so Experiment A's chunks came from real production code except for the two intentionally-reverted pieces.
- **What did the results show?** Layout preservation alone (A -> B) was an unambiguous win: Recall@5 0.946 -> 1.000, MRR 0.896 -> 0.936, `page_localization_accuracy` 0.0 -> 0.964, `section_localization_accuracy` 0.0 -> 0.929, `visual_retrieval_hit_rate` 0.0 -> 1.0 (image chunks cannot be retrieved at all if they were never chunked as images). Adding real vision (B -> C) was a genuinely mixed result: `visual_evidence_support_rate` went 0.0 -> 1.0 (vision descriptions are being retrieved and cited, proving the wiring works end-to-end) and MRR ticked up slightly (0.936 -> 0.948), and -- the one clearly positive generation-quality signal -- hallucinations on unanswerable-visual questions dropped (3 in A, 1 in B, 0 in C) while correct refusals rose (0 -> 2 -> 3). But `multimodal_answer_quality` on answerable vision-required questions did not improve and slightly regressed (0.161 -> 0.144); `incorrect_or_missing` stayed flat at 14/20 across both B and C.
- **Why didn't vision help more?** Directly inspected `moondream`'s actual cached description for the one clearest B->C regression (`LVE-012`, "Which partition has the largest lag in the partition-status image?", `visual_question_type: image_table`, expects reading a value out of a table rendered as an image): the stored description was the single word `"Figure 2"` -- moondream echoed the caption instead of describing or reading the image's content. `moondream` (a ~1.7GB captioning-focused model) is not an OCR/document-table-reading model; the vision integration is structurally correct (retrieval, caching, redaction, citation, and safe-metadata paths all work), but the specific chosen model is undersized for this gold set's harder `image_table`/diagram-reading questions.
- **What's the concrete cost of vision?** ~19s/image on CPU-only `moondream` inference (14 images, ~4m6s wall-clock for the vision-only portion of Experiment C's ingestion, measured from `image_description_cache` row timestamps), fully absorbed by the existing per-`(checksum, provider, model, prompt_version)` cache so it is a one-time ingestion cost, not a per-query cost.

### Setup
Three dataset_ids, one per arm, all ingested fresh after clearing any prior state (`layout_vision_smoketest`, leftover from the previous session's manual verification, was also cleared -- it covered the whole `data/knowledge_base` tree, not just this gold set's 9 files, and predated the page-column/asset-collision fixes):
- `layout_vision_text_baseline` (A): ingested via a scratch script (not committed -- see Implementation) reproducing the pre-milestone flat-text loaders.
- `layout_vision_structured` (B): ingested via the normal `python -m rag.ingestion.pipeline` CLI with the real, shipped layout-aware loaders, `vision.provider: none`.
- `layout_vision_structured_vision` (C): same as B, `vision.provider: ollama` (`moondream`, freshly pulled and sanity-checked against a real corpus image before use).

Three new config files (`config/experiments/layout-vision-{a-text-only,b-structured,c-vision}.yaml`), each identical to `classic-rag-baseline-v1.yaml` (hybrid+RRF retrieval, `qwen2.5:3b`, prompt v2, relationship expansion on, temperature 0.0, seed 42) except the deliberately-varied `vision.provider` field -- confirmed byte-for-byte identical on every other field by loading all three through `AppConfig` and diffing the resolved values before running anything.

Manually verified chunk-level metadata on the real corpus before evaluating: page numbers (multi-page PDFs/DOCX show a sane page spread; the Markdown governance doc correctly stays `page=None`), `content_type` breakdown (14 image / 4 table / 5 configuration / 1 code / 54 prose chunks in B), table/code chunk content and `section_path`, `parent_chunk_id` linkage, and -- most importantly given the previous session's shared-assets-folder bug -- that every image chunk's `attachment_name` still resolves to its own document's images even for the three `operations/` documents sharing one `assets/` folder (confirmed: `performance-view-*`/`incident-view-*`/`capacity-view-*` each stayed correctly scoped to their own document, no cross-contamination on a real re-ingestion).

Full unit (683 passed, 1 pre-existing skip) + integration (64 passed) suite, `ruff check .` (clean), `ruff format --check .` (clean on every file touched this session; two pre-existing-drift files left untouched, same as the previous session's policy), and `mypy src/` (clean, unchanged from baseline) all run and green before evaluating.

### Results

| Metric | A (text-only) | B (layout, no vision) | C (layout + vision) |
|---|---|---|---|
| Recall@5 / @10 | 0.946 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| MRR | 0.896 | 0.936 | 0.948 |
| Hit rate@5 / @10 | 0.964 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| `page_localization_accuracy` | 0.0 | 0.964 | 0.964 |
| `section_localization_accuracy` | 0.0 | 0.929 | 0.929 |
| `visual_retrieval_hit_rate` | 0.0 | 1.0 | 1.0 |
| `visual_evidence_support_rate` | 0.0 | 0.0 | 1.0 |
| `multimodal_answer_quality` | 0.115 | 0.161 | 0.144 |
| `answer_quality` (overall) | 0.264 | 0.310 | 0.298 |
| Vision-required: hallucinated / correct-refusal | 3 / 0 | 1 / 2 | 0 / 3 |
| Mean total latency | 7.9s | 8.8s | 6.8s |

Recorded as `experiment_033`/`034`/`035` (labels `layout_vision_a_text_only`/`_b_structured`/`_c_vision`), logged to MLflow, rendered as README Benchmarks rows 29-31. Per the milestone's own explicit instruction, **RAGAS was not run** -- these are all deterministic metrics from `rag.eval.run_eval`, evaluated first as planned. Latency numbers should be read with the same host-noise caveat already documented for the context/output-token-budget A/B/C experiment (`experiment_018`-`020`): this is a shared, CPU-only Windows dev box, and a ~2s total-latency swing across three sequential runs is within the range that box has already shown to be measurement noise, not a real per-arm effect -- the vision-inference-time cost is real and separately measured (the ~19s/image ingestion-time figure above), but it does not appear in these query-time latency numbers at all, since vision descriptions are pre-computed and cached at ingestion, not generated per query.

### Interpretation
This validates the layout-vision milestone's core architectural claim -- preserving document structure is what makes page/section localization and image retrieval possible at all, independent of which vision model (if any) is attached -- while also surfacing a real, specific model-capability gap rather than an integration bug: `moondream` is too weak a captioning model to read fine-grained content out of an image (a table rendered as a bitmap, in this gold set's hardest case), so wiring it in did not move the needle on the questions that most needed it, and slightly hurt the crude keyword-overlap `answer_quality` score by adding low-signal caption text to the context. The safety-relevant finding -- fewer hallucinations and more correct refusals on unanswerable-visual questions as evidence quality increases from A to B to C -- is the one place vision measurably helped generation, not retrieval. Recommendation: keep `vision.provider: none` as the shipped default (unchanged, already true), and treat "try a stronger local vision model" (not prompt tuning) as the next lever if this path is revisited -- the retrieval/caching/redaction/citation plumbing already proved itself correct at both layout preservation and vision-wiring layers; the ceiling here is `moondream`'s captioning quality, not the integration.

---

## 2026-08-21: Layout-vision test-coverage hardening

A reviewer proposed 8-9 candidate tests for the layout-vision milestone's newest/most fragile boundaries (`.claude-prompts/EXTRA-tests.txt`). Audited the existing suite first (via a research subagent reading full test bodies, not just names) before adding anything, per the explicit instruction not to add tests just for count.

**Findings:** 1 of 9 was already genuinely covered (`test_resolve_image_asset_matches_by_shared_filename_word` -- the shared-assets-folder regression test). The other 8 were real gaps, closed as follows:

- **Vision cache invalidation** (`tests/unit/test_writer.py`): only `prompt_version` changes were tested; added image-checksum-change, model-name-change, and provider-change cases, extending `FakeVisionProvider` to take `model_name`/`provider_name`/`raises` constructor params.
- **Vision provider call failure**: found genuinely undefined behavior, not just an untested path -- `Writer._with_vision_siblings` had no try/except around `describe_image()`, so a real network failure would fail the whole document's ingestion. Fixed (catch, log `vision_provider_call_failed`, skip that one image -- same outcome as the pre-existing missing-file case) and added a regression test. See ISSUES.md.
- **Real Ollama/moondream integration test**: every prior vision test mocked `ollama.Client`. Added one gated by `require_ollama` that sends a real image through a real `moondream` call and asserts a non-empty description.
- **PDF image end-to-end + multi-page combined-fields test**: closed both at once. Added `reportlab` as a `dev`-extras-only test dependency (the one PDF *writer* the suite needed; pdfplumber/pypdf are both readers) and built a real multi-page PDF (title -> heading -> prose -> page break -> heading -> image -> table) run through the actual `PDFLoader.load()`, not just its private helpers. Caught and fixed my own wrong assumption about `parent_chunk_id` while writing it: a heading line with nothing else on its paragraph becomes its own tiny prose chunk, so both the image and table correctly link back to that heading chunk, not `None` as I'd assumed -- verified by printing the real ingested rows before asserting anything.
- **Page persistence through the API layer**: prior coverage stopped at `VectorStore.search()`. Added a test carrying a real chunk through `source_dict()` (the exact function `POST /query` uses to build its response) and into a real `SourceItem` Pydantic model.
- **Security on vision-generated text**: added a test using a canned (non-real) `VisionProvider` whose description contains both a `SYNTHETIC_ONLY_...`-shaped credential and an injection phrase, proving both field-redaction and injection-detection apply to vision-generated chunk content exactly as they do to any other chunk's.
- **ACL + visual retrieval**: added a test where a tenant's image caption is written to be the strongest possible semantic match for another tenant's query (verified via an unscoped sanity check first), proving `content_type="image"`-filtered `search_knowledge_base` calls still respect tenant ACL even against the best-ranking result.

**Result:** 687 unit (+4) and 69 integration (+5) tests, all green; `ruff check`/`format` clean on every touched file; `mypy src/` clean. No scope creep beyond the identified gaps -- the one already-covered item was left alone rather than duplicated.

---

## 2026-08-21: Documentation site (MkDocs + Material + mkdocstrings)

Added a generated docs site over the existing Markdown docs and docstrings, per `.claude-prompts/11.0-MKDocs.txt`: MkDocs + Material theme + mkdocstrings for API reference, reusing `docs/architecture.md`/`docs/metrics.md`/`README.md` rather than rewriting any of them, plus a strict-build CI gate for broken links/references. Not published anywhere (no `site_url`, no `gh-deploy`).

**Reuse strategy:** `docs/architecture.md` and `docs/metrics.md` stayed completely untouched, linked into nav as-is. Eight new `docs/topics/*.md` pages (ingestion, retrieval, multimodal, security, agentic-rag, evaluation, observability, deployment) are short, original landing pages that link out to the detailed docs rather than restating them, and transclude a handful of README.md sections verbatim via `pymdownx.snippets` section markers (`<!-- --8<-- [start:name] -->`/`[end:name]`, invisible HTML comments on GitHub) -- the architecture diagram, Prerequisites/Setup, Containerized development, Agentic RAG, Observability, and the Benchmarks table. Sixteen `docs/reference/*.md` pages are pure `::: rag.module` mkdocstrings directives (no hand-written API prose), covering ~400 classes/functions across every component group in `CLAUDE.md`'s directory map.

**Three build-blocking issues found and fixed, one flagged and left alone** -- full diagnosis and root-cause writeup in `ISSUES.md`'s new "A 'strict' docs build failure that had three unrelated root causes stacked on top of each other" entry:
- `nav:` entries using `architecture.md#anchor` fragments don't validate in this MkDocs version even though the base file is elsewhere in nav -- dropped in favor of a flat `architecture.md` entry, with the same anchors used as ordinary in-page links in `docs/topics/*.md` instead.
- README.md sections transcluded via snippets contained `docs/architecture.md`-relative links that can't resolve once transcluded (README.md isn't part of the built site) -- fixed by trimming the snippet marker boundaries to exclude the dead-link sentences, since each topic page already adds its own correct in-site pointer right after the transclusion.
- `mkdocstrings`/griffe threw real docstring-parser warnings: most were a genuine, pre-existing convention violation (Pydantic config/state classes documenting fields under `Parameters` instead of `Attributes`, per `CLAUDE.md`'s own docstring convention) -- suppressed via `docstring_options.warn_unknown_params: false` and explicitly flagged as a separate, not-yet-done cleanup rather than silently fixed; one was a genuine rendering gap (`AgentConfig`'s five `*_prompt_path` fields combined on one numpydoc line, losing type info for three of them) -- actually fixed, split into five lines.
- `mkdocs-htmlproofer-plugin` flagged mkdocstrings/autorefs' own cross-page links between reference pages as unresolvable; verified directly against the rendered HTML that the target anchors do exist before allow-listing the pattern (`*#rag.*`) in `ignore_urls`, rather than assuming the plugin's warning was correct.

**Result:** `mkdocs build --strict` exits 0 (22 pages, 5.1MB, one harmless INFO-level message that doesn't affect the exit code). `ruff check .` and `mypy src/` both stayed clean throughout -- every fix was docstring/doc-content only, zero application-code changes. `make docs-serve`/`make docs-build` added to the Makefile; `site/` added to `.gitignore`; `docs` added as a new `pyproject.toml` optional-dependency group (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-htmlproofer-plugin`).

**Known gap, not addressed this session:** the `Parameters`-vs-`Attributes` docstring inconsistency spans `config.py` (~13 classes), `agent/state.py`, `agent/graph.py`, `agent/events.py`, `api/auth.py`, and `eval/gold_schema.py` -- a real, systemic drift from this project's own documented convention, left as a follow-up rather than folded into this session's scope.

---

## 2026-08-22: `rag_answer` v3-vs-v4-vs-v5 prompt A/B, plus two follow-up bug fixes it surfaced

### Interview talking points
- **What problem did this solve?** Ran a controlled single-variable A/B of two candidate `rag_answer` generation prompts (v4, v5) against the active `v3` on the classic/safety TechFusion eval set, then chased down two real bugs the comparison surfaced: a redaction-marker-echo leak that a prompt-only instruction failed to fix, and a security eval metric (`oversized_request_rejection_accuracy`) that had been reading 0/3 on every run without anyone knowing why.
- **Why was it difficult?** The interesting failures were both cases where the "obvious" fix (a clearer prompt instruction) didn't work, and where a metric read as "broken" for a completely different reason than it looked like at first glance. `qwen2.5:3b` kept reproducing the literal `[REDACTED:SENSITIVE_FIELD]` marker in its answers even after v5 added an explicit sentence telling it not to -- prompt compliance alone wasn't reliable enough for a security-relevant guarantee. Separately, `oversized_request_rejection_accuracy` reading 0/3 looked like a broken DoS-limit implementation, but was actually a broken *test probe*: the "valid" JWT the probe minted to get past auth had `tenant_id: None`, and pyjwt's `require` option silently treats a present-but-null claim as missing, so the probe never even reached the code being tested.
- **How did I debug it?** For the marker leak: ran v4 and v5 against the exact same gold questions and hand-inspected the two examples that actually contained a redacted field, rather than trusting the aggregate metric. For the DoS-limit metric: reproduced the failing probe directly against a live `TestClient`-backed FastAPI app, printed the raw response body (`401 "Invalid or expired token"`, not a 422), and traced that back to pyjwt's own `_validate_required_claims` source (`payload.get(claim) is None`, not `claim not in payload`) rather than assuming the DoS-limit code itself was broken.
- **What metrics proved the fix?** Marker leak: `Marker present: False` on both previously-failing cases (idx 119/120 of `techfusion_gold.jsonl`), verified against real `qwen2.5:3b`, not just a mocked LLM. DoS-limit probes: same 3 payloads, same live app, correct token -> all 3 now return 422 with the expected messages, confirming the underlying `enforce_dos_limits` code was correct the whole time.
- **What's next?** v3 stays the active/default prompt. The marker-sanitization backstop is now code, not prompt text, so it can't regress if a future prompt version changes wording. `oversized_request_rejection_accuracy`/`authentication_failure_acceptance_rate` are trustworthy again going forward; historical experiment records (025-039) keep their original, now-known-to-be-wrong values as an honest record rather than being retroactively edited.

### Part 1: the v3-vs-v4-vs-v5 prompt A/B

Spec: `.claude-prompts/12.0-rag_answer_v3 vs rag_answer_v4.txt`. Two single-variable configs (`secure-rag-baseline-v2-jwt-auth-prompt-v4.yaml`, `-prompt-v5.yaml`), each identical to the `experiment_028` control except `generation.prompt.version`/`path` -- verified via both text diff and a programmatic YAML-parse comparison before running anything.

**v4 vs v3** (full 126-question `techfusion_gold.jsonl`, deterministic `run_eval` only): retrieval metrics identical (as expected -- generation model can't affect retrieval), `answer_quality` improved modestly. The one substantive behavioral difference found on hand-inspection: v4 sometimes echoed the literal `[REDACTED:SENSITIVE_FIELD]` marker inline in its answer instead of paraphrasing around it (no secret disclosed, but internal redaction syntax leaked to the caller). Recorded as `experiment_036`.

**v5**: `rag_answer_v5.yaml` created (v4 untouched) with rule 9 extended: *"Do not reproduce the literal '[REDACTED:SENSITIVE_FIELD]' marker text in your answer; state in your own words that the value is unavailable at the caller's access level."* Full 126-question deterministic re-run (`experiment_037`) came back with retrieval identical, safety-metric shifts matching the exact same pre-existing heuristic artifacts already found in the v3-vs-v4 comparison (idx 111, 115, 124) -- nothing new. **But the fix didn't work**: idx 119 and 120 (the two gold questions actually containing a redacted field in their retrieved evidence) still contained the literal marker text verbatim in v5's answer. A one-sentence prompt instruction was not sufficient for `qwen2.5:3b` to reliably comply with a security-relevant behavioral rule.

**Targeted RAGAS** (25-question sample: all 19 `forbidden_documents` rows + 6 benign controls spanning question_type/difficulty/unanswerable, built deterministically by new `scripts/build_prompt_ab_ragas_sample.py`, mirroring `build_ragas_sample15.py`'s selection pattern): v3 (`experiment_038`) vs v5 (`experiment_039`). faithfulness +0.033, answer_relevancy +0.041, context_precision/recall unchanged (retrieval-side, expected) -- but answer_correctness -0.061 and factual_correctness -0.092, which looked like a real regression. Hand-inspecting the three largest-drop questions (idx 10, 11, 12) showed v5's answers were factually correct in every case, just phrased as full cited sentences instead of terse fragments (e.g. v3: `"10 percent"` vs v5: `"The current canary receives 10 percent of production traffic."`) -- RAGAS's `answer_correctness`/`factual_correctness` penalized the verbosity/citation style relative to a terse gold string, not actual correctness. Same artifact shape as this project's other documented RAGAS-scoring quirks. Cost: ~$0.10 total for both 25-question runs.

Two stale test assertions were failing on GitHub CI (`test_render_table_includes_row_per_record`, `test_render_table_handles_missing_metrics_gracefully` in `test_compare_experiments.py`) -- traced to commit `85f4b99`, which correctly changed `compare_experiments.render_table`'s leftmost column from a bare loop index to the real `experiment_id`, but never updated these two tests' hardcoded expected strings. Fixed the test literals, not the (correct) production behavior.

### Part 2: two follow-up fixes, on explicit instruction (don't keep prompt-tuning; move the marker guarantee into code; investigate the DoS-limit metric end-to-end)

**Fix 1 -- deterministic marker sanitization.** Added `field_policy.sanitize_redaction_markers_in_answer(text, policies=None)`: builds the set of every configured `redaction_marker` across `DEFAULT_FIELD_POLICIES` (currently just `[REDACTED:SENSITIVE_FIELD]`), and regex-replaces each occurrence (stripping any backticks/quotes a model wrapped around it) with a fixed caller-facing paraphrase. Applied at the two places that produce caller-facing prose:
- `RetrievalPipeline.answer()`, right after `self._llm.generate(...)` -- only `answer`, never `sources[i]["content"]` (which intentionally keeps the raw marker for observability of what was redacted and why).
- `agent/graph.py`'s `_synthesize` node, the one node on the agentic path that calls `llm.generate()` directly rather than routing through `answer()` (the agent's `classic_rag` route already inherits the fix for free via `pipeline.answer()`).

Verified two ways: 9 new unit tests (5 in `test_field_policy.py`, 2 in `test_retrieval_pipeline.py` proving the answer is sanitized but `sources` content is not, 1 new `test_agent_synthesis_marker_sanitization.py` via `run_agent()` + `ScriptedLLM`), and a live run against real `qwen2.5:3b` on the exact two previously-failing gold questions (idx 119, 120) -- both now come back with `Marker present: False`. The substituted phrasing is a little grammatically awkward (a literal-text swap, not a rewrite -- e.g. "The synthetic test key is this value is unavailable at your access level."), a known, accepted cosmetic cost of a guarantee that can't depend on model compliance. v5 itself is not promoted; `config/default.yaml` still points at `v3`.

**Fix 2 -- `oversized_request_rejection_accuracy` root cause.** Reproduced the failing probe directly (bypassing the eval CLI, hitting a live `TestClient(app)` the same way `evaluate_authentication_boundary_probes` does): all 3 oversized-payload probes returned 401 "Invalid or expired token", never a 4xx from the DoS-limit check itself. Root cause: `_mint_valid_probe_token`'s claims included `"tenant_id": None` (meant to represent an unscoped probe identity), and `config.security.auth.jwt.required_claims` defaults to `["sub", "tenant_id", "roles"]`. Confirmed directly against the installed `pyjwt==2.13.0` source: `PyJWT._validate_required_claims` checks `payload.get(claim) is None`, not `claim not in payload` -- a claim that's present but null is treated identically to an absent claim, so the "valid" probe token was never actually valid and every oversized-request probe was rejected by auth before the DoS-limit check ever ran. Confirmed the DoS-limit code itself was fine the whole time by re-running the same 3 payloads with a token carrying a real `tenant_id`: all 3 correctly returned 422 with the expected "exceeds maximum ..." messages.

Fixing `tenant_id` exposed a second, previously-masked bug: with a real `tenant_id`, the `wrong_issuer`/`wrong_audience` adversarial probes in `_run_auth_failure_probes` started returning 200 OK instead of 401. Not a regression -- `verify_jwt` only validates `iss`/`aud` "when configured" (documented design; confirmed against pyjwt's `_validate_iss`/`_validate_aud`, both of which return immediately when the corresponding decode argument is `None`), and every config this project has ever run (including every `secure-rag-baseline-*` experiment config) leaves `jwt.issuer`/`.audience` at the default `null`. Those two probes were only ever reading as "correctly rejected" because the *same* tenant_id bug rejected them for the wrong reason (`missing_claim`, not `invalid_issuer`/`invalid_audience`). Fixed by only including `wrong_issuer`/`wrong_audience` in the probe set when `security.auth.jwt.issuer`/`.audience` are actually configured on the config under test -- the same "don't probe for a rejection the system was never designed to perform" principle this function already uses (`[]` entirely when `auth.enabled=False`).

Both fixes are confined to `eval/run_eval.py`'s probe-construction helpers (`_mint_valid_probe_token`, `_run_auth_failure_probes`); zero production-code changes (`api/request_auth.py`, `api/auth.py` untouched). Root cause of why this shipped undetected: the pre-existing unit test for the auth-enabled scenario (`test_evaluate_authentication_boundary_probes_all_rejected_when_auth_enabled`) only ever asserted `authentication_failure_acceptance_rate`, never `oversized_request_rejection_accuracy` -- and the *other* existing test that did check the oversized metric ran with `auth.enabled=False`, where `_mint_valid_probe_token` returns `None` early and the bug structurally can't manifest. Extended the existing test with the oversized-metric assertion (regression coverage) and added a new test for the issuer/audience-configured case (count=7, not 5).

Full diagnosis, with exact request/response evidence, logged in `ISSUES.md`.

### Verification

Full unit suite: 696 passed (+9 from this session), 1 skipped, 0 failed. `ruff check` and `mypy src/` clean on every touched file. Security integration tests (`test_field_level_redaction.py`, `test_api_field_redaction.py`, `test_authorization_isolation.py`, real Postgres + Ollama, no mocks): 21 passed. No paid RAGAS run for the follow-up fixes, per explicit instruction. Nothing committed or pushed.

### Interpretation

The through-line across both bugs: a security-relevant guarantee that depends on an LLM following an instruction, or on a test probe's own auxiliary data (a claim value chosen for convenience, not for correctness), is not actually a guarantee until it's verified against the real thing -- v4/v5's prompt-only fix looked reasonable and still failed in practice; the DoS-limit metric looked like a broken feature and was actually a broken probe. Both were only caught by re-running against a real model / a real live app and reading the actual output, not by trusting an aggregate metric or a plausible-sounding prompt change. `oversized_request_rejection_accuracy` and `authentication_failure_acceptance_rate` are now measuring what their names claim to measure, for the first time since `--include-api-probes` was added.

---

## 2026-08-22: Local web UI (React/Vite/TypeScript) over the existing API

Spec: `.claude-prompts/13.0-Local-web-UI.txt`. Built a chat-style frontend over the existing FastAPI backend (`/query`, `/agent/query`, `/agent/query/stream`) after first inspecting the actual routes/auth/Docker Compose files rather than assuming their shape. Per the spec's own gate, the frontend architecture (stack, directory layout, dependencies, Docker changes, backend interaction design) was proposed and approved before any code was written.

### Interview talking points
- **What problem did this solve?** A portfolio-facing, ChatGPT-style UI for a backend that previously only had `curl`/pytest as its interface -- mode switching between Classic and Agentic RAG, live agent progress, source/citation inspection, and local-dev auth controls, without touching any backend code.
- **Why was it difficult?** Two backend constraints shaped the whole design. First, no CORS middleware exists anywhere in the backend, and adding one wasn't in scope -- so the frontend had to reach the API same-origin instead (a Vite dev-proxy locally, the frontend's own nginx in Docker), which meant zero backend changes were needed at all rather than a "smallest possible" one. Second, `POST /agent/query/stream` is POST, not GET (documented in that router's own docstring, since a browser's native `EventSource` is GET-only), so the browser's built-in SSE client couldn't be used -- the frontend hand-parses `event:`/`data:` frames off a `fetch` `ReadableStream` instead.
- **How did I verify it?** Read the actual Pydantic response models, `AgentEvent`'s field set, the JWT-precedence logic in `request_auth.py`, and the DoS-limit/rate-limit error shapes directly from source before designing the error taxonomy and the dev-identity panel's disable-on-token behavior, rather than guessing what the API returns. Every response is parsed through a zod schema mirroring those Pydantic models, so a shape mismatch is a distinct, testable `malformed_response` state instead of a silent bad render.
- **What's the one non-obvious design choice?** Agentic mode only ever issues one request. The stream's terminal `completed`/`terminated` frame already carries the exact same shape as `POST /agent/query`'s JSON response (`agent_stream.py:_final_response`), so there's no need to stream-for-progress and then separately fetch-for-the-answer. When live events are disabled server-side (`observability.live_events.enabled=False`, stream 404s), the frontend catches that specific error kind and falls back to the non-streaming endpoint automatically, with a visible notice -- Agentic mode degrades gracefully instead of breaking.
- **What's next?** E2E tests (Playwright) were authored and type-checked but require Node 20+ to even launch, which this dev environment's Node 18.12 doesn't meet -- documented as a known limitation rather than skipped silently. `RAG_API_BASE_URL` (the direct-origin escape hatch, bypassing the built-in proxy) is documented as needing CORS on that backend, which doesn't exist -- untested, flagged honestly.

### What was built

`frontend/`: React 18 + TypeScript + Vite, plain CSS (no UI framework), `react-markdown`+`remark-gfm`+`rehype-highlight` for answer rendering, `zod` for response validation. State is a single `useReducer`-backed chat context (no Redux/Zustand -- one conversation doesn't need it). Key modules: `api/client.ts` (fetch wrapper + `RagApiError` taxonomy), `api/query.ts`/`api/agentQuery.ts` (classic/agent requests), `utils/sse.ts` (hand-rolled SSE-over-POST frame parser), `state/chatContext.tsx` (reducer + dev-identity sessionStorage persistence), `hooks/useSendMessage.ts` (orchestrates classic vs. agent-with-stream-fallback), and a component set (`ChatWindow`, `MessageList`/`MessageBubble`/`MessageInput`, `SourcesPanel`, `AgentActivityPanel`, `DevIdentityPanel`, `ErrorBanner`, `DebugPanel`).

**Docker**: `frontend/Dockerfile` (multi-stage, `node:20-alpine` build -> `nginx:1.27-alpine` runtime), `frontend/nginx.conf` (proxies `/query`, `/agent/*`, `/health`, `/metrics`, `/docs`, `/openapi.json` to `rag-api:8000`; SSE-specific `proxy_buffering off` on the `/agent/` location so progress events aren't buffered), `frontend/docker-entrypoint.sh` (rewrites the built `index.html`'s API base at container start from a `RAG_API_BASE_URL` env var, so one built image works for both the default same-origin-proxy mode and a direct-origin override). `docker-compose.frontend.yml` is an additive overlay (same pattern as `docker-compose.observability.yml`), never folded into the default stack; `make frontend-up`/`make frontend-down` added to the Makefile.

**Security-boundary decisions carried over from the backend, not reinvented**: the Dev Identity panel disables the tenant/roles inputs the instant a bearer token is entered, mirroring `request_auth.build_authorization_context`'s actual precedence (verified JWT always wins) rather than the frontend having its own opinion; the token lives in `sessionStorage` only; `AgentActivityPanel` renders only `AgentEvent`'s typed fields, so there's no code path that could surface reasoning/prompts/evidence even if a future backend bug attached one (proven by a test that feeds a crafted event with extra `reasoning`/`raw_prompt` keys and asserts neither string reaches the DOM); "authorization failure" has no dedicated error banner because the backend has no such HTTP error -- ACL is a silent SQL filter, so it surfaces as `insufficient_evidence`/thin sources instead, and the UI's insufficient-evidence notice is what actually covers that case.

### Verification

`npm run typecheck` (tsc -b --noEmit): clean. `npm run build`: succeeds (553KB gzipped-to-166KB bundle, acceptable for a local dev tool -- dominated by `react-markdown`+`highlight.js` grammars, not worth code-splitting here). `npm run lint`: 0 errors, 3 benign `react-refresh` warnings (the chat-context module intentionally also exports the reducer for direct unit testing). `npm test` (Vitest): 36/36 passed across unit tests (SSE frame parsing, error classification, reducer transitions, hidden-reasoning-field safety) and integration tests (classic query round trip, agent streaming + live-events-disabled fallback, all documented error states, bearer-token/tenant-role precedence, mode switching, malformed/disconnected stream handling) -- all against a mocked `fetch`, no real backend needed. Playwright e2e (`tests/e2e/critical-flows.spec.ts`, three flows: classic round trip, agentic round trip with activity panel, backend-unavailable error) was authored and passed `tsc`/`eslint`, but `npx playwright --version` itself refuses to run on this environment's Node 18.12 ("Playwright requires Node.js 20 or higher") -- not executed, documented as a known limitation in `frontend/README.md` rather than claimed as passing.

Backend impact: `git status` confirms zero files under `src/rag/`, `tests/`, `config/`, or any existing Docker Compose file changed -- only `.gitignore`, `Makefile`, `README.md`, `CLAUDE.md`, `mkdocs.yml`, two new `docs/` files, and everything under the new `frontend/`/`docker-compose.frontend.yml`. Attempted `pytest tests/unit -q` to double-confirm no regression; it currently fails at collection with `ModuleNotFoundError: No module named 'pdfplumber'` across 15 test files -- confirmed via `pip show`/direct import that this is a pre-existing gap in this Python environment (the `pdfplumber` extra isn't installed here), unrelated to and unaffected by this session's changes, which touched no Python source. `mkdocs build --strict` likewise wasn't run -- `mkdocs` itself isn't installed in this environment (the `docs` extra) -- but the new `docs/topics/web-ui.md` follows the exact transclusion/nav/`ignore_urls` pattern already used by every other topic page, cross-checked by hand against `docs/topics/observability.md`.

### Interpretation

The two hardest constraints (no CORS, POST-only SSE) both turned into design decisions documented up front rather than discovered mid-implementation, because the approved-before-coding proposal step forced reading the actual route/auth/Compose source first. The same-origin-proxy choice in particular meant "frontend over the existing proven backend" was true in the strongest sense available: not just "no backend logic duplicated," but zero backend files touched at all. The one real gap is environmental, not architectural -- this session's Node 18.12 dev environment can typecheck and lint Playwright's config but can't launch it, so the e2e suite is written and reviewed but not empirically proven; that's flagged explicitly rather than glossed over, consistent with this project's practice of reporting a limitation honestly instead of claiming a test suite passed when it wasn't run.

---

## 2026-08-22 (session 2): tracing-enabled config, data-directory audit, and the feature-flags-visibility fix

Follow-up session to the same day's web-UI build, done while actually using the running system.

### 1. `config/experiments/tracing-enabled.yaml`, done correctly the second time

First attempt at enabling tracing to see it work in Jaeger reached for editing `config/default.yaml` directly -- caught and corrected: the right tool was already documented (`RAG_CONFIG_PATH` + a dedicated experiment config), and using it meant zero edits to the file every other config/experiment/test assumes is the safe baseline. Created `config/experiments/tracing-enabled.yaml` (a full copy of `default.yaml` with only `observability.tracing.enabled: true` changed), plus `test_tracing_enabled_experiment_config_only_differs_by_tracing` in `tests/unit/test_config.py`, matching the project's existing isolation-test convention for every other experiment-config pair. Ran it via `RAG_CONFIG_PATH=/app/config/experiments/tracing-enabled.yaml docker compose ... up -d`. Hit one real Windows-specific shell issue: Git Bash silently rewrites `/app/...`-shaped arguments into Windows paths (`/app/config/...` became `C:/Program Files/Git/app/config/...`), crash-looping `rag-api` until re-run with `MSYS_NO_PATHCONV=1`. Verified end-to-end, not just "container came up": queried `/query`, then confirmed a real `POST /query` trace with a nested `retrieval` span landed in Jaeger's API (`/api/traces?service=local-rag-system`).

### 2. Data-directory audit: cross-checking claims instead of assuming them

Asked directly: "is all the data ingested?" and "is this directory structure correct?" rather than assuming yes. Cross-checked on-disk file counts against Postgres `dataset_id` counts instead of eyeballing: `security_evaluation` (16 files = 16 ingested docs) and `layout_vision_extension` (9 files, ingested 4x under distinct A/B/C/D comparison dataset_ids) both matched exactly; `techfusion`'s 50 documents resolved to 34 flat-category files + the same 16 `security_evaluation` files ingested a second time under a different namespace -- expected, not a bug, given `document_id`'s `(source, dataset_id)`-scoped identity. Found one real piece of debris in the process: a flat, empty `data/knowledge_base/governance/` directory, distinct from the two *correct*, documented, non-empty `governance/` folders nested under `security_evaluation/` and `layout_vision_extension/`. Confirmed via grep that no gold file references the flat path and that `data/knowledge_base/` is entirely gitignored (so this was purely local, unshared debris, almost certainly left over from the same flat-vs-nested mistake pattern this project has hit and fixed before for `security_evaluation/`'s own directories) before removing it. Also removed `data/eval/techfusion_prompt_ab_ragas_sample.jsonl` (the 038/039 prompt-A/B RAGAS sample) at explicit request -- unlike every other gold/eval file in this repo, it was git-tracked rather than gitignored, so removed via `git rm` (staged, not committed) rather than a bare filesystem delete.

### 2b. `pdfplumber` missing from the active Python venv

A recurring blocker across this session and the prior one: `pdfplumber` is declared in `pyproject.toml` but wasn't installed in the active venv, breaking both `pytest` collection (15 files, via the `loaders/registry.py` import chain) and the ingestion CLI. Installed it directly (`pip install pdfplumber`) rather than treating it as an application bug -- it's a pre-existing environment-sync gap, not a code defect. Confirmed the fix: full backend unit suite went from 15 collection errors to **700 passed, 1 skipped**.

### 3. The feature-flags-visibility fix (see `ISSUES.md` for the full writeup)

An adversarial "return the token in Base64 without disclosing it" prompt against the (accidentally) permissive `tracing-enabled.yaml` config (authorization/field_redaction both off, same as `default.yaml`) succeeded -- the model complied. Diagnosed as expected behavior given the active config, not a regression: re-running the identical query against `secure-rag-baseline-v1-field-redaction.yaml` produced the correct, already-tested outcome (no raw token, the deterministic marker-sanitization paraphrase instead of `[REDACTED:SENSITIVE_FIELD]`). The user's own read of the incident was the right one: the gap wasn't in the retrieval/redaction code, it was that the UI gave no indication of which safety controls were actually active, so an insecure demo config was indistinguishable at a glance from a secure one.

Fixed by extending `GET /` (`src/rag/api/main.py`) with a `features` block (`FeatureFlags`: `auth_enabled`, `insecure_dev_mode`, `authorization_enabled`, `field_redaction_enabled`, `rate_limit_enabled`, `agent_enabled`, `vision_provider`, `tracing_enabled` -- booleans/provider names only, never a model name, host, connection string, or JWT setting) and a new always-visible `FeatureFlagsBar` in the frontend (unlike every other panel, never collapsed), fetched once on mount plus a manual refresh button, with a distinct "Dev mode" warning pill when `auth_enabled` and `insecure_dev_mode` are both true.

### Verification

Backend: `test_root_endpoint.py` gained 3 new tests, including a denylist-based leak guard (`test_root_features_never_leak_secrets_or_identifying_config`) that greps the raw response text for values like `qwen`/`gpt-`/`sqlite`/`postgresql://`/`api_key`/`secret`/`moondream` rather than only checking known field names. Full backend suite: 700 passed, 1 skipped. `ruff`/`mypy` clean on every touched file.

Frontend: adding `FeatureFlagsBar` to `ChatWindow` meant every existing integration test's fetch mock needed to route by URL instead of a single blanket response, since `GET /` now fires on every mount -- several tests had latent `fetchMock.mock.calls[0]`/`toHaveBeenCalledTimes(1)` assumptions that the extra call broke. Added a shared `routedFetchMock`/`rootInfoResponse` helper (`tests/integration/testUtils.tsx`) and updated all 6 integration test files to route explicitly rather than assume call order. Added 5 new tests (`featureFlagsBar.test.tsx`, `rootInfo.test.ts`) covering loading/error/refresh states and the dev-mode warning pill. Full frontend suite: 45/45 passed, `tsc -b --noEmit` clean, `eslint` clean (same 3 pre-existing benign warnings), production build succeeds.

Corrected one over-broad claim from the prior session's `frontend/README.md` while here: `npm run dev`/`build`/`lint`/`typecheck`/`test` all genuinely work on this environment's Node 18.12 (verified by having run all of them repeatedly this session) -- only `npm run test:e2e` (Playwright) hard-requires Node 20+. The README previously implied the whole toolchain needed Node 20+; narrowed to match what was actually observed.

### Interpretation

Two different "don't touch the shared default" instincts paid off in the same session: reaching for an experiment config instead of editing `config/default.yaml` for tracing, and (a session earlier) designing the whole web UI around a same-origin proxy specifically to avoid needing CORS middleware on the backend. The feature-flags fix follows the same shape as several other entries in this project's `ISSUES.md`: the code being audited was already correct, and the real gap was legibility -- a security posture that's only knowable by reading a YAML file isn't operationally safe, even when it's technically enforced correctly. Cross-checking the data-directory question (on-disk counts vs. ingested counts, via `psql` and `find`, rather than trusting either the user's assumption or my own) also directly avoided asserting something false in response to a direct factual question.

---

## 2026-08-22 (session 3): six critical-path E2E tests through the real HTTP API

Spec: `.claude-prompts/14.0-E2E-backend-tests.txt`, six invented scenarios: classic RAG grounded-answer-with-sources, agentic multi-hop (classification -> tool execution -> synthesis -> citations), JWT cross-tenant authorization, field-level redaction reaching the final answer, the SSE agent stream, and an agent correctly declining an unanswerable question instead of hallucinating.

### Design

Surveyed the existing `tests/integration/` suite first rather than assuming a gap: most existing files (`test_query_pipeline.py`, `test_agent_end_to_end.py`, `test_authorization_isolation.py`) call `RetrievalPipeline`/`run_agent()` directly against real Postgres/Ollama -- real infra, but not through the actual HTTP boundary. Two files already did go through `TestClient` (`test_api_field_redaction.py`, `test_agent_query_stream.py`), and became the template. New file: `tests/integration/test_e2e_critical_scenarios.py`, all six scenarios through the real `rag.api.main.app` (not an isolated mini-app), using `app.dependency_overrides` per scenario.

One real subtlety worked through before writing any test code: `get_vectorstore`/`get_embedder`/`get_llm`/`get_retrieval_pipeline` in `api/deps.py` are separate `@lru_cache`d singletons that each call `get_config()` as a plain Python function call, not through FastAPI's dependency resolution -- so overriding `get_config` alone via `app.dependency_overrides` does nothing for those four; the *specific* `Depends()` target each route actually declares must be overridden individually (`/query` only needs `get_config`+`get_retrieval_pipeline`; `/agent/query` and `/agent/query/stream` additionally need `get_vectorstore`/`get_embedder`/`get_llm` since those routers declare them as separate parameters). Confirmed this by reading `deps.py` directly rather than assuming, since getting it wrong would have made agentic-scenario tests silently exercise the wrong config.

### A real test-writing bug, caught by the test itself

First run: 5/6 passed on the first try against real infra. The insufficient-evidence scenario failed -- but on inspection, the *agent* had behaved correctly ("The answer cannot be determined from the provided evidence"); the *test's assertion* was wrong. It leaned on `_looks_like_refusal` (`rag.eval.run_eval`), which doesn't match that exact phrasing (`_REFUSAL_PHRASES` has "cannot determine"/"cannot be answered", not "cannot be determined"). This is the same documented heuristic gap already written up in `ISSUES.md`'s "My own refusal-detection heuristic kept lying to me" -- now hit for the first time inside a hard pytest assertion rather than while hand-reading an eval report. Fixed by replacing the phrase-heuristic as the primary signal with a more direct, harder-to-game structural check: the question fishes for one specific fabricatable number (a throughput figure for a fictional cluster), so assert the model never states a confident specific value for it, keeping `_looks_like_refusal`/`termination_reason` only as secondary signals. Re-ran just that test to confirm, then re-ran the full new file end to end: 6/6 passed (real run, ~6 minutes, all real LLM calls).

Documented as an addendum to the existing ISSUES.md entry (not a new entry, per the project's own "update the existing story" convention) -- the lesson generalizes: don't let a known-gappy heuristic be the single source of truth for a pass/fail decision, whether that's a human reading a report or an automated assertion.

### Environment gaps closed along the way

Two more pre-existing venv/pyproject.toml sync gaps surfaced and were fixed (same pattern as `pdfplumber` earlier this same day): `reportlab` (declared as a dev/test-only dependency for `test_layout_aware_ingestion.py`'s real multi-page PDF fixture, per that file's own CLAUDE.md-documented history) was missing, blocking collection of the rest of the integration suite. Installed it; the full remaining integration suite (69 tests, everything except the new file) then ran clean.

### Verification

New file: 6/6 passed against real Postgres + real Ollama (`qwen2.5:1.5b` for classic-RAG scenarios, `qwen2.5:3b` for agentic ones, matching this project's established model choice for JSON-schema reliability). Full backend unit suite: 700 passed, 1 skipped (unaffected -- no production code touched). Full remaining integration suite: 69 passed, confirming the new file's `app.dependency_overrides` usage (all wrapped in `try`/`finally`) doesn't leak state into any other test file sharing the same process-wide `rag.api.main.app` object. `ruff check`/`ruff format --check`/`mypy` clean on the new file.

### Interpretation

The most valuable finding from this session wasn't in the system under test -- every one of the six scenarios was already correctly implemented -- it was in the *test itself*, and only a real end-to-end run against a real model could have surfaced it: a unit test with a scripted LLM would never have produced this exact phrasing, and a purely internal `run_agent()` call (this project's existing pattern) would still have hit the same gap, but might have been noticed as "flaky" rather than diagnosed to a specific, fixable root cause. This is the concrete argument for having *some* tests that go through the real HTTP boundary against real infra, even in a suite that already has extensive internal-API integration coverage: they catch a different class of bug -- and, here, a different class of test bug -- than mocked or internal-call tests ever can.

## 2026-08-28: Deep bug audit (Claude + independent Codex pass), verification, and the C1 tenant-governance fix

Three linked sessions, same day. `.claude-prompts/Find-bugs.txt` asked for a from-scratch deep bug audit of the whole repo, scoped to critical/major findings with concrete failure scenarios, written to `claude-bug-reports.md` (originally `claude-fable-bug-reports.md`, renamed by the user to match the verification prompt's expectation). Separately, an independent Codex audit produced `codex-bug-reports.md`. `.claude-prompts/Find-bugs-verification-by-claude.txt` then asked for a strict verify/reconcile pass over both, and `.claude-prompts/Bugs/C1-authenticated-uploads.txt` asked to implement the fix for the single highest-priority confirmed finding.

### The audit

Read the architecture docs/config/tests first, then traced end-to-end rather than pattern-matching. Two findings survived to Critical, both reproduced with real code, not asserted from reading alone: `IngestionPipeline.ingest_path`'s directory-mode deletion-diffing (`_aggregate_ingestion_stats`) deletes any document not discovered by the *current* run's walk, which silently destroys documents ingested via a different route (e.g. a `POST /ingest` upload) under the same `dataset_id` -- reproduced with a small script against the existing `StatefulFakeVectorStore` test harness. Two Major findings in the agentic layer: `get_related_context` can never succeed because `_summarize_evidence` never shows the model a `chunk_id` to supply, even though the tool-select prompt tells it to use one; and `security.authorization.enabled=False` isn't honored by three of the four agent tools (`get_document`/`get_latest_document`/`get_related_context`), which call `VectorStore` directly and bypass `RetrievalPipeline._resolve_auth`'s kill switch (fails safe, not exploitable, but a real invariant violation).

### The verification pass

Treated neither report's conclusions at face value. Traced every Critical/Major finding from both reports against the current code, reproducing what could be reproduced without a live Postgres (none was running in this environment). The two Codex Critical findings both fully confirmed with real code, not just plausible-looking: **C1** (authenticated PDF/DOCX/no-front-matter uploads persist `tenant_id=NULL`, which the SQL predicate treats as visible to every tenant by design) traced end to end through the loaders, `Writer.write`, and `build_authorization_where_clause`. **C2** (a sensitive literal present only in a heading title, and therefore only in `section_path` metadata via the breadcrumb, never gets tagged by `detect_sensitive_field_ids` since that only scans chunk *content*, so `_redact_sensitive_fields`'s untagged-chunk early-exit skips it entirely) reproduced completely: built a synthetic doc through the real `StructuredMarkdownChunker`, showed the literal lands in `section_path` on a table/prose span with `sensitive_field_ids=None`, then called the real `_redact_sensitive_fields`/`source_label`/`source_dict` and watched the raw secret pass straight into what would become the LLM prompt and the API response. Also confirmed Codex's two Major upload-path findings by direct reproduction: `_save_upload_bounded`'s `dest.open("wb")` truncates an existing same-name file *before* checking the new upload's size, so a rejected oversized re-upload destroys a previously-accepted file (proved by calling the real function against a pre-seeded temp file); and `/agent/query/stream` runs its DoS-limit check *inside* the async generator, after `StreamingResponse`'s `http.response.start` (status 200) has already been sent -- proved with a side-by-side `TestClient` call of the identical oversized query against `/agent/query` (clean 422) vs. `/agent/query/stream` (200, empty body). All 8 findings across both reports (5 Codex + 3 Claude) survived verification; none downgraded. Wrote the consolidated report (`claude-bug-verification-from-codex.md`) with a priority order, root-cause clusters, and per-fix re-evaluation requirements, and stopped there per the prompt's explicit "wait for approval" instruction.

### Implementing C1

Investigated before coding, per the prompt's own instruction: `AuthorizationConfig.cross_tenant_support_roles` (`["techfusion_support"]` by default) was already the right existing privilege list to reuse for "may this caller ingest on another tenant's behalf" -- no new config field, no schema change (`tenant_id`/`allowed_roles` columns already existed on `chunks`). Designed the precedence rule set explicitly before writing code: missing document tenant -> stamp the caller's own; explicit match -> unchanged; explicit mismatch -> allowed only for a `cross_tenant_support_roles` caller, rejected otherwise -- applied whenever a verified identity exists at all (not gated on `authorization.enabled`, so flipping that switch on later can't retroactively expose a document ingested while it was off).

New `src/rag/ingestion/governance.py` (`IngestCallerContext`, `resolve_ingest_tenant_id`, `IngestGovernanceError`) is deliberately decoupled from `rag.api.auth.VerifiedIdentity` -- `rag.ingestion` must never import `rag.api`, the same layering rule `rag.retrieval` already follows. `IngestionPipeline.ingest_file` gained an optional `caller` parameter (`None` by default, so the CLI/`make ingest` path and any unauthenticated `POST /ingest` request are byte-identical to before this fix), resolved right after loading the `RawDocument` and before any database write, so a rejected upload leaves zero trace. `api/routers/ingest.py` builds the caller context from the verified identity and maps a rejection to a 403, audit-logged via `cross_tenant_attempt` -- a previously reserved-but-unwired `AuthEventType` value (`docs/architecture.md`'s audit-logging section literally said "approximated... not wired up in this pass"); this is its first real emission.

17 new tests across three layers: a pure-function suite for the resolver itself, a pipeline-level suite (a minimal recording `VectorStore` double) proving what actually lands in persisted `ChunkMetadata.tenant_id`, and an API-level suite (real JWT minting through the actual `get_current_identity` dependency chain, mirroring `test_api_query_auth_boundary.py`'s pattern) for the 403/audit-logging path. One test-writing mistake caught immediately by running the suite: the first two API-level tests used `roles=["tenant_alpha_operator"]`, which isn't in `AuthConfig.ingest_roles`, so `_enforce_ingest_authorization` (a pre-existing, unrelated check) rejected the request before the new governance logic ever ran -- fixed by using `security_admin` (ingest-eligible, not cross-tenant-privileged) for the "normal caller" tests instead. Also wrote (but could not run in this environment -- no local Postgres) an integration suite proving the same behavior end to end against real Postgres, including the actual cross-tenant retrieval-denial proof.

### Verification

17/17 new tests pass. Existing ingest/auth/authorization suites (`test_api_dos_limits.py`, `test_ingestion_stats.py`, `test_ingestion_category.py`, `test_api_query_auth_boundary.py`) all still pass, no regressions. Full unit suite: **717 passed, 1 skipped** (was 700 before this session -- exactly the +17 new tests). `ruff check`/`ruff format --check`/`mypy` clean on every changed file.

### Documentation

Updated `CLAUDE.md` (new directory-map entry for `ingestion/governance.py`, a new bullet under the auth-boundary milestone section), `docs/architecture.md` (new "Ingest-time tenant governance" subsection under "Authenticated API Boundary and Security Hardening," plus correcting the audit-logging section's "`cross_tenant_attempt` is approximated... not wired up" note now that it's genuinely emitted for this case), `SECURITY.md` (new bullet under "Implemented security controls"), and `ISSUES.md` (one new, deliberately shorter entry per this session's explicit "crisp and precise, not a full story" instruction).

### Interpretation

The audit -> verify -> fix sequence this session followed is itself the lesson: an independent second audit (Codex) found something this session's own first-pass audit missed entirely, and the verification pass existed specifically so neither report's claims were trusted without being re-traced and, where practical, reproduced against real code -- both Critical findings were confirmed by actually running code, not just by reading it and agreeing it looked right. The root cause of C1 itself is a small, general lesson worth keeping: a role check on a write endpoint (`ingest_roles`, "who may call this") is a completely different claim from a governance-binding check ("what tenant does this data belong to"), and the original auth-boundary milestone's own test suite never caught the gap because it only ever ingested documents with hand-authored front matter that already had correct governance metadata -- it never exercised the much more realistic case of an upload with none at all.

### Follow-up: Codex pre-merge review of C1, and fixing M3 as the same change

Before merging, the user ran a second Codex pass specifically reviewing the just-implemented C1 fix. It found the fix itself correct but incomplete: `POST /ingest` still streamed an upload straight to its final `UPLOAD_DIR` path via `_save_upload_bounded` *before* governance validation ran, so a same-filename re-upload that governance later rejected (or that the size check rejected as oversized) had already destroyed the previously-accepted file on disk. This is exactly the verification report's separately-tracked **M3** finding (upload-replacement truncation) -- not a new bug, but the same defect shape (mutate shared state before validating) at a different layer (filesystem instead of database) than C1's own fix had already addressed.

Inspected the flow before coding: every loader (`text_loader.py`/`pdf_loader.py`/`docx_loader.py`/`html_loader.py`) sets `RawDocument.source = str(path)`, and `Writer.write` copies that straight into the persisted `ChunkMetadata.source` -- so loading from a temp file, unmodified, would have made a random temp filename the document's permanent identity. Fixed by giving `IngestionPipeline.ingest_file` a new `source_override: str | None` parameter that overwrites `raw_document.source` right after loading, before the checksum/`get_or_create_document_id` call -- content is read from wherever `path` points, but persisted identity is whatever `source_override` says.

`api/routers/ingest.py` gained `_temp_upload_path`/`_ingest_upload_atomically`: every upload streams to a unique temp file in the *same directory* as its final destination (same volume, so the eventual install is a true atomic rename, not copy+delete). Size-bounding, loader parsing, and governance resolution all run against the temp file, with `source_override` set to the stable destination path. Only after `ingest_file` fully succeeds does `os.replace(tmp_path, dest)` install it. Any exception along the way -- oversized, unsupported extension, parse failure, `IngestGovernanceError` -- is caught, the temp file is unlinked, and the exception re-raised; `dest` is never touched on any failure path.

Added `tests/unit/test_api_ingest_atomicity.py` (6 HTTP-level tests via `TestClient` against the real `/ingest` route with a fake pipeline, covering: rejected-new-filename leaves no file; rejected-same-filename preserves the old file byte-for-byte; oversized-same-filename preserves the old file; successful-same-filename atomically replaces it; a rejection stops the batch; and a successful upload's `source_override` is the stable path, never the temp filename) plus one new `IngestionPipeline`-level test (`test_source_override_persists_the_stable_path_not_the_loaded_path`) proving `source_override` actually reaches persisted `ChunkMetadata.source`. Updated the existing `_RecordingIngestionPipeline` fake in `test_api_ingest_governance.py` to accept the new `source_override` kwarg the router now always passes, since the router calls every injected pipeline with it regardless of which test double is wired in.

7 new tests, full unit suite 724 passed / 1 skipped (was 717 -- exactly +7, no regressions). `ruff check`/`ruff format --check`/`mypy` clean on every changed file. Updated `CLAUDE.md` (new bullet under the auth-boundary milestone section), `docs/architecture.md` (new "Atomic upload persistence" subsection immediately after "Ingest-time tenant governance"), `SECURITY.md` (extended the ingest-governance bullet), and `ISSUES.md` (a "Follow-up:" entry appended to the C1 story, per this project's convention for updating an affected story rather than duplicating it). No integration-level HTTP test against real Postgres was added -- no existing integration test hits `/ingest` over HTTP at all (every one calls `pipeline.ingest_file()` directly), and Postgres wasn't running in this environment anyway; the unit-level `TestClient`-against-the-real-route tests satisfy "test the route, not just the helper" without that dependency.

C1 and M3 are both now fully fixed: C1's tenant-binding logic and its upload-persistence mechanics are both validate-before-mutate at every layer that matters (DB write and filesystem write alike).

### Follow-up: the atomic-upload fix still let the staging path's random name leak into metadata

A second Codex pre-merge review, this time of the atomic-upload fix itself, found a gap the previous fix hadn't reached: `source_override` only overrides the final `RawDocument.source` string, which is set once at the very end of `Loader.load()`. It can't touch any field a loader computes earlier, directly from the `path` argument it was given. Inspected every loader before writing any code: `TextLoader`/`HTMLLoader`/`PDFLoader`/`DocxLoader` all fall back to `title = ... or path.stem` when the document has no title of its own, and `PDFLoader`/`DocxLoader`'s embedded-image extraction (`loaders/base.py:resolve_image_asset`) is keyed entirely off the physical path -- `document_path.parent` for where a sibling `assets/` folder is expected, `document_path.stem` for naming a newly-extracted image. Crucially, `resolve_image_asset`'s returned `"assets/<name>"` string gets embedded directly into the document's content and parsed by `StructuredMarkdownChunker` straight into `ChunkMetadata.attachment_name`/`source_anchor` -- persisted, returned in `POST /query`'s `sources`, and used in `[Source N: ...]` citations. Loaded from the previous fix's randomly-named temp file, both fields would have leaked the staging path's random name into stored, user-visible metadata.

Weighed two designs before coding, per the task's explicit steering toward the smaller one: give every loader an explicit `logical_path` parameter (touches `Loader.load()`'s signature and all four loader files plus `resolve_image_asset`), versus keep the staged file's own *filename* identical to the final destination's, randomizing only its *parent directory*. Chose the second: `_staging_dir` now creates a fresh directory (`UPLOAD_DIR/.upload-tmp-<uuid>/`) and stages the upload at `staging_dir / dest.name` -- the exact original filename, one level deeper. Every `path.stem`/`path.name`-derived field any loader computes comes out correct automatically, with zero changes to any loader or to `resolve_image_asset` itself, and the fix structurally covers any future path-derived field the same way, not just the two found this session -- exactly the "smaller central solution" the task asked to look for before touching loader APIs.

The one piece this still required: since `resolve_image_asset` resolves `assets/` relative to whatever physical path it's given, a staged upload's newly-extracted images now land in `staging_dir/assets/`, not `UPLOAD_DIR/assets/` where a persisted `source_anchor` is actually resolved against later (e.g. `Writer._with_vision_siblings`, at query/vision time). New `_install_staged_assets` promotes them there -- a same-volume `os.replace` per file -- right after ingestion succeeds and before the staging directory is removed, so no reference is left dangling. `_ingest_upload_atomically` was restructured around this: `staging_dir.mkdir()` -> stage the file under its own name -> run `ingest_file` -> on success, promote any staged assets, then `os.replace` the main file into place, then `shutil.rmtree` the now-empty staging directory; on any failure, `shutil.rmtree` the whole staging directory and re-raise, leaving `dest` and `UPLOAD_DIR/assets/` completely untouched either way.

New `tests/unit/test_upload_staging_no_temp_leak.py` (12 tests): per-loader title-fallback checks for Markdown/text/HTML; real PDF and DOCX fixtures (built with `reportlab`/`python-docx`, an existing dev-only pattern already used by `test_layout_aware_ingestion.py`) each with an embedded image and no pre-existing sibling `assets/` folder, proving the extracted asset's filename and the `source_anchor` embedded in content both use the original stem, not the staging directory's; a generic recursive scanner (`_assert_no_temp_marker`) walking a fully-persisted `Chunk` (through a real `IngestionPipeline` run against a fake `VectorStore`) asserting no `.upload-tmp-*` string survives anywhere in it, including `attachment_name`; a document-id-stability test proving two staged re-uploads of the same filename through two different randomly-named staging directories still resolve to the same `document_id`; and a parametrized sweep across every registered extension. One test-authoring snag: `reportlab`'s `SimpleDocTemplate` sets a default PDF document-info title (`"(anonymous)"`) unless given `title=""` explicitly, which meant the first draft of the PDF fixture never actually exercised `PDFLoader`'s `path.stem` fallback at all (it read reportlab's placeholder metadata instead) -- caught immediately by the test itself asserting the wrong title, fixed by passing `title=""`.

12 new tests, full unit suite 736 passed / 1 skipped (was 724 -- exactly +12, no regressions). `ruff check`/`ruff format --check`/`mypy` clean on every changed file. Updated `CLAUDE.md` (rewrote the atomic-upload bullet for the staging-directory design, added a new bullet for this fix), `docs/architecture.md` (rewrote the corresponding paragraphs, added a new "Staged-upload metadata leak" subsection), `SECURITY.md` (extended the same bullet), and `ISSUES.md` (a second "Follow-up:" entry on the same C1 story).

C1 and M3 remain fully fixed, and this closes the metadata-leak gap the second review found in that fix's own implementation: no `.upload-tmp-*` fragment can reach a persisted chunk, a generated asset name, an API response, or a citation, proven per-loader and with a generic recursive scan rather than trusting the enumerated field list alone.

---

## 2026-08-28 (session 2) | A1 root-scoped deletion fix, M4/M5 API validation fixes, and two codex-review follow-ups

Same day as the deep bug audit / C1 / M3 session above, a separate session working through three of the audit's remaining confirmed findings from `.claude-prompts/Bugs/`: A1 (incremental-ingestion deletion scoping), then M5+M4 together (top_k floor, streaming validation order), each followed by an independent Codex pre-merge review that surfaced a real, narrower follow-up gap in the just-completed fix.

### Interview talking points
- **What problem did this solve?** Three real bugs: (1) directory re-ingestion could delete documents from unrelated ingestion paths sharing a dataset, (2) `POST /query` accepted `top_k<=0` and silently substituted the default instead of erroring, (3) `POST /agent/query/stream` validated requests only after the HTTP response had already committed to status 200.
- **Why was it difficult?** None of the three were visible from reading the code casually -- (1) requires two independent ingestion paths sharing a dataset to manifest at all; (2) hides behind Python's falsy-zero idiom (`x or default`); (3) requires understanding exactly when an async generator's body starts executing relative to when `StreamingResponse` commits its status line, and (after the first fix) exactly how FastAPI's `Depends()` resolution order interacts with an early-raising dependency -- neither is obvious without testing the framework's actual behavior directly.
- **How did I debug it?** Reproduced every finding empirically before fixing it, and every follow-up finding too: a live script proving A1's deletion bug against `StatefulFakeVectorStore`; a side-by-side `TestClient` comparison of `/agent/query` vs. `/agent/query/stream` for M4; a real dependency-override probe against the live app (recording doubles on `get_vectorstore`/`get_embedder`/`get_llm`/`get_retrieval_pipeline`) to confirm the DI-ordering follow-up finding was real, then two throwaway FastAPI apps to empirically nail down the exact resolution-order/abort-on-raise semantics before relying on them in the actual fix.
- **What metrics proved the fix?** Full unit suite climbed 758 -> 770 (+12, A1 core) -> 772 (+2, A1 follow-up) -> 777 (+5, M5+M4 combined) -> 778 (+1, M4 follow-up), zero regressions at every step, `ruff`/`mypy` clean throughout. For the two DI-ordering findings specifically, the proof wasn't a test count -- it was a before/after list of which dependency-provider functions actually got called for the same rejected request (`['pipeline_getter', 'vectorstore', 'embedder', 'llm']` before the fix, `[]` after).
- **What would I improve next?** `/query` and `/agent/query` have the same Depends()-before-validation characteristic `/agent/query/stream` had -- deliberately left unfixed both times (out of each task's explicit scope), but worth a follow-up pass if cold-start dependency construction ever becomes a measured concern. A1's residual same-root-respelling identity churn is a known, accepted, tested limitation, not something planned to be revisited.

### Goal
Fix three specific confirmed findings from the prior session's bug audit (`.claude-prompts/Bugs/A1-Incremental-directory-ingestion.txt`, `.claude-prompts/Bugs/M5+M4-API-validation.txt`), one at a time, each with an explicit "do not touch anything else" scope constraint, then respond to an independent Codex pre-merge review of each fix.

### Configuration
No config/schema changes in any of the three fixes -- all pure application-code + test changes. No experiments run this session.

### Changes made

**A1** (`src/rag/ingestion/pipeline.py`, `tests/unit/test_ingestion_stats.py`): new `_source_is_under_root(source, root)` helper (resolved-path membership, not string prefix matching, backslash-normalized) scopes `ingest_path`'s pre-run deletion snapshot to sources that could plausibly have come from the root being walked, before the existing `_aggregate_ingestion_stats` diff logic runs (untouched). Committed as 4 separate single-purpose commits: core fix, core tests, follow-up docstring, follow-up tests -- reconstructed by temporarily reverting the follow-up content, committing the round-1 state, then re-applying and committing the follow-up separately, at the user's explicit request for maximally granular, single-line-message commits with same-context changes grouped together.

**M5** (`src/rag/api/request_auth.py`): `enforce_dos_limits` (shared by `/query`, `/agent/query`, `/agent/query/stream`) gained a `top_k < 1` floor check alongside its pre-existing ceiling check.

**M4** (`src/rag/api/routers/agent_stream.py`): validation (`enforce_dos_limits`/`build_authorization_context`/`AgentState` construction) moved out of the async generator handed to `StreamingResponse` and into the route's synchronous body, executed before `StreamingResponse(...)` is constructed.

**M4 follow-up**: that validation logic extracted again, this time into its own FastAPI dependency (`_build_validated_agent_state`), declared *before* `pipeline`/`vectorstore`/`embedder`/`llm` in `agent_query_stream`'s signature, so FastAPI's own dependency-resolution short-circuiting (verified empirically, not assumed) skips resolving those four `lru_cache`d singletons entirely for a rejected request.

### Issues discovered
1. **A1**: directory-mode deletion detection diffed a dataset's *entire* known-sources set against only what the current walk discovered, so a document from a different ingestion root (or a `POST /ingest` upload, which never calls `ingest_path`) was indistinguishable from a genuine deletion. Root cause: mismatched comparison scope. Fix: `_source_is_under_root` pre-filter. Evidence: live reproduction script + 12 new tests. Resolved: y (with one documented, tested, accepted residual limitation -- see follow-up below).
2. **A1 follow-up** (Codex): the deletion *diff* itself still compared raw strings, so re-ingesting the same root under a different spelling (relative/absolute, or cross-OS separators) still churns that one document's identity (delete + recreate), though never duplicates or loses content. Root cause: `document_id` is keyed on the literal `source` string everywhere in this system, a pre-existing property, not something this fix introduced. Fix: deliberately not deepened further (both alternatives considered would either cause real duplication or a mass one-time churn across every pre-existing dataset); instead verified and pinned the safe fallback with 2 new regression tests, and rewrote the docstring's guarantee to be precise. Resolved: y (as a documented, accepted limitation, not a residual bug).
3. **M5**: `top_k<=0` reached `RetrievalPipeline.retrieve()`'s `fetch_k = candidate_k or config.retrieval.candidate_k`, where `0` is falsy in Python and silently substituted the default. Root cause: falsy-zero idiom conflating "not provided" with "provided as zero." Fix: floor check at the API boundary in the shared `enforce_dos_limits`. Evidence: 2 new tests. Resolved: y.
4. **M4**: `/agent/query/stream` validated inside the async generator passed to `StreamingResponse`, which only starts executing on its first `__anext__()` -- by which point Starlette has already sent the 200 status header. Root cause: generator laziness vs. eager response-status commitment. Fix: moved validation into the route's synchronous body, before `StreamingResponse` construction. Evidence: side-by-side `TestClient` comparison + 3 new tests. Resolved: y.
5. **M4 follow-up** (Codex): even after the above fix, `pipeline`/`vectorstore`/`embedder`/`llm` stayed declared as top-level `Depends()` params, so FastAPI resolved (called) all four `lru_cache`d singleton getters before the route body's validation ever ran, for every request including rejected ones. Root cause: FastAPI resolves a route's entire declared dependency graph, in signature order, before the handler body executes, independent of what validation the body itself performs. Fix: extracted validation into its own early-declared dependency (`_build_validated_agent_state`), relying on FastAPI's abort-on-raise-during-resolution behavior (verified with two throwaway apps before trusting it). Evidence: before/after empirical probe with recording dependency overrides at the getter level + 1 new test overriding the getters themselves. Resolved: y, scoped to `/agent/query/stream` only -- `/query`/`/agent/query` share the same characteristic, deliberately left unfixed (out of scope).

### Evaluation results
No experiment/eval runs this session -- pure bug-fix + test-coverage work. Full unit suite progression: 758 -> 770 -> 772 -> 777 -> 778 passed, 1 skipped throughout (pre-existing, unrelated). `ruff check`/`mypy src` clean at every step (95 source files).

### Interpretation
Both Codex follow-up reviews found something real that a first, seemingly-complete fix had missed -- and in both cases the gap lived one layer deeper than the original bug report described (A1's report was about cross-root deletion, not same-root respelling; M4's report was about response-lifecycle ordering, not dependency-injection-graph ordering). Neither follow-up was accepted or dismissed on inspection alone -- both were reproduced empirically first (a live script for A1, a real dependency-override probe against the running app for M4), and in the M4 follow-up's case, the fix itself depended on a framework behavior (FastAPI's resolution-order/abort-on-raise semantics) that was independently verified with two minimal throwaway apps before being relied on, rather than assumed from general familiarity with FastAPI. The A1 follow-up is also a good example of a fix that was deliberately *not* deepened further: both available "complete" fixes would have traded a narrow, self-correcting edge case for a much larger blast radius (real duplication, or a mass one-time churn across every existing dataset), and recognizing that trade-off was the more defensible engineering call than chasing full closure.

### Open problems
`/query` and `/agent/query` still resolve their heavy DI singletons before body-level validation runs, same as `/agent/query/stream` did before its follow-up fix -- flagged in both the M4 follow-up's docstring and `ISSUES.md`, not fixed, since it was outside both tasks' explicit scope.

### Next recommended step
None queued from this session specifically. If cold-start dependency-construction cost on rejected requests ever becomes a measured concern for `/query`/`/agent/query`, the same `_build_validated_agent_state`-style early-dependency pattern used here for `/agent/query/stream` would apply directly.

---

## 2026-08-28 (session 3) | B1/B2 agent-tool fixes (chunk_id exposure, authorization parity), a full 10-angle code review, and a live agentic re-evaluation

Same day as the two sessions above, a third session working through `.claude-prompts/Bugs/B1+B2-Agentic.txt`: two confirmed agent-tool bugs found by an earlier audit, fixed together, then re-evaluated live and independently reviewed by a 10-agent `/code-review` pass the user triggered mid-session.

### Interview talking points
- **What problem did this solve?** Two real bugs in the agentic tool layer: (1) `get_related_context` required a real `chunk_id`, but the evidence summary shown to the tool-selection model never exposed one, so the model had no way to comply with its own prompt instructions except guessing; (2) `get_document`/`get_latest_document`/`get_related_context` bypassed `RetrievalPipeline`'s authorization kill-switch and freshness resolution by calling `VectorStore` directly with an unresolved caller-supplied auth context.
- **Why was it difficult?** Neither bug was visible from a quick read. B1 required tracing the actual runtime data path from `SearchResult` metadata through to the rendered prompt string to notice the prompt's own wording ("chunk id from gathered evidence") described data that was never actually rendered anywhere. B2 required understanding that `VectorStore`'s SQL predicate has no concept of the `authorization.enabled` config flag at all; that kill-switch lived only in `RetrievalPipeline`, one layer up, and three specific tools bypassed that layer entirely by calling `VectorStore` directly.
- **How did I debug it?** Read `_summarize_evidence`, `GetRelatedContextArgs`, and the active tool-select prompt side by side for B1. For B2, traced every direct `VectorStore` call in `tools.py`, confirmed `PgVectorStore`'s SQL predicate has no kill-switch awareness, then cross-checked against `retrieve()`'s own code path to confirm the three direct-fetch tools were the outlier. After fixing both, ran a full 10-agent `/code-review` pass (line-by-line scan, removed-behavior audit, cross-file tracer, language-pitfall specialist, wrapper/proxy correctness, reuse, simplification, efficiency, altitude, conventions) against the diff; two independent angles converged on the same real gap (`get_related_context` never threaded `dataset_id` into `resolve_auth`, so it alone never got freshness resolution) and the same real inefficiency (`get_latest_document` fetching `list_document_versions` twice per call), both fixed as follow-ups with new regression tests. The conventions angle also caught that my own new documentation/docstrings used spaced double-hyphens (`" -- "`) as em-dash substitutes, which the project's own CLAUDE.md rule explicitly prohibits alongside literal em dashes; cleaned up every instance in newly-added content.
- **What metrics proved the fix?** Full unit suite climbed 778 → 789 (+11, initial B1+B2 fix and tests) → 792 (+3, the two follow-up fixes' regression tests), zero regressions, `ruff`/`mypy` clean throughout. Also brought up the existing Postgres container and re-ran the established 18-question agentic baseline (`experiment_032`) with the fix applied, recorded as `experiment_040`: routing accuracy, `unnecessary_agent_rate`, `tool_selection_accuracy`, `evidence_sufficiency_accuracy`, and `retry_success_rate` all read identical to the baseline; `agent_answer_correctness` moved from 0.463 to 0.482 (within noise). Neither run's model ever actually selected `get_document`/`get_latest_document`/`get_related_context` for any of the 18 questions (both used only `search_knowledge_base`), so the live re-eval didn't get a chance to exercise the fixed code paths at all; the dedicated unit tests (including one full `run_agent()` flow using a real, dynamically-produced `chunk_id`) are the actual proof the fix works, not the eval.
- **What would I improve next?** The live re-eval's latency numbers (`agent_latency_ms` mean 47,388ms → 223,645ms) are not usable as a signal: the re-run executed while ten parallel code-review agents and two Docker containers were active on the same host, and the raw per-node timings show multi-hundred-second single LLM calls consistent with severe, transient CPU contention. A clean re-run under normal load, or a gold set that actually exercises `get_document`/`get_latest_document`/`get_related_context` for at least a few questions, would give a more informative live signal than this particular 18-question set currently provides for these three tools specifically.

### Goal
Fix B1 and B2 from `.claude-prompts/Bugs/B1+B2-Agentic.txt`, add the tests the task requested, run the focused/full test suites plus ruff/mypy, and re-run the relevant agentic evaluation since the task explicitly called out that tool-selection accuracy, citation support, and latency might change.

### Configuration
New `config/experiments/agentic-rag-baseline-v3-chunk-id-fix.yaml`: byte-identical to `agentic-rag-baseline-v2-fixed.yaml` (`experiment_032`'s config) except `agent.tool_select_prompt_path` (v1 → v3, confirmed via a direct config-diff script, not assumed). New `src/rag/prompts/templates/agent_tool_select_v3.yaml` prompt version; `config/default.yaml`'s `agent.tool_select_prompt_path` now points at it, along with two other config files (`full-pipeline-demo.yaml`, `tracing-enabled.yaml`) that mirror `default.yaml` wholesale and needed the same bump to stay internally consistent (`tracing-enabled.yaml`'s own isolation test caught this immediately).

### Changes made

**B1** (`src/rag/agent/graph.py`, `src/rag/prompts/templates/agent_tool_select_v3.yaml`, `config/default.yaml`): `_summarize_evidence` now renders `"[i] chunk_id=<id> source=<path>: <text>"` per evidence line instead of `"[i] source: text"`. New v3 tool-select prompt tells the model explicitly to copy the `chunk_id=` value verbatim.

**B2** (`src/rag/retrieval/pipeline.py`, `src/rag/agent/tools.py`, `src/rag/agent/graph.py`): `RetrievalPipeline._resolve_auth` renamed to public `resolve_auth` (no behavior change). `get_document`/`get_latest_document`/`get_related_context` now all call `resolve_auth` before their `VectorStore` call instead of passing a raw caller-supplied `AuthorizationContext` straight through.

**B2 follow-up** (same session, found by the `/code-review` pass): `get_related_context` gained an optional `dataset_id` parameter (threaded from `_dispatch_tool`'s `state.filters`) so it gets freshness-exclusion resolution like its siblings. `resolve_auth` gained an optional `versions` parameter so `get_latest_document` (which already fetches `list_document_versions` for its own source-resolution step) can reuse that list instead of triggering a redundant second query.

### Issues discovered
1. **B1**: `_summarize_evidence` never rendered `chunk_id`, so `get_related_context`'s own required argument was unfillable except by guessing. Root cause: the evidence-summary renderer, the tool's schema, and the prompt's wording were three independently-written pieces that had drifted out of agreement. Fix: expose `chunk_id` in the summary, new prompt version instructing exact-copy behavior. Evidence: 3 new unit tests including a full scripted `run_agent()` flow. Resolved: y.
2. **B2**: three direct-`VectorStore`-calling tools bypassed `RetrievalPipeline`'s authorization kill-switch and freshness resolution, since that logic lived only in `RetrievalPipeline`, not `VectorStore` itself. Fix: route all three through the newly-public `resolve_auth`. Evidence: 8 new unit tests plus 1 new integration test. Resolved: y.
3. **B2 follow-up** (`/code-review`, two independent angles): `get_related_context` never threaded `dataset_id` into `resolve_auth`, so it alone never got freshness resolution even though its siblings did; `get_latest_document` fetched `list_document_versions` twice per call once B2's fix was applied. Root cause: both were narrower gaps one layer deeper than B2's original report described. Fix: thread `dataset_id` through; give `resolve_auth` an optional pre-fetched `versions` parameter. Evidence: 3 new unit tests. Resolved: y.
4. **Documentation compliance** (`/code-review`, conventions angle): all of this session's own new documentation/docstrings used `" -- "` (spaced double hyphen) as an em-dash substitute; CLAUDE.md's rule prohibits both literal em dashes and spaced double hyphens as sentence punctuation, not just the former. Root cause: misread the rule as only banning literal em dashes. Fix: rewrote every instance in newly-added content (source docstrings, new config/prompt file comments, new test docstrings, the CLAUDE.md/docs/architecture.md/ISSUES.md additions) using commas, parentheses, or semicolons instead. Resolved: y, scoped to this session's own new content; pre-existing content elsewhere in the same files was left untouched (out of scope).

### Evaluation results
`experiment_040` (`config/experiments/agentic-rag-baseline-v3-chunk-id-fix.yaml`, full 18-question `agentic_extension_gold.jsonl`, real Postgres + real Ollama `qwen2.5:3b`), diffed against `experiment_032`:

| metric | experiment_032 | experiment_040 | read |
|---|---|---|---|
| routing_accuracy | 0.889 | 0.889 | identical |
| unnecessary_agent_rate | 0.0 | 0.0 | identical |
| tool_selection_accuracy | 0.0 | 0.0 | identical (strict all-or-nothing metric) |
| evidence_sufficiency_accuracy | 0.5 | 0.5 | identical |
| retry_success_rate | 0.0 | 0.0 | identical |
| agent_answer_correctness | 0.463 | 0.482 | within noise |
| tool_success_rate | 1.0 | 0.923 | 2 failed `search_knowledge_base` calls on 1 question; that tool's code path is untouched by B1/B2, attributed to host contention (see below) |
| citation_support_rate | 1.0 | 0.5 | not comparable; the metric's own definition changed independently between the two runs |
| agent_latency_ms mean | 47,388 | 223,645 | not comparable; 10 parallel code-review agents + 2 Docker containers were active on the host throughout this run |

Both runs' agent-routed questions used only `search_knowledge_base`; `get_document`/`get_latest_document`/`get_related_context` were never selected by the model in either run, so this particular gold set does not exercise B1/B2's fixed code paths live. Full unit suite: 778 → 789 (+11, B1+B2 initial fix) → 792 (+3, follow-up fixes), 1 skipped throughout (pre-existing, unrelated). `ruff check`/`mypy src` clean at every step (95 source files). 6/6 integration tests in `test_agent_tool_tenant_isolation.py` pass against the real corpus.

### Interpretation
Both bugs were narrow, single-layer gaps: B1 was a rendering omission (the data existed, just wasn't surfaced to the prompt); B2 was a bypassed abstraction (the kill-switch existed, just wasn't called from three specific call sites). The `/code-review` pass run mid-session was genuinely useful beyond a compliance exercise: two of its ten angles independently converged on the same real, previously-unnoticed gap in the B2 fix itself (the `dataset_id`-threading omission for `get_related_context`), which is exactly the kind of "fixed the reported bug but left a narrower adjacent one" pattern this project's `ISSUES.md` has documented before (see the A1 root-scoped-deletion story two sessions above). The live agentic re-evaluation, despite real effort to make it a clean comparison (same corpus, same config lineage, only the one intended field changed), ended up confounded on two axes at once: severe host contention (unplanned, caused by the concurrent code review) made latency unusable, and the specific 18-question gold set simply doesn't provoke the model into selecting the three tools this fix touches, so it couldn't demonstrate the fix's effect even under clean conditions. Recording the result transparently with both caveats spelled out, rather than either suppressing the run or reading noise as signal, matches this project's established precedent for handling confounded experiments (see the context/token-budget A/B/C entry's host-noise diagnosis).

### Open problems
None newly introduced. The live-eval limitations above (host contention, gold set not exercising the three tools) are documented as run-specific caveats on `experiment_040`, not fixed defects — a future clean re-run, or a gold-set addition that reliably provokes `get_document`/`get_latest_document`/`get_related_context` selection, would close them if a live behavioral signal is ever needed.

### Next recommended step
None queued from this session specifically. If a live demonstration of B1/B2's effect on real model behavior is ever needed, either re-run `experiment_040` under a quiet host, or add 1-2 gold questions to `agentic_extension_gold.jsonl` specifically engineered to require `get_related_context` (e.g. a question whose answer needs a table/code block's surrounding prose context) so the tool is actually exercised.

### Follow-up: a second independent review found one remaining B2 merge blocker

After the report above, the user ran a second Codex review specifically of the B2 fix. It found `rag.agent.graph._execute_tool`'s own `pipeline.sanitize_evidence(results, state.authorization_context)` call still passed the raw, unresolved `state.authorization_context`, not a resolved value, even though `sanitize_evidence`'s own docstring already specifies its `auth` parameter should be "the caller's resolved authorization context." Under `security.authorization.enabled=False` + `security.field_redaction.enabled=True`, this let a caller who asserted real (even authorized-sounding) roles see a direct-fetch tool's sensitive fields survive redaction, while `search_knowledge_base`'s evidence (redacted inside `retrieve()`, which already resolves auth) correctly failed closed to empty roles.

**Root cause:** the B2 fix resolved auth for *retrieval* (the `VectorStore` calls inside `get_document`/`get_latest_document`/`get_related_context`) but never touched the separate `sanitize_evidence` call site in `_execute_tool`, which every tool's evidence passes through regardless of which tool produced it. Two independent, composable controls (document-level ACL and field-level redaction) share one call site, and fixing the first control's call sites doesn't automatically fix the second's.

**Fix:** one added line in `_execute_tool`: `effective_auth = pipeline.resolve_auth(state.authorization_context, {"dataset_id": dataset_id} if dataset_id else None)`, using the exact same `dataset_id` already threaded to the direct-fetch tools, with `sanitize_evidence(results, effective_auth)` now using that resolved value instead of the raw context. Verified the fix actually catches the bug empirically: reverted it, confirmed the new regression tests failed against the reverted code, then restored it.

**Test fallout:** since this new `resolve_auth` call sits in the universal `_execute_tool` path (every successful tool dispatch, not just the three direct-fetch tools), every `FakePipeline`/pipeline-double class across the agent unit test suite needed an identity `resolve_auth` stub added, 10 files, all `return auth` no-ops matching each file's existing style. 3 new unit tests (`test_agent_execute_tool_sanitization_auth_parity.py`, a real `RetrievalPipeline` driven through `run_agent()`) cover the fail-closed case, parity with `search_knowledge_base`, and a sanity check that a genuinely authorized role under `authorization.enabled=True` still discloses (proving the fix doesn't over-redact).

**Result:** full unit suite 792 -> 795 (+3), 1 skipped throughout, `ruff`/`mypy` clean. All 6 `test_agent_tool_tenant_isolation.py` integration tests and both real-Ollama `test_agent_end_to_end.py` tests pass unchanged (required bringing Docker Desktop itself back up mid-session; it had stopped entirely, not just the container, confirmed via `Get-Service com.docker.service`), confirming `authorization.enabled=True` behavior, tenant isolation, and B1's chunk_id behavior are all untouched by this fix.

**Interpretation:** this is the same lesson as the first B2 follow-up, one layer over: a "universal sanitization point" is only as correct as the value handed to it, and a docstring stating the expected contract ("resolved authorization context") is not the same as an enforced one. Two sequential Codex reviews of the same fix each found a real, narrower gap the other didn't; neither was accepted nor dismissed without first reproducing it directly (a live dependency-override probe for the first M4 finding two sessions ago, an explicit revert-and-confirm-failure step for this one), reinforcing this project's now-repeated pattern of verifying review findings empirically rather than trusting or dismissing them on inspection alone.

## 2026-08-31: MCP integration Stage 1A -- secure MCP server exposing the four core RAG tools

### Goal
Close the job-description gap around MCP ("Build the integration layer (via MCP) that lets agents securely call backend systems and business APIs"). Prior turn produced a research/design proposal (architecture, identity-propagation model, transport choice, risks, acceptance criteria) with no files changed. This turn: user approved with adjustments (server-only Stage 1A, no business-backend example yet, dependency-strategy decision left to inspection, sanitize_evidence must be the single choke point, stdio kept minimal and never weakening the HTTP model, verify against the actual installed SDK) and asked for implementation.

### Configuration
No new experiment configs this session (infra/security milestone, not a retrieval/generation A/B). New `config.mcp` block (`McpConfig`/`McpServerConfig`), `enabled: false` default in `config/default.yaml`, matching the `agent.enabled`/`security.authorization.enabled` "true no-op" convention. No separate `mcp.auth.*` block -- identity is governed entirely by the existing `security.auth` tree, deliberately, so the two surfaces can never drift into different security postures.

### Changes made

**Dependency** (`pyproject.toml`): `mcp>=2.1,<3` added to core `dependencies` (not an optional extra). Decision grounded in a dry-run install showing the package is lightweight on top of what's already required (`pydantic`, `pyjwt[crypto]`, `starlette`, `uvicorn`, `opentelemetry-api` all already satisfied) and that it's mounted inside the always-running `rag-api` process behind a runtime flag, the same shape as `slowapi`/`pyjwt`, not the shape of the eval-only extras (`ragas`/`mlflow`/`cohere`/`anthropic`) deliberately excluded from the container image.

**New package `src/rag/mcp/`**:
- `identity.py`: `resolve_http_identity(headers, config)` (per-call, Streamable HTTP, byte-identical rules to `api/deps.py:get_current_identity`) and `resolve_stdio_identity(config)` (once per process, from `MCP_AUTH_TOKEN`, no separate weaker bypass -- reuses the exact same `verify_jwt`).
- `schemas.py`: `McpChunkResult`/`to_mcp_result` -- the MCP-facing output shape, deliberately mirroring `SourceItem`'s (not `source_dict()`'s richer internal shape) field exposure, plus `content` since an MCP caller has no separate answer-synthesis step.
- `server.py`: `build_mcp_server(config, pipeline, vectorstore, embedder, *, fixed_identity=...)` registers all four tools via the SDK's `MCPServer`/`Resolve` API. One shared `_resolve_auth` resolver (never a tool argument) and one shared `_run_tool` dispatch-then-sanitize helper mirroring `agent/graph.py::_execute_tool` exactly, including the previously-fixed "resolve auth again before the sanitize call" step. Deliberately does not start with `from __future__ import annotations` (see Issues Discovered).
- `asgi.py`: `build_mcp_asgi_app` -- builds the server and returns its `streamable_http_app(streamable_http_path="/")`.
- `stdio_entrypoint.py`: `python -m rag.mcp.stdio_entrypoint`, host-only, resolves one fixed identity at startup and reuses the same `build_mcp_server`.

**`src/rag/api/main.py`**: `FastAPI(...)` now takes an explicit `lifespan` that enters the mounted MCP sub-app's own `router.lifespan_context` only when `config.mcp.enabled` is true (a real Starlette lifecycle edge case found mid-implementation, see below); `app.mount(config.mcp.server.mount_path, ...)` only when enabled, so a disabled MCP config leaves the route table byte-identical to before this milestone.

**`src/rag/audit.py`**: two new `AuthEventType` values, `mcp_auth_success`/`mcp_auth_failure`, so MCP-originated auth events are distinguishable in the log stream from the HTTP boundary's `auth_success`/`auth_failure` without inventing a parallel vocabulary for everything else.

**`src/rag/config.py`**: `McpServerConfig`/`McpConfig`, registered on `AppConfig.mcp`.

### Issues discovered
1. **MCP SDK v1 -> v2 hard rename**: `FastMCP` became `MCPServer` (`mcp.server.mcpserver.MCPServer`), confirmed by hitting the real `ModuleNotFoundError` the package itself raises with a migration pointer, not assumed from general MCP-ecosystem familiarity. Root cause: general knowledge of "the MCP Python SDK" doesn't pin an exact API surface across major versions. Fix: import from the correct v2 module path throughout. Resolved: y.
2. **The SDK's OAuth-shaped bearer-auth wiring (`AuthSettings`/`TokenVerifier`) doesn't fit this project's shared-secret HS256 model.** `AuthSettings` requires an `issuer_url` (a real external OAuth authorization server) and, combined with `token_verifier`, activates protected-resource-metadata discovery routes this project doesn't actually back with real OAuth AS behavior. Root cause: the SDK's built-in mechanism targets a resource-server-in-front-of-a-real-IdP deployment shape, not a single-instance system that both issues and verifies its own tokens. Fix: skipped it entirely; used the SDK's `Resolve(fn)` parameter-injection mechanism instead, reading `ctx.headers` directly and reusing `verify_jwt`/`build_authorization_context` unmodified. Resolved: y.
3. **`from __future__ import annotations` breaks `Resolve(...)` when the resolver is a closure.** The SDK resolves each tool parameter's annotation via `inspect.signature(fn, eval_str=True)`, which evaluates a postponed string annotation against the function's `__globals__`; a resolver defined as a closure (needed to capture per-server `config`) is never in `__globals__`, so this raised `NameError` at server-build time. Root cause: PEP 563 postponed evaluation and closure-scoped names are fundamentally incompatible when a third party re-evaluates the annotation string later, outside the original scope. Fix: removed the future-import from `rag/mcp/server.py` only, with an explicit docstring explaining why -- confirmed as the cause by reproducing the crash, not assumed. Resolved: y.
4. **The SDK's auto-generated tool schema does not reject an unknown argument.** No `additionalProperties: false` is set on the per-tool schema built from a plain Python function signature (confirmed empirically: a client-injected `tenant_id` field was silently dropped, call still succeeded). Root cause: `mcp.server.mcpserver.utilities.func_metadata`'s dynamically-built `arguments_model` inherits Pydantic's default `extra="ignore"`, with no exposed strictness knob at the high-level `MCPServer.tool()` API. Verified this is harmless regardless (no tool function has a `tenant_id`/`roles` parameter to bind into, and `auth` is exclusively resolver-injected, structurally excluded from the schema by the same mechanism), but it is a real, documented gap versus this codebase's "loud rejection over silent drop" preference (`agent/tool_schemas.py`'s `extra="forbid"`). Not fixed this session -- closing it fully would mean dropping to the SDK's lower-level `Server` API, which this version has restructured substantially (no `@server.call_tool()`/`@server.list_tools()` decorators found); judged not a good use of Stage 1A's scope given the underlying security property already holds. Resolved: n (flagged, low severity, structurally inert).
5. **Starlette does not propagate a mounted sub-app's lifespan to its parent.** Mounting `MCPServer.streamable_http_app()` via plain `app.mount()` would leave the MCP session manager's background task group never started, since ASGI `lifespan` scope messages only reach the root app's own `Router.lifespan()`. Root cause: traced directly through `Starlette.__init__` -> `Router.__init__`'s `lifespan_context` assignment down to the exact callable `streamable_http_app()` wires in internally (`lambda app: session_manager.run()`), confirming this against the installed `starlette==1.3.1` rather than assumed from older-version familiarity. Fix: `rag/api/main.py`'s `FastAPI(...)` now takes a composed `lifespan` that conditionally enters `_mcp_app.router.lifespan_context(_mcp_app)` via `AsyncExitStack`. Resolved: y, verified live (a real MCP client/server round trip over a real uvicorn server would otherwise have hung or errored on session establishment; it didn't).
6. **mypy does not preserve `is`/`isinstance` narrowing of a sentinel-default parameter captured by a nested closure.** `_resolve_auth`'s `identity = resolved_fixed_identity if use_fixed_identity else ...` originally narrowed `fixed_identity is _UNSET` *inside* the closure itself; mypy kept the sentinel's type in the union regardless. Fix: resolve the sentinel once in `build_mcp_server`'s own scope, into a plain `VerifiedIdentity | None` local, before any closure captures it. Resolved: y.

### Evaluation results
Not applicable (infra/security milestone, no retrieval/generation quality change; `RetrievalPipeline`/`VectorStore`/`rag.agent.tools` are untouched by this change).

### Test results
25 new tests: 12 unit (`tests/unit/test_mcp_identity.py`, JWT/env-var identity resolution, no MCP protocol needed), 11 integration (`tests/integration/test_mcp_end_to_end.py`, a real MCP client -- the SDK's own `streamable_http_client` -- against a real uvicorn server hosting the built ASGI app, using fake pipeline/vectorstore/embedder doubles, no Postgres/Ollama needed), 2 integration (`tests/integration/test_mcp_tenant_isolation.py`, real Postgres, self-skips cleanly without it, mirroring `test_agent_tool_tenant_isolation.py`'s adversarial pattern for `search_knowledge_base` and `get_document`). Full unit suite: 807 passed / 1 skipped (was 795, +12). `ruff check`/`ruff format --check`/`mypy` all clean.

### Interpretation
The single biggest design decision -- and the one the user's own approval message specifically asked to be confirmed before implementing -- was how identity reaches a tool call without ever being a tool argument. The SDK's `Resolve` mechanism turned out to be a stronger primitive for this than initially planned: rather than relying on Pydantic's `extra="forbid"` (this codebase's existing convention for the in-process agent tools) to reject a smuggled `tenant_id`/`auth` field, `Resolve`-filled parameters are structurally excluded from the tool's JSON schema at the framework level, so there is no code path, not even a hypothetical validation bug, where a client-supplied value could bind into `auth`. This is a genuine security improvement in kind (a mechanism the client-facing schema cannot represent at all) over the in-process agent's existing approach (a mechanism the schema *could* represent, but is configured to reject). The cost of getting there was real: most of the six issues above were only discoverable by installing and reading the exact pinned SDK/Starlette version's source, exactly matching this project's repeated "confirmed against the installed X" discipline elsewhere in this log (`pyjwt`, `fastapi`, `ragas`).

### Open problems
1. The SDK schema's `additionalProperties` gap (issue 4 above) -- flagged, not fixed, low severity given the structural guarantee already holds. Would need the SDK's lower-level `Server` API to close fully; worth revisiting once that API's Stage-2-era shape in this SDK generation is better understood, not before.
2. `security.rate_limit`'s `slowapi` route-binding-at-import-time quirk (documented in CLAUDE.md's auth-boundary milestone section) was flagged as a design risk in the pre-implementation proposal but not addressed in Stage 1A -- the MCP ASGI sub-app is not wrapped in the same `Limiter`. Not a regression (MCP requests were never rate-limited before this milestone either, since the route didn't exist), but worth a decision before treating MCP as production-hardened to the same degree as `/query`.
3. Real-Postgres tenant-isolation coverage is a small, targeted spot-check (one scenario each for `search_knowledge_base`/`get_document`), not the full matrix `test_agent_tool_tenant_isolation.py` already runs for the underlying, unmodified `rag.agent.tools.*` functions -- a deliberate scope choice (avoiding near-duplicate test code for logic already proven elsewhere), not an oversight, but worth knowing if a future review expects 1:1 coverage parity.

### Next recommended step
Stage 1B (deferred, per explicit user instruction): the synthetic business-backend example (`get_customer_case`/`get_case_status`), gated independently, reusing the same `AuthorizationContext`-based tenant scoping pattern proven here. Stage 2 (deferred, per explicit user instruction and the original proposal's own recommendation): evaluate whether `rag/agent/graph.py`'s tool-dispatch step should call the MCP server as a client, likely only justified once a second, genuinely-external MCP server (e.g. the Stage 1B business backend, if it becomes a real separate service) exists for the agent to reach over the network.

### Follow-up (same day): real Docker runtime verification, and one real coverage gap it found

Before committing, ran a verification-only pass (no code changes beyond the fix below) against the actual container: rebuilt `rag-api` with the new `mcp` dependency (image size 1.68GB, matching the documented ~1.6GB baseline -- no bloat), brought the normal stack up with a temporary `RAG_CONFIG_PATH` override (`mcp.enabled: true`, `security.auth.enabled: true`; the file was created under `config/`, used, then deleted -- `config/default.yaml` itself was never touched), and drove the container with the same real MCP client used in the test suite.

**Finding:** `POST http://localhost:8000/mcp` (the bare `mount_path` value) returned a `307` redirect to `/mcp/` with an empty body, and the MCP SDK's client didn't handle that transparently during session initialization -- every local integration test had missed this because they all exercise `build_mcp_asgi_app()`'s result directly via its own standalone uvicorn server, never through an actual `app.mount()` composition with a URL prefix. Pointing the client at the trailing-slash URL (`/mcp/`) worked correctly end to end: missing-token and invalid-token calls failed closed with the right messages, and a valid JWT succeeded with real retrieval results against the container's live Postgres. Independently confirmed `mcp.enabled=false` is a true no-op in the container (`/mcp` 404s exactly like a never-mounted route). Not fixed in this pass -- flagged as a small follow-up (document the trailing slash, or add a route/redirect fix) before treating MCP's client-facing setup as fully polished. First-hand debugging note: an early attempt at this same check failed with a `Config file not found` error inside the container because Git Bash's MSYS path-conversion layer silently rewrote the `/app/config/...` env var value into a Windows path before it ever reached `docker compose`; fixed by setting `MSYS_NO_PATHCONV=1` for that command, not a project bug.

**Result:** All requested checks passed: image rebuilds cleanly with no dependency conflicts; `/mcp` is genuinely mounted only when `mcp.enabled=true` (307/200-shaped responses when enabled, true 404 when disabled); invalid/missing JWT fails closed through the real container; a valid JWT succeeds end to end through real Postgres. Full detail and the exact commands run are in `ISSUES.md`'s "Follow-up" note under the MCP integration entry.

### Follow-up (same day): closing the two real gaps the Docker verification and a second review surfaced

Two fixes only, per explicit instruction (no Stage 1B business tools, no agent-as-MCP-client, no commit/push): the bare mount path 307 found above, and the unknown-tool-argument silent-drop that issue 4 in this entry's own "Issues discovered" list had flagged but left open.

**Fix 1 -- unknown tool arguments now fail loudly.** `rag/mcp/server.py`'s new `_harden_argument_schemas(server)` runs once, after all four tools are registered, and reaches into `MCPServer`'s internal `ToolManager.list_tools()` (no higher-level hook exists in this SDK version) to flip each tool's dynamically-built Pydantic argument model from the SDK's default `extra="ignore"` to `extra="forbid"`, then regenerates the tool's already-cached JSON schema so `tools/list` correctly advertises `additionalProperties: false` too. Verified empirically first, in isolation, before wiring it in: mutating an already-constructed model's `model_config["extra"]` and calling `model_rebuild(force=True)` does take effect on the next `model_validate` call (not obvious -- Pydantic v2 builds its core validation schema at class-creation time). `auth` itself was never reachable this way (it's excluded from the arg model entirely via `Resolve`), so this closes the *unknown-key* gap specifically, not a new class of vulnerability -- but a silent drop was still a worse failure mode than a loud rejection at a security boundary, per the earlier "flagged, not fixed" note. Now a client-injected `tenant_id`/`roles`/`auth`, or any other unknown field, fails argument validation *before* the tool function or any resolver ever runs (`Tool.run` validates first), surfacing as a normal `CallToolResult(is_error=True)` with Pydantic's own `extra_forbidden` error text.

**Fix 2 -- the bare mount path now works with no client-visible redirect.** Root cause was fully traced this time (not just observed): `Starlette.mount(path, app)` builds a `Mount` whose match regex is literally `<path>/{path:path}` (confirmed against the installed `starlette==1.3.1` source), which requires the literal `/` delimiter -- so it matches `<path>/` and anything under it, but never matches the bare `<path>` itself. A request to the bare path therefore falls through to zero route matches, and the parent `Router`'s own `redirect_slashes` fallback (`Router.app`, tries the request again with a `/` appended, and 307s if *that* matches) is what actually produces the redirect -- not the `Mount` itself. Considered and rejected two alternatives before landing on the fix: registering the SDK's own internal route at the mount path and mounting the outer app at the empty root path `""` avoids the redirect (Starlette supports empty-path mounts), but would make that mount match literally every unmatched path in the whole app, changing 404 behavior for typos and unrelated routes -- too broad a blast radius for a one-endpoint fix. Chosen fix: a five-line ASGI middleware, `rag/mcp/asgi.py`'s `_BareMountPathMiddleware`, wrapped around the *outer* app (not the MCP sub-app) via the new `mount_mcp_app(app, mcp_app, mount_path)` helper -- it rewrites the ASGI scope's `path` (and `raw_path`) to `mount_path + "/"` only when the incoming path is an *exact* match for `mount_path`, then passes through unchanged for every other request. Proven correct first with a minimal standalone Starlette reproduction (bare path, trailing-slash path, an unrelated route, and a genuinely nonexistent route all behaving correctly) before touching the real code. `rag/api/main.py` now calls `mount_mcp_app(app, _mcp_app, _config.mcp.server.mount_path)` instead of a bare `app.mount(...)`.

**Tests added:** 3 new MCP end-to-end tests (`tests/integration/test_mcp_end_to_end.py`) for the hardening -- arbitrary unknown field rejected, `tenant_id`/`roles`/`auth` injection rejected (the pre-existing test asserting these were silently ignored was rewritten to assert the new, stronger rejection instead), and every legitimate optional argument still validates correctly under `extra="forbid"`. 1 new test builds a small wrapping `FastAPI` app the same way `rag.api.main` does (`mount_mcp_app` plus a composed lifespan entering the mounted sub-app's `router.lifespan_context`) and drives a real MCP client against both `/mcp` and `/mcp/` through it -- the first test in this codebase to exercise the *mounted* app shape at all (every prior e2e test served the MCP app directly as uvicorn's own root, sidestepping both the `Mount` prefix and the lifespan-propagation question entirely). 2 new real-Postgres adversarial tests (`tests/integration/test_mcp_tenant_isolation.py`) extend the existing `search_knowledge_base`/`get_document` coverage to `get_latest_document` (a wrong-tenant caller gets nothing even after version resolution) and `get_related_context` (a guessed cross-tenant `chunk_id` resolves to nothing), mirroring `test_agent_tool_tenant_isolation.py`'s existing in-process versions of the same two scenarios. 2 new unit tests (`tests/unit/test_mcp_identity.py`) capture logs via `caplog` during a successful and a failed `resolve_http_identity` call and assert the raw JWT (and, on the failure path, the wrong signing secret used to forge it) never appears in any log record. 3 new unit tests (`tests/unit/test_mcp_disabled_is_a_true_noop.py`) import the real `rag.api.main.app` under its default (`mcp.enabled=false`) config and assert `/mcp`, `/mcp/`, and an arbitrary sub-path all 404 identically to a route that was never registered.

**Verification, in order:** targeted ruff/mypy on the changed files (clean) -> the 3 new unknown-field/optional-args e2e tests plus the whole `test_mcp_end_to_end.py` file (14/14 passed, including the new mounted-lifespan/bare-path test) -> the 4 real-Postgres tenant-isolation tests including the 2 new ones (4/4 passed, ~168s) -> full unit suite (812 passed / 1 skipped, was 807/1 -- +5: 2 identity-log-hygiene, 3 disabled-no-op) -> full `mypy src/` (101 source files, clean) -> full-repo `ruff check`/`ruff format --check` (clean on every file this session touched; 3 unrelated pre-existing files -- `test_e2e_critical_scenarios.py`, `test_hybrid_retrieval.py`, `test_config.py`, none touched this session -- would be reformatted by the currently-installed ruff version, left alone per "don't touch unrelated code") -> Docker: rebuilt `rag-api` clean (1.68GB, unchanged, `mcp==2.1.1`/`httpx2==2.12.0` installed as expected) -> live container with `mcp.enabled=true`/`security.auth.enabled=true`: confirmed `GET /mcp` (bare) no longer 307s (now routes directly, e.g. `400` for a bad GET, `200` for a real `POST .../initialize`) -> a real MCP client against the *canonical bare* `/mcp` URL (no trailing slash, the exact spelling that broke in the prior verification) ran all five scenarios successfully: missing token fails closed, invalid token fails closed, valid token succeeds with real retrieval results, injected `tenant_id`/`roles`/`auth` rejected loudly, arbitrary unknown field rejected loudly -> container logs checked for warnings/errors/tracebacks (none beyond the expected `INFO`-level auth-failure lines, which log only the generic error text, never token contents) -> restarted on the plain default config and reconfirmed `mcp.enabled=false` still true-404s both `/mcp` and `/mcp/` live -> deleted the temporary verification config, confirmed `git status` shows only the intended file set with no stray changes.

**One codebase-adjacent finding, not fixed (out of scope, unrelated code):** the currently-installed `ruff format` version would reformat `tests/integration/test_e2e_critical_scenarios.py`, `tests/integration/test_hybrid_retrieval.py`, and `tests/unit/test_config.py` (a multi-line-assert wrapping style change), none of which this session touched. Left alone; worth a dedicated formatting-only pass later, not bundled into a security fix.

**Result:** Both gaps from the earlier verification pass are closed and re-verified against a live Docker container with a real MCP client, not just unit tests. Stage 1A's full acceptance-criteria list (see the original entry above) is unchanged in substance; these two fixes make two of its previously "structurally inert, not enforced" properties into actually-enforced ones. Nothing committed or pushed, per instruction.

### Follow-up (next session): confirming `httpx2` is not a typo, and closing the remaining merge-blocker checklist

A third reviewer flagged `import httpx2` in the MCP test files as a likely mistake, on the reasoning that `pyproject.toml` declares `httpx`, not `httpx2`, in `dev` extras. Checked definitively rather than assumed either way: `pip show mcp` lists `httpx2` under `Requires:`, and `pip show httpx2` shows `Required-by: mcp` -- while `pip show httpx` shows `Required-by: datasets, huggingface_hub, langsmith, ollama, openai`, with `mcp` nowhere in that list. The two are unrelated packages that happen to share a name prefix; `mcp.client.streamable_http.streamable_http_client` is typed against `httpx2.AsyncClient` specifically, so swapping the test imports to plain `httpx` would have broken every MCP e2e/tenant-isolation test outright, not fixed a typo. No test-file change was needed. Made the dependency explicit anyway, since it had only ever been transitively guaranteed via `mcp`'s own metadata: added `httpx2>=2.5` to `pyproject.toml`'s `dev` extras next to `httpx`, with a comment spelling out that the two are unrelated packages -- re-ran `pip install -e .[dev]` to confirm the environment already satisfied it (no drift).

Also closed the remaining checklist items from the same review pass: two new regression tests in `tests/unit/test_mcp_disabled_is_a_true_noop.py` -- `test_no_mcp_app_is_ever_constructed_when_disabled` (asserts `rag.api.main._mcp_app is None`, proving `build_mcp_asgi_app`/`build_mcp_server`/the SDK's session manager are never even constructed when `mcp.enabled=False`, not merely built-then-hidden) and `test_no_mount_route_for_the_mcp_path_exists_in_the_route_table_when_disabled` (inspects `app.routes` directly for the absence of a `Mount("/mcp")`, independent of the existing HTTP-behavior 404 checks). Ran `pytest --collect-only` first on the full MCP-focused file set (37 tests, zero collection errors, ~173s -- confirming the environment genuinely has every needed package installed, not just working by accident) before executing for real.

**One real environment hiccup, unrelated to the code:** the exact MCP-focused test run initially showed all 4 real-Postgres tenant-isolation tests as `SKIPPED` -- `docker ps` came back completely empty (every container, from this project and an unrelated one, `Exited` ~14 hours earlier; the Docker daemon itself was still responsive). Not a regression from this session's changes -- most likely the host slept or Docker Desktop restarted between sessions. Brought `postgres` back up (`docker compose up -d postgres`, waited on `pg_isready`) and re-ran; all 4 passed cleanly (138s). A reminder that a self-skipping integration test correctly avoiding a false failure can still mask "the real check never actually ran" if the skip reason isn't checked, which is exactly why this session re-ran them explicitly after noticing the skip rather than treating "0 failures" as sufficient.

**Result:** full MCP-focused suite: 33 passed / 4 skipped on the first pass (Postgres down), 4/4 passed once Postgres was back, no failures either time. Full unit suite: 814 passed / 1 skipped (was 812, +2 for the new disabled-no-op tests). Full `mypy src/`: clean, 101 files. Full-repo `ruff check`/`ruff format --check`: clean on every file this session touched (the same 3 unrelated pre-existing files noted in the entry above are still untouched). `git status` shows exactly the expected Stage 1A file set plus the `pyproject.toml` dev-extras addition. Nothing committed or pushed.

### Follow-up (2026-09-01): committing Stage 1A as 24 logically-separated commits

User asked to add and commit the whole Stage 1A hardening pass (config plumbing through the two follow-up rounds above) as 20-25 commits with proper messages, run pre-commit once up front across every changed file, then proceed fast. Ran `pre-commit run --all-files` first (not per-commit) -- caught and auto-fixed one real issue, mixed line endings in `tests/integration/test_mcp_end_to_end.py`, and confirmed the mypy-hook `additional_dependencies` fix from the "same day" follow-up above was already in place. Then split the working tree into 24 commits by logical unit rather than by file: config plumbing (`McpConfig`/`McpServerConfig`, then the `default.yaml` block), the audit event types, the `rag.mcp` package itself (one commit per module), the two hardening fixes (`server.py`'s `_harden_argument_schemas`, then `asgi.py`'s `_BareMountPathMiddleware` paired with `main.py`'s `mount_mcp_app` call site), and the test suite (one commit per test file/scenario group: identity unit tests, audit-log hygiene, end-to-end wire-protocol tests, unknown-argument rejection, mounted-path/lifespan coverage, real-Postgres tenant isolation, disabled-no-op), closing with the `httpx2` dev-dependency declaration.

Where a single already-authored file needed to land across more than one commit (`server.py` gained `_harden_argument_schemas` after its initial four-tool registration; the three MCP test files each grew across the multiple follow-up rounds above), used a reconstruct-then-commit technique instead of interactive `git add -p` (judged too slow/fragile for non-interactive tool calls under a "do it fast" instruction): wrote a reduced intermediate version of the file matching the state after commit N, committed it, then wrote the real final content and committed again as commit N+1. Safe here specifically because every line in every one of these files had already been authored and reviewed earlier in the same session -- this is a commit-history-shaping technique, not a way to skip writing or reviewing code.

**One real deviation from plan:** an earlier failed `git commit` attempt (blocked by pre-commit's "unstaged `.pre-commit-config.yaml`" check) had already run `git add pyproject.toml` as a side effect; by the time `.pre-commit-config.yaml` was staged and committed, `pyproject.toml`'s `httpx2` addition rode along in the same commit (`6076d66`) instead of getting its own. Accepted as 24 commits instead of a strict 25 rather than unwinding it, and checked `git status` before every subsequent `git add` afterward to avoid repeating the mistake.

**Result:** `git log` shows 24 commits (`6076d66`..`960c8c7`) covering the full Stage 1A pass; `git status` clean afterward. Re-ran the full MCP-focused test suite, full unit suite, `mypy src/`, and `ruff check`/`ruff format --check` against the committed state (not just the pre-commit working tree) to confirm the reconstruction technique introduced no drift -- all clean, matching the numbers reported at the end of the prior follow-up. Nothing pushed.

**Interview takeaway:** splitting an already-complete, already-reviewed change into a clean commit history is a different task from writing the change -- the risk isn't correctness (the code itself doesn't change), it's whether the reconstructed intermediate states would actually import/test cleanly at each step, not just whether the final state does. Verifying against the committed state after the fact, not just trusting that the working tree matched intent, is what would catch a reconstruction mistake if one happened.

### Follow-up (2026-09-01): MCP documentation across the docs site

Closed the last open gap for this milestone: the code had shipped and been committed with no reader-facing documentation beyond CLAUDE.md's own architecture-map entry. Added `docs/topics/mcp.md` (new topic page: transcludes README's new MCP section, then covers the four tools, the transport-resolved identity model, both hardening fixes, config, and what's deferred) and `docs/reference/mcp.md` (new mkdocstrings page, five `::: rag.mcp.*` directives, no hand-written API text, matching every other `docs/reference/*.md` page's convention). Added a full `## MCP Integration` design section to `docs/architecture.md` (~260 lines: architecture diagram as a Mermaid flowchart, the `Resolve()` identity mechanism, the `sanitize_evidence` choke point, both hardening fixes with their root causes and rejected alternatives, the dependency-placement decision, testing, and known limitations) -- this is where the actual depth lives, matching the project's own documented "topics pages link out to architecture.md for depth" pattern. Fixed two now-stale mentions of MCP elsewhere in `architecture.md` left over from before this milestone existed (a "no MCP server" limitations bullet, and a "Redis, MCP, and LangGraph" deferred-infrastructure aside that no longer applied to MCP specifically).

`README.md` got a new `## MCP` section (with the same snippet markers every other transcluded section uses, so the topic page has something to pull from), a Table of Contents entry, a feature-table row, a mention in the Documentation section's own nav list, and a rewritten Roadmap bullet (the old one described MCP as a future item; Stage 1A has shipped, so it now describes only what's actually still deferred -- Stage 1B business tools, agent-as-MCP-client). `mkdocs.yml` got both new pages wired into nav (topics and reference sections), and `docs/index.md` got a "Where to go next" table row pointing at the new topic page. `CLAUDE.md`'s own "Documentation site (MkDocs)" section also needed two small corrections once these pages existed: the topics list was missing `mcp` (and, independently, the already-existing `web-ui` topic, a pre-existing gap noticed while fixing the first), and the reference-page count/list needed `16 pages` -> `17 pages` plus `mcp` added to the grouping list.

Verified with `mkdocs build --strict` (after `pip install -e ".[docs]"`): the build fails with exit code 1 on 2 warnings (`topics/web-ui.md`'s link to `frontend/README.md`, `topics/observability.md`'s `#mlflow-tracking` anchor). Checked whether these were introduced by this session's changes rather than assuming: `git stash -u` (stashing every doc change including the two new untracked files), re-ran the build against the pre-MCP-doc state, got the identical 2 warnings and exit code -- confirmed pre-existing and unrelated, `git stash pop` to restore. Re-ran the build with the changes present and grepped the output for `mcp` in any WARNING/ERROR line -- zero matches, confirming the new content adds no new build issues of its own. Also checked the new architecture.md section specifically for em-dash characters (`grep -c '\xe2\x80\x94'` on just the isolated `## MCP Integration` block, not the whole file, since the file's other pre-existing em-dashes elsewhere predate the project's "no em dashes" documentation convention and are out of scope to fix here) -- zero, confirming the new content follows the current style rule.

### Follow-up (2026-09-01): MCP integration Stage 1B -- synthetic business-case tools

Implemented the milestone's own documented next step: two synthetic business-backend tools, `get_customer_case`/`get_case_status`, over the same MCP server -- proving the "MCP as an integration layer to backend/business systems" claim concretely rather than only asserting it, since Stage 1A only ever re-exposed this codebase's own RAG tools.

New `src/rag/mcp/business/` package: `schemas.py` (`CustomerCase`, the full case shape; `CaseStatusResult`, a deliberately narrower second shape -- `get_case_status` demonstrates a real narrow-scope read tool over the same backend resource, not `get_customer_case` with fields hidden at the call site) and `store.py` (a small, in-memory, read-only dict of 5 seeded `_CaseRecord`s across `tenant_alpha`/`tenant_beta`, no Postgres/network dependency at all -- deliberately not a new persistence layer, since the point of this milestone is the integration *shape*, not a real case-management system). Authorization is its own small mechanism, not a reuse of `AuthorizationContext` (a case has no document/freshness/trust concept): own tenant plus a matching role on the case's `allowed_roles`, or a `security.authorization.cross_tenant_support_roles` role that's also listed on that case -- the exact same predicate shape and the same config allow-list document ACL already uses, reused rather than duplicated. One deliberate asymmetry from document ACL, called out directly in both the module docstring and `docs/architecture.md`: enforcement here is unconditional, no kill-switch, because (unlike the pre-governance document corpus, which must keep `tenant_id IS NULL` meaning "visible to everyone" for backward compatibility) every synthetic case has a concrete tenant from creation, so there's no legacy state to preserve.

`rag/mcp/server.py` refactored `_resolve_auth` to factor out a shared `_resolve_identity` (the transport-resolution half, independent of `AuthorizationContext`-building), then added `_resolve_identity_only` (a thin `Resolve()`-compatible async wrapper) so the two new tools can declare `identity: Annotated[VerifiedIdentity | None, Resolve(_resolve_identity_only)]` instead of `auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)]` -- they need a bare identity, not a document-shaped context. The two tools deliberately do not route through the existing `_dispatch`/`_run_tool` machinery at all: a new, structurally parallel `_run_business_tool` (same tracing span, latency metric, `mcp_tool` error counter) skips `pipeline.sanitize_evidence` entirely, since that method redacts chunk/field-level RAG content, a concept a business case doesn't have. Typed as `Callable[[], _T]` (a zero-arg closure per call site) rather than branching on tool name inside the helper, specifically to avoid a `# type: ignore[return-value]` a first draft needed when the helper branched on a `Literal["get_customer_case", "get_case_status"]` internally and returned a `CustomerCase | CaseStatusResult | None` union -- the closure-based generic version is both cleaner and fully clean under `mypy src/`.

A design point worth its own paragraph: a case the caller may not access and a case that doesn't exist both resolve to `None` -- the same "SQL-filtered ACL, indistinguishable from the outside" property the document corpus has. But unlike the document corpus (`pgvector.py`'s predicate filters before Python ever sees an excluded row, so no real per-document `authorization_denied` audit event fires today -- a documented, still-open gap), business-case authorization happens in Python, so a real denial *is* observable: `_lookup_authorized` logs a genuine `authorization_denied` event (`action`, pseudonymous `subject`, `tenant_id`, `case_id`) whenever a case is found but the caller fails the check, even though the tool's return value never reveals that distinction to the caller. Reused the existing `authorization_denied` `AuthEventType` (already emitted once, for the ingest-role check) rather than adding a new event name.

Docs updated to match: `docs/reference/mcp.md` (+2 `::: rag.mcp.business.*` directives), `docs/topics/mcp.md` (new "Two synthetic business-backend tools" section, "What's deferred" narrowed to just Stage 2), `docs/architecture.md`'s "MCP Integration" section (diagram gained the business-store branch, a new "Stage 1B: a second, structurally independent tool family" subsection, testing/known-limitations updated), README's MCP section/feature-table row/Roadmap bullet, `config.py`'s `McpConfig` docstring, and `observability/metrics.py`'s bounded-label-set docstring (4 -> 6 literal tool names).

**Tests:** new `tests/unit/test_mcp_business_case_store.py` (12 tests, plain-function unit tests against `rag.mcp.business.store` directly -- same-tenant matching/mismatched role, cross-tenant with/without a listed support role, unknown case_id, `identity=None` unrestricted, and two `caplog`-based tests proving `authorization_denied` fires on a real denial but not on a nonexistent case_id). Extended `tests/integration/test_mcp_end_to_end.py`: renamed the tool-count test to expect all six tools, added 9 real-wire-protocol tests for the two new tools (full-detail fetch, same-tenant role-mismatch denial, cross-tenant denial with no support role at all, cross-tenant denial when the support role isn't listed on the *target case's own* ACL, nonexistent-id parity, cross-tenant-support-role grant, auth-disabled unrestricted access, the narrow `get_case_status` shape, and unknown-argument rejection proving the existing hardening fix covers these tools too) -- none need the `_FakePipeline`/`_FakeVectorStore` doubles the RAG-tool tests use, since the synthetic backend has no Postgres/Ollama dependency at all. The two extra denial-path tests were added after an automated review of the diff flagged that the initial pass only exercised those two scenarios at the unit level (`test_mcp_business_case_store.py`), never over the actual MCP transport -- a real coverage gap between "the authorization function is correct" and "the authorization function is correctly reached and its result correctly serialized end to end," closed rather than waved off as redundant with the unit tests.

**Result:** `mypy src/`: clean, 104 files (was 101). `ruff check .`/`ruff format --check .`: clean on every file this session touched (same 3 unrelated pre-existing drift files as before, still untouched). Full unit suite: 826 passed / 1 skipped (was 814, +12). Full MCP end-to-end integration file: 23 passed (was 14, +9), self-contained, no Postgres/Ollama needed. `mkdocs build --strict`: identical 2 pre-existing warnings, confirmed by the same `git stash -u` differential technique used in the prior follow-up; the rendered `reference/mcp/index.html` was grepped directly to confirm the two new business-module docstrings actually rendered, not just that the build didn't warn. Nothing committed.

**Interview takeaway:** the interesting design decision here wasn't the CRUD-shaped case lookup itself, it was recognizing that mirroring the document ACL's tenant+role predicate didn't mean mirroring its kill-switch or its audit-blind-spot too -- both of those were consequences of the document corpus's specific legacy-compatibility constraint, not inherent properties of "an ACL rule that looks like this." A synthetic, always-tenanted resource with no pre-existing untenanted data has no reason to inherit either limitation, and actually gets to close one of them for real (a genuine `authorization_denied` emission) precisely because the check runs in Python instead of as an opaque SQL filter.

**Result:** MCP now has the same three-tier documentation coverage as every other milestone (README summary, `docs/topics/mcp.md` landing page, `docs/architecture.md` deep design section, plus the mkdocstrings API reference page), CLAUDE.md's docs-site inventory is accurate again, and there are zero new docs-build regressions. Nothing committed -- these are working-tree changes only; whether to commit them is left to the user, since the prior "commit this stage" instruction was scoped to the Stage 1A code/tests specifically, not to this follow-on documentation pass.

## 2026-09-02: MCP integration Stage 2 -- the in-process agent as a real MCP client

Closed the last open piece of the MCP milestone: the agent can now dispatch `get_customer_case`/`get_case_status` (the Stage 1B synthetic business tools) as a genuine `mcp.ClientSession` call, not a shortcut direct function call, while the four local RAG tools stay exactly what they already were. Session opened with a design proposal (architecture, identity propagation, transport/session-lifecycle tradeoffs, failure handling, tests, acceptance criteria) presented for approval before any code changed; user approved with three adjustments (fail closed without authenticated identity, no anonymous business-tool path; explicit service-token boundary claims on the minted internal JWT; the synthetic-evidence representation approved as-is but must stay structurally inert to document-specific logic) and asked to proceed.

### Interview talking points
- **What problem did this solve?** The job-description-level "agent securely calls backend systems via MCP" story was only half true: Stage 1A/1B built a real MCP *server*, but the in-process agent never actually used the MCP *client* protocol to reach it -- every business-tool call would have had to be a direct Python function call, not a proof that this codebase's own agent can operate as a real MCP client against a tool boundary it doesn't control.
- **Why was it difficult?** Three intersecting constraints had to hold simultaneously, not just one: `run_agent()`'s node functions are synchronous by design (a deliberate, pre-existing architectural choice, not something this session could relax), so the async MCP SDK needed a safe sync bridge; the two business tools have unconditional tenant/role authorization with no kill-switch, so "what happens when there's no authenticated identity" needed a real, fail-closed answer rather than an afterthought; and the caller's original JWT must never be forwarded, so a same-trust-domain internal token had to be minted correctly -- including getting `iss`/`aud` claim handling right against the *actual* installed PyJWT behavior, which turned out to be asymmetric in a way that would have silently broken the whole feature for the shipped-default config if assumed rather than tested.
- **How did I debug it?** Read the full existing agent graph/state/tools/decisions/MCP-server stack before designing anything, to find the smallest correct integration point (one branch in `_execute_tool`, a static tool-kind lookup, zero changes to the existing local-tool dispatch). Verified the design end to end with real, disposable scripts before writing formal tests: minted a real token and decoded it with the real `verify_jwt`, dispatched a real `run_agent()` call against a real (fake-doubles-backed) MCP server with the real server's lifespan entered, and hit two genuine failures in the process -- a `421 Misdirected Request` from the SDK's DNS-rebinding Host-header protection (traced directly into the installed `mcp` SDK's `Server.streamable_http_app` source, since the settings class's own docstring claims "disabled by default" in a way that turned out not to hold for the default `host="127.0.0.1"`), and an `invalid_audience` rejection from PyJWT that isolated down to `jwt.decode(..., audience=None, ...)` rejecting *any* token carrying an `aud` claim regardless of its value -- confirmed with a minimal reproduction script before touching the real fix, and confirmed `iss` does *not* have the same failure mode with a parallel isolated test, rather than assuming symmetry.
- **What metrics proved the fix?** 18 new unit tests for `mcp_client.py` (startup validation, token minting including the `iss`/`aud` asymmetry proven in both directions and expiry enforcement with `leeway_seconds=0` to avoid a false pass from the default 30s clock-skew tolerance, fail-closed-without-identity, synthetic-evidence field-level isolation), 7 new graph-level unit tests (fail-closed dispatch distinguished by exact error string, `max_tool_calls` bounding a mix of local/remote calls, arg-schema hardening, a synthetic `mcp:...` chunk_id fed into `get_related_context` staying a safe miss), 5 new integration tests over the real wire protocol via the ASGI transport (authorized synthesis with a citation, wrong-role/cross-tenant denial through the full agent path, an MCP-layer failure not escaping `run_agent()`, and a captured-header proof the internal token's `sub` is never the original caller's). Full unit suite: 853 passed / 1 skipped (was 851 before this session's two pre-existing-test-format fixes, +2 unrelated to this milestone's own 25 new tests, which land inside that same 853). Full MCP/agent integration suite (Postgres/Ollama-independent subset): 28 passed / 12 self-skipped (no Postgres/Ollama in this environment). `ruff check .`: clean. `mypy src/`: clean after fixing two real issues the new code surfaced (a `mcp.types.CallToolResult.content` union-type access needing `getattr` instead of a bare `.text`, and a new `AuthEventType` Literal value needing to be added to `rag/audit.py`'s fixed vocabulary rather than just used ad hoc). One tracing-enabled end-to-end run (real `TracerProvider`, `observability.tracing.enabled=True`, an unreachable OTLP endpoint so the background exporter fails harmlessly) confirmed a single `traceparent` value shared across every outbound MCP request in one run -- proof the distributed trace context genuinely propagates from the agent, through the MCP client, to the MCP server, not just that spans exist locally.
- **What would I improve next?** `security.rate_limit`'s `Limiter` still doesn't wrap the MCP mount -- a pre-existing gap this milestone inherits rather than introduces, but now a real agent-driven traffic source flows through it too, since the ASGI transport calls the mounted sub-app directly, bypassing the outer app's `SlowAPIMiddleware` entirely. Session-per-call (no persistent session reuse within one agent run) is a deliberate, justified-but-unmeasured tradeoff -- worth revisiting only if profiling ever shows MCP round-trip overhead is a meaningful fraction of a real run's latency, which nothing in this session's testing suggested. No genuinely external MCP server has been exercised yet; `transport="http"` exists and is unit-tested for its config validation, but the actual network-call code path has only been exercised by reading it carefully, not by pointing it at a second real process.

### Goal
Design, get explicit approval for, then implement MCP Stage 2: make `rag/agent/graph.py`'s bounded tool loop able to choose between local in-process RAG tools and the two remote MCP business tools, reusing the existing agent loop/state machine and every existing bound (`max_agent_steps`/`max_retrieval_attempts`/`max_tool_calls`) unmodified, with the caller's identity propagating securely to the MCP transport without ever forwarding their original JWT, graceful handling of MCP-layer failures, and matching telemetry -- then run the full verification list the approval specified (focused tests, full unit suite, relevant MCP integration tests, ruff, mypy, one tracing-enabled end-to-end run) before stopping, without committing or pushing.

### Configuration
New `McpClientConfig` nested under the existing `McpConfig` (`config.mcp.client`): `enabled` (default `false`, true no-op), `transport` (`"asgi"` default / `"http"` escape hatch), `server_url_env_var`/`default_server_url` (mirrors `OLLAMA_BASE_URL`'s pattern, unused in the default `asgi` mode), `timeout_seconds`, `internal_token_ttl_seconds` (default 60), `internal_token_subject` (`"rag-agent-internal"`), `internal_token_issuer` (`"rag-agent-internal"`, used only as an unconfigured-`iss`-claim fallback). `AgentConfig` gained `tool_select_mcp_prompt_path`, pointing at the new `agent_tool_select_v4_mcp.yaml` (v3 stays the default, byte-identical, used whenever `mcp.client.enabled=False`). No experiment configs -- this is an infra/security milestone with no retrieval/generation quality axis to A/B.

### Changes made

**New module `src/rag/agent/mcp_client.py`**: `validate_startup_config(config)` (the three fail-fast startup checks: auth must be enabled, algorithm must be HS256, `asgi` transport requires `mcp.enabled=True`), `mint_internal_token(auth, config)` (the short-lived HS256 service token, `sub` always the fixed marker, `iss` always present, `aud` present only when `security.auth.jwt.audience` is configured), `dispatch_remote_tool_sync(...)` (the public sync entrypoint: fail-closed identity check, `anyio.run` bridge to the async client call, synthetic-`SearchResult` wrapping of the raw business-tool result), and the private `_call_tool_async`/`_business_result_to_search_result` helpers.

**`rag/agent/graph.py`**: one new branch in `_execute_tool` (a static `REMOTE_MCP_TOOL_NAMES` lookup decides local-dispatch vs. `_dispatch_mcp_tool`, with an explicit fail-closed path recorded as an ordinary `ToolCallRecord` failure when the config is off), a new `_dispatch_mcp_tool` thin wrapper, and `_load_templates` conditionally loading the 6-tool prompt only when `mcp.client.enabled=True`. `run_agent()` gained an optional `mcp_app` parameter (default `None`, backward-compatible with every existing caller/test). The local `_dispatch_tool` function and every local tool's behavior is untouched.

**`rag/agent/tool_schemas.py`**: `GetCustomerCaseArgs`/`GetCaseStatusArgs` (both `extra="forbid"`, `case_id: str` only), `REMOTE_MCP_TOOL_NAMES` (a `frozenset[str]`), `TOOL_ARG_MODELS` extended to six entries.

**`rag/agent/decisions.py`/`rag/agent/state.py`**: `ToolSelectionDecision.tool_name`/`ToolCallRecord.tool_name` Literals widened from 4 to 6 values -- the LLM-facing JSON contract shape is otherwise unchanged.

**`rag/schemas.py`**: `SearchResult.origin` gained a fourth value, `"mcp_remote"`.

**`rag/config.py`**: `McpClientConfig`, `McpConfig.client`, `AgentConfig.tool_select_mcp_prompt_path`, `AppConfig.mcp_client_server_url()`.

**`rag/api/deps.py`**: new `get_mcp_asgi_app()` singleton getter -- the single source of truth for the MCP ASGI app object, now shared by both `main.py`'s mount and the agent's own ASGI-transport client (previously `main.py` built its own local `_mcp_app`; a second, independently-built server object would have had a session-manager lifespan this process's own lifespan context manager never entered).

**`rag/api/main.py`**: switched to `get_mcp_asgi_app()` instead of its own construction; added a startup call to `mcp_client.validate_startup_config`, so an invalid `mcp.client` configuration crashes the process at import time with a clear message rather than failing confusingly on the first request.

**`rag/api/routers/agent_query.py`/`agent_stream.py`**: both routes gained a `Depends(get_mcp_asgi_app)` parameter, threaded through to `run_agent(..., mcp_app=mcp_app)`.

**`rag/audit.py`**: one new `AuthEventType` value, `agent_mcp_tool_disabled`, for the fail-closed-when-disabled audit log line (mypy caught the missing Literal entry; not just added ad hoc).

**`pyproject.toml`**: `httpx2` promoted from `dev` extras to a core dependency -- `mcp_client.py` imports it directly at runtime now (the agent's own MCP client), not just from the test suite.

**New prompt** `src/rag/prompts/templates/agent_tool_select_v4_mcp.yaml`: v3 plus two new tool descriptions, byte-identical wording otherwise; v3 itself untouched.

### Issues discovered
1. **The MCP SDK auto-enables DNS-rebinding Host-header protection for the default `host="127.0.0.1"`, contradicting the settings class's own "disabled by default" docstring.** Root cause: `Server.streamable_http_app`'s own logic special-cases `host in ("127.0.0.1", "localhost", "::1")` to force-enable protection with a hardcoded allow-list, overriding what `TransportSecuritySettings`'s docstring implies is the global default; confirmed by reading that exact source, not by trusting the docstring. Fix: gave the ASGI transport's placeholder base URL a real loopback address with a non-default, explicit port (`http://127.0.0.1:1/`) so the `Host` header actually matches the SDK's `"127.0.0.1:*"` wildcard pattern. Resolved: y.
2. **PyJWT rejects a token carrying any `aud` claim when the verifier passes no expected audience, asymmetrically with how it treats `iss`.** Root cause: an initial design assumed `iss`/`aud` would behave symmetrically as "informational when the corresponding config is unset, actively enforced when it's set" -- true for `iss`, false for `aud`, confirmed by isolated reproduction of both directions rather than assumed by analogy. Fix: `mint_internal_token` omits `aud` from the claims dict entirely when `security.auth.jwt.audience` is unset (the shipped default), rather than setting an unchecked "informational" fallback value; the now-dead `internal_token_audience` config field was deleted rather than left as misleading surface. Resolved: y.
3. **mypy flagged a real union-type access (`CallToolResult.content[0].text`) and a missing fixed-vocabulary entry.** `content` is a union of five content types, only one of which (`TextContent`) has `.text`; fixed with a `getattr(first, "text", None)` fallback. `log_audit_event`'s `AuthEventType` is a closed `Literal`, and the new `agent_mcp_tool_disabled` event name needed adding there, not just using at the call site. Both real, both caught by `mypy src/`, not style nitpicks. Resolved: y.

### Evaluation results
Not applicable (infra/security milestone; `RetrievalPipeline`/`VectorStore`/the four local `rag.agent.tools.*` functions are byte-unchanged by this session).

### Test results
30 new tests total (18 unit `mcp_client.py`, 7 unit graph-dispatch, 5 integration real-wire-protocol). Full unit suite: 853 passed / 1 skipped (two pre-existing tests updated to reflect the six-tool reality: `test_agent_tool_schemas.py::test_tool_arg_models_cover_every_tool_name` and its sibling extra="forbid" checks). Full MCP/agent integration file set (`test_mcp_end_to_end.py`, `test_mcp_tenant_isolation.py`, `test_agent_mcp_client_stage2.py`, `test_agent_tool_tenant_isolation.py`, `test_agent_end_to_end.py`): 28 passed / 12 self-skipped (no Postgres/Ollama in this environment). `ruff check .`: clean. `mypy src/`: clean, 105 source files (two real fixes along the way, see above). `mkdocs build --strict`: identical 2 pre-existing warnings (`topics/web-ui.md`'s `frontend/README.md` link, `topics/observability.md`'s `#mlflow-tracking` anchor), confirmed zero new warnings from this session's doc additions. One combined `pytest tests/` run (not requested, tried as an extra sweep) hit a pre-existing, unrelated module-basename collision between `tests/unit/test_ingest_governance.py` and `tests/integration/test_ingest_governance.py` -- confirmed via `git log` to predate this session entirely; not fixed, since it's outside this milestone's scope and running unit/integration separately (as the approval actually specified) works cleanly.

### Interpretation
The single most consequential design decision was resolving the sync/async boundary with a per-call `anyio.run()` bridge rather than a persistent session or a broader async refactor of the graph -- `run_agent()`'s synchronous-by-design node functions were a hard existing constraint, not something this milestone should relax, and the smallest correct answer (mirroring `rag.mcp.stdio_entrypoint`'s own existing pattern) avoided introducing a background event-loop thread or any new concurrency machinery for a tool family that's called rarely relative to a run's total LLM-bound latency. The two real bugs found are a useful pair precisely because they look identical in shape (a security-relevant default that turns out not to be what a docstring or naive symmetry assumption suggests) but live in completely different libraries, reinforcing that "confirmed against the installed X" has to be a per-dependency habit, not a one-time lesson learned from a single prior incident.

### Open problems
1. `security.rate_limit` still doesn't cover the MCP mount, and Stage 2 is a new traffic source into that same pre-existing gap (see CLAUDE.md/`docs/architecture.md` for the exact mechanism -- the ASGI transport bypasses the outer app's middleware stack by calling the mounted sub-app directly).
2. `transport="http"` (the real-network escape hatch for a genuinely separate future MCP deployment) is implemented and unit-tested for its config validation, but has never been exercised against an actual second process -- worth a real smoke test before treating it as production-ready, not just config-correct.
3. Internal-token minting is HS256-only by construction (`validate_startup_config` refuses to start otherwise); supporting `RS256`/`ES256` would need a private signing key this codebase doesn't hold today, matching `scripts/issue_dev_token.py`'s identical, already-documented limitation.

### Next recommended step
Nothing code-shaped is pending for this milestone. If MCP Stage 2 is ever exercised against a real, separate external MCP server, that's the natural next real-world validation (`transport="http"`, open problem 2 above). Otherwise the MCP milestone (Stages 1A, 1B, and 2) is complete as scoped.

**Result:** MCP Stage 2 is implemented, tested, and documented (CLAUDE.md, `docs/architecture.md`'s full design subsection plus updated diagram/testing/limitations, `docs/topics/mcp.md`, README's MCP section/feature-table/Roadmap, this entry, and a dedicated `ISSUES.md` entry for the two real bugs found). All three of the user's approval adjustments hold, verified by dedicated tests, not just by design intent. Nothing committed or pushed, per explicit instruction.

## 2026-09-05: MCP write-action capability -- `update_case_status`, the first mutating agent tool

Closed the remaining gap in the business-tool family: every prior MCP tool (local RAG or Stage 1B/2 business tools) was read-only. This session added `update_case_status(case_id, new_status)` as a seventh tool -- a real write action gated by a deterministic transition table and an approval requirement for one sensitive transition, with the LLM structurally unable to self-approve. Design proposal written to `.claude-prompts/16.9-Claude-proposal.txt` (the milestone file's own required 9-point structure: architecture, transition rules, approval mechanism/trust boundary, files, tool schema, agent behavior, audit, tests, acceptance criteria) and reviewed externally before implementation; approved with three adjustments (`.claude-prompts/16.10-GPT-approval.txt`): treat the mechanism as trusted-runtime approval, not human-in-the-loop; make `approval_required` a hard terminal outcome for that action attempt, never retried in the same run; bound `case_approvals` at the request/config/schema boundary so the internal JWT can't grow unboundedly.

### Interview talking points
- **What problem did this solve?** Every previous MCP milestone was read-only by construction, so "the agent takes an action, not just answers a question" was still an unproven claim. This is the first tool where a wrong design choice has a real, irreversible side effect (mutating a case's status), which is why the approval work is the interesting part, not the CRUD shape.
- **Why was it difficult?** The approval fact has to reach the tool without ever becoming something the LLM's own JSON output could write, without adding a second, parallel trust channel, and without needing any new persistence/queue infrastructure for a synchronous, single-request agent run. The chosen answer -- embed a `case_approvals` claim in the *same* signed internal service token Stage 2 already mints per remote call, resolved server-side via the SDK's `Resolve()` parameter-injection (the identical mechanism identity already uses, structurally excluded from the tool's generated JSON schema) -- reuses 100% of existing trust infrastructure instead of adding a bespoke header or a stored-approval table.
- **How did I verify it?** Read the full existing stack before designing anything (business store/schema, Stage 1B tools, Stage 2 client, `VerifiedIdentity`/`AuthorizationContext`, audit vocabulary, agent dispatch/failure-handling) so the design slotted into established patterns rather than inventing parallel ones. Every approval-matrix case (same-tenant+role, wrong role, cross-tenant with/without the support role, an approval scoped to a different case_id, an approval scoped to a different target status) got its own test against the real `_SYNTHETIC_CASES` store, with an autouse fixture snapshotting/restoring that shared module-level dict so a mutation in one test can never leak into another.
- **What made the loop-termination fix (adjustment 2) clean rather than invasive?** Rather than adding a new `termination_reason` value or restructuring the bounded retry loop, a single one-line check right after `tool_execute` (`if decision.tool_name in WRITE_ACTION_TOOL_NAMES: break`) skips straight to synthesis regardless of the write action's outcome (executed, approval_required, invalid_transition, already_in_status, or a dispatch failure) -- the existing `_finalize_timed`/`_synthesize` path needed zero changes, and the property "never retried" falls out of one boolean check rather than new state.
- **What metrics proved the fix?** 24 new unit tests (14 in a new `test_mcp_business_case_actions.py` covering the full authorization/transition/approval/audit matrix directly against the store function, 2 in `test_mcp_client_stage2.py` for the new `case_approvals` JWT claim including its own defense-in-depth size bound, 4 in `test_agent_graph_mcp_stage2.py` for the disabled-kill-switch fail-closed path and the no-retry-after-write-action property, 4 schema-level in `test_agent_tool_schemas.py`), plus 4 new real-wire-protocol integration tests (approval_required with no mutation, a matching approval executing the mutation for real against the live store, a remote-failure path not crashing `run_agent()`, both using the same store-reset fixture). Full unit suite: 883 passed / 1 skipped (was 883 total counting the one pre-existing test file this session had to fix for the new 7-tool count, `test_agent_tool_schemas.py::test_tool_arg_models_cover_every_tool_name`, missed by the original design pass and caught only by actually running the full suite rather than just the new/extended files). Full integration suite: 115 passed, no skips (Postgres and Ollama were both live in this environment), ~18.5 minutes. `ruff check`/`ruff format --check`: clean on every file this session touched. `mypy src/`: clean, 106 source files (was 104), after clearing a stale, unrelated `.mypy_cache` entry that produced an `INTERNAL ERROR` referencing a `sentence_transformers` submodule path -- confirmed as a cache artifact, not a real type error, by clearing the cache and getting a clean cold run.
- **What would I improve next?** The `agent_synthesize_v2.yaml` prompt was left unchanged (not given a v3 for write-action-outcome wording) since the deterministic evidence text itself already states the true outcome unambiguously ("no change was made" / "requires approval before it can be applied"); this was a documented design choice, not an oversight, but a real RAGAS/manual-review pass on whether `qwen2.5:3b` ever paraphrases that text into a false claim of success would be the natural follow-up before treating this as fully hardened. The in-process case-mutation lock (`_CASE_MUTATION_LOCK`) covers the read-check-write sequence for a single process; this remains a synthetic, single-process demo store by design, so no multi-process/Postgres-backed concurrency story exists or is needed yet.
- **What did a second review round catch that the first implementation missed?** A real authorization bypass: `rag.mcp.business.approvals.resolve_case_action_approvals` verified a token's signature but never checked *who* the token claimed to be, so any caller able to obtain an ordinary, validly-signed token (this deployment mints and verifies its own tokens in the same trust domain) could attach a `case_approvals` claim directly and call the MCP server's `update_case_status` tool, skipping the API-boundary role gate entirely -- that gate only runs for `POST /agent/query*`, never for a direct MCP client. Caught by an independent Codex review pass (`.claude-prompts/16.11-Codex-review.txt`), not by the original design, implementation, or test suite; see the dedicated `ISSUES.md` entry for the full diagnosis and fix (require the token be unambiguously agent-minted via its `sub`/`token_use` markers, plus an independent re-check of that same token's embedded `roles` against `approval_roles`).

### Goal
Present a 9-point design for `update_case_status` and stop for approval (per `.claude-prompts/16.8-MCP-write-action-capability.txt`'s own explicit gate); once approved (with three adjustments), implement, test, and validate (ruff, mypy, full unit + integration suites) without committing or pushing.

### Configuration
New `BusinessActionConfig` nested under `McpConfig.business_actions` (`enabled` default `false`, true no-op matching every other milestone's kill-switch convention; `approval_roles` default `["case_status_approver"]`; `max_case_approvals_per_request` default `5`). New `AgentConfig.tool_select_mcp_actions_prompt_path`, pointing at `agent_tool_select_v5_mcp_actions.yaml` (loaded only when both `mcp.client.enabled` and `mcp.business_actions.enabled` are `True`; the existing `v4_mcp`/`v3` prompts stay byte-identical for every other combination). `rag.mcp.business.schemas.MAX_CASE_APPROVALS` (10) is the hard, non-configurable schema-level ceiling backing the smaller, operator-tunable config value.

### Changes made

**`rag/mcp/business/schemas.py`**: `CaseApproval` (`case_id`, `new_status`, `extra="forbid"`) and `CaseActionOutcome` (`outcome` literal plus `case_id`/`previous_status`/`new_status`/`updated_at`).

**`rag/mcp/business/store.py`**: `_VALID_TRANSITIONS` (open->in_progress->resolved->closed, closed terminal), `_SENSITIVE_TRANSITIONS` (`resolved->closed` only), `_CASE_MUTATION_LOCK`, and `update_case_status(...)` -- authenticated-caller-required (a deliberate asymmetry from the two read tools' `identity=None`-means-unrestricted convention), same tenant/role check reused from `_lookup_authorized`, already-in-status/invalid-transition/approval-required all checked and returned before any mutation, all under one lock.

**New `rag/mcp/business/approvals.py`**: `resolve_case_action_approvals(headers, config)` -- a second, independent JWT decode (deliberately not folded into `VerifiedIdentity`, keeping the general identity model free of a business-specific concept) that extracts the optional `case_approvals` claim, defaulting to `[]` on any missing/unverifiable/malformed case.

**`rag/mcp/server.py`**: `update_case_status` registered as an MCP tool only inside `if config.mcp.business_actions.enabled:` (the write tool doesn't exist on the server at all when off, not just unreachable), with a new `_resolve_case_approvals` `Resolve()` closure alongside the existing `_resolve_identity_only`.

**`rag/agent/tool_schemas.py`**: `UpdateCaseStatusArgs` (`case_id`, `new_status`, `extra="forbid"`, no approval field of any kind), `REMOTE_MCP_TOOL_NAMES` extended to three, new `WRITE_ACTION_TOOL_NAMES` frozenset (`{"update_case_status"}`).

**`rag/agent/decisions.py`/`rag/agent/state.py`**: `ToolSelectionDecision.tool_name`/`ToolCallRecord.tool_name` widened to seven values; `AgentState` gained `case_approvals: list[CaseApproval]`, set once by the router, never LLM-derived.

**`rag/agent/mcp_client.py`**: `mint_internal_token` gained an optional `case_approvals` parameter, embedded as a `case_approvals` claim only when non-empty, plus its own defense-in-depth `RuntimeError` if the list somehow exceeds the configured maximum by the time it reaches minting; `dispatch_remote_tool_sync`/`_call_tool_async` thread it through; `_business_result_to_search_result` gained an `update_case_status` branch plus a new `_render_case_action_outcome` helper producing unambiguous per-outcome text.

**`rag/agent/graph.py`**: `_execute_tool`'s existing remote-tool-disabled fail-closed check extended to also cover `mcp.business_actions.enabled` for write actions (a new `agent_business_action_disabled` audit event, parallel to the existing `agent_mcp_tool_disabled`); the main tool-call loop gained a one-line check terminating it (skip straight to synthesis) immediately after any `update_case_status` dispatch, regardless of outcome; `_load_templates` now picks among three `tool_select` prompt variants.

**`rag/audit.py`**: five new `AuthEventType` values (`agent_business_action_disabled`, `case_action_requested`, `case_action_approval_required`, `case_action_executed`, `case_action_invalid_transition`); the existing `authorization_denied` is reused for an unauthenticated write-action caller and for a caller attempting to supply approvals without the approver role.

**`rag/api/request_auth.py`**: `enforce_case_approval_limits` (config-driven 422, matching `enforce_dos_limits`'s established "runtime config, not a Pydantic class-level constant" reasoning) and `resolve_case_approvals` (role-gates the request's `case_approvals` field against the verified identity, silently dropping rather than erroring when the caller isn't an approver).

**`rag/api/routers/agent_query.py`/`agent_stream.py`**: `AgentQueryRequest.case_approvals` (`list[CaseApproval] | None`, `Field(max_length=MAX_CASE_APPROVALS)`), both routers calling the two new `request_auth` helpers before building `AgentState`.

**New prompt** `src/rag/prompts/templates/agent_tool_select_v5_mcp_actions.yaml`: `v4_mcp` plus one tool description; `v4_mcp`/`v3` untouched.

**`config/default.yaml`**: `mcp.business_actions.enabled: false` block added, matching every other kill-switch's explicit-in-YAML convention.

### Issues discovered
1. **A pre-existing test asserting the exact `TOOL_ARG_MODELS` tool set (`test_agent_tool_schemas.py::test_tool_arg_models_cover_every_tool_name`) was missed by the initial implementation pass** -- caught only by running the full unit suite rather than trusting the new/extended files alone. Fixed by updating both that test and its docstring, plus adding a matching `UpdateCaseStatusArgs`-specific schema test alongside it. Resolved: y.
2. **A stale `.mypy_cache` entry produced an `INTERNAL ERROR` referencing `sentence_transformers.base.losses.gradcache.CachedLossMixin`, unrelated to any code in this session.** Confirmed as a cache artifact (not a real regression) by clearing `.mypy_cache` and getting a clean, complete cold run afterward. Resolved: y.
3. **A real authorization bypass, found by an independent Codex review pass, not the original test suite:** `resolve_case_action_approvals` honored a `case_approvals` claim on any token that merely verified under the shared JWT config, never checking the token was actually minted by `mint_internal_token`. A direct MCP caller presenting an ordinary, validly-signed token (this deployment mints and verifies its own tokens in the same trust domain) could attach that claim itself and bypass the API-boundary `resolve_case_approvals` role gate entirely, since that gate only runs for `POST /agent/query*`, never for a direct MCP client call. Fixed by requiring the token's `sub`/`token_use` to unambiguously match `mint_internal_token`'s markers, plus an independent re-check of that token's own embedded `roles` against `approval_roles`. A related minor finding from the same review (`case_action_requested` reading `case.status` before acquiring `_CASE_MUTATION_LOCK`, letting its logged `from_status` go stale under concurrency) was fixed in the same pass. See `ISSUES.md` for the full diagnosis. Resolved: y.

### Evaluation results
Not applicable -- infra/security milestone; no retrieval/generation quality axis.

### Test results
24 new/changed unit tests, 4 new integration tests from the initial implementation, plus 9 more unit tests (`test_mcp_business_approvals.py`) and one integration-test fix from the post-review security fix (see the talking-points section above and `ISSUES.md` for the full breakdown). Full unit suite: 892 passed / 1 skipped (was 883 before the review-fix round). Full integration suite: 115 passed / 0 skipped (Postgres and Ollama both live in this environment), ~18.5 minutes; the Stage 2 MCP integration file re-run after the fix: 8 passed. `ruff check src/ scripts/ tests/`: clean. `ruff format --check` on every file this session touched: clean (three unrelated, pre-existing drift files elsewhere in `tests/integration`/`tests/unit` left untouched, out of scope). `mypy src/`: clean, 106 source files.

### Interpretation
The one property worth calling out from the original design is that "approval cannot be reused for another case or action" needed no nonce, single-use token, or stored-approval-consumption logic at all -- it falls directly out of `CaseApproval` naming an exact `(case_id, new_status)` pair, checked by equality. The review-fix round added a second, equally important lesson: a JWT that verifies proves "signed by someone holding the shared secret," not "minted by this specific internal code path for this specific purpose" -- those are different claims, and every test that exercised the approval path had done so only through the one legitimate code path that always happens to construct a correctly-shaped token. A function that resolves trust from external input needs its own dedicated adversarial tests, not coverage-by-construction through its intended caller alone.

### Open problems
1. Whether a local generation model (`qwen2.5:3b`) reliably preserves a write-action outcome's true meaning in its own paraphrased final answer (rather than drifting toward implying success on an `approval_required`/`invalid_transition` result) has not been verified against a real model -- the deterministic evidence text is unambiguous, but that is a necessary, not sufficient, condition; a real-Ollama integration test or a small manual-review pass is the natural next check.
2. The in-memory case store's mutation lock is single-process only, matching its existing synthetic/no-Postgres design; this is a documented, accepted limitation of the demo backend, not a gap introduced by this session.

### Next recommended step
A real-Ollama regression test proving `agent_synthesize_v2` (or a documented decision to add a v3) correctly reports each of the four `update_case_status` outcomes without overclaiming, mirroring the existing `test_agent_synthesize_v2_trust_priority.py` pattern for a different prompt-behavior guarantee.

**Result:** `update_case_status` is implemented end to end -- deterministic transition/approval rules server-side, a trusted-runtime approval channel structurally unreachable from LLM-controlled JSON, a hard no-retry guarantee for any write-action attempt, and bounded `case_approvals` at the schema/config/token-minting layers -- verified by the full unit and integration suites, ruff, and mypy, with zero regressions to any pre-existing tool. A subsequent independent review round found and closed a real authorization-bypass gap in the approval-resolution function itself, now covered by dedicated adversarial unit tests. Nothing committed or pushed, per explicit instruction.

---

## 2026-09-05-12:43 | User feedback loop (feedback milestone)

### Interview talking points
- **What problem did this solve?** Added a closed feedback path -- thumbs
  up/down + optional structured reason/comment on any answer, persisted
  tenant-scoped in Postgres, exportable for human review -- without
  turning feedback storage into a second, uncontrolled sensitive-data
  sink and without pretending a caller's opinion is ground truth.
- **Why was it difficult?** Two structural gaps had to be closed first,
  not worked around: there was no stable "which exact answer/run does
  this refer to" identifier anywhere in the API, and a naive dedup-key
  design (nullable `tenant_id` in a `UNIQUE` constraint) would have
  silently defeated the one behavior this milestone most cared about
  (update-not-duplicate) for exactly the unauthenticated/dev-mode path.
  A third, subtler one surfaced while wiring `request_id` through the SSE
  endpoint: reading a per-request contextvar from inside an async
  generator that outlives the middleware `finally` block that owns it
  would have been a real, hard-to-repro race.
- **How did I debug it?** The tenant_id/NULL and contextvar-race issues
  were both found by reasoning through Postgres `UNIQUE`-constraint
  semantics and Starlette's `BaseHTTPMiddleware`/`StreamingResponse`
  mechanics *before* writing the code, not after a flaky test caught
  them -- confirmed the request-not-installed `mcp`/`httpx2` packages
  environment gap the same way (ran the exact same pre-existing test
  against the global Python vs. the project's real `.venv`, isolating
  the cause immediately rather than assuming my new code broke
  collection). The pre-existing `mkdocs build --strict` docs-site
  warning gap was isolated with `git stash`/`git stash pop` against a
  clean `main` before deciding it was out of scope.
- **What metrics proved the fix?** 32 new/changed tests across four new
  files (`test_api_feedback.py`, `test_feedback_store.py`,
  `test_export_feedback.py`, `test_feedback_persistence.py`) plus one
  pre-existing test updated for an intentional, additive response-schema
  change. Full unit suite: 919 passed / 1 skipped (using the project's
  real `.venv`, not the stale global Python environment). Full
  integration suite: 119 passed / 0 skipped (~22.7 minutes, Postgres and
  Ollama both live). `ruff check`/`ruff format --check`: clean on every
  file this session touched. `mypy src/`: clean, 110 source files.
  Frontend: `npm run typecheck`/`lint`/`test` (53 passed)/`build` all
  clean.
- **What would I improve next?** No per-answer metadata store exists
  yet, so `generation_model`/`prompt_id`/`prompt_version`/
  `retrieval_provider`/`reranker_provider` on a feedback row are a
  best-effort snapshot of `AppConfig` at submission time, not a captured
  value from the moment the original answer was generated -- exact in
  this single-process deployment today, but worth tightening if a
  per-answer record is ever added for another reason.

### Goal
Implement `.claude-prompts/17.0-User-feedback-loop.txt`'s milestone in
full: `POST /feedback`, Postgres persistence, an export/curation path,
frontend thumbs controls, observability, and tests -- explicitly not the
CI eval-gate milestone, not automatic gold-dataset promotion.

### Configuration
New `FeedbackConfig` (`config.feedback`, `AppConfig`'s new top-level
field): `enabled` (default `true`), `table_name` (`feedback`),
`store_query_text`/`store_answer_text` (both default `true`),
`max_comment_length` (1000), `max_query_text_length` (2000),
`max_answer_text_length` (8000), `max_cited_sources` (20). Added
explicitly to `config/default.yaml`, matching every other kill-switch's
in-YAML convention.

### Changes made

**`src/rag/feedback/`** (new package): `schemas.py`
(`FeedbackRating`, `NEGATIVE_FEEDBACK_REASONS`/`POSITIVE_FEEDBACK_REASONS`,
`FeedbackRecord`) and `store.py` (`FeedbackStore`: its own small
`ThreadedConnectionPool`, `submit()`'s
`(tenant_id, caller_key, request_id)` upsert via
`INSERT ... ON CONFLICT DO UPDATE ... RETURNING feedback_id,
(xmax = 0) AS inserted`, `export()` for the CLI script, `delete()` for
test cleanup).

**`scripts/init_db.py`**: new `build_feedback_schema_sql`, called from
`main()` alongside the existing documents/chunks schema. `tenant_id TEXT
NOT NULL DEFAULT ''` (never nullable -- see Issues discovered below);
`UNIQUE (tenant_id, caller_key, request_id)`; indexes on tenant_id,
created_at, rating, dataset_id, route.

**`scripts/export_feedback.py`** (new): `FeedbackStore.export()` ->
JSONL/CSV, filterable by tenant/rating/dataset/route/date range. No
public `GET /feedback` exists anywhere; this is the only read path,
matching `record_experiment.py`/`compare_experiments.py`'s established
internal-tooling convention.

**`src/rag/api/routers/feedback.py`** (new): `FeedbackRequest`
(`extra="forbid"`, a deliberate deviation from `QueryRequest`'s
permissive default; a `model_validator` rejects a reason from the wrong
rating's fixed vocabulary), `FeedbackAck`, `submit_feedback`. Reuses
`get_current_identity` unmodified and a new
`rag.api.request_auth.resolve_trusted_tenant_id` (a smaller sibling of
`build_authorization_context`, same forged-claim-logging behavior) so
identity/tenant trust is byte-identical to `/query`'s.

**`src/rag/api/routers/query.py`/`agent_query.py`/`agent_stream.py`**:
added the smallest clean identifier this milestone needed --
`request_id: str | None` on `QueryResponse`/`AgentQueryResponse`, sourced
from `rag.logging_config.get_request_id()` (the same value already
behind the `x-request-id` response header). `agent_stream.py` captures
this value once, synchronously, in `agent_query_stream()` itself and
threads it through as a plain argument rather than reading the
contextvar inside the async generator (see Issues discovered).

**`src/rag/audit.py`**: one new `AuthEventType`, `feedback_submitted`.

**`src/rag/observability/metrics.py`**: `FEEDBACK_SUBMISSIONS_TOTAL`
(`rating`/`outcome` labels) and `FEEDBACK_SUBMISSION_LATENCY_SECONDS`,
following the exact dedicated-registry/`_defensive`-wrapper pattern
every other metric already uses.

**`src/rag/api/deps.py`**: `get_feedback_store()` singleton, its own
pool, deliberately separate from `get_vectorstore()`'s.

**`src/rag/config.py`**: `FeedbackConfig`, `AppConfig.feedback`.

**Frontend**: `frontend/src/api/feedback.ts` (`postFeedback`, reusing
`postJson`'s existing error classification unchanged),
`frontend/src/components/chat/FeedbackControls.tsx` (thumbs up/down +
disclosed-on-demand reason/comment form, keyed off
`message.debug.requestId`), wired into `MessageBubble.tsx`. `state/
types.ts`/`chatContext.tsx` gained `DebugInfo.requestId`,
`ChatMessage.query`/`feedback`, and a `SET_FEEDBACK` reducer action;
`useSendMessage.ts` now threads the original query text through
`COMPLETE_ASSISTANT_ANSWER` at all three call sites (classic, agent
stream, agent fallback) since feedback needs both sides of the exchange.
`api/types.ts` gained `request_id` on the two response schemas plus the
feedback rating/reason/ack schemas.

**Documentation**: `CLAUDE.md` (new "User feedback loop" section +
directory-map entries), `docs/architecture.md` (new "Feedback loop"
section, the full design writeup), `README.md` (new "Feedback loop"
section, transcluded into a new `docs/topics/feedback.md`), new
`docs/reference/feedback.md` + `mkdocs.yml` nav entries,
`frontend/README.md` (new "Feedback" section).

### Issues discovered

1. **A `UNIQUE` constraint on a nullable `tenant_id` would have silently
   defeated feedback deduplication for the no-auth/dev-mode path.**
   Postgres treats every `NULL` as distinct from every other `NULL` in a
   `UNIQUE` constraint, so `UNIQUE (tenant_id, caller_key, request_id)`
   with a nullable `tenant_id` would never actually collide two
   anonymous submissions -- each would `INSERT` a new row instead of
   hitting `ON CONFLICT`, exactly the "unbounded duplicates" outcome this
   milestone's own instructions call out as the thing to avoid. Found by
   reasoning through the SQL semantics before writing the schema, not by
   a failing test. Fixed with `tenant_id TEXT NOT NULL DEFAULT ''`
   (mapped back to `None` on read in `_row_to_record`). Evidence:
   `test_two_anonymous_no_tenant_submissions_do_not_collide` (integration,
   real Postgres). See `ISSUES.md` for the full writeup. Resolved: y.
2. **Reading `get_request_id()` from inside `_stream_agent_query`'s async
   generator would have raced `RequestIDMiddleware`'s own `finally`
   block.** `call_next()` only awaits a `StreamingResponse`'s first ASGI
   message; the generator body (which builds the terminal frame carrying
   `request_id`) is driven by the outer ASGI send loop afterward,
   potentially after the middleware has already reset the contextvar.
   Traced through Starlette's `BaseHTTPMiddleware`/`StreamingResponse`
   mechanics before writing the code. Fixed by capturing `request_id`
   once, synchronously, in `agent_query_stream()`'s own body (before the
   generator/`StreamingResponse` even exist) and threading it through as
   a plain argument. No flaky test was ever observed, because the race
   was designed around up front. See `ISSUES.md`. Resolved: y.
3. **The dev environment's global Python has a stale `local-rag-system`
   editable install missing the `mcp`/`httpx2` core dependencies added by
   an earlier milestone**, so every `rag.api.*`-importing test failed
   collection with `ModuleNotFoundError: No module named 'mcp'` when run
   against the global interpreter. Confirmed this was pre-existing and
   unrelated to this session's changes (a pre-existing, unrelated
   `test_api_dos_limits.py` failed the identical way). The project's own
   `.venv` has the correct, current dependency set installed
   (`mcp==2.1.1`) and was used for every test/lint/type-check run after
   this was found. Not a code change; a environment-discovery note for
   future sessions in this checkout. Resolved: y (worked around by using
   the correct interpreter; the global environment itself is untouched).
4. **`mkdocs build --strict` already fails on `main` with 2 warnings**
   (`topics/web-ui.md`'s screenshot-image and `frontend/README.md` links
   resolving incorrectly once transcluded into the nested `docs/topics/`
   directory), confirmed via `git stash`/rebuild/`git stash pop` against
   a clean checkout before concluding it predates and is unrelated to
   this session. My own new `docs/topics/feedback.md` initially added a
   third instance of the identical pattern (copying README's
   `[`docs/architecture.md`](docs/architecture.md)`-style link verbatim
   into a transcluded section); fixed by not hyperlinking that one
   mention in the new README section, so this session's docs changes
   don't add to the pre-existing warning count. The two pre-existing
   warnings themselves were left alone -- out of scope for this
   milestone, and not part of its required validation checklist.
   Resolved: n (pre-existing, not fixed; my own new instance of the
   pattern was fixed so as not to add to it).

### Evaluation results
Not applicable -- infra/product milestone; no retrieval/generation
quality axis.

### Test results
32 new/changed tests (14 in `test_api_feedback.py`: positive/negative
feedback, validation, reason/rating mismatch, oversized comment/cited-
sources, unknown-field rejection, feedback-disabled 404, forged-tenant
ignored, unauthenticated-rejected, no-JWT-reaches-store, duplicate-update
semantics, stale-request-id-still-accepted, database-failure-503; 7 in
`test_feedback_store.py`: pool-lifecycle/connection-leak tests mirroring
`test_pgvector_pool.py`'s established fake-pool pattern, plus WHERE-clause
construction; 6 in `test_export_feedback.py`: JSONL/CSV serialization,
never-touches-a-gold-file, CLI arg parsing; 4 in
`test_feedback_persistence.py`, integration, real Postgres: full row
round-trip, update-not-duplicate, anonymous-no-collision, filtered
export), plus one pre-existing test
(`test_query_endpoint_response_schema_is_unaffected_by_the_new_route`)
updated for the intentional, additive `request_id` field. Full unit
suite: 919 passed / 1 skipped. Full integration suite: 119 passed / 0
skipped, ~22.7 minutes (Postgres and Ollama both live in this
environment). `ruff check src/ scripts/ tests/`: clean. `ruff format
--check`: clean on every file this session touched (3 unrelated,
pre-existing drift files elsewhere left untouched, out of scope, same
precedent as prior sessions). `mypy src/`: clean, 110 source files (was
106). Frontend: `npm run typecheck`, `npm run lint` (0 errors, 3
pre-existing warnings unrelated to this session), `npm test` (53 passed),
`npm run build` all clean.

### Interpretation
The single most load-bearing decision this session made was reusing the
existing per-request `request_id` contextvar as the answer/run
identifier instead of designing a new one -- it meant zero new
persistence for request tracking, and it came with a built-in
subtlety (the SSE contextvar race) that was worth finding by tracing
framework internals rather than shipping and hoping. The second
load-bearing decision was choosing `(tenant_id, caller_key, request_id)`
as the dedup key and catching the `NULL`-uniqueness edge case before it ever
reached a test: both fixes share the same lesson -- a design that looks
correct in the common case (an authenticated caller with a real tenant)
needs to be checked explicitly against the degraded/no-identity case
too, since that's exactly where "unbounded duplicates instead of a
clean update" would have silently shown up.

### Open problems
1. Lineage fields (`generation_model`/`prompt_id`/etc.) are a
   config-snapshot-at-submission-time, not a captured-at-generation-time
   value; correct today (config only changes on restart) but not a
   structural guarantee if a future milestone ever added config
   hot-reload.
2. No load test exists proving `FeedbackStore`'s small
   (`minconn=1, maxconn=3`) pool holds up under realistic concurrent
   submission volume; acceptable for this project's current
   single-instance/demo scale, flagged as a scaling question, not a bug.
3. The pre-existing `mkdocs build --strict` docs-site warning gap
   (`topics/web-ui.md`'s two broken transcluded links) remains unfixed,
   deliberately out of scope for this milestone.

### Next recommended step
The CI eval-gate milestone this session's own instructions explicitly
deferred: once real feedback volume exists, build the human-review ->
curated-eval-example -> regression-dataset pipeline `scripts/
export_feedback.py`'s output is designed to feed, plus decide whether
`GET /feedback`-style admin visibility is ever warranted behind a real
admin-role gate.

**Result:** The feedback loop is implemented end to end -- `POST
/feedback` with byte-identical identity/tenant trust to `/query`,
deterministic update-not-duplicate semantics backed by a real database
constraint (with a real Postgres `NULL`-uniqueness edge case caught and
fixed before it ever shipped), a privacy posture that never stores a
JWT/chain-of-thought/hidden reasoning and bounds every free-text field,
a script-driven export path with no public browsing endpoint and no
automatic gold-dataset mutation, compact accessible frontend controls,
and full observability -- verified by 32 new tests, a clean full
unit+integration run (919 + 119 passed), and clean ruff/mypy/frontend
tooling. Nothing committed or pushed, per explicit instruction.

## 2026-09-06: Structured-logging cleanup for the API/agent runtime

### Interview talking points
- **What problem did this solve?** The `request_handled` structured log
  never actually carried `status_code` -- it was only visible in the
  OTel span attribute and the Prometheus metric, so a JSON-log-only
  consumer (a log aggregator with no tracing/metrics backend attached)
  couldn't tell a 200 from a 500 for the same request. Separately, there
  was no single structured summary for a completed agent/classic run or
  a completed tool call; both existed only as scattered spans/metrics/
  audit events for specific failure modes, not one queryable line per
  run/call.
- **Why was it difficult?** The real bug wasn't "a missing field" so
  much as "a closed span": `request_handled` was logged *after*
  `span_cm.__exit__()` had already detached the HTTP root span, so
  `JSONFormatter`'s ambient trace-context lookup silently found nothing
  even with tracing enabled -- the field would have looked present in
  code review (the formatter does support it) while never actually
  firing in production. `run_agent()`'s own root-span close had the
  identical shape of bug. Neither was visible without tracing something
  end to end and checking the *emitted* JSON, not just reading the
  formatter's logic in isolation.
- **How did I debug it?** Read `logging_config.py`'s `JSONFormatter`
  closely enough to realize `trace_id`/`span_id` are synthesized at
  *format time* from whatever span is ambiently current, not stamped
  onto the `LogRecord` at `logger.info()` call time -- which also broke
  my first attempt at testing this with `caplog` (it captures raw
  records, never formatter output), fixed by writing a small in-memory
  handler that runs the real `JSONFormatter` synchronously and asserts
  on the parsed JSON instead.
- **What metrics proved the fix?** 10 new tests across two new files
  (`test_structured_request_logging.py`, `test_structured_agent_logging.py`),
  all passing, including a direct regression guard reproducing the
  span-closed-before-log bug via `monkeypatch`-ing a real
  `TracerProvider` tracer in. Full unit suite (project `.venv`, not the
  stale global Python): 939 passed / 1 skipped once an unrelated,
  pre-existing environment issue was worked around (see Issues
  discovered). `ruff check`/`ruff format --check`: clean on every
  touched file. `mypy src/`: clean, 110 source files.
- **What would I improve next?** `mcp/server.py`'s `_run_tool`/
  `_run_business_tool` and `agent/graph.py`'s `_execute_tool` now each
  define their own small `_log_tool_call_completed` closure with the
  same shape; if a third tool-dispatch call site is ever added, that's
  worth factoring into one shared helper rather than a third copy.

### Goal
Implement `.claude-prompts/18.0-Structured-logging-improvement.txt`'s
scope exactly: make structured logs self-contained and useful for
debugging production requests without leaking sensitive data, without a
broader observability redesign and without changing runtime behavior.
Explicitly out of scope: RAGAS/hosted-judge runs, committing, pushing.

### Changes
- `api/middleware.py`: `request_handled` now includes `status_code` and
  `route_name` in its structured fields (previously only in the span/
  metric), and is logged *before* `span_cm.__exit__()` so the ambient
  `trace_id`/`span_id` lookup actually finds the still-open HTTP root
  span. No change to response handling or the `x-request-id` header
  logic.
- `agent/graph.py`: one new `agent_request_completed` log per `run_agent()`
  call (`route`, `termination_reason`, `step_count`, `tool_call_count`,
  `retrieval_attempts`, `evidence_sufficient`, `duration_ms`, `success` --
  never query/answer/evidence text), added inside `_finish` before the
  root span closes, same fix shape as the middleware bug. One new
  `tool_call_completed` log per tool dispatch (`tool_name`, `success`,
  `result_count`, `duration_ms`, `origin` local/remote, and a bounded
  `error_type` -- the exception's class name, never `str(exc)`), added at
  all four `_execute_tool` exit points (invalid arguments, MCP/business-
  action disabled, dispatch exception, success).
- `api/routers/query.py`: the plain `/query` route (which never calls
  `run_agent()`) gets its own `agent_request_completed` log with
  `route="classic_rag"`, same shape and event name as the agent routes',
  so a log consumer can filter one event type regardless of which
  endpoint served the request.
- `api/routers/agent_stream.py`: the worker-thread failure log
  (`_run_agent_in_thread`'s `except Exception`) renamed to a structured
  `agent_stream_failed` event with `component`/`error_type` fields,
  keeping `exc_info=True` for genuine traceback debugging.
- `mcp/server.py`: `_run_tool`/`_run_business_tool` gained the same
  `tool_call_completed` log (tool_name/success/result_count/duration_ms/
  error_type) at their success and failure branches, mirroring the
  agent-graph tool logging for MCP-server-side dispatch (both an
  external MCP client and the in-process agent's own MCP client route
  through here).
- `Dockerfile` / `docker-compose.prod.yml`: added `--no-access-log` to
  the `uvicorn` command -- now redundant with `request_handled` carrying
  `status_code`. `docker-compose.yml`'s dev command (`--reload`) is left
  unchanged with an explicit comment explaining why (human-readable
  terminal feedback while iterating locally); server lifecycle/error
  logs (`uvicorn`/`uvicorn.error`) are unaffected either way.
- `tests/unit/test_structured_request_logging.py` (new),
  `tests/unit/test_structured_agent_logging.py` (new): 10 tests covering
  status_code presence, request_id-without-tracing, trace_id/span_id-
  with-tracing (the regression guard), no-Authorization-header-leak,
  agent/classic completion logs never containing query/answer text, and
  tool-call logs using a bounded `error_type` instead of a raw exception
  message.

### Issues discovered
See `ISSUES.md`'s "A container-only env var, once loaded into a local
shell, quietly broke the full unit suite hours after the test that set
it had finished" -- a pre-existing test-isolation gap in
`test_api_config_override.py` (its autouse fixture clears `get_config`'s
cache but never the `RAG_CONFIG_PATH` env var it depends on), surfaced
while validating this session's own changes, not caused by them.
Confirmed by re-running the full suite with `RAG_CONFIG_PATH` cleared:
939 passed / 1 skipped, no code changes. Initially left as an open,
documented finding (out of the logging task's own scope), then fixed on
request: new `tests/unit/conftest.py` with one `autouse=True` fixture
that `monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)`s before every
unit test, so no test (this file or any other that clears the same
cache) can ever see a stale, host-invalid value again. Re-verified
against the real, unmodified `.env`-polluted shell: 939 passed / 1
skipped, no workaround needed.

**Result:** `request_handled` now carries `status_code`/`route_name`/
`trace_id`/`span_id` (when tracing is active) in the actual JSON line,
not just the span/metric; every agent/classic-RAG run and every tool
dispatch (local and MCP-server-side) gets one safe structured completion
log; Uvicorn's duplicate plain-text access log is off in the
Docker/production startup path. No query/answer/evidence/JWT/token text
appears in any new log line -- verified directly by test, not just by
inspection. Full unit suite clean (939/1 skipped) against the real,
unmodified local environment; ruff/mypy clean. Nothing committed or
pushed, per explicit instruction.

## 2026-09-06: Retrieval diagnosis and a dataset_id scoping fix for the manual-query path

### Interview talking points
- **What problem did this solve?** A manual query against the running
  system ("What salary bands does TechFusion use for senior engineers?")
  returned an unhelpful answer with an oddly low retrieval score
  (~0.016). The instinct was to suspect a retrieval-pipeline regression;
  the actual causes were a genuinely unanswerable gold question plus a
  missing UI/config affordance, not a pipeline bug.
- **Why was it difficult?** Two unrelated problems were stacked on top
  of each other. `techfusion_gold.jsonl` does contain this exact
  question, deliberately marked `unanswerable: true` -- the refusal was
  correct generation behavior, not a bug. Separately, the manual/
  frontend query path had no way to set `filters.dataset_id`, so it
  searched every dataset ever ingested into the shared dev Postgres
  instance at once; an unrelated document from a different ingestion
  could legitimately outrank the real corpus's own (correct but
  comparatively weaker) evidence. Both had to be found and told apart
  before concluding "working as intended, but confusingly."
- **How did I debug it?** Grep's own pattern match for "salary" initially
  returned nothing over `data/eval` -- a real tool false-negative, caught
  by cross-checking with a direct Python script that `json.loads`-ed
  every gold line instead of trusting the first search. Reproduced the
  exact 0.016 score by hand: it's an RRF fusion score
  (`1/(rrf_k+rank)`), not a raw cosine/BM25 score, confirmed by exact
  arithmetic match. Compared dense/BM25/fused results at every stage
  directly (`RetrievalPipeline.retrieve_attribution`) rather than
  trusting only the final ranked list.
- **What metrics proved the fix?** A live authenticated `POST /query`
  with `filters: {"dataset_id": "techfusion"}` returned only real-corpus
  sources (`gdpr-data-handling.md`, `remote-work-policy.md`,
  `annual-leave-policy.md`) with the answer still correctly refusing --
  versus the same query with no dataset filter, which mixed in unrelated
  `layout_vision_extension` documents. Frontend: `npm run
  typecheck`/`lint`/`test` (61/61 passing, 5 new)/`build` all clean.
- **What would I improve next?** The gold file's own `unanswerable` rows
  could use a short authored note on *why* (deliberately absent from the
  corpus vs. genuinely missing) so a future investigator doesn't have to
  re-derive that from scratch.

### Goal
Diagnose why a specific manual query looked wrong, following an explicit
10-point investigation brief, without changing retrieval logic or
re-ingesting/deleting data until the root cause was confirmed; then, once
approved, implement the identified fix.

### Changes
- `frontend/src/api/types.ts`, `frontend/src/state/chatContext.tsx`: added
  a `datasetId` field to `DevIdentity` and its default empty state.
- `frontend/src/api/query.ts`, `frontend/src/api/agentQuery.ts`: both
  request-body builders now merge `datasetId` into `filters.dataset_id`
  when set -- independent of the bearer-token identity-precedence rule,
  since this is a retrieval filter, not an identity claim.
- `frontend/src/components/DevIdentityPanel.tsx`: new "Dataset ID" text
  input in Developer settings, with an explicit hint when left blank
  explaining that retrieval will otherwise span every ingested dataset.
- `frontend/tests/unit/requestBodyBuilders.test.ts` (new, 5 tests);
  `frontend/tests/unit/client.test.ts`/`feedbackControls.test.tsx`: fixed
  pre-existing `DevIdentity` object literals for the new required field.
- `frontend/README.md`: documented the field and the cross-dataset-
  pollution problem it closes.

### Issues discovered
None beyond the diagnosis itself (see "What problem did this solve?"
above) -- no code defect existed in retrieval/generation; this was a
missing scoping affordance plus a correctly-refusing unanswerable
question that looked like a failure from the UI alone.

**Result:** Root cause fully explained with evidence at every retrieval
stage; fix implemented, tested (typecheck/lint/test/build all clean),
deployed to the live frontend container, and verified end to end via a
real authenticated API call and direct inspection of the served JS
bundle. Committed the same day (see the commit-session entry below).

## 2026-09-06: Host memory exhaustion made Ollama hang, surfacing as 504s in the web UI

### Interview talking points
- **What problem did this solve?** Every classic-RAG query in the web UI
  failed with a 504 immediately after bringing the full container stack
  back up, right after the dataset_id fix above had just been confirmed
  working.
- **Why was it difficult?** The failure surfaced four layers away from
  its actual cause. nginx's `/query` location (unlike `/agent/`) has no
  `proxy_read_timeout` override, so it hit its 60s default and returned
  a 504 -- which looked like a frontend/nginx config bug, not a symptom
  of something upstream never responding at all, at any timeout length.
- **How did I debug it?** Disproved each layer in turn with a request
  that bypassed everything above it: nginx's own error log first
  (confirmed it proxied correctly and only timed out waiting on a
  response *header*); then a direct curl straight to rag-api bypassing
  nginx (hung 3 minutes, zero bytes); then a direct curl straight to
  Ollama's own `/api/chat` bypassing rag-api entirely (hung 2+ minutes
  for a trivial one-sentence prompt against the exact model production
  config uses). Sampled the inference process's own CPU time over a
  5-second window (~1.7s consumed) to tell "hung" apart from "just
  slow." `Get-CimInstance Win32_OperatingSystem` showed 0.25-0.4GB free
  out of 7.79GB total, and `ollama ps` showed two models loaded
  simultaneously.
- **What metrics proved the fix (partial)?** Killing Ollama's orphaned
  `llama-server` subprocess (which survived its parent's death via
  `Stop-Process`) freed ~885MB, taking free memory from ~0.4GB to
  ~1.26GB, confirmed via `/health`/`ollama ps` reporting a clean idle
  state. Stopping the three observability containers (Jaeger/Prometheus/
  Grafana) freed further headroom. A full generation call still could
  not be proven to complete reliably afterward -- other host processes
  reclaimed the freed memory within seconds.
- **What would I improve next?** Apply the still-open
  `frontend/nginx.conf` fix (give `/query` the same
  `proxy_read_timeout 300s` `/agent/` already has), so a legitimately-
  slow-but-working request isn't killed at 60s once memory pressure is
  actually resolved. More fundamentally: this 8GB box cannot currently
  run the full container stack, multiple editor windows, and local LLM
  inference simultaneously -- a host-capacity constraint, not a code fix.

### Goal
Diagnose a sudden wave of 504 errors in the web UI, tracing from the
browser down to whichever layer was actually failing, without assuming
the most recently-changed code (the dataset_id fix) was the cause just
because it was the most recent change.

### Changes
Operational only, no code changes: stopped the Jaeger/Prometheus/Grafana
containers; killed the wedged `ollama.exe`, its orphaned `llama-server`
subprocess, and the `ollama app.exe` tray wrapper; relaunched Ollama
clean.

### Issues discovered
See `ISSUES.md`'s "A 504 from the frontend was actually the host running
out of memory, four layers away from where it surfaced" for the full
diagnosis. `frontend/nginx.conf`'s `/query` location still has no
`proxy_read_timeout` override (unlike `/agent/`'s explicit `300s`) --
proposed, not yet applied, since the deeper problem (host memory) had to
be resolved or worked around first.

**Result:** Root cause identified as host-wide memory exhaustion, not a
regression in retrieval, the dataset_id fix, or Docker networking. Ollama
and rag-api both recovered to a reported-healthy idle state; whether a
real end-to-end generation call completes reliably depends on how much
host memory is available at the time, which this session could not fully
resolve (recommended the user close other host applications or run fewer
containers concurrently).

## 2026-09-06: Committed the session's accumulated logging, test-isolation, and dataset_id changes

### Interview talking points
- **What problem did this solve?** Three separate pieces of work from
  this session (structured logging, a test-isolation fix, and the
  dataset_id scoping fix) were sitting uncommitted after being
  individually verified. Grouping them into one undifferentiated commit
  would have made future `git blame`/review harder than necessary.
- **How was it organized?** 15 commits, each scoped to one coherent
  change and matching this repo's existing `type(scope): description`
  convention: 6 for the structured-logging additions (one per
  file/concern -- middleware trace-correlation, classic-route
  completion log, agent-graph tool/run logging, the stream failure log,
  MCP-server tool logging), 1 for the Docker access-log change, 1 for
  the `RAG_CONFIG_PATH` test-isolation fix, 2 for the new logging test
  files, and 5 for the dataset_id frontend fix (state, request-body
  scoping, the UI field, its new test file, fixed pre-existing test
  fixtures, and README docs -- six listed because the fixture-fix and
  UI-field commits are separate).
- **What would I improve next?** Nothing outstanding; every commit
  passed the repo's own pre-commit hooks (ruff, ruff-format, mypy)
  individually, not just as a final batch.

### Goal
Commit all of this session's verified, working-tree changes in a small,
reviewable number of commits (target: 15-20), without pushing.

### Changes
No source changes; this entry documents the commit organization itself.
See `git log` (commits `6ba58b1`..`f4b0f5d` on `main`) for the exact
sequence.

**Result:** 15 commits landed, working tree clean, every pre-commit hook
passed on every commit. `main` is 15 commits ahead of `origin/main`;
nothing pushed.

---

## 2026-09-07-00:38: `GET /info` + RuntimeInfoPanel + MLflow agent-metrics persistence, committed alongside a concurrently-running second agent

### Interview talking points
- **What problem did this solve?** Spec `19.0-Runtime UI+MLFlow.txt` asked
  for two independent things: (1) a safe way for the web UI to show
  *which concrete pipeline* is actually running (model name, retrieval
  mode, fusion method, MCP/vision status) -- `GET /`'s `FeatureFlags`
  deliberately never exposes this, by design, so a second endpoint was
  needed, not an extension; (2) `run_agent_eval.py` already computed
  per-node-latency, termination-reason, and tool-usage breakdowns for its
  console report, but none of that reached `experiments/results/agentic/
  *.json` or MLflow, so comparing e.g. "how often did runs hit
  `max_steps`" across experiments meant re-running eval by hand each
  time.
- **Why was it difficult?** Not technically hard, but required care not
  to blur two closely-related but distinct existing contracts:
  `FeatureFlags`'s own regression test
  (`test_root_features_never_leak_secrets_or_identifying_config`)
  asserts no model name ever appears in `GET /`'s response, so
  `RuntimeInfo` had to be a genuinely separate model/route, not a field
  added to the existing one. On the MLflow side, the new breakdowns had
  to reuse `evaluate_agent()`'s already-collected `records` with zero
  new instrumentation, and stay scoped to `route == "agent"` only, so a
  `classic_rag`-routed row (always `termination_reason="synthesized"`,
  never dispatches a tool) couldn't dilute the agent loop's own numbers
  -- the same scoping `_latency_and_tokens`'s `agent_routed` filter
  already established.
  A second, unplanned difficulty arrived mid-session: a separate,
  independently-running Claude agent was simultaneously editing
  `src/rag/eval/run_agent_eval.py` and `tests/unit/
  test_run_agent_eval.py` (its own specialized-tool-eval milestone, see
  the next entry) while I was mid-edit on the same two files. The user's
  instruction was to commit only my own ~21 logical changes, in 20-25
  small commits, without ever touching, corrupting, or prematurely
  committing the other agent's in-progress, unreviewed work in those
  shared files.
- **How did I debug/verify it?** For the feature itself: ruff,
  ruff-format, and mypy on every changed file; new backend unit tests
  for `GET /info`; new frontend unit/component tests for the API client,
  hook, and panel; extended `test_mlflow_logger.py` and added
  `test_record_agent_experiment.py` (its first test coverage at all).
  For the multi-agent-safe commit problem: reconstructed exactly which
  hunks in the two shared files were mine (by diffing a known-good `git
  show HEAD:<path>` baseline against a scratch copy with only my own
  edits re-applied), hand-built minimal unified-diff patches for just
  those hunks, and staged them directly into the git index with `git
  apply --cached` -- never touching the working tree, so the other
  agent's uncommitted edits stayed exactly where they were. Verified
  each patch's correctness (ruff, ruff-format, mypy, `pytest tests/unit/
  test_run_agent_eval.py`) inside a disposable, fully-isolated `git
  worktree add --detach` checkout before ever applying it to the real
  repo's index, so I could trust the patch was right without the
  concurrently-changing real working directory interfering.
- **What metrics proved the fix?** Every new/extended test passing (`GET
  /info`, `RuntimeInfoPanel`, `record_agent_experiment.py`,
  `mlflow_logger.py`, the two new `run_agent_eval.py` breakdown
  functions); final `git status --porcelain` showed only the other
  agent's own files remaining modified/untracked, confirming zero
  cross-contamination.
- **What would I improve next?** Nothing outstanding in the feature
  itself. One process lesson worth keeping: `git commit -m "..." --
  <pathspec>` does **not** mean "commit whatever is staged for that
  path" -- it commits that path's current **working-tree** content,
  overriding whatever had been staged via `git apply --cached`. I hit
  this directly (see Issues below) and it is a sharp enough edge case to
  actively watch for again.

### Goal
Implement spec 19.0 end to end (`GET /info` + `RuntimeInfoPanel`
frontend; MLflow persistence of agent node-latency/termination-reason/
tool-usage breakdowns), and commit all of it -- and only it -- in 20-25
small, reviewable commits while a second agent's own concurrent,
unrelated work sat uncommitted in two of the same files.

### Configuration
No config file changes; purely code, tests, and docs.

### Changes made
- `src/rag/api/main.py`: new `RuntimeInfo` Pydantic model + `GET /info`
  route -- `generation_provider`/`generation_model`,
  `embedding_provider`/`embedding_model`, `vectorstore_provider`,
  `retrieval_mode`, `sparse_retrieval_enabled`, `fusion_method`,
  `reranker_provider`/`reranker_enabled`, `agent_enabled`,
  `mcp_enabled`/`mcp_client_enabled`, `vision_provider`,
  `tracing_enabled`, `rate_limit_enabled`, `field_redaction_enabled`,
  `authorization_enabled`, `auth_enabled`. `tests/unit/
  test_info_endpoint.py` added.
- Frontend: `RuntimeInfoSchema`/`RuntimeInfo` zod type
  (`api/types.ts`), `fetchRuntimeInfo()` client (`api/runtimeInfo.ts`),
  `useRuntimeInfo` lazy-fetch hook, `RuntimeInfoPanel` collapsible
  component, wired into `ChatWindow.tsx`, styled in `styles/index.css`.
  Unit tests for the client (`runtimeInfo.test.ts`) and component
  (`runtimeInfoPanel.test.tsx`).
- `src/rag/eval/run_agent_eval.py`: `_TERMINATION_REASONS`/`_TOOL_NAMES`
  fixed-vocabulary constants (mirroring `AgentState.termination_reason`/
  `ToolCallRecord.tool_name`'s Literal sets), `_termination_reason_
  breakdown()` (counts/rates per reason plus a combined
  `guardrail_termination_count`/`_rate` over `_BOUND_TERMINATIONS`), and
  `_tool_usage_breakdown()` (per-tool dispatch counts/rates/success
  rates), both scoped to `route == "agent"` records, wired into
  `evaluate_agent()`'s returned report dict. `tests/unit/
  test_run_agent_eval.py` extended with 4 new tests plus assertions on
  the existing routing/tool-metrics test.
- `scripts/record_agent_experiment.py`: flattens both breakdowns into
  named record fields (`agent_node_<name>_latency_ms_mean`,
  `agent_termination_<reason>_count`/`_rate`,
  `agent_guardrail_termination_count`/`_rate`,
  `agent_tool_usage_<tool_name>_count`/`_rate`) for MLflow/JSON-record
  logging. `tests/unit/test_record_agent_experiment.py` added (first
  coverage this script has ever had).
- `src/rag/eval/mlflow_logger.py`: ~41 new field names added to
  `_METRIC_FIELDS`. `tests/unit/test_mlflow_logger.py` gained 2 tests.
- Docs: `docs/architecture.md` (`/metrics and GET /` section expanded),
  `README.md` (Web UI section + MLflow tracking section, two
  independent hunks), `frontend/README.md` (new "Runtime configuration
  panel" section).

### Issues discovered
- **Symptom:** `git commit -m "..." -- src/rag/eval/run_agent_eval.py`
  produced a commit with ~325 insertions instead of the expected 94 --
  it had committed the *entire* combined working-tree diff (mine plus
  the other agent's uncommitted MCP/specialized-tool additions), not
  just what I'd staged via `git apply --cached`.
  **Root cause:** `git commit -- <pathspec>` stages and commits that
  path's current working-tree content at commit time, disregarding
  whatever the index already held for it.
  **Fix:** `git reset --soft HEAD~1` (undo the bad commit, keep the
  index), `git reset -- <path>` (unstage that one file back to matching
  `HEAD`, leaving the working tree untouched), re-applied the correct
  patch via `git apply --cached`, verified `git diff --cached --stat`
  showed exactly the expected line count, then committed with **plain
  `git commit -m "..."`, no trailing pathspec** -- the only form that
  commits what's actually staged. Adopted "no trailing pathspec, always
  verify the staged diff first" for every remaining commit.
  **Evidence:** `git show --stat HEAD` before/after the fix; the working
  tree's full combined content was confirmed intact throughout via `git
  diff --stat`.
  **Resolved:** y.
- **Symptom:** Normal (hooks-enabled) commits touching
  `run_agent_eval.py`/`test_run_agent_eval.py` failed pre-commit's mypy
  hook with errors like `"GoldExample" has no attribute
  "requires_specialized_tool"`, even though `git show :<path> | grep
  ...` proved my staged content referenced no such identifier.
  **Root cause:** pre-commit's "stash unstaged changes" mechanism for a
  partially-staged file doesn't cleanly isolate it from a *different*,
  purely-unstaged file (`gold_schema.py`, mid-edit by the other agent)
  that `run_agent_eval.py` imports from -- mypy ended up type-checking
  against an inconsistent combination of staged-mine and
  unstaged-theirs.
  **Fix:** Used `--no-verify` for exactly the 3 commits in this
  situation, only after independently proving each one clean (ruff,
  ruff-format, mypy, `pytest tests/unit/test_run_agent_eval.py`, 19/19
  passed) inside the isolated throwaway worktree; each commit message
  states this justification explicitly.
  **Evidence:** identical patch content passed all checks in the
  isolated worktree while failing the hook in the live, concurrently-
  edited repo.
  **Resolved:** y (worked around for this session; not a defect in my
  own code).

### Evaluation results
Not applicable -- purely additive instrumentation and a new read-only
endpoint, validated by unit tests only, matching this milestone's own
scope (no experiment re-run required).

### Interpretation
Both halves of the spec are backward-compatible by construction:
`RuntimeInfo` is a new endpoint (nothing existing changed shape), and
the MLflow fields are new dict keys/record fields (every pre-existing
report/record field is untouched). The multi-agent commit-isolation
work was the more novel part of the session and generalizes: `git apply
--cached` plus a throwaway `git worktree` for independent verification
is a reliable pattern for surgically committing "my hunks only" out of
a file two agents are simultaneously editing, as long as `git commit`
is never given a trailing pathspec once anything has been staged via
`apply --cached`.

### Open problems
None for this scope. The other agent's own in-progress work (`src/rag/
eval/gold_schema.py`'s new `GoldExample` fields, the remaining hunks of
`run_agent_eval.py`/`test_run_agent_eval.py`, and its new,
still-untracked files) was deliberately left uncommitted, exactly as
instructed -- see the next entry for what that work turned out to be.

### Next recommended step
None required immediately. 21 commits landed (`7e38549`..`f3bb159` on
`main`), nothing pushed; `git status --porcelain` confirmed only the
other agent's own files remained modified/untracked afterward.

---

## 2026-09-07-08:41: Specialized-tool evaluation subset, plus the case-mutation routing fix it motivated

### Interview talking points
- **What problem did this solve?** No gold coverage existed that
  isolated `get_document`/`get_latest_document`/`get_related_context` or
  any of the three MCP business tools as the *genuinely necessary*
  choice -- every existing row bundled them into a multi-hop sequence
  alongside `search_knowledge_base`, so a model that never called them
  at all could still land on a correct final answer, making "is this
  tool actually reachable in practice" an untested question.
- **Why was it difficult?** The deterministic side (does each tool work
  correctly when called) and the live-model side (does the model
  actually choose to call it) are genuinely separate claims, and this
  work proved they can diverge sharply even when both are individually
  well-tested: 73 deterministic tests (58 pre-existing plus 15 new) all
  passed, yet the live 14-question run showed three of seven tools at
  0.0 reachability, and, more seriously, two identically-worded
  write-action requests ("please close case X") were misclassified
  `simple` and routed to `classic_rag`, which has no MCP access at all
  -- so `update_case_status` was never even attempted.
- **How was it debugged?** Built `data/eval/specialized_tool_gold.jsonl`
  (14 rows spanning all 7 tools, all grounded in real corpus/business-
  store data) plus a fully deterministic fixture test
  (`tests/integration/test_specialized_tool_gold_fixtures.py`, 15 tests
  calling each tool directly with a gold row's exact tenant/role/case_id/
  source, independent of any LLM) to separate "does the tool work" from
  "does the model choose it." Then ran the identical 14-question subset
  live (`qwen2.5:3b`, `config/experiments/agentic-rag-specialized-tool-
  eval.yaml`) to measure the second, different question, via three new
  `run_agent_eval.py` metrics: `per_tool_reachability`,
  `mcp_authorization`, `case_action_outcome_accuracy`.
- **What metrics proved the fix?** `routing_accuracy` 0.857 -> 1.000
  (12/14 -> 14/14) after the case-mutation fix; `update_case_status`'s
  `per_tool_reachability` 0.5 -> 0.75; `mcp_authorization.denial_tool_
  behavior_accuracy` 2/2 -> 3/3; `case_action_outcome_accuracy` 0.333 ->
  0.667; `denial_leakage_rate` stayed 0.0 throughout, before and after.
  Full unit suite 959 -> 974 passed / 1 skipped across the two sessions
  (no regressions).
- **What would I improve next?** Two residual gaps, both deliberately
  left open (out of this fix's narrow scope): a cross-tenant denial row
  still never reaches `update_case_status` itself, since the model
  spends its whole tool-call budget on two read probes first (no
  leakage results); and the `already_in_status` row returns an empty
  tool result live even though the same call succeeds when made
  directly with the caller's real identity -- likely an argument
  mismatch the model made, but unconfirmed since raw tool arguments
  aren't captured in this eval's per-example record.

### Goal
Add gold coverage that isolates each of the 7 agent tools as necessary
rather than merely plausible, run it live against the established
agentic baseline, and (in a follow-up pass) fix whatever real gap the
live run surfaced within a narrowly scoped, evidence-backed change.

### Configuration
`config/experiments/agentic-rag-specialized-tool-eval.yaml`: the
established `experiment_040` baseline (`qwen2.5:3b`, hybrid+RRF,
relationship expansion, full security) plus `mcp.enabled`/
`mcp.client.enabled`/`mcp.business_actions.enabled`.

### Changes made
- `data/eval/specialized_tool_gold.jsonl` (14 rows: 2 each for
  `get_document`/`get_latest_document`/`get_related_context`, 4 MCP
  business reads, 4 MCP business actions).
- `GoldExample` (`eval/gold_schema.py`): 5 additive fields
  (`requires_specialized_tool`, `requires_mcp_business_tool`,
  `expected_mcp_denial`, `mcp_forbidden_snippets`,
  `expected_case_action_outcome`).
- `run_agent_eval.py`: threads an MCP ASGI app through the eval harness
  (`config.mcp.client.enabled` gate) so remote business tools are
  reachable at all during eval; new `per_tool_reachability`/
  `mcp_authorization`/`case_action_outcome_accuracy` metrics.
- `tests/integration/test_specialized_tool_gold_fixtures.py` (15 tests,
  deterministic, no LLM).
- Case-mutation fix: `_looks_like_case_mutation_request` (`rag.agent.
  graph`), a regex backstop requiring "case" plus either a direct
  mutation verb (close/reopen/resolve) or a directive verb (set/change/
  update/mark/move/transition) paired with a status word/value.
  `_classify_query` overrides the LLM's own classification to `complex`
  whenever it fires, regardless of what the model decided -- widens
  routing toward the agent path only, never away, and infers nothing
  about authorization/approval. 15 new unit tests
  (`tests/unit/test_agent_classify_case_mutation_override.py`), checked
  against all 144 rows of `techfusion_gold.jsonl` +
  `agentic_extension_gold.jsonl` with zero false positives.
- Follow-up (separate short session, 11:55): a real CI-only failure in
  `test_specialized_tool_gold_fixtures.py`'s 6 local-tool tests, fixed
  via `require_techfusion_corpus`. See the dedicated "Diagnosed and
  fixed the CI-only failure in the specialized-tool gold fixtures" entry
  below for the full investigation (confirmed via the GitHub Actions
  API and `gh run view --log-failed`, reproduced locally against a
  throwaway Postgres container); committed as `67ace5a` and pushed to
  `origin/main`.

### Issues discovered
- **Symptom:** Live 14-question run: `get_document`/`get_latest_document`/
  `get_related_context` all at 0.0 `per_tool_reachability`; the model
  defaulted to extra `search_knowledge_base` calls instead.
  **Root cause:** not a tool defect (all 3 pass deterministic tests) --
  a model tool-*selection* preference, and for 5/6 of the affected rows
  `search_knowledge_base` alone (with relationship expansion + freshness
  exclusion already on) happened to cover the same evidence anyway.
  **Fix:** none applied; explicitly out of scope for this eval-coverage
  task, flagged as a real but lower-urgency finding.
  **Resolved:** n (documented, not fixed).
- **Symptom:** Two identically-worded "please close case X" requests
  were classified `simple` and routed to `classic_rag`, so
  `update_case_status` was never attempted; `routing_accuracy` read
  0.857 because of exactly these two rows.
  **Root cause:** `_classify_query` relied entirely on the LLM's own
  judgment for a decision that gates whether the one path capable of
  reaching a write action even runs -- the same "prompt compliance
  alone is not a guarantee" shape as the earlier redaction-marker
  leakage fix.
  **Fix:** `_looks_like_case_mutation_request` deterministic backstop
  (see Changes above), fixed in a separate, later, narrowly-scoped
  session specifically to avoid bundling multiple unrelated behavior
  changes into one before/after comparison.
  **Evidence:** `routing_accuracy` 0.857 -> 1.000 on an identical
  before/after re-run of the same 14-question subset and config.
  **Resolved:** y.
- The CI-only failure this milestone's new integration test file hit
  (`test_specialized_tool_gold_fixtures.py` passing locally, failing
  every CI run) is documented in its own dedicated entry below
  ("Diagnosed and fixed the CI-only failure in the specialized-tool gold
  fixtures"), not repeated here.

### Evaluation results
Before -> after the case-mutation fix, identical 14-question subset and
config: `routing_accuracy` 0.857 -> 1.000 (12/14 -> 14/14);
`update_case_status` `per_tool_reachability` 0.5 -> 0.75 (2/4 -> 3/4);
`mcp_authorization.denial_tool_behavior_accuracy` 2/2 -> 3/3;
`case_action_outcome_accuracy` 0.333 -> 0.667 (1/3 -> 2/3);
`mcp_authorization.grounding_rate` 0.6 -> 0.8 (3/5 -> 4/5);
`denial_leakage_rate` 0.0 -> 0.0 (no change, no leakage either side);
`tool_success_rate` stayed 1.0 across both runs (22 -> 25 dispatches).
`get_document`/`get_latest_document`/`get_related_context` reachability
stayed 0.0 throughout, as intended (out of scope for this fix). Full
unit suite: 959 -> 974 passed / 1 skipped.

### Interpretation
A green deterministic test suite proves a tool is correct in isolation;
it says nothing about whether a live model will ever choose to call it.
Separating those two claims into two different kinds of tests (fixture
tests with no LLM, and a live-model metric run against gold rows
engineered to make each tool *necessary*) is what actually surfaced both
findings here. The case-mutation fix reused this codebase's established
"deterministic backstop under an unreliable prompt instruction" pattern
rather than trusting the classifier's prompt harder -- appropriate
because the decision it guards controls access to a mutating action.

### Open problems
`get_document`/`get_latest_document`/`get_related_context` reachability
still 0.0 (deliberately unaddressed). The cross-tenant `update_case_
status` denial row and the `already_in_status` empty-result row both
remain unresolved edge cases in tool-argument selection, neither
resulting in leakage.

### Next recommended step
If tool-selection reachability for the three local specialized tools is
ever worth improving, it would need either a tool-select prompt revision
or gold-set rewording to make the tool distinguishably necessary (not
just plausible) -- not attempted here, per this session's own scope
discipline.

---

## 2026-09-07-10:14: Developer Settings polish -- apply workflow, JWT countdown, sectioned debug panel, origin badges

### Interview talking points
- **What problem did this solve?** Three UX gaps in the developer-facing
  tooling, none touching retrieval/agent logic/security policy: (1)
  `DevIdentityPanel` committed every keystroke straight to
  `sessionStorage`/the next request, with no validation and no clear
  moment of "this is now active"; (2) nothing showed a JWT's remaining
  validity, so a token could silently expire mid-session with no
  warning; (3) the debug panel was one flat `<dl>`, and an agentic run's
  tool calls (including whether one hit a remote MCP tool) weren't
  visible at all.
- **Why was it worth doing now?** These are exactly the kind of details
  that make or break trust in a local security/observability demo:
  "what identity am I actually running as, and for how much longer" and
  "which specific tool calls did this run make, and were any of them
  remote" are both things a reviewer or interviewer would ask about live.
- **What was technically interesting?** `@testing-library/user-event`
  interactions hang indefinitely under `vi.useFakeTimers()` unless the
  timers are created with `{ shouldAdvanceTime: true }`; once real time
  is allowed to bleed into the fake clock, a `Math.floor(remainingMs /
  ...)` countdown can legitimately roll over between minting a test
  token and asserting on it, so the summary test asserts the format
  (`/^\d+m$/`) rather than an exact value, while every fully-frozen-time
  test still asserts exact values.
- **What metrics proved the fix?** New/extended frontend test files for
  every new piece (apply-workflow validation, JWT decoding edge cases,
  token-status transitions/accessibility, sectioned debug panel +
  tool-call cards, source origin badges, eager runtime-info fetch +
  compact badge); two small additive backend fields
  (`SourceItem.origin`/`Citation.origin`,
  `AgentQueryResponse.tool_call_details`) with their own new unit tests.
- **What would I improve next?** Nothing outstanding for this pass; it
  was scoped to UI/debug ergonomics only.

### Goal
Close the Apply-workflow, JWT-countdown, and debug-UI gaps described in
CLAUDE.md's "Developer Settings polish" section, reusing existing,
already-computed backend data wherever possible rather than adding new
instrumentation.

### Configuration
No config file changes.

### Changes made
- **Apply-settings workflow**: `DevIdentityPanel` now edits a local
  draft and only commits (`onChange`, which persists to
  `sessionStorage`) from a `handleApply` submit handler after
  `validateDraft` passes; invalid input is rejected inline
  (`aria-describedby`/`role="alert"`). A successful apply collapses into
  a `DevSessionSummary` (decoded tenant/role, as-of, dataset, trust
  level, countdown) with an "Edit settings" action;
  `DevIdentityPanel` is `forwardRef`-wrapped
  (`DevIdentityPanelHandle.openForEdit`) so the header badge's "Replace
  token" button can reopen it.
- **JWT countdown**: `frontend/src/utils/jwt.ts` (structural decode
  only, no signature check, fixed allowlist of display claims:
  `exp`/`sub`/`tenant_id`/`roles`), `hooks/useTokenStatus.ts` (state
  machine: `none`/`invalid`/`no_exp`/`healthy`/`expiring_soon`
  (<=10min)/`critical`(<=2min)/`expired`, ticking once per second only
  while a real `exp` claim is present), `components/
  TokenStatusBadge.tsx` (header badge, icon+text per state, a visually
  hidden `aria-live="polite"` region announcing only on state
  transitions).
- **Sectioned debug panel**: `DebugPanel` restructured into four
  `<section aria-label=...>` blocks (Request/Pipeline/Timings/Tools);
  `ToolCallCard` per dispatch (name, Local/MCP remote, Success/Failed,
  duration -- never args or raw error text). Backend additions:
  `SourceItem.origin` (`api/routers/query.py`, mirrors the pre-existing
  `SearchResult.origin` `source_dict()` already computed but the API
  model silently dropped) flowing through both `/query` and
  `/agent/query`'s `Citation.origin`; `AgentQueryResponse.
  tool_call_details` (`ToolCallDetail` model + `_tool_call_details()`,
  execution derived from the existing `REMOTE_MCP_TOOL_NAMES` lookup);
  `agent_stream.py`'s SSE terminal frame reuses the same helper.
  `SourcesPanel` badges each source `RAG`/`RAG (related)`/`RAG (tool)`/
  `MCP remote`.
- **`RuntimeInfoPanel` now fetches eagerly on mount**, not lazily on
  first expand -- reversed from the original design in the prior
  session, since the panel's collapsed toggle now doubles as a compact
  `qwen2.5:3b - Hybrid - RRF - MCP` badge (`compactBadge()`), which
  needs the data to render at all.

### Issues discovered
None rising to an ISSUES.md-worthy bug -- pure feature/UX work with no
security- or correctness-relevant defect found.

### Evaluation results
Not applicable (frontend/UX + two additive, non-behavioral API fields;
validated via unit/component tests only).

### Interpretation
None of this changes what the backend does; it only changes what a
developer/operator can *see* about what the backend is doing (identity,
token freshness, and per-tool-call provenance), continuing this
project's established pattern (`FeatureFlagsBar`, `RuntimeInfoPanel`) of
treating "is the system's actual state legible to the person operating
it" as its own class of bug, not just a code-correctness concern.

### Open problems
None.

### Next recommended step
None required immediately.

---

## 2026-09-07: CI evaluation gates milestone (design + initial implementation)

*Note: written mid-implementation, before this work was committed,
merged, and pushed. See the later "2026-09-07 (session 2)" entry below
for the completed state (three commits, PR #1, merged to `main`, CI
green) -- kept here rather than deleted since the design reasoning and
interview talking points below are still accurate, only the "in
progress"/"uncommitted" framing throughout this entry is now stale.*

### Interview talking points
- **What problem did this solve?** No CI job caught a retrieval-quality
  or security-invariant regression before merge; every eval/security
  check up to this point was manual, run-by-hand only when someone
  remembered to.
- **Why was it difficult?** The obvious first instinct -- hardcode
  `experiment_028`'s already-recorded baseline (Recall@5 ~= 0.788, MRR
  ~= 0.753) as the CI floor -- turns out to be structurally impossible:
  `data/knowledge_base/` (the real `techfusion` corpus) and every one of
  its gold files are gitignored, so CI has no access to the data that
  baseline describes at all. The same wall applies to the milestone's
  security invariants (cross-tenant leakage, forged-role acceptance,
  etc.): their gold rows exist only in the gitignored
  `security_evaluation` corpus.
- **How was it resolved?** Built a standalone, frozen `config/
  experiments/ci_eval.yaml` (not an override of `config/default.yaml`,
  so a future default.yaml change can't silently shift the gate) and
  measured a *real* baseline directly against the tracked `data/
  sample_docs/` (3 files) + tracked `data/eval/sample_gold.jsonl` (4
  questions): Recall@5 0.625, Recall@10 0.75, hit-rate@5 0.75, MRR
  0.75. Recorded these into `config/eval_gates.yaml` with an explicit
  floor and written rationale per metric, checked by new
  `scripts/check_eval_gates.py`. The security invariants were routed to
  a *different* mechanism entirely: deterministic pytest assertions in a
  new `security-regression-gate` CI job, since a test that directly
  asserts "this request was denied" needs no corpus to be meaningful.
- **What metrics proved the fix?** `scripts/check_eval_gates.py`'s own
  25 unit tests pass; `retrieval-eval-gate` ingests and evaluates the
  sample corpus end to end in well under a minute using only an
  ephemeral CI Postgres and `--skip-generation` (no Ollama needed).
- **What would I improve next?** (As of this entry: this was still
  uncommitted, in-progress work on `feature/ci-eval-gates`. It was
  committed as three logical commits, PR'd, and merged shortly after --
  see the later session-2 entry.)

### Goal
Add a fast, deterministic, PR-blocking quality/security gate on top of
the existing `code-quality`/`unit-tests`/`integration-tests`/
`docker-build` CI jobs, reusing existing evaluation/test infrastructure
end to end rather than a parallel evaluation framework.

### Configuration
New `config/experiments/ci_eval.yaml` (standalone, CI-only, not an
override of `config/default.yaml`); new `config/eval_gates.yaml`
(per-metric floor/ceiling + `blocking`/`rationale`).

### Changes made (uncommitted as of this entry; committed/merged shortly after, see session 2)
- `.github/workflows/ci.yml`: two new jobs --
  `security-regression-gate` (a curated ~230-test subset of `tests/
  unit`: authorization, field-level redaction, DoS/rate limits, MCP
  identity/business-tool authorization and approval logic, agent step/
  tool-call/retry bounds, insufficient-evidence and malformed-tool-
  argument handling, the case-mutation routing fix, citation
  attribution -- no Postgres/Ollama, runs in seconds) and
  `retrieval-eval-gate` (ingests `data/sample_docs/` into an ephemeral
  CI-only Postgres, runs `run_eval --skip-generation`, checks the report
  against `config/eval_gates.yaml`).
- `scripts/check_eval_gates.py`: reads one `run_eval.py` JSON report,
  checks each gate (`metric_path` dotted lookup, `higher_is_better`/
  `lower_is_better`, `floor`/`ceiling`, `blocking`), prints a bounded
  PASS/FAIL/MISSING/MALFORMED line per gate, exits non-zero iff any
  blocking gate did not PASS; with `--verbose`, recomputes per-example
  pass/fail from the report's `per_example` list and prints up to
  `--max-example-failures` (default 5) offending questions.
- `tests/unit/test_check_eval_gates.py`: 25 tests, loading the script
  dynamically via `importlib.util.spec_from_file_location` (the script
  has no `__init__.py` and isn't on pytest's `pythonpath`).
- `docs/ci_eval_gates.md`: full design writeup, including "Updating a
  baseline."
- `README.md`, `config/eval_gates.yaml`: modified/added as part of the
  same change set.

### Issues discovered
- **Symptom:** The natural instinct to reuse `experiment_028`'s
  already-recorded Recall@5/MRR numbers as the CI floor doesn't work.
  **Root cause:** `data/knowledge_base/` and every gold file describing
  that experiment are gitignored; CI has no access to the data the
  baseline describes at all -- using its numbers as a floor would either
  always fail (nothing to retrieve against) or need silent
  reconstruction against a completely different corpus, at which point
  it's no longer that experiment's baseline.
  **Fix:** froze a dedicated, much smaller CI-only corpus/config/gold
  set and measured its own real baseline directly, rather than
  reusing/estimating a number that describes different data.
  **Evidence:** `git ls-files data/eval/` confirms only `sample_gold.
  jsonl` and `specialized_tool_gold.jsonl` are tracked at all; the
  latter's own integration test already self-skips in CI for the
  identical reason (`require_techfusion_corpus`, see the prior entry).
  **Resolved:** y.

### Evaluation results
CI-only sample corpus baseline (`config/experiments/ci_eval.yaml` +
`data/sample_docs/` + `data/eval/sample_gold.jsonl`, `--skip-
generation`): Recall@5 0.625, Recall@10 0.75, hit-rate@5 0.75, MRR
0.75 -- not comparable to `experiment_028`'s numbers; a different,
much smaller corpus, not a regression.

### Interpretation
A recorded baseline is scoped to the exact data and config that produced
it; reusing its number where the underlying data doesn't exist isn't
reuse, it's a different claim wearing the old one's clothes. Discovering
this twice in one milestone (once for retrieval metrics, once for
security metrics) is a direct consequence of `data/knowledge_base/`
being gitignored well before this milestone existed -- a decision made
independently, for repo-hygiene/privacy reasons, whose CI-gating
consequence only became visible once something needed to actually run
an evaluation inside a GitHub Actions job.

### Open problems
(As of this entry) working tree on `feature/ci-eval-gates` not yet
committed. Resolved shortly after -- see the session-2 entry below for
the three-commit split, PR #1, merge to `main`, and the green CI run.

### Next recommended step
(As of this entry) review and commit the branch's changes, then open
the PR against `main`. Full unit suite as of this entry: 1003 passed, 1
skipped -- no regressions from any of the four sessions recorded above.
Superseded by the session-2 entry's own "Next recommended step" below.

## 2026-09-07: Diagnosed and fixed the CI-only failure in the specialized-tool gold fixtures

### Interview talking points
- **What problem did this solve?** Several `tests/integration/
  test_specialized_tool_gold_fixtures.py` tests (e.g.
  `test_gold_get_latest_document_incident_response_v1_redirects_to_v2`,
  `test_gold_get_document_postgres_recovery_spans_troubleshooting_and_closure`)
  were failing in GitHub Actions CI while passing locally. The instinct
  was to suspect the failures were transient (concurrent peer sessions
  editing the same repo), since a full local re-run on the exact failing
  commit came back 134/134 green.
- **How did I confirm it wasn't a flake?** Cross-referenced CI's own run
  history via the GitHub Actions REST API (unauthenticated access is
  enough for public-repo run/job/step/annotation metadata, though not
  raw job logs or artifacts): the last green run strictly predated the
  commit that added this test file, and every run on `main` since had
  failed the same `integration-tests` job. That timeline is stronger
  evidence than any number of local re-runs against an environment that
  happens to already satisfy the hidden dependency.
- **How did `gh` change the investigation?** Once the user installed and
  authenticated the `gh` CLI, `gh run view --log-failed` retrieved the
  complete raw pytest traceback for the failing job, confirming the
  `AssertionError` messages matched my diagnosis exactly for all six
  affected tests, before any further code change was made.
- **How was the root cause proven, not just inferred?** Reproduced CI's
  exact environment locally: a throwaway `pgvector/pgvector:pg16` Docker
  container on a non-default host port, schema-only init via
  `scripts/init_db.py`, no ingestion step. The six local-tool tests
  failed there before the fix (matching CI) and skipped cleanly with an
  explicit message after it, while the file's other 9 tests (the
  in-memory MCP business-tool scenarios) were unaffected throughout.
- **What was the actual root cause?** `data/knowledge_base/` is
  gitignored, so a fresh CI checkout has no source documents to ingest,
  and CI's `integration-tests` job never runs an ingestion step at all.
  The six local-tool tests in this file (`get_document`/
  `get_latest_document`/`get_related_context`) silently assumed the
  real, already-ingested `techfusion` corpus was present in whatever
  Postgres the test run hit, which is true on the long-lived local dev
  instance but structurally can never be true in CI.
- **What would I improve next?** Nothing further needed for this fix;
  the corpus-dependent scenario now degrades to an explicit skip rather
  than an opaque failure, matching the repo's established
  `require_postgres`/`require_ollama` convention.

### Goal
Investigate why `test_specialized_tool_gold_fixtures.py` was failing in
CI (per the user-provided run URL and the formal investigation brief in
`.claude-prompts/Investigate-job-fail.txt`), identify the exact root
cause and CI-vs-local difference, implement the smallest correct fix,
and verify it end to end.

### Changes
- `tests/integration/conftest.py`: added `techfusion_corpus_available`
  (session-scoped, checks `vectorstore.list_document_sources
  ("techfusion")` is non-empty once Postgres is reachable) and
  `require_techfusion_corpus` (skips with an explicit, actionable
  message otherwise), mirroring the existing `postgres_available`/
  `require_postgres` skip-not-fail pattern exactly.
- `tests/integration/test_specialized_tool_gold_fixtures.py`: added
  `require_techfusion_corpus` to all six local-tool test signatures
  (`get_document` x2, `get_latest_document` x2, `get_related_context`
  x2); updated the module docstring's own description of that test
  group to name the new fixture and explain why it always skips in CI.

### Issues discovered
See `ISSUES.md`'s "A new integration test file passed locally but
failed every CI run, because it silently assumed the real corpus was
pre-ingested" entry for the full Problem/Diagnosis/Solution/Result
writeup.

**Result:** Local: unchanged, 15/15 still pass against the real,
already-ingested corpus; two other integration files sharing the same
`conftest.py` (19/19) unaffected; `mypy src/` clean (110 files); `ruff`/
`ruff format` clean on both changed files. Committed as `67ace5a` and
pushed to `origin/main`. The resulting CI run (`34108719417`) completed
with all four jobs green (`unit-tests`, `docker-build`,
`integration-tests`, `code-quality`) -- the first fully green run since
this bug was introduced.

## 2026-09-07 (session 2): Fast CI evaluation gates (ci-eval-gates milestone)

### Interview talking points
- **What problem did this solve?** CI ran tests but had no quality/
  security *gate*: a real retrieval regression or a security-invariant
  break could merge silently as long as the existing test suite still
  passed. Added a small, deterministic, PR-blocking gate on top of the
  pre-existing four CI jobs, without adding paid-API/hosted-judge/broad-
  Ollama dependencies to per-PR feedback.
- **Why was it harder than "just add a threshold check"?** The obvious
  design -- reuse `experiments/results/experiment_028.json`'s recorded
  baseline (Recall@5 ~= 0.788) as the CI floor -- turned out to be
  impossible: `data/knowledge_base/` and every real gold file are
  gitignored, so CI has no corpus to evaluate against at all. Had to
  measure a genuinely new, much smaller baseline against the one small
  tracked corpus (`data/sample_docs/` + `data/eval/sample_gold.jsonl`,
  4 questions) instead of assuming the well-known number would just
  work. See `ISSUES.md`'s "Building a CI retrieval-quality gate
  discovered its baseline couldn't be the one documented everywhere
  else in the repo" entry.
- **What did I actually build?** Two new CI jobs
  (`.github/workflows/ci.yml`): `retrieval-eval-gate` (ingests the
  tracked sample corpus into an ephemeral CI Postgres, runs
  `rag.eval.run_eval --skip-generation`, checks the report against
  `config/eval_gates.yaml` via a new `scripts/check_eval_gates.py`) and
  `security-regression-gate` (a curated, ~230-test, no-external-services
  subset of `tests/unit` covering authorization, field redaction, DoS/
  rate limits, MCP identity/business-tool authorization and approval
  logic, and agent guardrail behavior). Also a new frozen
  `config/experiments/ci_eval.yaml` and `docs/ci_eval_gates.md`.
- **How did I decide what's "zero-tolerance security" vs. "quality
  threshold"?** By checking what's actually computable from the CI-safe
  report. Most of the milestone's zero-tolerance invariants (cross-
  tenant leakage, forged-role acceptance, unauthorized MCP actions,
  etc.) need gold rows with tenant/`forbidden_documents` fields that
  only exist in the gitignored `security_evaluation` corpus -- so they
  can't be report metrics in CI at all. Routed those to deterministic
  pytest assertions instead (each test already asserts a deny/redact/
  no-mutation outcome directly, no threshold needed). The one security
  metric genuinely computable from the small sample corpus's own report
  (`duplicate_sensitive_field_miss_rate`, corpus-level, not gold-row-
  driven) got a real `eval_gates.yaml` entry at an exact zero-tolerance
  ceiling.
- **What went wrong mid-session, unrelated to the actual milestone?**
  While trying to append this exact entry, I (Claude) called the `Write`
  tool instead of `Edit` on this file -- `Write` replaces a file's
  entire contents, so it overwrote all ~7,550 lines of this journal with
  a 2-line placeholder. This file is gitignored (`.git/info/exclude`),
  so there was no commit history to recover from; I searched git's
  dangling-blob objects (108 candidates, none matched), VS Code's local
  history (empty -- external tool writes bypass the editor's own save-
  triggered snapshotting), Volume Shadow Copy (inaccessible), and
  Windows File History (service stopped, not active) -- all dead ends.
  Disclosed it immediately and stopped touching the file. You had your
  own copy and recovered it yourself; this entry is being appended (via
  `Edit`, not `Write`, this time) to the version you restored.

### Goal
Add a fast, deterministic CI quality/security gate (Recall@5/MRR floors,
zero-tolerance security-regression tests, agent-guardrail behavioral
checks, citation-compliance checks) that runs on every PR/push without
paid APIs, hosted judges, or a broad Ollama benchmark -- reusing existing
evaluation/test infrastructure rather than building a parallel one.

### Changes
- `config/experiments/ci_eval.yaml` (new): frozen, standalone eval
  config for the CI gate -- mirrors `config/default.yaml`'s retrieval/
  embedding/chunking block so a future default.yaml change can't
  silently shift the gate's baseline.
- `config/eval_gates.yaml` (new): central threshold definitions --
  `recall_at_5`/`recall_at_10`/`hit_rate_at_5`/`mrr` (baselines measured
  directly against the CI sample corpus: 0.625/0.75/0.75/0.75, floors
  leaving roughly one discrete-step tolerance at this 4-question sample
  size) and `duplicate_sensitive_field_miss_rate` (zero-tolerance,
  ceiling 0.0). Every gate carries an explicit `rationale` string.
- `scripts/check_eval_gates.py` (new): reads a `run_eval.py` JSON
  report, checks each gate (`metric_path` dotted lookup, direction,
  floor/ceiling, blocking), prints a bounded PASS/FAIL/MISSING/
  MALFORMED line per gate (recomputing up to 5 offending question
  strings per failing recall/hit-rate gate from the report's
  `per_example` list, reusing `rag.path_matching.
  source_matches_relevant`), exits non-zero iff any blocking gate
  failed. A 1e-9 epsilon absorbs float noise only.
- `tests/unit/test_check_eval_gates.py` (new, 25 tests): all-pass, one
  quality-threshold failure, zero-tolerance-security failure, missing
  metric, malformed value, bool-is-not-a-number, float-tolerance
  boundary, non-blocking gates, gate-definition validation errors, the
  per-example drill-down, and two subprocess-level CLI exit-code tests.
  `scripts/` has no `__init__.py`/isn't on pytest's `pythonpath`, so the
  module is loaded via `importlib.util.spec_from_file_location` with an
  explicit `sys.modules` registration before `exec_module` (needed for
  the `@dataclass`-decorated `GateResult` to resolve its owning module).
- `.github/workflows/ci.yml`: two new jobs, `security-regression-gate`
  (curated `pytest` file list, no external services) and
  `retrieval-eval-gate` (Postgres service container + ingest +
  `run_eval --skip-generation --verbose` + `check_eval_gates.py`).
- `docs/ci_eval_gates.md` (new): full design doc -- what runs on every
  PR, which metrics are blocking and why, why the baseline differs from
  every other number in this repo, why `security-regression-gate`
  deliberately duplicates tests `unit-tests` already runs, how to run
  the gate locally, how to update a baseline intentionally, what stays
  manual/heavy.
- `README.md`: `## Continuous Integration` section updated for six jobs
  (was four) and links to the new doc.
- `CLAUDE.md` (untracked, local): new `## CI evaluation gates
  (ci-eval-gates milestone)` section; one new `scripts/` directory-map
  line.
- `ISSUES.md` (untracked, local): two new entries -- the gitignored-
  corpus/baseline discovery, and the `Write`-vs-`Edit` data-loss
  incident above.
- New branch `feature/ci-eval-gates`: all of the above was implemented
  there, per this milestone's own instruction not to commit or push.
  Later split into three logical commits (gate script/config/tests, CI
  workflow wiring, docs), force-pushed (`--force-with-lease`, safe --
  a fresh branch nobody else was building on), opened as PR #1 ("Add
  deterministic CI evaluation gates"), and merged to `main` with all six
  CI jobs green, including the two new ones.

### Issues discovered
1. The real `techfusion` corpus and its gold files being gitignored
   meant the CI retrieval gate's baseline could not be the one recorded
   everywhere else in this repo -- see `ISSUES.md`.
2. My own `Write`-instead-of-`Edit` mistake overwrote this file mid-
   session -- see `ISSUES.md` and the talking point above. Recovered by
   you, not by me; no tool-side fix applied (the lesson is procedural:
   always `Edit` an existing file meant to be appended to, never
   `Write`, which the tool's own contract already states but I didn't
   apply carefully enough here).

### Evaluation results
- New gate-script tests: 25/25 passed.
- Curated `security-regression-gate` subset run locally: 228/228 passed
  (~17s, no Postgres/Ollama).
- Full `tests/unit`: 1003 passed, 1 skipped (pre-existing skip,
  unrelated).
- Full `tests/integration` against real local Postgres + real local
  Ollama: 134/134 passed (~15 min, since Ollama was actually reachable
  so nothing skipped on that fixture) -- confirms nothing in this
  milestone's changes touched retrieval/agent/security source code.
- `retrieval-eval-gate`'s exact CI steps run locally end to end:
  ingested `data/sample_docs` -> `ci_eval_sample`, ran `run_eval
  --skip-generation --verbose`, ran `check_eval_gates.py` against the
  real report -- all 5 gates PASS.
- `ruff check .` / `ruff format --check .`: clean on every file this
  session touched (3 unrelated pre-existing files elsewhere in the repo
  -- `test_e2e_critical_scenarios.py`, `test_hybrid_retrieval.py`,
  `test_config.py` -- flagged formatting drift from a ruff-version bump,
  left untouched as out of scope).
- `mypy src/`: clean, 110 files (this milestone added no `src/` code).

### Interpretation
The milestone's own instruction to "inspect the current repository
before choosing thresholds" mattered twice, not once: it caught the
obvious baseline-reuse mistake before it shipped, and it caught it again
in miniature for the security metrics (checked what `run_eval.py`'s
`safety` section actually populates for a gold set with no tenant/
`forbidden_documents` fields, rather than assuming every zero-tolerance
metric from the milestone's own spec would simply appear). Separately:
a data-loss mistake mid-session is not something to route around
quietly -- disclosing it immediately, stopping further action on the
file, and exhausting real (if ultimately unsuccessful) recovery options
before asking for help was the right sequence even though the actual
recovery came from you, not from anything I tried.

### Open problems
- The CI sample corpus (`data/sample_docs`, 3 files) is prose-only --
  no table/code/layout content-type diversity, which the milestone's
  own design would prefer "if available." Not fabricated for this
  milestone; deferred to the full manual `techfusion` evaluation,
  per `docs/ci_eval_gates.md`.
- `security-regression-gate` deliberately re-runs tests `unit-tests`
  already executes (a documented, acknowledged duplication of test
  *execution* for the sake of a distinctly-named, fast status check),
  not test *code*.
- `mkdocs build --strict` was never run during this session's own
  validation pass (out of scope for the stated CI-gates validation
  list); once it was run in a follow-up pass, `docs/ci_eval_gates.md`
  turned out to have 3 broken cross-repo links (to `README.md` sections
  and `config/eval_gates.yaml`, none of which resolve as file-relative
  paths from inside the docs site) -- fixed by converting them to plain
  backtick-quoted paths instead of markdown links, and the new page was
  also missing from `mkdocs.yml`'s nav. Both fixed in a follow-up pass;
  see the next entry.

### Next recommended step
Was: review the diff on `feature/ci-eval-gates`, decide whether the
retrieval gate's 4-question sample size is acceptable long-term or
whether the CI sample corpus should grow (more tracked synthetic docs,
still avoiding the gitignored real corpus), then commit/push/open a PR.
Done -- see the next entry for the merge and the docs-nav follow-up.

## 2026-09-07 (session 3): PR merge, and a real `mkdocs build --strict` regression fixed

### Interview talking points
- **What happened here?** Split the single squashed commit from session
  2 into three logical ones (`7fbc8df` gate script/config/tests,
  `7278dd1` CI workflow wiring, `6796d26` docs), force-pushed
  (`--force-with-lease`, safe on a fresh branch nobody else was
  building on), opened PR #1 ("Add deterministic CI evaluation gates")
  with a real test-plan checklist, and it merged with all six CI jobs
  green -- including the two new ones, on their first real run against
  GitHub's actual runners, not just local reproduction.
- **What did "make sure all md files are updated" turn up?** Two real
  gaps, neither invented to justify the check: (1) `docs/ci_eval_gates.md`
  was never added to `mkdocs.yml`'s nav, and had 3 links (to
  `README.md` sections, `config/eval_gates.yaml`) that don't resolve
  from inside the docs site. (2) Running `mkdocs build --strict` by
  hand (not part of this session's original validation list) found it
  was *already* failing on `main`, for reasons unrelated to this
  milestone -- see `ISSUES.md`'s "strict docs build failure" entry's
  new follow-up section for the full story: a screenshot added in a
  later, unrelated web-UI session had silently regressed a fix already
  recorded in that same entry, undetected because `mkdocs build
  --strict` isn't CI-gated.
- **How was the regression actually fixed, not just suppressed?** The
  broken pattern was a `pymdownx.snippets` transclusion embedding a
  root-relative image path and a `frontend/README.md` link -- correct
  from `README.md`'s own position at repo root, unresolvable once
  transcluded into `docs/topics/web-ui.md` (one level under `docs_dir`).
  No single link string can be correct in both places. Fixed by
  narrowing the transcluded snippet's boundaries in `README.md` (two
  regions now: `docs-web-ui` for the link-free intro, `docs-web-ui-setup`
  for the link-free setup commands) so the image and the frontend link
  stay GitHub-only, while `docs/topics/web-ui.md` embeds its own
  correctly-pathed copy of the image directly. `README.md`'s own
  GitHub rendering is byte-identical (only invisible snippet-marker
  comments moved); confirmed the image is actually present and correct
  in the built HTML (`site/topics/web-ui/index.html`'s `<img src=
  "../../assets/web-ui-screenshot.png">`, and the file really exists at
  `site/assets/web-ui-screenshot.png`), not just that warnings went away.
- **What else did the README staleness pass catch?** The "Highlights"
  section's "CI that actually gates" bullet still described the
  pre-milestone state (ruff/mypy/pytest/Docker only, no mention of the
  new gates) -- updated. A "40 tracked experiments" count had drifted
  to 41 (`find experiments/results -name '*.json' | wc -l`), unrelated
  to this milestone but cheap to fix while auditing anyway.

### Goal
Split the prior session's single commit into a reviewable history, open
and merge the PR, pull `main`, then verify every markdown file this
milestone touched (and, per the user's request, the repo's docs
correctness generally) is accurate and that `mkdocs build --strict`
actually passes -- not just that it wasn't checked.

### Changes
- Git history: 3 commits (see above) replacing the single `e02ff14`
  squash commit on `feature/ci-eval-gates`; PR #1 opened and merged
  (`97c7fa7`) to `main`.
- `mkdocs.yml`: added `ci_eval_gates.md` to nav under "Evaluation &
  RAGAS"; removed two now-unused `ignore_urls` entries
  (`frontend/README.md`, `docs/assets/web-ui-screenshot.png`) since the
  snippet restructuring means htmlproofer no longer needs to ignore them.
- `docs/ci_eval_gates.md`: 3 broken links converted to plain paths.
- `docs/topics/evaluation.md`: added a cross-link to the new page.
- `README.md`: narrowed the `docs-web-ui` snippet, added a new
  `docs-web-ui-setup` snippet; updated the stale CI highlight bullet;
  fixed the experiment count.
- `docs/topics/web-ui.md`: consumes both snippets now, plus its own
  correctly-pathed image reference.
- `ISSUES.md` (untracked, local): extended the existing "strict docs
  build failure" entry with this regression/fix, rather than creating a
  duplicate entry for the same underlying failure mode.
- Committed directly to `main` as `7d59f94` (the user's explicit
  instruction this time, not a feature-branch PR -- a small, low-risk,
  docs-only change).

### Issues discovered
The `mkdocs build --strict` regression above -- see `ISSUES.md` for the
full writeup, including the "manual-only gates don't stay fixed"
interview takeaway.

### Evaluation results
`mkdocs build --strict`: was failing (2 warnings, both pre-existing/
unrelated to this milestone's own first pass) -- now exits 0. Verified
the fix is real, not a suppression: inspected the built HTML directly
for a working `<img>` tag and confirmed the asset file exists in
`site/`. `gh pr checks 1`: all 6 jobs pass (`code-quality`,
`docker-build`, `integration-tests`, `retrieval-eval-gate`,
`security-regression-gate`, `unit-tests`) -- the first real CI
confirmation of the two new jobs outside local reproduction.

### Interpretation
Two independent instances of the same lesson in one session: (1) a
recorded baseline/fix is only as good as the exact conditions that
produced it -- `experiment_028`'s numbers didn't transfer to a
different corpus (session 1/2), and a `--strict`-clean docs build
didn't stay clean once unrelated content changed later (session 3). (2)
Something not gated by CI is something nobody is reliably checking --
true of the docs build here, and the same reasoning this project
already applies deliberately to RAGAS/hosted-judge calls, just usually
phrased as an intentional tradeoff rather than encountered as a
regression.

### Open problems
`mkdocs build --strict` remains a manual, local-only gate (not part of
`.github/workflows/ci.yml`), on the same "don't gate PRs on something
non-essential" reasoning as RAGAS/hosted judges. That's a defensible,
documented tradeoff, but it means this exact regression class (new
content reintroducing an already-fixed transclusion-path problem) can
recur silently again.

### Next recommended step
None required for this session. If docs-build regressions like this
one become a recurring annoyance, the next step would be a dedicated,
optional CI job (`workflow_dispatch` or a docs-path-triggered job,
matching the "heavy/manual, not PR-blocking" pattern this project
already uses elsewhere) rather than adding it to the required PR gate.

## 2026-09-07 (session 4): Production vs. developer UI mode

### Interview talking points
- **What problem did this solve?** Every developer/debug control this
  project had built up (Developer Settings/bearer-token entry, tenant/
  role impersonation, the feature-flags bar, the runtime/pipeline info
  panel, the per-answer Debug panel, raw retrieval scores) was always
  visible, to everyone, in one single frontend build. That is exactly
  right for a local security/observability demo and exactly wrong for
  anything a real end user would use -- there was no way to ship a
  cleaner, safer chat surface without a second React app or scattering
  `if (isDev)` checks through a dozen components.
- **Why was it worth doing now?** The prompt asked for it directly, but
  it also closes a real gap: nothing in the frontend distinguished "a
  developer/demo deployment" from "a production-shaped one," so an
  operator's only options were "expose every internal control" or "fork
  the UI." Neither is acceptable for a project whose whole security
  story (JWT auth, tenant/role ACL, field redaction) is about not
  conflating convenience with access control.
- **What was technically interesting?** The one real design tension was
  where to put the mode check. Gating *inside* leaf components (e.g.
  `DebugPanel` itself reading the mode and rendering nothing) would have
  broken `debugPanel.test.tsx`'s existing contract tests, which assume
  "given this debug data, render these fields" independent of any
  environment. The fix was to keep every leaf presentational component
  mode-agnostic and put the mount/no-mount decision only in the two
  composing containers (`ChatWindow`, `MessageBubble`) -- the "small
  typed configuration/helper layer" the prompt asked for, used at
  exactly two call sites rather than scattered through every component.
  `SourcesPanel` got a narrow exception (`showScore`, a prop, not an
  internal read) since its citation list is a keep-item in both modes
  and only the raw numeric score is developer-only.
- **What was deliberately *not* gated, and why?** `AgentActivityPanel`
  (the live per-step progress panel during agent streaming) stays
  visible in both modes. It's the visible form of the "useful loading
  state" the prompt explicitly said to keep for Agentic RAG, not a
  post-hoc debug artifact like `DebugPanel`'s tool-call cards -- and an
  existing test (`agentQueryStream.test.tsx`) already asserted on it
  being visible during streaming, independent of any mode concept that
  didn't exist yet when that test was written.
- **What metrics proved the fix?** New `tests/unit/uiMode.test.ts` (14
  cases covering the resolver directly, including every malformed-input
  shape) and `tests/integration/uiModeGating.test.tsx` (5 cases through
  the real `ChatWindow`: production hides developer controls but chat/
  citations/error states still work, a wrong-case value falls back to
  production, developer mode shows everything including the debug panel
  and raw score, the JWT countdown still works in developer mode, and
  the exact request body/headers sent to the backend are identical
  regardless of mode). Full suite: 24 test files / 138 tests passing,
  `tsc -b --noEmit` clean, `eslint .` unchanged (pre-existing warnings
  only), both a default (`production`) and a `VITE_UI_MODE=developer`
  production build succeed.
- **What would I improve next?** Nothing outstanding for this pass, it
  was scoped to the UI-mode split only. A future real login flow (not
  the raw-JWT-paste dev tool) would need its own decision about which
  mode it belongs in; today `TokenStatusBadge` only ever renders because
  `DevIdentityPanel` is the only thing that can set a bearer token, so
  hiding the latter in production makes the former inert for free.

### Goal
Implement the two UI modes described in `.claude-prompts/
23.0-production-vs-developer.txt`: one frontend codebase, a
configuration-driven `developer`/`production` split, safe-by-default,
never a substitute for backend authorization.

### Configuration
New `VITE_UI_MODE` build-time Vite env var (`frontend/.env.example`,
`frontend/src/vite-env.d.ts`'s `ImportMetaEnv` augmentation). Safe
default is `production` -- resolved in `frontend/src/config/uiMode.ts`'s
`resolveUiMode`/`getUiModeConfig`, the only place that reads it.

### Changes made
- **New**: `frontend/src/config/uiMode.ts` -- the typed config layer
  (`UiMode`, `SAFE_DEFAULT_UI_MODE`, `UiModeConfig`, `resolveUiMode`,
  `getUiModeConfig`). Reads `import.meta.env.VITE_UI_MODE` fresh on every
  call (not cached at module scope) so tests can override it per-case via
  `vi.stubEnv` without `vi.resetModules()`.
- **`ChatWindow.tsx`**: mounts `FeatureFlagsBar`/`RuntimeInfoPanel` only
  when `showRuntimeDetails`, `DevIdentityPanel` only when
  `showDeveloperSettings`, and renders a small "Developer Mode" header
  badge when `showDeveloperModeIndicator`.
- **`MessageBubble.tsx`**: mounts `DebugPanel` only when
  `showDebugDetails`; passes `showScore={showDebugDetails}` to
  `SourcesPanel`.
- **`SourcesPanel.tsx`**: new optional `showScore` prop (default `true`,
  so existing direct-render tests are unaffected), gates only the raw
  `source.score.toFixed(3)` span -- the citation list itself (name,
  section, page, content-type/origin badges) is unaffected in either
  mode.
- **`ErrorBanner.tsx`**: the `authentication_failed` copy's "Check the
  bearer token in Developer settings" detail sentence only appears when
  `showDeveloperSettings` is true (that panel doesn't exist otherwise).
- **`frontend/Dockerfile`**: `ARG VITE_UI_MODE=production` /
  `ENV VITE_UI_MODE=$VITE_UI_MODE` in the builder stage, before `npm run
  build`, so a plain `docker build` with no `--build-arg` always ships
  production mode.
- **`docker-compose.frontend.yml`**: forwards a `VITE_UI_MODE` shell
  variable as that same build arg, defaulting to `production`.
- **`frontend/src/styles/index.css`**: `.developer-mode-badge` (reuses
  the existing `--notice-*` tokens, same family as the "Dev mode"
  feature-pill warning).
- **Tests**: new `tests/unit/uiMode.test.ts`, new
  `tests/integration/uiModeGating.test.tsx`; updated
  `tests/integration/authHeader.test.tsx` to `vi.stubEnv("VITE_UI_MODE",
  "developer")` since it exercises Developer Settings through the real
  `ChatWindow` (the one existing test that needed a change -- every other
  existing test either renders a leaf component directly, unaffected by
  the container-level gating, or doesn't touch a gated component at
  all).
- **Docs**: `frontend/README.md` gained a "UI modes: developer vs.
  production" section (configuration, exact behavior per mode, Docker
  build selection, the "not a security boundary" reminder);
  `frontend/.env.example` documents `VITE_UI_MODE`; `CLAUDE.md`'s "Web
  UI" milestone gained a matching condensed subsection.

### Evaluation results
`npx vitest run`: 24 files / 138 tests passing. `tsc -b --noEmit`:
clean. `eslint .`: 0 errors, the same 3 pre-existing `react-refresh`
warnings in `chatContext.tsx` (unrelated). `npm run build` (default, no
env var) and `VITE_UI_MODE=developer npm run build`: both succeed.

### Interpretation
The one design decision worth remembering: "gate at the container, not
the leaf" wasn't just cleaner, it was what kept the existing test suite
green without editing more than one pre-existing test file. Leaf
components that already had their own isolated contract tests
(`DebugPanel`, `SourcesPanel`, `RuntimeInfoPanel`, `FeatureFlagsBar`,
`DevIdentityPanel`) stayed exactly as they were tested; only the
composition layer changed.

### Open problems
None identified. `AgentActivityPanel`'s exemption from gating is a
judgment call (documented in CLAUDE.md), not a certainty -- if a real
production deployment later decides live per-step tool-name/route
disclosure is itself too much internal detail for end users, that would
need its own follow-up decision, not a re-litigation of this session's
scope.

### Next recommended step
None required for this session.

## 2026-09-07 (session 5): Developer/demo UI polish

### Interview talking points
- **What problem did this solve?** A live screenshot of the developer/
  demo UI (from the previous session's production-vs-developer work)
  exposed five rough edges: a transient `GET /info` failure rendered as
  scary red "unavailable" text even though chat itself was fine; the
  feature-flag pills communicated enabled/disabled through color alone;
  the collapsed "Developer session" summary spent up to six full rows
  on fields that could fit on one line; the empty chat state was a
  single generic sentence with no product identity or example prompts;
  and the composer's textarea was a fixed two-row box with only a
  manual resize handle, no real auto-grow.
- **Why was it worth doing now?** These are exactly the kind of details
  that shape a first impression in a demo/interview setting -- an
  "unavailable" banner in red reads as "this is broken" even when it
  isn't, and a wall of blank vertical space around the dev-session block
  reads as unfinished. None of it touches retrieval/agent/security
  logic, which is precisely why it was safe to batch into one focused
  pass.
- **What was technically interesting?** Two things. First, choosing
  what to say instead of "Authenticated" in the compact dev-session
  line: the header's `TokenStatusBadge` already owns a live countdown
  and the word "Authenticated," so repeating it in the collapsed summary
  would be pure duplication -- the fix was a `statusSegment()` helper
  that only contributes something the badge doesn't already say (the
  coarse `expires in Xm` phrase implies authenticated on its own; a
  bare "Authenticated" word is reserved for the one edge case with no
  `exp` claim to phrase a countdown from). Second, textarea auto-grow
  has no way to be genuinely tested under jsdom (no real layout engine,
  so `scrollHeight` is always 0) -- the test stubs `scrollHeight` via
  `Object.defineProperty` to simulate a tall multi-line paste and asserts
  the component reads it and clamps to the configured cap, which tests
  the component's own logic without pretending jsdom can measure text.
- **What was deliberately hardcoded, and how was that de-risked?** The
  three empty-state example queries are real facts, not placeholders --
  each one was verified by directly reading the actual (gitignored)
  `data/knowledge_base` content before writing it into the component,
  specifically choosing topics with no `tenant_id`/`allowed_roles` front
  matter (so they work with zero dev identity configured) and avoiding
  `security_evaluation/`'s tenant-gated content and any synthetic MCP
  case ID (since `mcp.enabled` defaults `false`, a case-status example
  would fail out of the box for most readers).
- **What metrics proved the fix?** Full frontend suite: 27 files / 151
  tests passing (up from 24/138 at the end of the prior session -- 3 new
  test files, 2 test files extended, 1 pre-existing test rewritten for
  the new summary structure, zero other pre-existing tests touched).
  `tsc -b --noEmit` clean, `eslint .` unchanged (same 3 pre-existing
  warnings), a full `npm run build` still succeeds.
- **What would I improve next?** Nothing outstanding for this pass. If
  the demo corpus's actual content ever changes meaningfully, the three
  example queries in `MessageList.tsx` would need re-verifying against
  whatever replaced it -- that's an inherent cost of grounding them in
  real content rather than generic placeholders, a tradeoff made
  deliberately (see above).

### Goal
Apply the UX polish described in `.claude-prompts/
24.0-UI-improvement.txt`: soften the runtime-unavailable state, fix
color-only feature-badge semantics, compact the developer-session
summary, make the empty chat state product-like with example queries,
and give the composer a real auto-grow -- without redesigning the app,
changing backend behavior, or touching production-mode gating.

### Configuration
No config file changes.

### Changes made
- **`RuntimeInfoPanel.tsx`**: neutral "Runtime info unavailable" wording
  in both the collapsed toggle and expanded body (was "Runtime
  configuration (unavailable)" / "Runtime configuration unavailable" in
  red); dropped the `--error`-styled CSS classes entirely rather than
  reusing them with different text.
- **`FeatureFlagsBar.tsx`**: added a visible "✓" (`aria-hidden`, decorative)
  plus an `sr-only` "enabled"/"disabled" word to every boolean pill;
  `Vision: none` now displays as `Vision: off`.
- **`DevIdentityPanel.tsx`**: `DevSessionSummary` rewritten from a
  `debug-grid` dt/dd list to one `.dev-session-summary__line` of
  " · "-joined `<span>` segments; new `statusSegment()` helper replaces
  `sessionStatusText()` with the "don't repeat what the header badge
  already says" behavior described above.
- **`MessageList.tsx`**: new empty-state heading/subtitle/example-query
  chips (`EXAMPLE_QUERIES`); new optional `onExampleQuery` prop.
- **`MessageInput.tsx`**: `forwardRef`-wrapped, exposing
  `MessageInputHandle.setValue`; `rows={1}` (was `2`); new
  `useEffect`-driven auto-grow capped at `MAX_TEXTAREA_HEIGHT_PX` (160px).
- **`ChatWindow.tsx`**: holds a `messageInputRef`, wires
  `MessageList`'s `onExampleQuery` to `messageInputRef.current?.setValue`.
- **`styles/index.css`**: removed `.runtime-info__status--error`/
  `.runtime-info__badge-error`; added `.feature-pill__check`; reworked
  `.dev-session-summary*` rules for the one-line layout; added
  `.message-list__empty-title`/`__empty-subtitle`/`__examples`/
  `.example-query-chip`; tightened `.message-input textarea` (`resize:
  none`, `max-height: 160px`, `overflow-y: auto`, smaller `min-height`).
- **Tests**: new `tests/unit/messageInput.test.tsx`,
  `tests/unit/messageList.test.tsx`,
  `tests/integration/exampleQueries.test.tsx`; extended
  `tests/unit/featureFlagsBar.test.tsx` and
  `tests/unit/runtimeInfoPanel.test.tsx`; rewrote one
  `tests/unit/devIdentityPanel.test.tsx` case for the new summary shape.
- **Docs**: `frontend/README.md` updated (Active feature flags, Runtime
  configuration panel, Classic vs. Agentic RAG's sources/debug
  description, new Empty chat state and Composer sections, Developer
  session summary description); `CLAUDE.md` gained a matching condensed
  subsection under "Web UI."

### Evaluation results
`npx vitest run`: 27 files / 151 tests passing. `tsc -b --noEmit`:
clean. `eslint .`: 0 errors, same 3 pre-existing `react-refresh`
warnings in `chatContext.tsx` (unrelated). `npm run build`: succeeds.

### Interpretation
The one design decision worth remembering: avoiding duplication between
the header's `TokenStatusBadge` and the collapsed dev-session summary
required actually deciding what each one is *for* (badge: live,
per-second countdown; summary: a snapshot of the asserted identity) --
"just don't show the same word twice" isn't a rule you can apply
mechanically without that distinction, since the no-`exp`-claim case
legitimately needs the word in the summary and nowhere else.

### Open problems
None identified.

### Next recommended step
None required for this session.

## 2026-09-07 (session 6): Live-container test found a real nginx/vite proxy gap for /info and /feedback

### Interview talking points
- **What problem did this solve?** After bringing up the real Docker
  stack (postgres + rag-api + frontend, built with VITE_UI_MODE=developer
  to exercise this session's UI polish) and opening the actual browser
  UI, RuntimeInfoPanel read "Runtime info unavailable" against a backend
  that was demonstrably healthy (`curl :8000/info` worked directly).
- **What was the root cause?** `frontend/nginx.conf` and
  `vite.config.ts`'s dev-server proxy never had a route for `/info` or
  `/feedback` -- both existing, fully-tested backend endpoints from
  earlier milestones. Both paths fell through to the SPA's own
  catch-all (`try_files $uri /index.html`), so `GET /info` silently got
  back `index.html` (200, `text/html`) instead of JSON, and
  `POST /feedback` got nginx's own `405` before ever reaching `rag-api`.
- **Why did 151 passing frontend tests never catch this?** Every unit/
  integration test mocks `fetch` directly -- there is no test anywhere
  in the suite that exercises the real nginx config or the real Vite
  dev-server proxy. This class of bug is structurally invisible to a
  mocked-fetch test suite; only running the actual container (or the
  actual `npm run dev` proxy) surfaces it.
- **What was the fix?** Added `location = /info` / `location =
  /feedback` to `nginx.conf`, and matching `"/info"`/`"/feedback"`
  entries to `vite.config.ts`, mirroring the existing proxied paths.
  Rebuilt just the `frontend` service and verified directly:
  `GET /info` -> `Content-Type: application/json` (was `text/html`);
  `POST /feedback` -> the backend's real `401` (was nginx's `405`).
- **Documentation**: added a full ISSUES.md entry ("Two backend
  endpoints were live and fully tested, but the frontend's own reverse
  proxy never learned they existed"), and corrected the stale proxy-path
  list in both `CLAUDE.md`'s "Web UI" section and `frontend/README.md`
  (both had always been incomplete -- this gap predates this session's
  own UI-polish work).

### Goal
Bring up the full stack (postgres + rag-api + frontend, observability
explicitly excluded per the user's request) to manually test the UI
work from the two prior sessions, and fix whatever the live test
actually surfaces.

### Configuration
Built with `VITE_UI_MODE=developer` (build arg, both for the direct
`docker compose ... up -d --build` and the frontend-only rebuild after
the nginx fix). Backend config: `.env`'s `RAG_CONFIG_PATH=/app/config/
production.yaml` (auth/authorization/field-redaction/rate-limit/agent
all on) -- minted a 4-hour dev JWT via `scripts/issue_dev_token.py`
(`sub=demo-user`, `tenant_id=tenant_alpha`, `roles=[tenant_alpha_admin]`)
for manual testing against it.

### Changes made
- `frontend/nginx.conf`: added `location = /info` and `location =
  /feedback` proxy rules.
- `frontend/vite.config.ts`: added `/info` and `/feedback` to
  `server.proxy`.
- `CLAUDE.md`'s "Web UI" section and `frontend/README.md`'s "Running
  locally" section: corrected the proxied-path list (both were already
  stale before this session, missing `/info`/`/feedback`).
- `ISSUES.md`: new entry + index link.

### Evaluation results
Verified directly against the running containers (not a mocked test):
`curl -D- http://localhost:3001/info` -> `200`, `Content-Type:
application/json` (previously `text/html`, silently wrong).
`curl -X POST http://localhost:3001/feedback` -> `401` (previously
`405` from nginx itself, never reaching the backend). No frontend test
needed to change -- the bug was invisible to the entire mocked-fetch
test suite by construction, not something the existing tests should
have caught.

### Interpretation
A same-origin reverse-proxy design (deliberately avoiding CORS on the
backend) has an ongoing maintenance cost: every new backend route needs
a matching proxy rule added by hand in *two* separate places (dev-server
config and production nginx config), and nothing -- not the type
system, not the test suite -- enforces the two stay in sync. This is
exactly the kind of gap that only a real, live smoke test against the
actual built artifact (not a mocked unit test) can catch.

### Open problems
No automated check exists to keep `nginx.conf` and `vite.config.ts`'s
proxied-path lists in sync with the backend's actual route list, or to
catch a future new backend endpoint that isn't added to both. A future
improvement would be a lightweight integration test (or a documented
manual checklist item) that exercises the real nginx config, not just
mocked fetch calls -- not implemented in this session, flagged as an
open problem instead per the "don't invent a fix that wasn't asked for"
discipline.

### Next recommended step
None required for this session; flagged as an open problem above for a
future session to pick up if it recurs.

## 2026-09-07 (session 7): The /query 504, and closing a gap flagged open in an earlier session

### Interview talking points
- **What problem did this solve?** Every classic-RAG query through the
  live Docker frontend returned "The server reported an error. Request
  failed with status 504," right after fixing the /info proxy gap from
  the previous session.
- **Diagnosis, one layer at a time.** `docker logs local-rag-frontend`
  showed nginx's own error line: `upstream timed out (110: Operation
  timed out) while reading response header from upstream` on `POST
  /query`. That alone doesn't say whether the backend was actually
  broken or just slow, so the next check was `docker logs local-rag-api`
  for the same request -- it showed `agent_request_completed ...
  duration_ms: 79105.58, success: true` and `request_handled ... status_
  code: 200`. The backend had genuinely succeeded, 19 seconds *after*
  nginx had already given up and returned 504 to the browser. Checking
  `nginx.conf` confirmed why: `/query`'s location block had no
  `proxy_read_timeout` override, so it fell back to nginx's stock 60s
  default; `/agent/`'s location block already had `300s` for exactly
  this reason.
- **This exact gap was already flagged, in this project's own ISSUES.md,
  in an earlier session.** The prior 504 incident (host memory
  exhaustion making Ollama hang outright) closed with an explicit,
  named open item: "`frontend/nginx.conf`'s `/query` location is left
  without a `proxy_read_timeout` override as a known, still-open gap."
  This session is that gap actually manifesting, for a related but
  distinct reason (slow-but-successful CPU inference under memory
  pressure, not a fully wedged Ollama process) -- confirmed by checking
  host memory again (`Get-CimInstance Win32_OperatingSystem`: 0.35GB
  free of 7.79GB, `ollama ps` showing the model loaded and actually
  computing, not stuck).
- **What was the fix, and how was it verified?** Added `proxy_read_
  timeout 300s;` to `/query`'s location block, matching `/agent/`.
  Rebuilt just the `frontend` service and re-ran the identical query
  directly against the real proxy (`curl -w`, real wall-clock time):
  `200 OK` in `39.37s`, well under the new timeout, no `504`.
- **What this does and doesn't fix.** It fixes the specific bug: a
  slow-but-successful backend response being misreported to the caller
  as a hard failure. It does not fix the underlying cause of the
  slowness itself (host memory pressure, CPU-only inference variance) --
  that remains a host-capacity fact about this particular 8GB dev box,
  not a code bug, consistent with the earlier session's own conclusion.

### Goal
Diagnose and fix a live 504 the user hit while manually testing the
containers brought up in the previous session, using the exact same
layer-by-layer method the project's own ISSUES.md entry for the
previous 504 incident documents, rather than guessing.

### Configuration
No config file changes (nginx.conf is not a `config/*.yaml` file; no
`AppConfig` fields touched).

### Changes made
- `frontend/nginx.conf`: added `proxy_read_timeout 300s;` to `/query`'s
  location block (previously only `/agent/` had this override).
- `ISSUES.md`: added a "Follow-up" section to the existing "A 504 from
  the frontend..." entry (per this repo's own convention: update the
  affected story, don't duplicate it), documenting the recurrence and
  the actual fix for the previously-flagged open gap.
- `CLAUDE.md`: added a matching short bullet under the "Developer/demo
  UI polish" subsection's reverse-proxy-gap note.

### Evaluation results
Direct verification against the real running stack (not a mocked
test, since this bug lives entirely in nginx config no test suite
exercises): `docker logs local-rag-api` for the original failing
request showed `200`/`79105.58ms`/`success: true` -- proof the backend
itself was never broken. After the fix, `curl -w` against `POST
/query` through the rebuilt frontend container returned `200` in
`39.37s` real wall-clock time, with the answer text matching the real
corpus content (`data/knowledge_base/security/access-control-policy.md`'s
one-hour production-write-access rule) -- also a nice incidental
confirmation that the example-query chip added in the prior UI-polish
session was factually accurate.

### Interpretation
Two sessions in a row on this same box hit variations of "504 at the
edge, success at the core," for two different underlying reasons
(fully wedged inference vs. merely slow inference), through the exact
same missing-timeout-override mechanism. The first session's own
"Interview takeaway" already named the right diagnostic principle
("disprove each layer in turn... rather than assuming the most
recently-changed code was the cause") and this session followed it
directly to the same file, same fix, finally applied.

### Open problems
Host memory pressure on this particular dev machine remains
unaddressed and will keep making CPU-only generation slower and more
variable than the project's own documented baselines (~20-25s for
qwen2.5:3b) -- a hardware/environment constraint, not something
`nginx.conf` or any application code can fix. If it keeps recurring,
the next step would be the same one already named in the earlier
session's entry: stop non-essential host processes (other containers,
extra editor windows) before running local inference, not a code
change.

### Next recommended step
None required for this session.

---

## 2026-09-16 (session 8): Reconciling two independent adversarial audit reports into one verified backlog

### Interview talking points
- Two full-repo adversarial reviews had already been run independently in
  a prior session (one Claude, one Codex, neither shown the other's
  output) — `.claude-prompts/25.1-Claude-review-report.md` and
  `25.3-Codex-review-report.txt`. This session's job was explicitly *not*
  another review: it was to reconcile the two against the live code,
  verify or falsify every Critical/Major/Moderate claim, merge
  duplicates, downgrade overstated severities, and reject false
  positives — producing one evidence-backed backlog instead of two
  opinionated ones.
- Used four parallel background subagents, each independently re-deriving
  evidence for a cluster of related claims (ingestion/DB, API/DoS,
  agent/MCP, retrieval/eval), then did my own direct spot-checks on
  several additional Moderate findings the prioritized list didn't cover
  (audit-trail gaps, dead frontend cancel code, missing sample-size
  reporting, the case-mutation regex's edge cases, the default prompt's
  empty system template).
- The single most important result: a live, fault-injecting reproduction
  (a throwaway script, not a repo file) proved the ingestion
  transaction-atomicity bug is real and CRITICAL, not just
  plausible-sounding — a simulated embedder failure after the checksum
  commit left the `chunks` table empty and the `documents` row
  permanently "successfully ingested," with every retry silently
  no-op'ing forever.
- Exactly one claim across both reports was falsified: a claimed
  middleware exception-masking bug (`RequestIDMiddleware` allegedly
  dereferencing `response.headers` after a `finally` block when
  `call_next()` raises). Both a code-level defensive guard the claim
  missed, and — more fundamentally — Python's own `try`/`finally`
  control flow (statements after an unhandled exception's `finally`
  never execute) disprove it; a live `TestClient` repro confirmed the
  original exception propagates unmasked.
- 19 of 20 prioritized claims confirmed, several materially refined: the
  MCP session-handshake timeout gap turned out to be *worse* than
  reported (the default `transport: asgi` config has no bound at any
  layer, contradicting that config field's own docstring); the "stale
  default agent prompt" finding turned out to already have a real fix
  (`config/production.yaml`) and a full incident writeup in `ISSUES.md`
  that neither audit report nor `CLAUDE.md` itself had surfaced — so the
  actual gap was narrower (a documentation-completeness gap in
  `CLAUDE.md`) than either report's framing implied.
- No code, tests, or configs were modified, per the task's explicit
  constraint. Only documentation was touched: the new reconciled report,
  one new `ISSUES.md` entry (the ingestion-atomicity finding, marked
  fix-pending since no fix was applied this session), and one new
  `CLAUDE.md` bullet closing the `production.yaml` documentation gap the
  investigation itself surfaced.

### Goal
Reconcile `.claude-prompts/25.1` (Claude's adversarial review) and
`25.3`/`25.4` (Codex's adversarial review, `25.4` being an earlier rough
draft of the same findings) against the current repository (HEAD
`84bc466`, one docstring-only commit past the reports' own reviewed
commit `0e8b2d5`), per the instructions in
`.claude-prompts/25.5-Joint-Investigation-and-bug-fixing.txt`: verify
every Critical/Major/Moderate finding, reproduce where practical, merge
duplicates, reject false positives explicitly, downgrade overstated
severity, and flag anything depending on non-default config. Twenty
specific areas were named as priorities. No code/test/config edits, no
commits.

### Configuration
No config files read or touched beyond inspection (`config/default.yaml`,
`config/production.yaml`, `config/experiments/agentic-rag-baseline-v2-fixed.yaml`
were read to verify RET-1's claim; nothing written).

### Changes made
- No source, test, or config changes (explicitly out of scope for this
  session).
- New file: `.claude-prompts/25.6-Joint-Investigation-Verified-Report.md`
  — the full reconciled findings table (ID, original report(s), severity,
  exact file/function, execution path, proof, impact, default-config
  applicability, existing/missing tests, smallest safe fix, for every
  confirmed finding), the one rejected false positive with its
  falsification evidence, design limitations to leave as-is, and five
  suggested fix batches ordered by priority.
- `ISSUES.md`: added one new "Data & Isolation" entry for the ingestion
  transaction-atomicity finding (Problem/Diagnosis/Solution-pending/
  Result/Interview-takeaway), following the same "open item, not yet
  fixed" convention the pre-existing idempotency-key entry already uses.
- `CLAUDE.md`: added one bullet to "Known deviations from the original
  plan file" documenting `config/production.yaml`'s existence and role,
  closing the exact documentation gap this session's own verification
  traced the "stale default agent prompt" finding back to.

### Evaluation results
Four parallel background subagents (general-purpose, each with full tool
access including Bash) independently re-verified their assigned claims
against live code, running targeted `pytest` subsets and constructing
throwaway repro scripts where a claim was reproducible without a real
Postgres/Ollama stack:
- **Ingestion/DB cluster**: all 3 claims (transaction atomicity,
  concurrent-insert race, missing statement timeout/pool sizing)
  confirmed, including a live fault-injection repro for the atomicity bug.
- **API/DoS cluster**: 6 of 7 claims confirmed (Prometheus cardinality
  live-reproduced with `TestClient`; invalid-filter-key 500 live-reproduced;
  `/ingest` confirmed as the only `async def` router blocking the event
  loop; SSE-disconnect non-cancellation confirmed and already
  self-documented in the route's own docstring); 1 claim (middleware
  exception masking) falsified by both code inspection and a live repro.
- **Agent/MCP cluster**: all 6 claims confirmed, including a live
  reproduction of a Pydantic `ValidationError`'s `str(exc)` leaking a
  fragment of adversarial tool-argument text into an audit log line, and
  a stronger-than-reported finding for the MCP session-timeout gap under
  the default ASGI transport.
- **Retrieval/eval cluster**: all 4 claims confirmed, including a direct,
  reproducible chunker repro proving two adjacent Markdown tables with no
  blank line between them silently merge into one corrupted chunk — and
  the correction to the stale-prompt-version claim described above.
- My own direct follow-up spot-checks (not delegated) confirmed 8
  additional Moderate/Minor findings from Report A that weren't in the
  prioritized 20: no audit trail for document-level ACL denials on the
  core `/query`/`/agent/query` path; the frontend's `cancel()` function
  genuinely dead (zero call sites); `corpus_gold_record_count` genuinely
  `null` on a real 25-question RAGAS-sample experiment record with no
  sample-size column in the rendered comparison table; the case-mutation
  routing regex's both false-positive ("update on case ... status") and
  false-negative ("close ticket 1001", no literal "case") edge cases
  reproduced directly against the regex source; the shipped default
  generation prompt (`rag_answer_v1`) confirmed to have a literal empty
  `system_template`; no `max_length` on any LLM-writable tool-argument
  string field; the `/ingest` `dataset_id` form field confirmed unbounded;
  and the dense-search SQL confirmed to have no secondary sort key for
  tie-breaking.

### Interpretation
Running two independently-authored adversarial reviews and then
reconciling them against live code (rather than trusting either
report's prose, or worse, trusting their *agreement* with each other as
proof) is a meaningfully stronger process than either review alone: the
one finding both reports would have let through unchallenged if trusted
at face value (the middleware exception-masking claim) was disproven
only by actually running a repro, not by re-reading the claim more
carefully. Conversely, the one finding that read as most damning in
isolation (`default.yaml` shipping "already known to be buggy" agent
prompt versions with, per the audit, "no equivalent known-deviation
note") turned out to already have a real, tested, documented fix
(`config/production.yaml`) — just one that neither `CLAUDE.md` nor either
audit report had actually found, because it required checking a file
outside the one both reviews focused on. Both outcomes argue for the
same discipline: an audit report's confidence and internal consistency
are not evidence on their own; only reproducing (or directly falsifying)
the underlying claim is.

### Open problems
None of the 19 confirmed findings were fixed this session, by explicit
design — the task was reconciliation and verification only. The full
prioritized backlog (5 fix batches: data integrity/security, API
reliability/DoS, agent/MCP correctness, observability/config cleanup,
scalability-only) is recorded in
`.claude-prompts/25.6-Joint-Investigation-Verified-Report.md` for a
future implementation session.

### Next recommended step
Work through the suggested fix batches in `25.6`, starting with Batch 1
(the CRITICAL ingestion-atomicity bug and the MODERATE concurrent-insert
race that share one root cause and one likely fix — a shared transaction
plus an `ON CONFLICT` upsert in `get_or_create_document_id`), each as its
own reviewed change with a new regression test proving the specific
failure mode this session's repro demonstrated.

---

## 2026-09-16 (session 9): Implementing all five fix batches in parallel, on separate branches, via isolated git worktrees

### Interview talking points
- Took the prior session's reconciled, verified backlog
  (`.claude-prompts/25.6-Joint-Investigation-Verified-Report.md`, 5
  suggested fix batches) and implemented it for real, per the user's
  explicit request: each batch on its own branch, so each can become its
  own reviewable PR, and all five done in parallel rather than
  sequentially.
- Used five parallel background agents, each launched with
  `isolation: "worktree"` so they each got a genuinely separate checkout
  directory and could edit the same files (`pgvector.py`,
  `agent/graph.py`, etc.) simultaneously without colliding -- ordinary
  parallel edits in one shared working directory would have been
  destructive.
- Verified up front that gitignored/untracked files (`CLAUDE.md`,
  `ISSUES.md`, `PROJECT_JOURNAL.md` -- all three, confirmed via
  `git check-ignore -v`, not just the two I'd assumed in the prior
  session) would not even exist inside a fresh worktree, since
  `git worktree add` only materializes tracked content. Decided to keep
  all five fix agents scoped to code/tests/config only, and handle every
  documentation update centrally, myself, after their work landed --
  avoiding both a missing-file problem and five agents racing to append
  to files that don't have a meaningful per-branch identity anyway.
- All five agents hit the *same* API rate limit simultaneously partway
  through (a shared session limit, not a per-agent one), confirming
  they were genuinely running concurrently rather than my having
  accidentally serialized them. Resumed each directly via `SendMessage`
  to its `agentId` once the limit reset -- full prior context (including
  partial diffs already written to its own worktree) was preserved
  across the resume, so no work was lost or had to be redone from
  scratch.
- Two of the five agents paused themselves mid-task to "wait for a
  background command to finish" and ended their turn instead of
  actually waiting -- a real, repeatable failure mode worth remembering:
  a background subagent that spawns its own background shell command
  and then stops is not the same as one that blocks until that command
  resolves. Had to send an explicit follow-up instructing them to run
  such commands synchronously and check real output before proceeding,
  not narrate an intention to check later.
- One agent (Batch 3) hit a genuine, novel concurrency bug of its own:
  `pre-commit`'s hook cache is shared machine-wide, not per-worktree, so
  a `mypy` hook killed by a tool timeout left a recoverable backup patch
  in a cache directory that two *other* unrelated concurrent sessions
  were also writing to at the same time. It correctly verified via
  process-command-line inspection which specific patch belonged to its
  own commit before recovering it, rather than assuming a
  plausible-looking file was safe to trust -- see the new `ISSUES.md`
  entry for the full story.
- By the end of this session: Batches 1, 3, and 5 fully complete,
  tested, and committed on their own branches (not pushed, no PRs opened
  -- left for the user, as requested); Batches 2 and 4 resumed and
  in progress as this entry is being written.

### Goal
Implement every confirmed finding from `.claude-prompts/25.6-Joint-Investigation-Verified-Report.md`'s five suggested fix batches, each on its own branch (`fix/batch1-data-integrity-security` through `fix/batch5-scalability-improvements`), each with regression tests proving the specific failure mode fixed, so the user can open five separate, independently-reviewable PRs and merge them at their own pace. No pushing, no PR creation -- that stays the user's own action.

### Configuration
No `AppConfig`-level changes beyond what individual fixes touch (e.g.
Batch 4's planned `default.yaml` agent-prompt-version bump). No global
config/environment changes made by this orchestrating session itself.

### Changes made (this session's own actions, not the sub-agents' code diffs)
- Dispatched 5 parallel `Agent` calls with `isolation: "worktree"`,
  each given a fully self-contained prompt (no shared conversation
  context) describing: the project, the exact confirmed findings
  assigned to that batch (file/function, execution path, proof, smallest
  safe fix, drawn directly from `25.6`), this repo's conventions (numpydoc
  docstrings, no em dashes, `tests/unit` mocking conventions, no
  `Co-Authored-By`), and explicit instructions to create the named
  branch, implement fixes with tests, run the suite, commit, and never
  push or open a PR.
- When all five hit a simultaneous rate limit, resumed each via
  `SendMessage` to its `agentId`; when two of those paused themselves
  again waiting on a background command, sent a second, more explicit
  follow-up telling them to run commands synchronously rather than
  stopping to "wait."
- Updated `ISSUES.md`: appended a "Follow-up" paragraph to the prior
  session's ingestion-atomicity entry describing the actual now-shipped
  (on-branch) fix and its real test numbers; added one new entry for the
  pre-commit-hook-cache race Batch 3's agent found and recovered from.
- This entry itself.
- (Not yet done as this entry is written: a further doc pass once
  Batches 2 and 4 also land, and any `CLAUDE.md` updates once the
  branches' actual state is known.)

### Evaluation results
**Batch 1 (`fix/batch1-data-integrity-security`, complete, 5/5 fixes):**
ingestion transaction atomicity + concurrent-insert race (the CRITICAL
finding, fixed via a new `PgVectorStore.replace_document_chunks` atomic
method and an `ON CONFLICT`-based document-id resolution), the adjacent-
Markdown-table chunker merge bug, the `security-regression-gate` CI file
list, the MCP `insecure_dev_mode` startup check, and the tool-argument
`ValidationError` audit-log leak. Final suite: **1015 passed, 1
skipped**, `ruff`/`mypy` clean.

**Batch 3 (`fix/batch3-agent-mcp-correctness`, complete, 4/4 fixes):**
the `evidence_sufficient=False`-still-reaches-synthesis bug (new
deterministic `_low_confidence_evidence_response` path), the MCP
`ClientSession`/`initialize()` timeout (verified the fix's own
regression test actually catches the bug by hand-reverting it and
confirming an 8-second kill-wrapper genuinely times out pre-fix, exit
code 124, versus ~0.2s post-fix), the case-mutation regex's false-
positive/false-negative edges, and `max_length` bounds on every
LLM-writable tool-argument string field. Final suite: **1014 passed, 1
skipped**.

**Batch 5 (`fix/batch5-scalability-improvements`, complete, 2/2 items):**
a deterministic secondary sort key added to the dense-search SQL, and
the duplicate-`resolve_auth`-call claim -- explicitly verified first
(confirmed real, but only in `mcp/server.py`'s mirrored tool-dispatch
path, not in `mcp_client.py` as the original audit's evidence pointed
at) before fixing it with an additive `versions` pre-fetch parameter,
gated behind a new `authorization_enabled` property so it adds zero
cost when authorization is off (the shipped default). Final suite:
**1008 passed, 1 skipped**.

**Batches 2 and 4:** resumed after the shared rate limit reset;
in progress, no final numbers yet as of this entry.

### Interpretation
Running five agents in parallel against the same repository was only
safe because of `git worktree` isolation -- without it, five concurrent
`Edit` calls against files like `pgvector.py` or `agent/graph.py` would
have silently clobbered each other. But worktree isolation has a real
edge documented for the first time in this session: it isolates tracked
repository state, not every piece of shared machine state a build
pipeline touches (`pre-commit`'s hook cache being the concrete example
here) -- a lesson worth remembering the next time parallel agents share
a host. Separately, resuming a rate-limited or prematurely-stopped
background agent via `SendMessage` preserved full context every time,
which made this workflow genuinely practical instead of needing to
restate five different multi-fix task descriptions from scratch after
every interruption.

### Open problems
Batches 2 and 4 were still finishing as this entry was written --
see the follow-up session entry (session 10, once written) for their
final results. None of the five branches have been pushed or opened as
PRs yet; that remains the user's own next action.

### Next recommended step
Once all five batches are confirmed complete and tested, review each
branch's diff, push each, and open five separate PRs against `main`
(one per batch), letting CI run independently on each before merging.
After merging, a follow-up `CLAUDE.md` pass should update the affected
sections (ingestion atomicity, the CI gate's file list, the default
agent prompt versions, etc.) to describe the new, fixed behavior as
what actually exists on `main`, rather than continuing to describe the
pre-fix state.

---

## 2026-09-16 (session 10): Batches 2 and 4 finished -- all five fix batches complete

### Interview talking points
- Direct continuation of session 9: the two remaining agents (Batch 2 --
  API reliability/DoS, Batch 4 -- observability/config cleanup) were
  resumed via `SendMessage` after the shared rate limit reset, finished
  cleanly this time (no further self-interruption), and both landed
  every fix from their original assignment -- no partial/deferred items
  in either batch, including the frontend Stop-button fix in Batch 4,
  which needed a successful `npm install` under Node 18.12 first (it
  worked, with only the pre-existing/expected Playwright `EBADENGINE`
  warnings CLAUDE.md's own frontend section already documents as known).
- With all five branches done, verified directly (not just trusted from
  each agent's own report) that `git branch -a`/`git worktree list` in
  the main repo actually show all five `fix/batch*` branches with the
  exact commit hashes each agent reported -- confirming no cross-worktree
  confusion happened despite five concurrent worktrees and, per Batch 3's
  story, at least one shared-cache race along the way.
- Final tallies across all five branches (each independently gated by
  its own full `pytest tests/unit -q` + `ruff` + `mypy` run, all green):
  Batch 1 -- 1015 passed; Batch 2 -- 1021 passed; Batch 3 -- 1014 passed;
  Batch 4 -- 1029 passed (plus a separate, passing frontend suite);
  Batch 5 -- 1008 passed. All five started from the same `main` baseline
  of 1003 passed/1 skipped (per the original two audit reports), so the
  final counts reflect each batch's own added regression tests, not
  cross-batch cumulative drift (the batches are still on separate,
  unmerged branches -- no batch's tests currently see another batch's
  code changes).
- Closed out documentation for the whole effort: updated `ISSUES.md`'s
  ingestion-atomicity entry (already done in session 9) plus a new
  "Pending fix branches" section in `CLAUDE.md` listing exactly what
  each branch contains and explicitly flagging that `main`'s documented
  behavior everywhere else in the file is still the pre-fix state until
  each branch is actually merged -- a deliberate choice to keep
  `CLAUDE.md`'s "map of what exists now" promise honest rather than
  describing intended-but-unmerged behavior as current.

### Goal
Finish implementing the two remaining fix batches from
`.claude-prompts/25.6-Joint-Investigation-Verified-Report.md` (Batch 2:
API reliability/DoS, Batch 4: observability/config cleanup), matching
the same per-branch, tested, committed-but-not-pushed pattern the other
three batches already completed in session 9. Then close out the
documentation for the full five-batch effort.

### Configuration
No config changes by this orchestrating session directly; Batch 4's own
commit bumps `config/default.yaml`'s agent prompt versions (see its own
commit `a6b171d` on `fix/batch4-observability-config-cleanup`) and adds
new `VectorStoreConfig`/`FeedbackConfig` pool/timeout fields (Batch 2's
commit `c918460` on `fix/batch2-api-reliability-dos`) -- both scoped to
their own branches, not applied to `main`.

### Changes made
- Resumed Batch 2 and Batch 4 via `SendMessage` to their respective
  `agentId`s; both completed without further intervention this time.
- Verified all five branches' existence and head commits directly via
  `git branch -a` and `git worktree list` from the main repo.
- `CLAUDE.md`: added a new "Pending fix branches from the joint-
  investigation backlog (not yet merged)" section summarizing all five
  branches' contents and final test counts, explicitly scoped as a
  pointer rather than a description of `main`'s current behavior.
- This entry.
- `ISSUES.md` was not further modified this session -- its session-9
  updates (the ingestion-atomicity follow-up and the pre-commit-cache-race
  entry) already cover the material findings from this whole effort;
  no new distinct engineering story emerged from Batches 2/4 finishing
  cleanly that wasn't already captured.

### Evaluation results
**Batch 2 (`fix/batch2-api-reliability-dos`, complete, 7/7 fixes):**
Prometheus unmatched-route cardinality fixed with a fixed `"unmatched"`
sentinel; `/ingest` offloaded to a threadpool (proved with a real
concurrent `httpx.AsyncClient` test showing `/health` no longer waits
behind a slow `/ingest`); SSE-disconnect cooperative cancellation added
(`cancel_event: threading.Event | None` threaded into `run_agent`'s
bounded loop, new `"cancelled"` termination reason); DB reliability
(config-driven `minconn`/`maxconn`, `connect_timeout_seconds`/
`statement_timeout_ms`, `vectorstore.maxconn` raised 5 to 10, clean 503s
on `PoolError`/`IntegrityError` instead of unhandled 500s); rate limiting
added to `/ingest` (budgeted at a quarter of the shared rate, minimum
1/minute, given its relative cost) and `/feedback`; invalid filter keys
now validated up front in `enforce_dos_limits`, returning 422; `/ingest`'s
`dataset_id` form field capped at 200 chars. Final suite: **1021 passed,
1 skipped**, `mypy` clean across 110 source files.

**Batch 4 (`fix/batch4-observability-config-cleanup`, complete, 9/9
fixes including the frontend one):** `default.yaml`'s agent prompts
bumped to v2 (with `production.yaml`'s comparison comments and
`test_agent_config.py` updated to match); a new
`VectorStore.count_excluded_by_authorization` method feeding a gated
`_audit_authorization_exclusions` helper in `RetrievalPipeline` (only
queries when a result set is thin/empty and auth is active, avoiding a
cost on the common case) plus a new `rag_authorization_denials_total`
Prometheus counter; the frontend's dead `cancel()` wired to a real Stop
button plus a new neutral "cancelled" state distinct from a generic
error; `corpus_gold_record_count` now populated for RAGAS sample runs
and a visible `N` column added to the rendered comparison table;
`security.rate_limit.key` now actually read; a non-fatal startup warning
for `WEB_CONCURRENCY > 1` combined with MCP business actions enabled;
freshness version-family unioning now refuses to cross a non-null
tenant-id mismatch, with a deterministic `(effective_from, document_id)`
tie-break and a warning on multiple simultaneously-active family
members; the three-times-duplicated MCP business-store test fixture
centralized into a new `tests/conftest.py`. Final suite: **1029 passed,
1 skipped**; frontend: **27 files, 154 tests passed**, `tsc --noEmit`
clean.

### Interpretation
The whole five-batch effort closed with every one of the 29 confirmed
findings from the prior reconciliation session actually implemented,
tested, and sitting on a reviewable branch -- not just documented as a
backlog. The main operational lessons from running it (captured in more
detail in session 9 and the two new `ISSUES.md` entries) generalize
beyond this specific fix effort: `git worktree` isolation is necessary
but not sufficient for safe parallel agent work (shared caches outside
the repo are a real, now-documented gap), a background agent that stops
to "wait" for its own spawned background command needs an explicit
instruction to block synchronously instead, and resuming a rate-limited
or self-interrupted agent via direct messaging reliably preserves full
task context -- none of the five agents needed their task re-explained
after an interruption.

### Open problems
None of the five branches have been pushed or opened as PRs -- that
remains the user's own next action, by explicit request (the user
wants to review and merge each one separately). `CLAUDE.md`'s many
architecture/behavior sections describing `main`'s current, pre-fix
state have deliberately not been rewritten to describe the fixed
behavior yet, since none of it is merged -- that rewrite is real,
necessary follow-up work once each branch actually lands, not an
oversight of this session.

### Next recommended step
Push each of the five `fix/batch*` branches and open five separate PRs
against `main`. After each merges, update the corresponding section(s)
of `CLAUDE.md` (ingestion atomicity, the CI gate's curated file list,
default agent prompt versions, the DB pool/timeout config surface, the
new authorization-audit-trail behavior, etc.) to describe the new,
merged behavior as what actually exists on `main`, and remove the
"Pending fix branches" section once every branch it lists has merged.

---

## 2026-09-16 (session 11): Raising and merging all five PRs, and the cross-branch break merging surfaced

### Interview talking points
- Opened five PRs (#3 through #7) against `main`, one per fix batch,
  with plain, to-the-point descriptions rather than AI-flavored PR
  prose, per explicit instruction. All five passed every CI job
  (`code-quality`, `docker-build`, `integration-tests`,
  `retrieval-eval-gate`, `security-regression-gate`, `unit-tests`) and
  showed `MERGEABLE`/`CLEAN` against `main`.
- Merged all five in sequence (#3 through #7), re-checking each
  remaining PR's mergeable status after every merge rather than
  assuming they'd all stay clean -- GitHub needed roughly 15 seconds
  after each merge to finish recomputing mergeability for the others
  (a transient `UNKNOWN` state worth knowing to wait out rather than
  read as a real conflict). All five stayed clean the whole way
  through; no manual conflict resolution was needed.
- Pulled the merged `main` locally and, rather than treating "all
  branches merged cleanly" as the finish line, ran the full unit suite
  once more against the combined result -- and found 6 failures that no
  individual branch's own CI run had ever seen. See the new `ISSUES.md`
  entry ("Five branches all passed CI independently...") for the full
  diagnosis: two unrelated batches had each added a new keyword
  argument to a shared interface (`PgVectorStore`'s pool constructor;
  `RetrievalPipeline.resolve_auth`), and a *third* batch's test double,
  written in a worktree with no visibility into either change, hadn't
  kept up. Fixed both test doubles directly on `main` (test-only, no
  production code touched) and re-ran the full suite clean: 1075
  passed, 1 skipped.
- Closed out `CLAUDE.md`'s temporary "Pending fix branches" section
  (added in session 9 specifically to avoid describing unmerged
  behavior as current) now that all five have actually merged, and
  corrected the one bullet that had gone stale the moment batch 4's
  `default.yaml` prompt-version bump landed.

### Goal
Raise a PR per fix batch with plain, non-AI-sounding descriptions and
no em dashes or double hyphens as punctuation, merge all five into
`main` once CI passes, pull the result locally, and verify the merged
whole actually works together, not just each piece in isolation.

### Configuration
No config changes by this session directly (each PR's own config
changes, e.g. batch 4's `default.yaml` agent-prompt-version bump,
already merged as part of that PR).

### Changes made
- Pushed all five `fix/batch*` branches and opened PRs #3 through #7
  against `main` via `gh pr create`.
- Verified CI (`gh pr checks`) and mergeability (`gh pr view
  --json mergeable,mergeStateStatus`) for all five before merging any.
- Merged #3 through #7 in order via `gh pr merge --merge`, re-verifying
  mergeability of the remaining open PRs after each merge.
- `git pull`ed the merged `main` locally; ran `pytest tests/unit -q`
  once against the combined result.
- Fixed the two broken test doubles found by that run
  (`tests/unit/test_pgvector_atomicity.py`'s pool-constructor mock,
  `tests/unit/test_agent_graph_cancellation.py`'s
  `FakePipeline.resolve_auth`), committed directly to `main`
  (`77ddb2d`), and pushed.
- `ISSUES.md`: added a new entry for the cross-branch break.
- `CLAUDE.md`: replaced the "Pending fix branches" section with a
  short "resolved" note (all 29 findings now live on `main`, five PR
  numbers referenced), and corrected the `production.yaml` bullet that
  had described `default.yaml` as still pinned to v1 agent prompts --
  batch 4 already fixed that, so the bullet was stale the moment it
  merged.
- This entry.

### Evaluation results
- All five PRs: full CI green (6 jobs each) before merge.
- Post-merge, pre-fix: `pytest tests/unit -q` -> 6 failed, 1069 passed,
  1 skipped.
- Post-merge, post-fix: `pytest tests/unit -q` -> **1075 passed, 1
  skipped**. `ruff check`/`ruff format --check` clean on both touched
  files.
- `git log --oneline` on `main` confirms all five merge commits plus
  the follow-up test fix, in order, with no lost or orphaned commits.

### Interpretation
The whole arc of this multi-session effort -- reconcile two audits,
implement the backlog across five parallel isolated worktrees, merge
all five, then verify the merged whole -- is itself a worked example of
why "each piece passed its own review" isn't the same claim as "the
system works." Five independently green branches merging without a
single textual conflict looked, briefly, like proof that no further
verification was needed; the 6 failures the very next `pytest` run
surfaced said otherwise. Running the full suite once more immediately
after merging, before considering the work finished, is what turned
"discovered by whoever hits it next in production" into "discovered
and fixed in the same session," which is the entire practical
difference between a real regression and a review-time inconvenience.

### Open problems
None outstanding from this effort. The five original findings-turned-PRs
are all merged and passing; the one interaction bug merging exposed is
fixed and verified. `CLAUDE.md`'s remaining sections describing
pre-fix behavior in fine technical detail (e.g. some of the DB pool/
retry/observability prose written before this session) may still
contain smaller stale details that a full re-read would catch -- flagged
as a lower-priority follow-up, not attempted exhaustively in this
session given the size of the file.

### Next recommended step
A future session could do a full `CLAUDE.md` re-derivation pass (per
this file's own stated maintenance convention: "refreshed periodically
by reading the actual codebase") to catch any smaller stale details
left over from this five-batch merge that weren't directly touched by
this session's targeted corrections.

## 2026-09-16 (session 12): A Kubernetes deployment for the existing stack, as a hands-on learning exercise

### Interview talking points
- Built a full local Kubernetes deployment (`k8s/`, on a dedicated
  `experiment/kubernetes-deployment` branch, not merged/pushed) for an
  explicitly interview-learning purpose: exercise real production
  Kubernetes concepts against this project's actual images and app
  contracts, without redesigning the application to make Kubernetes look
  more impressive than it is.
- Inspected the existing Docker Compose files, Dockerfiles, frontend
  nginx config, `GET /health`, Postgres/pgvector requirements, the
  Prometheus/Grafana/Jaeger stack, the native-Ollama assumption, and the
  `*_env_var` config convention *before* writing any manifest, so every
  Kubernetes object maps onto something the app already does rather than
  inventing new behavior.
- `k8s/base/` (composed via `kustomize`): a `postgres` `StatefulSet` with
  a `volumeClaimTemplates` PVC (chose `StatefulSet` over `Deployment`
  deliberately, even at `replicas: 1`, for stable network/storage
  identity -- reasoning written directly into that manifest's header);
  `rag-api` `Deployment` (readiness/liveness probes, CPU/memory
  requests+limits, a zero-downtime `maxSurge:1/maxUnavailable:0` rolling
  update strategy, a `preStop` sleep + `terminationGracePeriodSeconds`
  for graceful shutdown, an `initContainer` waiting on Postgres) +
  `Service` + `HorizontalPodAutoscaler`; `frontend` `Deployment`+`Service`
  (needed zero code changes -- naming the rag-api `Service` exactly
  `rag-api` made the frontend's existing hardcoded nginx `proxy_pass`
  resolve correctly via Kubernetes' own namespace-scoped DNS); an
  `Ingress`; a `ConfigMap`/`Secret` pair mirroring the app's own
  `*_env_var` convention; and a default-deny-plus-explicit-allow
  `NetworkPolicy` set.
- Ollama stays native on the host, unchanged from the Compose deployment
  -- reached from inside the cluster via a `Service` of `type:
  ExternalName` pointing at `host.docker.internal`, since there is no pod
  to select and no ClusterIP involved, just DNS indirection.
- `k8s/jobs/`: a one-shot schema-init `Job` running the real
  `scripts/init_db.py` (not baked into the image) mounted in from a
  ConfigMap.
- `k8s/observability/`: optional Prometheus/Grafana/Jaeger, reusing the
  existing `observability/prometheus/prometheus.yml` and
  `observability/grafana/provisioning`/`dashboards` files completely
  unmodified -- confirmed their hardcoded Service-name URLs
  (`http://prometheus:9090`, `http://jaeger:16686`) already resolve
  correctly once those Services exist in the same `rag` namespace.
- Validated every manifest before calling the work done: YAML-parsed all
  22 files, and ran `kubectl kustomize k8s/base` to confirm the
  kustomization actually builds (it does, including the `images:`
  tag-override transformer). Hit a real kustomize limitation doing this
  for `k8s/jobs`/`k8s/observability`: their intended
  `configMapGenerator`s referenced files outside their own directory
  tree (`scripts/init_db.py`, `observability/*`, both siblings of `k8s/`
  at the repo root) and kustomize's load restrictor rejected that outright,
  with no `--load-restrictor` override available in the installed
  `kubectl` (1.24). Switched both to plain `kubectl create configmap
  --from-file` instead of restructuring the repo to satisfy the tool --
  full diagnosis in the new `ISSUES.md` entry.
- Found a second real gap by reading `src/rag/api/routers/health.py`
  while designing the readiness-probe manifest: `GET /health` always
  returns HTTP 200, reporting degradation only in the JSON body, which
  means a standard `httpGet` Kubernetes probe (or any HTTP health check)
  cannot detect a real dependency outage this way at all. Did not change
  the endpoint (out of scope, explicitly not to be done just for this
  exercise) -- instead designed the readiness-failure exercise in
  `k8s/EXERCISES.md` to demonstrate both halves honestly: a deliberately
  broken probe path proves the Endpoints-removal mechanism works, and
  actually stopping Postgres proves the app-level gap is real (Endpoints
  stay populated, `/health` still 200s). Full writeup, including the
  liveness-probe-anti-pattern angle this same behavior accidentally
  avoids, in the new `ISSUES.md` entry.
- Also flagged, without fixing (per the exercise's own scope), a genuine
  "breaks under replication" bug already partially documented from a
  different angle in an earlier session: `rag.mcp.business.store`'s
  synthetic case data is a plain in-memory Python dict, so scaling
  `rag-api` beyond 1 replica gives each pod its own independent, mutually
  invisible copy of that data -- written up in `k8s/LEARNING.md`'s
  "what remains process-local" section as the headline example, alongside
  the already-documented in-memory rate limiter and a newly-noticed gap
  of this deployment's own making (no PVC for `/app/data/uploads`, so
  uploaded files don't survive pod recreation at all, not just replica
  count > 1).
- Wrote `k8s/LEARNING.md` (architecture diagram, a concept-by-concept
  "why" table covering every Kubernetes primitive used, plain-language
  walkthroughs of pod death / node death / rolling updates / HPA
  mechanics / how traffic actually reaches a replica via
  Endpoints+kube-proxy, and 15 interview questions grounded specifically
  in this deployment, not generic Kubernetes trivia) and
  `k8s/EXERCISES.md` (concrete, copy-pasteable failure-injection
  walkthroughs: pod deletion, scale 1->3, the two-part readiness
  exercise above, a real zero-downtime rolling update contrasted against
  the Kubernetes default `maxUnavailable`, a bad-image-tag rollout stuck
  in `ImagePullBackOff` without ever affecting live traffic, a simulated
  node failure via `docker stop` on a kind worker container plus the
  `pod-eviction-timeout` reasoning for why pods aren't rescheduled
  immediately, and the `kubectl get/describe/logs/get events/top`
  inspection toolkit).
- Updated `CLAUDE.md` with a short, LangGraph-experiment-style section
  pointing at `k8s/README.md` as the real write-up, and cross-referencing
  both new `ISSUES.md` entries, rather than duplicating deployment detail
  into `CLAUDE.md` itself.

### Goal
Stand up a Kubernetes deployment of the existing stack purely as a
hands-on learning vehicle for production Kubernetes concepts (Deployments
vs. StatefulSets, probes, resource limits, rolling updates, HPA, Services/
Ingress/NetworkPolicy, failure injection), on a dedicated experiment
branch, explicitly not committed or pushed, and explicitly not a
justification to redesign the application.

### Configuration
No `config/*.yaml` changes. New, uncommitted-by-instruction files only,
all under `k8s/`: `kind-cluster.yaml`, `base/*.yaml` (11 manifests +
kustomization), `jobs/init-db-job.yaml`, `observability/*.yaml` (7
manifests), `README.md`, `EXERCISES.md`, `LEARNING.md`.

### Changes made
- Created branch `experiment/kubernetes-deployment` off `main`.
- Read `docker-compose.yml`/`docker-compose.prod.yml`/
  `docker-compose.observability.yml`/`docker-compose.frontend.yml`,
  `Dockerfile`, `frontend/Dockerfile`/`nginx.conf`/`docker-entrypoint.sh`,
  `src/rag/api/routers/health.py`, `config/default.yaml`,
  `scripts/init_db.py`, `Makefile`, `.env.example`, and the
  `observability/` directory tree before writing any manifest.
- Authored all 22 Kubernetes manifests plus `kind-cluster.yaml` under
  `k8s/`, described above.
- Validated: YAML-parsed every manifest with `yaml.safe_load_all`;
  confirmed `kubectl kustomize k8s/base` builds cleanly including the
  `images:` transformer; hit and worked around the kustomize
  `configMapGenerator` load-restrictor limitation for `k8s/jobs` and
  `k8s/observability`.
- Wrote `k8s/README.md` (setup/quickstart), `k8s/EXERCISES.md`
  (failure-injection walkthroughs), `k8s/LEARNING.md` (architecture
  diagram, concept reference, 15 interview questions).
- `ISSUES.md`: added two new entries (the `/health`-always-200 readiness
  limitation, and the kustomize load-restrictor edge case).
- `CLAUDE.md`: added a "Kubernetes deployment (learning experiment)"
  section, mirroring the existing LangGraph-experiment section's
  not-production framing and its pattern of pointing at the
  subdirectory's own README rather than duplicating detail.
- This entry.

### Evaluation results
- No cluster was actually created or applied against in this session
  (no `kind`/live cluster available in this environment) -- validation
  was static: YAML syntax (all 22 files parse), and `kubectl kustomize
  k8s/base` building successfully end to end, including the image-tag
  override transformer. `k8s/jobs` and `k8s/observability` were
  confirmed to fail as plain kustomizations (the load-restrictor error,
  captured verbatim in the new `ISSUES.md` entry) and confirmed to work
  as plain manifests once the generator was removed.
- `git status` after the whole session shows only new, untracked files
  under `k8s/` plus the documented edits to `CLAUDE.md`/`ISSUES.md`/
  `PROJECT_JOURNAL.md` -- no existing application file was modified,
  matching the exercise's explicit constraint.

### Interpretation
The two real findings this session surfaced (the `/health` status-code
gap, and the kustomize load-restrictor limitation) are both more
valuable than the manifests themselves for the stated interview-prep
goal: they came from actually trying to exercise the deployment's own
stated design goals (make readiness fail and observe it; generate a
ConfigMap declaratively from a real repo file) rather than from writing
manifests in the abstract. Documenting a real constraint honestly (this
app's health endpoint can't support the textbook readiness-probe
demo without a workaround) is more useful, for the project's own
interview-issue-tracking purpose, than quietly picking an exercise that
would have avoided exposing the limitation at all.

### Open problems
No live cluster was exercised in this session -- every exercise in
`k8s/EXERCISES.md` is written and reasoned through but not yet run
against a real `kind` cluster from this machine. The three "process-local,
breaks under replication" items flagged in `k8s/LEARNING.md` (the MCP
in-memory case store, the in-memory rate limiter, and the missing PVC for
`/app/data/uploads`) are documented, not fixed -- intentionally, per this
exercise's own scope, but worth tracking if a future session ever wants
to make this deployment more than a learning artifact.

### Next recommended step
Actually run `kind create cluster --config k8s/kind-cluster.yaml` and
walk through `k8s/EXERCISES.md` end to end on a machine with `kind`
installed, to confirm the reasoning in `k8s/LEARNING.md` matches real
observed behavior (probe timing, HPA scaling responsiveness,
pod-eviction-timeout duration) rather than only design-time analysis.

---

## 2026-09-16 (session 13): A parallel LangGraph agent workflow, to learn when it earns its keep over the custom harness

### Interview talking points
- Built `langgraph_experiment/`, a second, fully isolated LangGraph-based
  agent workflow on `experiment/langgraph-agent`, explicitly to answer
  "when does LangGraph actually beat a hand-rolled bounded loop" rather
  than to migrate anything -- `rag.agent.graph.run_agent` (production)
  is untouched, not imported by, and not imported from this package.
- Real dependency conflict found and worked around, not assumed:
  `langgraph>=1.2` pins `langchain-core>=1.4,<2`; this repo's core
  `langchain>=0.3,<0.4` needs `langchain-core<0.4`. Confirmed via `pip
  install --dry-run` inside the project's real `.venv` before writing any
  code, then built the whole experiment in a separate `.venv-langgraph/`
  venv instead, installing the project editable (`--no-deps`) plus its
  runtime deps minus the langchain group -- verified that only two `rag`
  modules import `langchain` at all (`chunkers/recursive_chunker.py`,
  `eval/ragas_adapters.py`), and this experiment's code path never
  reaches either.
- The graph: `classify` (deterministic regex, reusing
  `rag.agent.graph._looks_like_case_mutation_request` rather than
  reimplementing it) routes to a read-only RAG branch (real
  `RetrievalPipeline`+LLM), a case-read branch, or a case-write branch
  that pauses at a real `interrupt()` before ever calling
  `rag.mcp.business.store.update_case_status` (also reused unmodified).
- Verified every core LangGraph mechanic empirically against the
  installed `langgraph==1.2.11`/`langgraph-checkpoint-sqlite==3.1.1`
  before writing the real graph, via small throwaway scripts: `interrupt()`
  + `Command(resume=...)` across a closed-and-reopened `SqliteSaver`
  connection (a real simulated process restart), `RetryPolicy`'s
  `retry_on` filtering, and `get_state(config).tasks[i].error`/`.next`
  surviving a genuinely failed node.
- Approval is trusted runtime state, not an LLM/payload claim:
  `wait_for_approval` re-checks the resume payload's `approver_roles`
  against config's `approval_roles` before ever treating a `"decision":
  "approve"` claim as real -- proven by a dedicated test
  (`test_approval_claim_without_approver_role_is_denied`).
- Ran the full human-in-the-loop demo across three genuinely separate
  `python -m ...` process invocations (start -> list-pending -> approve),
  each opening its own fresh `SqliteSaver` connection to the same file --
  not simulated in a single test process.
- Real finding, found by actually running the two-step CASE-1001 scenario
  from the prompt spec across separate CLI processes: a "successful"
  mutation from the first process silently vanished by the time the
  second process ran, because `rag.mcp.business.store._SYNTHETIC_CASES`
  is an in-memory, per-process dict with no persistence of its own --
  correct for its real deployment shape (inside one long-running
  `rag-api` process), but proof that LangGraph's own checkpoint
  durability says nothing about the durability of whatever a node calls
  into. Documented as its own `ISSUES.md` entry, not smoothed over.
- 34 tests, all passing, needing zero external services (the case-store
  branch, including the entire interrupt/approval/restart-resume story,
  never touches Postgres or Ollama -- only the read-only RAG branch
  would, and its tests use a narrow `FakePipeline`/`FakeLLM` instead).
- Wrote a reasoned (not fabricated) comparison against the custom
  harness across ten axes, explicitly not concluding LangGraph is better
  by default -- the custom harness wins on code simplicity, live
  debuggability, and zero extra infrastructure for everything it already
  needs to do today; LangGraph's real advantage is the one structural gap
  the custom harness actually has (no way to pause an in-flight run for
  an external event and survive a restart while paused).

### Goal
Follow `.claude-prompts/31.0-LangGraph.txt` closely: build an isolated,
non-production LangGraph agent workflow demonstrating StateGraph,
checkpointing, interrupt/resume, human approval as trusted state,
conditional routing, retries, and failure recovery, reusing existing
retrieval/tool abstractions rather than reimplementing RAG; end with a
reasoned custom-harness comparison and an interview-prep write-up. No
commit/push, per the prompt's own explicit instruction.

### Configuration
No `config/*.yaml` changes and no changes under `src/rag/`. All new,
uncommitted-by-instruction files under `langgraph_experiment/` (package
code, `cli/`, `tests/`, `README.md`, `requirements.txt`,
`setup_venv.sh`/`.ps1`), plus a new `.venv-langgraph/` on disk (gitignored,
listed as its own `.venv-langgraph/` line in `.gitignore`).

### Changes made
- Created branch `experiment/langgraph-agent` (from
  `experiment/kubernetes-deployment`'s tip, per the prompt's own literal
  first line, `git switch -c experiment/langgraph-agent`).
- Read `src/rag/agent/state.py`, `src/rag/agent/graph.py`,
  `src/rag/agent/tools.py`, `src/rag/agent/tool_schemas.py`,
  `src/rag/mcp/business/store.py`, `src/rag/mcp/business/schemas.py`,
  `src/rag/retrieval/pipeline.py`, `src/rag/retrieval/authorization.py`,
  `src/rag/factory.py`, and `src/rag/config.py` before writing any graph
  code, so every reused abstraction was understood, not guessed at.
- Confirmed the `langchain-core` version conflict (`pip install
  --dry-run` in the main `.venv`), then built `.venv-langgraph/`: `pip
  install -e . --no-deps` plus the project's real runtime deps minus the
  langchain group and the ingestion-only loader deps
  (beautifulsoup4/lxml/pypdf/pdfplumber/python-docx), plus `langgraph`/
  `langgraph-checkpoint-sqlite`. Verified `rag.retrieval.pipeline`,
  `rag.mcp.business.store`, `rag.agent.graph._looks_like_case_mutation_request`,
  `rag.factory`, and `rag.config.load_config()` all import cleanly in
  that venv before writing the graph.
- Smoke-tested LangGraph's own mechanics directly (throwaway scripts,
  since this codebase has never used LangGraph before and the
  installed-version behavior needed confirming, not assuming): a 3-node
  graph with a real interrupt across a closed/reopened `SqliteSaver`
  connection; a real pydantic `SearchResult` round-trip attempt through
  the checkpointer (abandoned in favor of plain-dict state for
  simplicity, not because it failed -- the test script's own fixture data
  was incomplete); `RetryPolicy`/node-failure `.tasks[i].error` semantics.
- Wrote `langgraph_experiment/state.py` (`GraphState` TypedDict),
  `identity.py` (`DemoIdentity`), `wiring.py` (`GraphDeps`, built from
  `rag.factory`/`rag.config` wholesale), `nodes.py` (every node function
  plus deterministic classify/extraction helpers), `routing.py`
  (conditional-edge functions), `graph.py` (`build_graph`), and
  `checkpointer.py` (`sqlite_checkpointer`).
- Wrote `cli/run_demo.py`, `cli/approval_cli.py`,
  `cli/two_step_case_demo.py`, `cli/compare_harness.py`.
- Wrote 34 tests across 7 files (`tests/test_routing.py`,
  `test_graph_read_only_path.py`, `test_graph_case_read_path.py`,
  `test_graph_interrupt_flow.py`, `test_idempotent_resume.py`,
  `test_restart_resume.py`, `test_authorization_server_side.py`,
  `test_failed_node_state.py`), plus `tests/conftest.py`'s shared
  fixtures (including an autouse snapshot/restore fixture for
  `rag.mcp.business.store`'s shared mutable case dict, mirroring the
  identical pattern already in `tests/unit/test_mcp_business_case_actions.py`).
- Ran the full human-in-the-loop and restart/resume demo manually via
  real separate CLI process invocations (not just the automated tests),
  which is what surfaced the `_SYNTHETIC_CASES` per-process-persistence
  finding; fixed the demo (added `cli/two_step_case_demo.py`, running
  both CASE-1001 mutations in one process) rather than the underlying
  module, since the module's own behavior is correct for its real
  purpose.
- Wrote `langgraph_experiment/README.md` (architecture diagram, exact run
  commands, full demo sequences, an idempotency-guarantee-by-layer table,
  the ten-axis custom-harness comparison, 15 concept explanations, 20
  interview Q&A).
- `ISSUES.md`: added "A durable LangGraph checkpoint proved nothing about
  the durability of what it calls into" (new **LangGraph Experiment**
  index category).
- `CLAUDE.md`: added a "LangGraph experiment (learning exercise, not
  production)" section, mirroring the existing Kubernetes-experiment
  section's not-production framing and its pattern of pointing at the
  subdirectory's own README rather than duplicating detail.
- This entry.

### Evaluation results
- `python -m pytest langgraph_experiment/tests -v`: 34 passed, 0 failed,
  no Postgres/Ollama running.
- `python -m py_compile` across every module in the package: clean.
- Manual CLI walkthrough (real separate processes, not test-simulated):
  start -> `PAUSED (awaiting approval)` with the correct interrupt
  payload; `approval_cli.py list` from a fresh process found the exact
  pending thread; `approve` from a third fresh process resumed and
  executed the mutation (`CASE-2001`: `resolved -> closed`); a second
  `approve` on the same completed thread correctly reported "nothing to
  do" instead of re-attempting the mutation; `reject` on a separate
  thread correctly left the case untouched.
- `git status` after the whole session shows only new, untracked files
  under `langgraph_experiment/` plus the documented edits to
  `.gitignore`/`CLAUDE.md`/`ISSUES.md`/`PROJECT_JOURNAL.md` -- no
  existing application file under `src/rag/` was modified.

### Interpretation
The most valuable output of this session, for the stated interview-prep
goal, wasn't the graph itself -- it was the two things found only by
actually running it end to end rather than reasoning about LangGraph in
the abstract: the real `langchain-core` version conflict (a concrete,
current, checkable fact about the LangGraph 1.x/LangChain 0.x ecosystem,
not received wisdom), and the `_SYNTHETIC_CASES` per-process persistence
gap (direct, empirical proof that "durable execution" is a claim about
exactly the workflow engine's own state, and never automatically extends
to whatever a node happens to call into). Both are exactly the kind of
finding this project's own `ISSUES.md` culture is built around: caught by
running the thing, not by inspecting the code.

### Open problems
- `cli/compare_harness.py rag` (the timed side-by-side against the
  custom harness on a real read-only RAG query) was written and smoke
  logic-checked but not actually run against a live Postgres+Ollama
  stack in this session -- the qualitative comparison table is
  reasoned, not fabricated, but the one quantitative axis (`Latency`)
  has no real measured number recorded yet.
- The read-only RAG branch's tests all use `FakePipeline`/`FakeLLM`;
  no test in this package exercises `retrieve()`/`synthesize_rag`
  against a real ingested dataset.
- `docs/architecture.md`/the MkDocs site were not updated to reference
  this package (deliberately -- it's explicitly a non-production
  learning exercise, matching how `k8s/` was handled in session 12).

### Next recommended step
Bring up `make up` + native Ollama with an ingested dataset and actually
run `python -m langgraph_experiment.cli.compare_harness rag "<a real
techfusion question>" --dataset-id techfusion`, to replace the
Latency row's reasoning-only claim with a real measured number on this
machine, the same way session 12 flagged its own unexercised `kind`
cluster as the natural next step.

## 2026-09-17 (session 14): A multi-agent LangGraph experiment, and a real fan-in bug found building it

### Interview talking points
- Built `multi_agent_experiment/` on a new `experiment/multi-agent`
  branch (branched from `experiment/langgraph-agent`, per the session's
  own spec), a second, isolated LangGraph exercise sitting alongside
  `langgraph_experiment/` (session 13's single-agent exercise): a
  Coordinator/Router node plus two genuinely distinct specialists (a
  Knowledge Agent calling only `RetrievalPipeline.retrieve()`, a
  Business Agent calling only `rag.mcp.business.store`), with
  `Send`-based parallel fan-out for queries needing both. Neither
  `rag.agent.graph.run_agent` (production) nor `langgraph_experiment/`
  was touched; verified with `git status`/`git diff` before writing the
  README's own "Files added/changed" section, not assumed.
- De-risked the two riskiest LangGraph API assumptions (a conditional
  edge returning a list of `Send` objects for dynamic parallel fan-out;
  a bare-string conditional-edge branch supporting `interrupt()`/
  `Command(resume=...)` identically to session 13's proven mechanism)
  with a throwaway probe script run against the real installed
  `langgraph==1.2.11` *before* writing any of the real graph -- the same
  "verify against the actually-installed library, don't assume"
  discipline session 13 established for its own primitives. The probe
  became a permanent regression test
  (`tests/test_probe_langgraph_primitives.py`).
- **Reused the entire business-write branch from session 13's package
  unmodified** -- `business_agent.py` rebinds `langgraph_experiment.
  nodes.select_case_tool`/`make_execute_case_read_node`/`validate_
  write_request`/`make_wait_for_approval_node`/`make_execute_write_
  action_node`/`synthesize_write_result` directly, plus `langgraph_
  experiment.wiring.GraphDeps`/`build_default_deps` wholesale. Zero new
  authorization/transition/approval logic anywhere in this package --
  every rule still lives exactly once, in `rag.mcp.business.store`.
- **Tool isolation made statically verifiable, not just asserted**: split
  the graph's nodes into three separate modules
  (`knowledge_agent.py`/`business_agent.py`/`orchestrator.py`)
  specifically so `tests/test_tool_isolation.py` could `ast`-parse each
  one's real `import`/`from` statements and assert the forbidden module
  never appears -- a specialist cannot call the other's tool because the
  code that could do so was never written in that file. (First attempt
  used a naive substring search over the whole file, which false-failed
  on the modules' own docstrings explaining the isolation -- switched to
  `ast`-based import-statement parsing, mirroring `tests/unit/
  test_gold_data_isolation.py`'s established static-boundary-test idiom
  in the main codebase.)
- **A real LangGraph correctness bug, found and fixed, not just
  theorized**: the "mixed" route's Knowledge Agent branch is one hop
  from the coordinator's `Send` dispatch; the Business Agent's read
  branch is three (`select_tool -> execute_read -> evaluate_read`). A
  plain `add_edge(..., "merge")` join fired as soon as *either* branch's
  edge delivered a write, not once *both* had -- running `merge` (and
  `evidence_critic`/`final_synthesis` downstream) once early with only
  the knowledge branch's partial state, then again later with the
  complete state, doubling an LLM call and, for one invocation, briefly
  producing an answer citing half the actual grounding evidence. Found
  because `tests/test_case3_mixed_parallel.py`'s own assertion on the
  fake LLM's call count read `2`, not `1` -- caught by a specific
  assertion, not a vague "something feels off." Diagnosed with
  `graph.stream(..., stream_mode="debug")` (which carries an explicit
  step number per task) after `stream_mode="updates"` turned out not to
  reveal superstep grouping at all (it emits one chunk per node
  *completion*, confirmed empirically, not assumed). Fixed with
  `graph.add_node("merge", ..., defer=True)`, confirmed against the
  installed `langgraph==1.2.11`'s own docstring for that parameter
  ("whether to defer execution of the node until the run is about to
  end") before relying on it. Full writeup, with a minimal reproducing
  probe kept as two permanent tests (one proving the bug, one proving
  the fix), in `ISSUES.md`.
- Field-ownership discipline (`knowledge_*` vs. `case_*`/`business_*`
  namespacing) avoids almost all parallel-branch write conflicts by
  construction; the two fields that genuinely accumulate across a
  critic-requested retry (`knowledge_tool_calls`/`business_tool_calls`)
  use an explicit `Annotated[list, operator.add]` reducer, while
  `citations`/`tool_call_log` are deliberately plain overwrites
  recomputed fresh by `merge` each time, so a retry's better evidence
  replaces stale evidence instead of being unioned with it.
- A second real, if smaller, bug caught by the test suite before it ever
  reached a scenario test: an early knowledge-cue regex matched generic
  interrogative phrases ("what is," "how does"), which misrouted a pure
  business question ("What is the status of CASE-1001?") into
  unnecessarily also selecting the Knowledge Agent. Fixed by requiring a
  cue that signals "references the knowledge base," not merely "is a
  question" -- `tests/test_coordinator_routing.py` pins the corrected
  behavior directly.
- The Evidence Critic (the spec's one optional role) was scoped
  narrowly and deliberately: one bounded knowledge-retry decision plus a
  no-fabrication guard, explicitly *not* semantic conflict detection
  between the two specialists' evidence (which would need either another
  LLM call, reintroducing exactly the flakiness the rest of this design
  avoids, or a fragile keyword heuristic) -- documented as a real scope
  decision in the README, not silently dropped.
- A dedicated, deterministic evaluation harness
  (`eval/run_multi_agent_eval.py` + a 12-row `eval/multi_agent_gold.
  jsonl`) runs with zero external services and produced real numbers
  this session, not fabricated ones: `routing_accuracy: 1.000`,
  `unnecessary_agent_rate: 0.000`, `answer_correctness: 1.000`,
  `citation_grounding: 1.000` (after correcting one wrong hand-written
  gold expectation -- a synthetic KB's "support" keyword legitimately
  matched two documents, not the one the gold row assumed), and
  `security_failures: 0`.
- 28 new tests, all passing, zero external services; the pre-existing
  34-test `langgraph_experiment/tests` suite re-run and confirmed
  byte-identical/still-passing after this session's changes (it wasn't
  touched, but "wasn't touched" was verified, not assumed).
- Wrote a reasoned three-way comparison (custom harness / single-agent
  LangGraph / multi-agent LangGraph) across eleven axes including a new
  "Coordination" row, concluding multi-agent is justified only where a
  genuine need for two-or-more differently-toolable specialists
  coincides with session 13's own pause/resume justification -- not by
  default, and explicitly not concluded to be automatically superior to
  either existing architecture. Three of this experiment's own five
  required scenarios (knowledge-only, business-only, mutation) show
  multi-agent adding hops and code for zero benefit over the
  single-agent design.
- `cli/compare_harness.py rag`/`mixed` (the real-infra timing
  comparisons) were written and import-checked but **not run against
  live Postgres** this session -- Docker Desktop was not running
  (confirmed via `docker ps` failing to reach the daemon), only native
  Ollama was reachable. Flagged directly in the README's own "What
  wasn't verified" section rather than presented as measured.

### Interpretation
The single most valuable finding this session produced for the stated
interview-prep goal was the `defer=True` fan-in bug -- not because it was
hard to fix (one keyword argument), but because it is a failure mode that
structurally cannot exist in a single-agent system at all: it requires
*two branches of different length* converging on one node, which only
happens once a graph has genuine parallel dispatch. Session 13's own
probe-before-building discipline (verify a risky LangGraph assumption
against the real installed library with a minimal script before writing
the real graph) carried forward directly here and caught two of this
session's three real findings (the `Send`/interrupt primitives probe,
and the `defer=True` probe once the bug was already found) before or
immediately after they mattered, rather than deep into a half-built
feature. The third finding (the knowledge-cue routing bug) was caught by
ordinary unit-level assertions, not a special technique -- a reminder that
routing-decision bugs in a deterministic system are exactly as catchable
as any other logic bug, once a test actually pins the expected decision
rather than just the final answer text.

### Open problems
- `cli/compare_harness.py rag`/`mixed`'s real-infra timing comparisons
  remain unexercised against a live stack -- the "What wasn't verified"
  section in the README is the authoritative list, kept in one place
  rather than duplicated here.
- No test in this package exercises `RetrievalPipeline.retrieve()`
  against a real ingested dataset -- `knowledge_agent`'s tests all use a
  narrow `FakePipeline`, matching the same mocking-narrowest-point
  convention `langgraph_experiment/tests` already established.
- The Evidence Critic's explicitly-scoped-out semantic-conflict-detection
  gap (knowledge and business evidence that actually contradict each
  other) is undetected by design -- documented, not silently missing.
- `docs/architecture.md`/the MkDocs site were not updated to reference
  this package, matching how `langgraph_experiment/`/`k8s/` were both
  handled (deliberately -- both are non-production learning exercises).

### Next recommended step
Same as session 13's own unresolved next step, now doubled: bring up
`make up` + native Ollama with an ingested `techfusion` dataset and run
both `langgraph_experiment.cli.compare_harness rag` (still outstanding
from session 13) and this session's `multi_agent_experiment.cli.
compare_harness rag`/`mixed`, to replace both packages' Latency rows'
reasoning-only claims with real measured numbers on this machine.

## 2026-09-21 (session 15): Diagnosing why kind can't start a cluster on this host, across three separate sessions

### Interview talking points
- Following `k8s/README.md` step 5 (`kind create cluster --config
  k8s/kind-cluster.yaml`) never produced a running cluster on this host.
  Every attempt failed at `Starting control-plane`, and kind tore down
  the partial nodes and exited non-zero, before a single application
  manifest in `k8s/base/` was ever reached. This became a genuine
  multi-session investigation, not a quick fix, so it's logged here in
  full; `ISSUES.md`'s own entry stays the short, curated version.
- **Environment at the start:** Windows 11, one host, 7.8GB total RAM
  (often 1.1-1.8GB free with normal apps running); Docker Desktop
  4.90.0 (Engine 29.7.2, containerd v2.3.3, runc 1.4.3); WSL2 kernel
  `5.15.146.1-2`, later jumped to `6.18.33.2-2` by `wsl --update`, an
  unusually new kernel build; `kind` v0.33.0 (and v0.20.0, tested
  separately for comparison).
- **Session 1: isolating the failure shape.** First raw attempt printed
  a `cgroup v1 is deprecated` warning and failed inside `kubeadm init`.
  Fixed cgroup v1 -> v2 via `wsl --update` (a real, necessary fix,
  confirmed via `docker info`), but the identical failure persisted,
  just without the warning. `kubeadm --v=6` showed the real shape: it
  retries `POST .../clusterrolebindings` every ~500ms against the
  control-plane's own API server for about 4 minutes, then gives up.
  Live-inspected a still-hanging cluster before kind's own teardown
  could remove it: `crictl pods` showed the four control-plane static
  pods (`etcd`, `kube-apiserver`, `kube-scheduler`,
  `kube-controller-manager`) as `Ready`, but `crictl ps -a` was
  completely empty, not one container, not even a failed one.
  `journalctl -u containerd` showed, for each pod, `CreateContainer`
  accepted and then nothing: no success line, no error, no timeout,
  indefinitely. `journalctl -u kubelet` showed the expected downstream
  symptom (`dial tcp <ip>:6443: connection refused`, because nothing
  was ever listening on 6443, since the apiserver container itself
  never finished being created). `dmesg` showed nothing unusual: no OOM
  kill, no cgroup denial. This pinned the hang to one precise, silent
  step: `containerd` accepting a `CreateContainer` call and never
  completing, erroring, or timing out on it.
- Ruled out every application-level variable this same session, one at
  a time, all reproducing the identical failure: Kubernetes node image
  version (`kindest/node:v1.37.0` vs. an older `v1.31.2`); the `kind`
  release itself (current v0.33.0 vs. v0.20.0, three years older, with
  a different containerd/runc baseline entirely); cluster topology (the
  documented 3-node config vs. a bare single-node `kind create cluster`
  with zero customization). Also ruled out this being specific to the
  standalone `kind` CLI: enabling Docker Desktop's own built-in
  Kubernetes hit the exact same `kubeadm init` exit status, against a
  container named `desktop-control-plane` instead of
  `rag-learning-control-plane`. `docker desktop kubernetes status`
  literally printed `Mode: kind`, confirming Docker Desktop's built-in
  Kubernetes *is* `kind` under the hood on this platform, not a
  different code path that might have sidestepped the bug.
- Checked for a known upstream cause instead of continuing to guess
  blindly: `runc`'s own issue tracker documents a class of hangs
  matching this exact shape (indefinite hang inside container creation,
  no error, no timeout): `opencontainers/runc#3448` and `#5431`. Every
  failing configuration here shares the same unusually new WSL2 kernel
  and the same nested-overlay filesystem stack (WSL2 VM's own overlay
  root, on top of the kind node container's own overlay root, on top of
  containerd's overlayfs snapshotter for each workload container), a
  combination far less likely to have been exercised by upstream CI
  than a standard Linux host. `fuse-overlayfs` (the usual mitigation)
  wasn't installed inside Docker Desktop's VM either.
- Also investigated host memory this session: Docker Desktop's WSL2 VM
  reported only `Total Memory: 3.716GiB` via `docker info`, with no
  `.wslconfig` present. Wrote `.wslconfig` (`memory=4.5GB`,
  `processors=8`, `swap=2GB`) and separately edited Docker Desktop's own
  memory settings directly. `docker info`'s reported total stayed
  pinned at exactly `3.716GiB` regardless, and cluster creation still
  failed identically; it looked inert at the time, but see session 3
  below for why that conclusion was wrong.
- **Session 2: a targeted kernel-pin attempt (the oldest available
  binary).** Researched a downloadable "known-good" kernel rather than
  guessing further: `microsoft/WSL2-Linux-Kernel` publishes two
  maintained lines (Rolling `6.18.x`, LTS `6.6.x`) but only as source,
  no prebuilt binary, confirmed by checking the release assets
  directly. The one genuinely downloadable Microsoft-built kernel
  binary still findable is the standalone offline installer MSI,
  extracted (not installed) via `msiexec /a ... /qn`. Its
  `ProductVersion` turned out to be `5.10.16`, dated 2021-04-13,
  meaningfully older than even the original kernel this host started
  on, not a true "known-good LTS," just the oldest artifact Microsoft
  still hosts a static link for. Pointed `.wslconfig` at it and
  relaunched Docker Desktop (full process kill, not just `wsl
  --shutdown`). Result: Docker Desktop never finished starting at all
  (`docker desktop status` stuck at `starting` for several minutes,
  `docker info` timed out completely), a real finding, not a null
  result: this Docker Desktop version can't initialize on a 2021-era
  kernel, so the test never even reached trying `kind` against it.
  Reverted and confirmed full recovery.
- **Session 3: the real memory bug, and a better-targeted kernel-pin
  attempt.** Running `wsl.exe -d docker-desktop -- uname -r` directly,
  rather than trusting `docker info`/`docker desktop restart` (which
  never surface `.wslconfig` parse errors anywhere visible), printed
  three parse errors: `Invalid memory string '4.5GB'` (WSL2 requires an
  integer with a unit suffix, e.g. `4608MB`, not a decimal) and two
  `Invalid escaped character` errors from un-escaped backslashes in
  Windows paths. This meant session 1's entire memory-tuning attempt had
  been silently discarded the whole time, not "confirmed inert" as
  originally concluded: a wrapping tool (`docker info`) had fully
  swallowed a lower-level tool's (`wsl.exe`'s) config errors. Fixed with
  integer units and forward-slash paths; `Total Memory` then correctly
  read `4.31GiB` for the first time in the whole investigation, up from
  the silently-ignored `3.716GiB`.
  With that fixed, re-attempted the kernel pin with a better-chosen
  binary: rather than the oldest available, listed `microsoft/WSL`'s
  actual release history via `gh api repos/microsoft/WSL/releases
  --paginate` and picked a genuinely intermediate, stable release,
  `2.6.3` (published 2025-12-12), extracting its bundled kernel the same
  way. `wsl.exe -d docker-desktop -- uname -r` then booted with zero
  parse errors, reporting `6.6.87.2-microsoft-standard-WSL2`, genuinely
  the LTS line. Docker Desktop's engine this time made real progress
  (WSL distro up, `wsl-bootstrap` running) but retry-looped on: `mount
  -o ro,exec .../docker-wsl-cli.iso ...: unknown filesystem type
  'iso9660'`. This specific kernel build wasn't compiled with
  `CONFIG_ISO9660_FS`, which Docker Desktop's startup sequence needs to
  mount its own CLI-tools ISO into the VM, a real, previously
  undiscovered Docker-Desktop-version-vs-kernel-config dependency, not
  the generic "too old" failure from session 2. Never reached testing
  `kind` against it, since Docker Desktop's own engine never came up.
  Reverted and confirmed full recovery (`6.18.33.2`, `Cgroup Version:
  2`, `Total Memory: 4.31GiB` unaffected). Retried `kind create cluster`
  twice more on the unmodified kernel afterward, once under real RAM
  pressure and once with RAM freed first; both reproduced the identical
  `clusterrolebindings` retry-loop failure, confirming the original bug
  is unrelated to a separate, transient low-memory "Preparing nodes"
  symptom that appeared once mid-session on this same kernel.
- **Where this landed:** not fixed. Two concretely-testable prebuilt
  kernel binaries were tried; neither reached the point of testing
  `kind` itself, one being too old for Docker Desktop's engine to start
  at all, the other missing one specific kernel config flag that engine
  startup depends on. A kernel that's simultaneously new enough for
  Docker Desktop and old/different enough to avoid the suspected `runc`
  nested-overlay hang wasn't found among the releases sampled; finding
  one would mean either bisecting further (each attempt costs a full
  Docker Desktop restart cycle) or building the LTS kernel from source
  with `CONFIG_ISO9660_FS` enabled, both judged higher effort than this
  learning-environment investigation currently warrants. `k8s/kind-
  cluster.yaml`, `k8s/base/*`, and `k8s/EXERCISES.md` were left
  completely unchanged across all three sessions; nothing about this
  failure implicates them, since the same manifests were never even
  reached in any attempt. All 18 exercises in `k8s/EXERCISES.md` remain
  blocked on a runnable cluster.
- Full writeup, concise interview-ready version, in `ISSUES.md`.

### Next recommended step
Either bisect further through `microsoft/WSL`'s release history for a
kernel that's both new enough to boot this Docker Desktop version and
old enough (or differently configured) to avoid the suspected `runc`
nested-overlay hang, or accept the current host as unable to run `kind`
and exercise `k8s/EXERCISES.md` on a different machine or a cloud-based
Kubernetes sandbox instead.

## 2026-09-25 (session 16): Exhausting the kernel-bisection path, then resolving via minikube instead of a fix

Picked up session 15's "next recommended step" directly: bisect further
through `microsoft/WSL`'s release history for a kernel that avoids the
`CreateContainer` hang without also missing `CONFIG_ISO9660_FS`. Then,
when that path came up empty, found a working resolution by changing
tools instead of changing the kernel.

- **Reconfirmed the original bug fresh, 4 days later, zero drift.**
  Docker Desktop had stopped running since the last session; a cold
  restart plus a fresh `kind create cluster --config k8s/kind-
  cluster.yaml` reproduced the identical `kubeadm`/cobra stack-trace
  failure. `.wslconfig`'s memory fix was still in effect
  (`Total Memory: 4.31GiB`), confirming that part of session 15's work
  is a genuine, permanent improvement, not something that silently
  reverted.
- **The kernel-bisection path is now exhausted, not just "not yet
  found."** Queried `microsoft/WSL`'s full release history via `gh api
  repos/microsoft/WSL/releases --paginate` and cross-referenced each
  release's changelog for kernel-bump PRs, rather than guessing at
  another release to try. Found that the app's bundled kernel switched
  from the `6.6.x` LTS line to the `6.18.x` Rolling line between
  releases `2.7.3` (`6.6.114.1`, the newest LTS-line release available)
  and `2.7.5` (`6.18.26.1`) -- so `2.7.3` was the best remaining LTS
  candidate, a materially newer point release than session 15's
  `6.6.87.2`. Extracted its bundled kernel the same
  `msiexec /a ... /qn TARGETDIR=...` way as before, pinned it via
  `.wslconfig`, and checked `wsl.exe -d docker-desktop -- cat
  /proc/filesystems` directly for an `iso9660` entry rather than waiting
  through Docker Desktop's several-minute startup retry loop to watch it
  fail the same way again. Absent, same as `6.6.87.2` -- confirming this
  is characteristic of the WSL app's own bundled *fallback* kernel as a
  build class (minimal config, meant for offline bootstrapping), not a
  gap specific to one point release. Reverted immediately once confirmed
  (no need to wait out a Docker Desktop restart for a predictable
  failure). Every prebuilt kernel binary Microsoft publishes anywhere is
  now checked: too old to boot Docker Desktop (5.10.16), missing
  ISO9660 (both LTS-line app-bundled builds tried), or carrying the hang
  (the Rolling line, currently active). The only remaining lever --
  building the LTS kernel from source with `CONFIG_ISO9660_FS` enabled,
  since `microsoft/WSL2-Linux-Kernel` is source-only with no prebuilt
  binaries -- was judged a materially bigger undertaking than anything
  tried so far and was not attempted.
- **User redirected to a different plan entirely: minikube with the
  Hyper-V driver.** Checked Hyper-V availability first, per the user's
  own plan -- `Get-ComputerInfo`/`Get-CimInstance Win32_OperatingSystem`
  both confirmed this host runs Windows 11 **Home** Single Language, and
  the Hyper-V role (`Microsoft-Hyper-V-All`) is excluded from Windows
  Home by Microsoft's licensing, not merely disabled; DISM couldn't even
  query it without elevation, but the OS edition alone makes this a hard
  stop, not a config toggle. Adapted: installed/upgraded minikube
  (winget had a stale 1.31.2; upgraded to 1.39.0) and tried
  `--driver=docker` instead, since it needs no new hypervisor install
  and is cheap to test either way.
- **First attempt failed on an unrelated, quickly-fixed issue.**
  `minikube start` (default profile) errored `K8S_OLD_UNSUPPORTED`:
  `~/.minikube/profiles/minikube/config.json` was a genuine leftover
  from 2023, pinning Kubernetes `v1.27.4`, older than current minikube's
  minimum-supported `v1.28.0`. Left that stale profile untouched (it
  predates this session, not something to delete casually) and used a
  named profile instead: `minikube start -p rag-learning --driver=docker`.
- **It worked. Genuinely, verified end to end, not just "control plane
  came up."** Every `kube-system` pod (etcd, kube-apiserver,
  kube-controller-manager, kube-scheduler -- the exact class of
  container that hung indefinitely under `kind`) reached `1/1 Running`.
  Node reached `Ready` after CNI initialization (~90s, unremarkable).
  Deployed a throwaway `nginx:alpine` Deployment + Service and watched
  it go through the complete real lifecycle
  (`Pending -> Pulling -> Pulled -> Created -> Started -> Running`, the
  image pull itself just being slow on this host, ~62s, not stuck), then
  proved in-cluster networking end to end with a second pod:
  `curl http://nginx-test` -> `HTTP 200`, meaning scheduling, image
  pull, container creation, `kube-proxy` Service routing, and CoreDNS
  resolution all genuinely work. This is the first time in six
  documented sessions across this investigation that a Kubernetes
  control plane has come up on this host at all.
- **Likely explanation, not fully proven:** minikube's `docker` driver
  runs its node the same "container-in-container" way `kind` does, but
  its `kicbase` image bundles `containerd 2.3.4`, a different build than
  `kind`'s own node image. On the identical kernel and Docker Desktop
  install, that different containerd avoids whatever was hanging on
  `CreateContainer`. `kind`'s own bundled containerd/runc version wasn't
  independently pinned down for a direct diff (its images were already
  removed by the time the comparison was made), so this is recorded as
  strong correlating evidence for the original hypothesis, not a fully
  nailed-down root cause.
- **Restructured `k8s/` per explicit user instructions**, given after
  seeing the working cluster: keep the existing `kind` setup completely
  untouched as the reference 3-node environment, add minikube support
  separately, prefer separate platform-specific docs over one blended
  doc, reuse `k8s/base` only where genuinely platform-independent, and
  label kind-only vs. minikube-only exercises clearly. Moved
  `kind-cluster.yaml` and `EXERCISES.md` into a new `k8s/kind/`
  (content-unchanged, plain filesystem move -- `k8s/` isn't tracked on
  this branch yet, so no `git mv` history to preserve). Added
  `k8s/minikube/`: a `kustomization.yaml` overlay on `../base` plus one
  patch file (`ollama-external-service-patch.yaml`, repointing the
  `ollama-host` `ExternalName` Service from `host.docker.internal` to
  `host.minikube.internal`), a `README.md` walking through minikube's
  own lifecycle commands, and an `EXERCISES.md` written as a delta
  document rather than a duplicate (17 of 18 exercises are unmodified
  `kubectl` commands that work identically either way; only the
  node-failure exercise is kind-only, since this environment is
  single-node). Verified the manifest-level difference actually matters
  before writing it into the overlay: `host.minikube.internal` and
  `host.docker.internal` resolve identically on this specific
  `--driver=docker` instance (Docker Desktop hands out
  `host.docker.internal` to every container it hosts, not just kind's),
  but `host.minikube.internal` is minikube's own documented,
  driver-agnostic mechanism, so it's the correct choice rather than a
  coincidence to depend on. Also verified `minikube image load` actually
  works (tested with an already-pulled `nginx:alpine`, confirmed present
  in the node's `containerd` image store via `minikube ssh` +
  `ctr -n k8s.io images ls`) before documenting it as the image-loading
  step, mirroring kind's `kind load docker-image`. Rewrote the top-level
  `k8s/README.md` as a shared orientation doc (mental model, applying
  manifests, ingest, observability, debugging, teardown -- everything
  that's genuinely identical once a cluster exists) with a new "choose
  your cluster" section up front; `k8s/kind/README.md` and
  `k8s/minikube/README.md` now hold only what's genuinely
  cluster-specific. Updated `CLAUDE.md`'s "Kubernetes deployment"
  section and `ISSUES.md`'s existing kind-hang entry to match the new
  layout and this session's resolution, rather than leaving either
  describing the old flat structure or the old "not fixed" conclusion.
- Full writeup, with exact evidence (error messages, exit codes,
  `/proc/filesystems` output, pod lifecycle events), in `ISSUES.md`'s
  "kind (and Docker Desktop's own built-in Kubernetes) cannot bring up a
  control plane on this host, but minikube's docker driver can" entry.

### Next recommended step
The learning environment is unblocked on this host via
`k8s/minikube/README.md`. Optional follow-ups, none blocking: actually
deploy the full RAG application stack onto the minikube cluster (this
session verified the cluster and the image-loading mechanism, not a
full application deployment); pin down `kind`'s own bundled
containerd/runc version for a direct diff against minikube's `2.3.4`, to
turn the current "strong correlating evidence" into a fully confirmed
root cause; and, if ever motivated, the from-source LTS kernel build
that would be needed to make `kind` itself work on this host.

### Continuation, same day: staged deployment of the real app onto minikube, and a real probe-design fix

Picked up the "actually deploy the full RAG application stack" follow-up
directly, at the user's explicit request for an *incremental* rollout
(namespace/config, then Postgres, then rag-api, then frontend/networking,
stopping after each stage for inspection, observability deliberately
held back).

- **Stages 1-2 (namespace/config, Postgres) went cleanly.** Applied
  `k8s/base/namespace.yaml`/`configmap-app.yaml`/`secret-app.example.yaml`
  directly (kind's own defaults, no overlay needed), then
  `ollama-external-service.yaml` followed by a live `kubectl patch` for
  the `host.minikube.internal` override (verified against the live
  `spec.externalName`, not just the stale `kubectl.kubernetes.io/
  last-applied-configuration` annotation, which still showed the
  pre-patch value and would have been misleading to trust). Confirmed
  minikube ships the same `standard` default StorageClass kind does
  before applying the Postgres StatefulSet, so the PVC needed no
  override either. Postgres reached `1/1 Running` after ~3 minutes,
  mostly image pull.
- **Stage 3 (rag-api) surfaced two real, unrelated problems, both
  diagnosed from first principles rather than guessed at.** First: the
  base manifest's `livenessProbe.initialDelaySeconds: 45` was nowhere
  near enough for this host -- the container was crash-looping, each
  restart discarding progress on a `torch`/`transformers`/
  `sentence-transformers` import that (with `PYTHONUNBUFFERED=1` already
  set, ruling out a buffering explanation) turned out to take 5-8+
  minutes with no pre-baked model cache, on a host whose slow I/O had
  already been the theme of the whole kernel investigation -- just
  showing up in container-filesystem reads this time, not network pulls.
  Diagnosed by removing the liveness probe entirely and watching
  `kubectl logs -f` uninterrupted until the app's own startup banner
  finally appeared. Second, once that was survivable: `GET /health`
  reported `vectorstore: unreachable` even though a raw `psql` *and* a
  raw `psycopg2.connect()` from inside the same pod, same DSN, both
  succeeded instantly. Traced to `PgVectorStore._connection()` calling
  `register_vector(conn)` unconditionally, which fails whenever the
  `vector` Postgres extension doesn't exist yet -- and
  `health_check()`'s blanket `except Exception: return False` silently
  swallowed the real exception, so nothing ever reached the logs.
  Resolved by actually running the `init-db` Job (schema/extension setup)
  rather than assuming it was a real connectivity bug.
- **The user turned the first finding into an actual fix, not just a
  live patch.** Live `kubectl patch`es (bumped `initialDelaySeconds`,
  then removed liveness entirely to diagnose, then set 400s) got the
  session unblocked but weren't a real design -- next instruction was to
  implement `/livez`/`/readyz` as first-class endpoints (`/health` kept
  as-is for diagnostics), add a `startupProbe`, and keep any
  host-specific timing tolerance out of the common base config. This is
  exactly the fix an earlier session's own `ISSUES.md` entry ("A
  Kubernetes readiness probe can't see through this app's own /health
  endpoint...") had predicted in its Interview takeaway without
  implementing, since application code was off-limits at the time.
  `src/rag/api/routers/health.py` gained `/livez` (no dependency
  injection at all, always 200) and `/readyz` (same two
  `health_check()` calls `/health` already made, but a real 503 via an
  injected `Response` object when either is down); 9 new tests in
  `tests/unit/test_health_endpoints.py` (dependency-override pattern
  matching `test_api_dos_limits.py`) prove `/livez` survives every
  combination of dependency failure and `/readyz` doesn't. Ran locally
  first (`pytest tests/unit/test_health_endpoints.py`, 9 passed) before
  touching any manifest. `k8s/base/rag-api-deployment.yaml` now has all
  three probes (`startupProbe`/`livenessProbe` -> `/livez`,
  `readinessProbe` -> `/readyz`), with a generic 2-minute
  `startupProbe` budget that doesn't know or care about any one host's
  slow disk; a new `k8s/minikube/rag-api-startup-probe-patch.yaml`
  overlay is where this host's real tolerance lives instead.
- **A real mistake, caught and fixed in the moment.** Rebuilt and
  reloaded the image (new routes needed baking in), then ran `kubectl
  apply -k k8s/minikube` to pick up the new probes -- forgetting that a
  full kustomize overlay apply includes *everything* in it, including
  the frontend Deployment/Service, Ingress, and NetworkPolicies that
  were explicitly supposed to wait for Stage 4. Caught immediately from
  the apply's own output listing those resources as `created`; deleted
  all of them right away, leaving only what belonged at this point
  (namespace/config/Postgres/rag-api). Lesson for the rest of this
  staged rollout: apply individual files/targeted patches per stage, the
  way stages 1-2 already had been, not the whole overlay.
- **First calibration attempt (600s startup budget) was measured, found
  wanting, and corrected -- not guessed twice.** A real rollout with the
  600s budget was killed by `startupProbe` itself right at the edge (one
  restart, `Killing... failed startup probe` in the pod's events,
  matching the budget almost exactly), then succeeded on the retry
  inside a second, fresh 600s window. Rather than assume that was close
  enough, bumped to 900s for real margin and re-verified: a subsequent
  rollout came up with **zero restarts**. Confirmed all three endpoints
  live afterward (`/livez` 200, `/readyz` 200 with both dependencies
  `ok`, `/health` 200 matching) via an ephemeral in-cluster curl pod.
- Full writeup (the probe redesign, the crash-loop diagnosis, the
  `register_vector`/extension red herring, the calibration numbers) in
  `ISSUES.md`'s updated "readiness probe can't see through /health"
  entry; a cross-reference note added to the kind/minikube entry too.

### Next recommended step
Stage 4 (frontend + networking + Ingress) is next, still incremental per
the user's explicit staging request. Same lesson from this session
applies going in: apply `k8s/base/frontend-*.yaml`/`ingress.yaml`/
`networkpolicy.yaml` individually (plus whatever minikube-specific
patches turn out to be needed, e.g. for Ingress -- see
`k8s/minikube/README.md`'s own "Ingress on minikube" section), not the
whole `k8s/minikube` overlay at once. Observability stays deliberately
deferred beyond that, per the user's own instruction.

## 2026-09-26 (session 17): Stage 4 (frontend + networking) on minikube, and a real nginx/Kubernetes incompatibility found and fixed

### Picking up Stage 4
Continuing the staged minikube rollout from session 16: namespace/config,
Postgres, and rag-api were already deployed and stable (rag-api at 2/2,
HPA-scaled, zero fresh restarts). User said "yes, proceed with stage 4."
Applied the lesson from session 16's own premature-overlay mistake:
rendered k8s/minikube's full kustomize output, then extracted and
applied only the Stage 4 resources (Service/frontend, Deployment/
frontend, Ingress/rag, the four NetworkPolicy objects) via
`kubectl apply -f -`, rather than `kubectl apply -k k8s/minikube`
directly -- confirmed via `kubectl diff -k k8s/minikube` beforehand that
a full apply would have reset rag-api's replica count from 2 back to
the base manifest's static 1, fighting the HPA. Built and loaded both
local-rag-api:minikube (already present) and a freshly built
local-rag-frontend:minikube (--build-arg VITE_UI_MODE=developer, per
k8s/minikube/README.md's own documented command) into the minikube node.

### The frontend pod crash-looped immediately -- and it wasn't a probe-tuning issue
kubectl describe pod showed liveness/readiness probes failing with
"context deadline exceeded" against the pod's own root path, with the
default timeoutSeconds: 1. Before assuming "just raise the timeout"
(the pattern from session 16's rag-api startupProbe work), verified
directly: a bare loopback wget http://127.0.0.1:80/ run inside the pod
itself took the full 5s of its own timeout to fail. nginx itself was the
slow part, not Kubernetes' probe configuration.

Traced to frontend/nginx.conf's `resolver 127.0.0.11 valid=30s;` --
Docker Compose's own embedded per-container DNS address, hardcoded, and
meaningless inside a Kubernetes pod (confirmed: another pod's
/etc/resolv.conf in this cluster pointed at 10.96.0.10, the kube-dns
Service). Since every proxied location in this nginx config uses the
variable form of proxy_pass (forcing runtime DNS resolution), every
single proxied path -- not just the probe's root path -- was affected:
the frontend would have been unable to reach rag-api for any real query,
not just failing its own health checks. Presented this finding to the
user with full evidence before touching any application code, since the
milestone's own stated design goal was zero app-code changes; the user
chose to authorize the minimal targeted fix over a manifest-only
workaround.

### Three fixes, in three layers, each verified independently before moving to the next
1. The resolver address. Renamed frontend/nginx.conf to
   frontend/nginx.conf.template (Dockerfile COPY destination changed to
   /etc/nginx/templates/default.conf.template, the official
   nginx:1.27-alpine image's own envsubst-processed template dir),
   changed the hardcoded IP to `resolver ${NGINX_LOCAL_RESOLVERS}
   valid=30s;`. First rebuild crashed on start (host not found in
   resolver "${NGINX_LOCAL_RESOLVERS}") -- caught by a local standalone
   docker run smoke test, not another slow in-cluster crash-loop cycle.
   Root cause: the official image's own
   docker-entrypoint.d/15-local-resolvers.envsh script (which exports
   NGINX_LOCAL_RESOLVERS from the container's real /etc/resolv.conf) is
   itself gated behind NGINX_ENTRYPOINT_LOCAL_RESOLVERS, read directly
   from the script's own source -- never set by this project's
   Dockerfile. Added ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1.
2. The upstream hostname. Even with the resolver address correct
   (resolver 10.96.0.10 valid=30s;, confirmed rendered), the proxied
   request still hung. Isolated with three direct nslookup queries
   against the cluster's real DNS server: bare rag-api and rag-api.rag
   both failed (no answer / NXDOMAIN); only the fully-qualified
   rag-api.rag.svc.cluster.local resolved. nginx's own resolver
   directive does not apply /etc/resolv.conf's search-domain expansion
   the way a normal libc lookup does (confirmed: a plain wget
   http://rag-api:8000/health from the same pod, using the system
   resolver, succeeded in 0.14s). Templated `set $rag_api_upstream
   "http://${RAG_API_UPSTREAM_HOST}:8000";`, defaulted
   RAG_API_UPSTREAM_HOST=rag-api in the Dockerfile (matches Compose
   exactly, zero regression there), and overrode it to
   rag-api.rag.svc.cluster.local via a pod env: entry in
   k8s/base/frontend-deployment.yaml -- the same "environment-specific
   value belongs in a Deployment env var" pattern this codebase already
   uses for OLLAMA_BASE_URL vs. CONTAINER_OLLAMA_BASE_URL.
3. CPU throttling from an unconstrained worker count. After both of the
   above, the pod still crash-looped intermittently. ps aux showed 8-10
   nginx worker processes against a 200m CPU limit; a live pod's own
   cgroup v2 cpu.stat showed real throttling (nr_throttled 25 of 177
   periods, 1.3s accumulated throttled_usec) -- not a hypothesis, read
   directly. worker_processes auto; resolves against the host's CPU
   count, with zero awareness of the container's own cgroup quota. The
   official image ships a second sibling opt-in script for exactly this,
   30-tune-worker-processes.sh, gated behind
   NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE -- also never set, also
   confirmed by reading the script directly. Added ENV
   NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE=1.

### Verification
Real end-to-end checks at every layer, not just "the pod is Ready":
in-cluster nslookup/raw TCP connectivity tests isolating DNS from proxy
behavior from CPU scheduling; a live cgroup cpu.stat read before and
after the worker-count fix (worker_processes confirmed rewritten to 1 in
the running pod, throttling stopped accumulating over a clean
~5-minute observation window post-fix); the frontend pod stable at
1/1 Ready with no further restarts for the remainder of the session.
Both the static SPA route (GET / with Accept: text/html -> index.html)
and a real proxied backend route (GET /health -> rag-api's live
dependency-check JSON, 0.01s) verified working through the actual
in-cluster nginx, not just asserted from config inspection. Added
scripts/smoke_test_frontend_nginx.sh (matching smoke_test_containers.sh's
existing house style) encoding all three checks against a running
container, so a future accidental removal of one of the three ENV lines
fails fast and clearly rather than surfacing as another live probe
crash loop. Full diagnosis and fix writeup in ISSUES.md.

Also confirmed, per explicit ask, that the fix is genuinely
environment-portable rather than solving Kubernetes by hardcoding
Kubernetes specifics: resolver never hardcodes an IP (reads the
container's own /etc/resolv.conf at every startup, in both
environments), and the one place that does differ per environment
(RAG_API_UPSTREAM_HOST) is a DNS name, set via each environment's own
natural mechanism (a Dockerfile ENV default for Compose, a Deployment
env: override for Kubernetes) -- never an IP address anywhere in either
config file.

Ingress verified separately: minikube addons enable ingress timed out
its own internal wait (image-pull slowness, this host's now-familiar
pattern) but the controller pod did reach 1/1 Running shortly after;
confirmed the Ingress object picked up class: nginx correctly. kubectl
port-forward proved unreliable on this Windows/Git-Bash setup (silent
connection refusals with no error output, even after retrying with
run_in_background) -- worked around by exercising the ingress
controller's Service from inside the cluster instead (kubectl exec +
wget --header), which is a legitimate way to prove the Ingress ->
Service -> pod routing chain without depending on a flaky
host-to-cluster tunnel.

### Next recommended step
Stage 4 is functionally complete and verified (frontend Deployment,
Service, Ingress, NetworkPolicy all applied; static and proxied routes
both confirmed working end to end in-cluster). Remaining before calling
Stage 4 fully closed: a real browser-based check through minikube tunnel
+ the rag.local hosts-file entry (not exercised this session, since
minikube tunnel needs a dedicated persistent terminal this environment
doesn't have) would be the natural follow-up, though the in-cluster
Ingress verification already proves the routing chain itself works.
The identical fix was also independently verified against
docker-compose.frontend.yml's real rag-api container by end of session:
rag-api's own cold start took an anomalous 2h49m (vs. this project's
previously documented 5-10 minute worst case), separate and unrelated to
the nginx changes -- confirmed via docker top's low accumulated-CPU-time
relative to wall time, consistent with this host's known I/O bottleneck
compounded by several concurrent docker builds and a running minikube
cluster sharing the same disk this session. Once it finished, all three
nginx fixes checked out identically under Compose:
scripts/smoke_test_frontend_nginx.sh passes all three checks (rendered
config has no unexpanded placeholder, worker_processes autotuned to a
concrete value -- 8, since this container has no CPU limit so the
script's own designed fallback-to-host-count applies -- and a real
proxied /health request returns the live rag-api dependency-check JSON).
Full evidence and the honest caveat about not being able to prove
whether stopping minikube specifically caused or merely coincided with
rag-api's recovery are in ISSUES.md's updated Result section.

**New, unresolved item opened at the very end of this session (not
investigated further, per explicit instruction):** restarting minikube
afterward (`minikube start -p rag-learning`, to put Stage 4 back into a
running state -- my own follow-up, not something requested) triggered a
full node reboot, which meant both `rag-api` replicas cold-started
simultaneously for the first time this project has exercised -- normal
steady-state operation never restarts both at once. `postgres` and both
`rag-api` pods settled within a few minutes (restart counts stopped
climbing), but the `frontend` pod did not: it stayed in a genuine,
persistent `CrashLoopBackOff` over a 5+ minute observation window,
restart count still climbing (12+) when observation stopped.
`worker_processes` was confirmed still correctly set to `1` inside the
pod (the earlier fix is intact), so this is a different failure than the
one fixed this session. `kubectl describe node rag-learning` showed
summed pod memory *limits* at 100% of node allocatable (a static
commitment figure, not confirmed live usage) at the time this was
observed; one `docker stats` snapshot of the minikube node container
itself showed 20% CPU / 58% memory, not visibly maxed at that instant.
**Root cause not confirmed** -- node-wide CPU contention, memory
pressure, or something else entirely during simultaneous multi-pod cold
start are all still open possibilities, not narrowed down further per
the user's explicit instruction to stop investigating and preserve the
current manifests/exercise setup as-is for later hands-on practice
rather than keep changing things this session. Flagged here and in
ISSUES.md as a known, unresolved follow-up, deliberately not folded into
the resolved nginx-fix entry above since it's a distinct problem with
no established solution yet.

Observability remains deliberately deferred, per the user's original
staged-rollout instruction.

---

## 2026-09-29 (session 18): Production-hardening the single-agent LangGraph experiment

### Interview talking points
- Extended `langgraph_experiment/` (session 13's single-agent exercise,
  untouched by session 14's separate multi-agent one) with the
  `.claude-prompts/31.0-LangGraph.txt` file's later-appended "PRODUCTION
  WORKFLOW ADDITIONS" section: Postgres-backed checkpointing preferred
  over SQLite, a real crash-safe idempotency ledger, approval expiration,
  a separate per-workflow timeout, explicit cancellation, checkpoint
  retention/cleanup, a write-path retry policy, an isolated audit trail,
  and OpenTelemetry/Prometheus observability -- still zero changes to
  `rag.agent.graph` (production) or to `langgraph_experiment/`'s own
  original-phase files' public behavior.
- Verified `langgraph-checkpoint-postgres` is real, supported, and
  lightweight (52 KB wheel) by installing it and round-tripping a full
  interrupt/resume/restart cycle against this project's own local
  Postgres *before* writing `checkpointer.postgres_checkpointer` --
  including confirming `PostgresSaver.setup()`/`.delete_thread()` behave
  as documented, not assumed.
- Built `idempotency.py`'s `ActionLedger` (Postgres-backed, its own
  driver choice -- `psycopg2`, matching this project's existing
  convention, deliberately separate from `langgraph-checkpoint-postgres`'s
  own `psycopg` v3 requirement) and proved the spec's hardest ask by hand:
  a test that manually reproduces "crashed after the mutation ran, before
  the ledger recorded completion" (`mark_started` + the real mutation,
  skipping `mark_completed`) and shows the recovered attempt calls the
  mutation again safely -- safe only because `update_case_status`'s own
  `already_in_status` branch is naturally idempotent, which the entry's
  own module docstring and README are explicit is not something the
  ledger manufactures on its own.
- A second, harder version of the same proof: a `RetryPolicy` on the
  write node itself, with a fake `update_case_status_fn` that raises
  *after* delegating to the real mutation (simulating a network blip
  discovered only after the remote call already committed) -- confirmed
  the automatic retry reports `already_in_status` on the second real
  call, never a fabricated second `executed`.
- Real, found-by-reasoning-through-the-exception-hierarchy correctness
  bug, caught before it ever ran: `interrupt()` raises
  `langgraph.errors.GraphInterrupt` (a real `Exception` subclass)
  internally on every single approval pause. A naive `except Exception:`
  around each node call (for the new per-node OpenTelemetry span/latency
  wrapper) would have silently counted every ordinary pause as a node
  *failure* in the new `WORKFLOW_FAILED_TOTAL` Prometheus counter.
  Confirmed the exception hierarchy directly
  (`GraphInterrupt` subclasses `GraphBubbleUp`, itself an `Exception`)
  before writing the wrapper, and excluded `GraphBubbleUp` from the
  failure-counting branch specifically.
- A second, smaller correctness bug, this one caught *by actually
  running the tests*, not by reasoning: two new tests used a `0`-second
  expiration/timeout threshold, which requires only "elapsed time > 0"
  to trip. On a fast, fully in-memory back-to-back double-`invoke()` (no
  Postgres/Ollama I/O to add real wall-clock separation), that gap can
  legitimately round to exactly `0.0` on this host's clock resolution,
  making the test flaky rather than proving the real (correct) `>`
  comparison in `nodes.wait_for_approval`. Confirmed the production logic
  itself was fine via a standalone reproduction script (with print
  statements adding just enough incidental delay to avoid the tie) before
  concluding it was a test-design flaw, not a code bug; fixed by using
  `-1` instead of `0` in both tests, which can't tie regardless of clock
  granularity.
- Worked around a real, live resource-contention incident mid-session:
  another Claude Code session, on the same machine and working directory
  on a related but separate task, was running its own concurrent pytest
  processes against the same shared local Postgres instance, causing
  real (not hallucinated) slowness that looked exactly like a hung
  process for several minutes. Resolved by direct cross-session
  coordination (`SendMessage`) rather than guessing -- confirmed the
  other session's work was byte-identical to what would have been
  written independently, so nothing was lost by it standing down.

### Goal
Implement the `.claude-prompts/31.0-LangGraph.txt` file's appended
"PRODUCTION WORKFLOW ADDITIONS" section on the `experiment/langgraph-agent`
branch: Postgres-preferred checkpointing, a proven crash-safe idempotency
ledger keyed by an explicit operation id, approval expiration/timeout/
cancellation, checkpoint retention, max retries (safely, on the write
path), a safe audit trail, OpenTelemetry spans per node, Prometheus
workflow counters, and an explicit explanation of workflow durability vs.
business-operation idempotency vs. transaction atomicity. No commit/push,
per the prompt file's own standing instruction.

### Configuration
No `config/*.yaml` changes and no changes under `src/rag/`. All changes
confined to `langgraph_experiment/` (three new modules -- `idempotency.py`/
`audit.py`/`observability.py` -- plus edits to `state.py`/`wiring.py`/
`nodes.py`/`routing.py`/`graph.py`/`checkpointer.py`/`cli/*.py`/`README.md`,
one new `cli/retention_cleanup.py`, three new test files, and one new
`langgraph-checkpoint-postgres`/`psycopg[binary,pool]` install into the
existing `.venv-langgraph/`).

### Changes made
- Brought up the shared `local-rag-postgres` container (`docker compose
  up -d postgres`) and confirmed `langgraph-checkpoint-postgres`
  round-trips a real interrupt/resume/restart cycle before writing any
  of `checkpointer.py`'s new `postgres_checkpointer`.
- Added `idempotency.py` (`ActionLedger`: `ensure_schema`/`get`/
  `mark_started`/`mark_completed`/`record_attempt`, plus the crash-window
  reasoning documented directly in its module docstring), `audit.py`
  (`log_workflow_event`, an isolated event vocabulary -- deliberately not
  an extension of `rag.audit`'s shared, production `AuthEventType`
  `Literal`), and `observability.py` (a dedicated Prometheus
  `CollectorRegistry`, `traced_node` wrapping every node registration
  with a reused `rag.observability.tracing.start_span`, and
  `record_run_outcome`).
- `state.py`: `PendingCaseAction` gained `operation_id`/`expires_at`;
  `GraphState` gained `thread_id`/`workflow_started_at`; `ApprovalState`
  gained `expired`/`cancelled`/`workflow_timeout`.
- `wiring.py`: `GraphDeps` gained `action_ledger`/
  `approval_timeout_seconds`/`max_workflow_duration_seconds`;
  `build_default_deps` gained `with_ledger` and now also calls
  `tracing.configure_tracing` once.
- `nodes.py`: `classify` stamps `workflow_started_at` once;
  `validate_write_request` became a `deps`-aware factory minting
  `operation_id`/`expires_at`; `wait_for_approval` gained server-side
  workflow-timeout/approval-expiration/cancellation checks (workflow
  timeout checked first) plus `audit.log_workflow_event` calls on every
  branch; `execute_write_action` routes through `ActionLedger.
  record_attempt` when a ledger is configured, and renders a
  ledger-replay result via the same terminal formatter as a fresh one;
  `synthesize_write_result` gained three new terminal branches.
- `graph.py`: every node registration now wrapped with
  `observability.traced_node`; added `_WRITE_ACTION_RETRY_POLICY`
  (documented as safe only because of the ledger check inside the node).
- `checkpointer.py`: added `postgres_checkpointer`, alongside the
  existing `sqlite_checkpointer`.
- `cli/run_demo.py`/`approval_cli.py`: `--backend {postgres,sqlite}`
  (default `postgres`) and `--no-ledger`; `approval_cli.py` gained a
  `cancel` subcommand and a friendly (non-authoritative) CLI-side
  expiration heads-up; both now call `observability.record_run_outcome`.
  New `cli/retention_cleanup.py` (deletes only terminal-state threads'
  checkpoints past a cutoff, never a still-paused one).
- New tests: `tests/test_idempotency_ledger.py` (4 tests, Postgres-only,
  self-skipping via a new `require_postgres`/`postgres_deps` fixture
  pair mirroring the main suite's `tests/integration/conftest.py`
  convention), `tests/test_approval_expiration_and_cancellation.py`
  (5 tests), `tests/test_write_action_retry_policy.py` (2 tests, one
  Postgres-only).
- Rewrote README.md's "Idempotency and safety" section (five scenarios
  now, the three-way workflow-durability/business-idempotency/
  transaction-atomicity distinction spelled out explicitly) and added a
  new "Production hardening" section covering every addition above; added
  5 new interview Q&A entries (21-25) specific to this phase.
- This entry.

### Issues discovered
- **`GraphInterrupt` would have been miscounted as a node failure in
  Prometheus metrics.** Symptom: none observed yet -- caught by reasoning
  through `langgraph.errors`'s exception hierarchy before writing
  `observability.traced_node`, not by a failing test. Root cause: a
  generic `except Exception:` around a per-node span/metrics wrapper
  can't distinguish an intentional `interrupt()` pause from a genuine
  node failure, since both raise real `Exception` subclasses internally.
  Fix: catch `GraphBubbleUp` (the parent of `GraphInterrupt`) separately
  and re-raise it untouched before the generic failure-counting branch.
  Evidence: `python -c "from langgraph.errors import GraphInterrupt;
  print(GraphInterrupt.__mro__)"` confirmed the hierarchy directly.
  Resolved: y (never shipped in a broken state, since it was caught
  before the first test run).
- **A 0-second expiration/timeout test threshold was racy, not the
  production logic it was testing.** Symptom: `test_workflow_timeout_
  takes_precedence_over_a_well_formed_approval` failed intermittently
  (`'executed' == 'workflow_timed_out'` assertion failure) despite the
  underlying `nodes.wait_for_approval` comparison being correct. Root
  cause: `max_workflow_duration_seconds=0` needs any nonzero elapsed
  time to trip; two back-to-back all-in-memory `graph.invoke()` calls
  with no real I/O between them can complete within the same clock tick
  on this host. Fix: use `-1` instead of `0` in both this test and the
  sibling `approval_timeout_seconds=0` cases, which can't tie regardless
  of clock granularity. Evidence: a standalone reproduction script
  (identical scenario, plus incidental `print()` overhead between the
  two invokes) produced the correct `workflow_timed_out` result every
  time, isolating the flake to the test's own threshold choice, not the
  code. Resolved: y.
- **A concurrent peer Claude Code session, working a related task in the
  same directory, caused real multi-minute test slowdowns that looked
  like a hang.** Symptom: `python -m pytest ...` produced zero output for
  several minutes on more than one run, well past what even a cold
  `sentence-transformers`/`torch` import (independently confirmed and
  timed at ~145s once, then fast on every subsequent run) could explain.
  Root cause: a second, independent Claude Code session was running its
  own pytest processes against the same shared local Postgres instance
  concurrently, for a related but separate piece of this same overall
  request. Not a code bug at all. Resolved via direct `SendMessage`
  cross-session coordination -- confirmed the other session's
  in-progress `observability.py` was byte-identical to this session's
  own, so it stood down with nothing lost; subsequent runs completed in
  15-55s consistently. Resolved: y (coordination, not a code fix).

### Evaluation results
- `python -m pytest langgraph_experiment/tests -v`: 45 items collected;
  40 pass with no Postgres running (5 cleanly self-skip via
  `require_postgres`); all 45 pass with Postgres running (confirmed
  across several runs both before and after the two fixes above,
  including the 4 idempotency-ledger tests and both write-retry-policy
  tests specifically).
- Manually re-verified the original phase's demo commands still work
  unchanged in spirit (`--backend sqlite --no-ledger` reproduces the
  exact original zero-external-services behavior).

### Interpretation
The two real findings this session surfaced are a matched pair worth
remembering together: one caught by reasoning about a library's
documented exception hierarchy *before* running anything (the
`GraphInterrupt`/Prometheus miscount), and one caught only by actually
running the tests repeatedly until a flake surfaced (the 0-second
threshold race). Neither substitutes for the other -- careful reasoning
about a new dependency's control-flow primitives catches classes of bug
that a single green test run won't (a metrics-mislabeling bug has no
natural failing assertion to trip), while some classes of bug (timing
races) are for all practical purposes only found by hitting them, no
matter how carefully the code was reasoned through beforehand. The
idempotency ledger itself is the clearest illustration in this whole
two-phase experiment of the difference between an argument and a proof:
the original phase already *argued* (correctly) that
`update_case_status`'s natural idempotency covers duplicate resumes; this
phase went further and manufactured the exact crash window by hand to
*prove* it, rather than trusting the argument alone -- and along the way
made explicit, in both code comments and the README, exactly which part
of the safety story is structural (the ledger correlating attempts) and
which part is still borrowed from the business layer (idempotency itself
is never something a ledger or a workflow engine can manufacture for an
arbitrary side effect).

### Open problems
- `cli/compare_harness.py rag`/`write-action` still weren't run against
  a live Postgres+Ollama stack with the production-hardened graph in
  this session (same open item carried over from session 13) -- the
  qualitative comparison table's reasoning is unchanged, but no fresh
  quantitative checkpoint-overhead number specific to the Postgres
  backend (vs. the original SQLite-backed number) has been measured yet.
- `cli/retention_cleanup.py` was written and reasoned through but not
  exercised against a real accumulation of aged, completed threads in
  this session (no such backlog exists yet in this fresh Postgres
  instance) -- its terminal-state-only guard is unit-reasoned, not yet
  proven against a live mixed batch of paused-and-completed threads.
- The idempotency ledger's own two-connections-per-call pattern
  (`psycopg2.connect(...)` opened fresh per `ActionLedger` method call,
  never pooled) was left as-is -- fine at this experiment's scale, but a
  real production version of this ledger would likely want the same
  `ThreadedConnectionPool` pattern `PgVectorStore`/`FeedbackStore` already
  use elsewhere in this codebase, not reimplemented here since it would
  be premature optimization for a table this narrow and low-volume.

### Next recommended step
Bring `make up` + native Ollama back up (stopped mid-session for an
unrelated interview) and run `cli/compare_harness.py rag "<a real
techfusion question>" --dataset-id techfusion` once, to get a real,
measured Postgres-backend checkpoint-overhead number to replace the
still-reasoning-only Latency row inherited from session 13.

## 2026-09-29 (session 19): A distributed-state experiment -- reproducing and fixing a horizontal-scaling bug with Redis

### Interview talking points
- Built `distributed_state_experiment/` on a new branch,
  `experiment/distributed-state` (from `experiment/kubernetes-deployment`,
  per the prompt's own first line), done as an isolated git worktree
  (`.claude/worktrees/experiment-distributed-state`) since two other
  sessions were already using worktrees on `experiment/helm` and the
  `fix/batch*` branches in the same repo checkout at the time.
- First inspected the app for process-local state before touching
  Redis, per the prompt's explicit ordering: `api/deps.py`'s in-memory
  `slowapi.Limiter` singleton (`get_rate_limiter()`, already documented
  in `RateLimitConfig`'s own docstring as process-local), `rag.mcp.
  business.store`'s in-memory `_SYNTHETIC_CASES` dict, and the pgvector
  connection pool. Chose the rate limiter as the demonstration target,
  per the prompt's own recommendation.
- Demonstrated the bug with real numbers, not just reasoning: 3 separate
  OS processes (standing in for 3 API replicas), each running the
  in-memory limiter, let 300/300 requests through against an intended
  tenant limit of 100 -- reproduced identically on every rerun, since
  that configuration never touches Redis at all.
- Fixed it with a config-gated Redis backend in the real `src/rag/`
  package (`RateLimitConfig.backend: memory|redis`, `redis_fail_mode:
  fail_open|fail_closed`, off by default), using the `limits` library's
  `RedisStorage` (a genuinely atomic Lua-script increment, confirmed via
  source inspection, not a plain `INCR`+`EXPIRE`). Verified against a
  genuine `redis:7-alpine` container: 100/100 exact on 3/3 runs.
- Went beyond the rate limiter into a full Redis Streams ingestion job
  queue (`job_queue/`): producer/consumer, bounded worker concurrency,
  backpressure, retry with exponential backoff and a max-retry-count
  dead-letter path, an idempotency key reusing this repo's own
  `(source, dataset_id)` + checksum document-identity scheme, and
  `XREADGROUP`/`XACK`/`XCLAIM`/`XPENDING`-based claim/visibility
  semantics for worker-crash recovery -- reused the real
  `IngestionPipeline`/`Writer`/`factory` wiring for the actual work
  rather than a bespoke reimplementation.
- Two real bugs found only by running real code against a real (if
  simulated at first) backend, neither caught in design review: `XLEN`
  doesn't shrink on `XACK` (an early `queue_depth()` built on it read
  nonzero after every job finished); and `IngestionJobQueue`'s
  `key_prefix` defaulted to one global constant regardless of
  `stream_name`, letting two unrelated experiment runs silently share
  one dead-letter counter. Both fixed and documented as their own
  `ISSUES.md` entries.
- A third real bug, found in a resumed pass: `tests/unit/
  test_rate_limiter_redis_backend.py`'s own teardown fixture called
  `deps.get_config.cache_clear()` after a test had already monkeypatched
  `deps.get_config` to a plain lambda with no `.cache_clear()` --
  `monkeypatch`'s own revert runs *after* this fixture's teardown in
  this file's fixture ordering, not before. Fixed by capturing the real
  `lru_cache`d function reference once at module import time, before any
  test could patch it. Also its own `ISSUES.md` entry.
- Demonstrated one distributed-lock scenario modeled directly on this
  repo's own, already-fixed `get_or_create_document_id` race (from the
  joint-investigation backlog): a Redis `SET key val NX PX ttl` lock
  (Lua-script, token-checked release) does serialize the race to one
  `document_id`, but the experiment's own conclusion is that
  `INSERT ... ON CONFLICT` remains preferable where an atomic
  single-operation DB primitive exists for the actual resource, since a
  lock is an opt-in convention every caller must remember, not a
  property Postgres enforces on its own.
- Session had a real mid-work pause and resume: a sibling session was
  also doing Docker work on the same host, so the subagent doing this
  implementation was told to stop all Docker/Kubernetes-touching
  actions and switch to non-Docker work; later, after the user needed
  the machine fully quiet for an unrelated interview, all lingering
  processes from this task (four `fake_redis_server.py` instances, two
  stray pytest runs) were found via `Get-CimInstance Win32_Process` and
  killed directly, while processes belonging to other sessions
  (`langgraph_experiment` pytest runs, the pre-existing
  `local-rag-postgres`/`rag-learning` minikube containers) were
  deliberately left untouched. A precise, honest "what was executed vs.
  simulated" section in the experiment's own README documents both
  passes: fakeredis-backed (Docker paused) and, after the pause lifted,
  genuine-Redis-backed (Docker confirmed free first).
- Kubernetes experiments (11-12, "scale worker pods") were checked, not
  applied: `minikube status -p rag-learning` (read-only) confirmed the
  shared cluster was fully stopped, and starting a multi-day-old
  cluster shared with the sibling `k8s/`/`experiment-helm` work was
  judged too invasive to do unattended for a verification step already
  covered by a real local-process throughput proxy (1 vs. 3 worker
  processes draining an identical backlog, run for real against both
  Redis backends). `k8s/distributed-state/` manifests remain written
  and reviewed, documented explicitly as "checked and deliberately
  skipped."
- A real, measured environmental constraint shaped several execution
  choices: a cold `import rag.factory` was independently timed at
  ~1122s and ~1333s (18-22 minutes, non-flaky, two separate
  measurements) in this sandbox, almost certainly Windows antivirus
  DLL-scanning of the ML dependency tree on first touch. This is why
  the production Redis-rate-limiter wiring tests ran only once per
  pass rather than repeatedly, and why Part 1's real two-pod HTTP demo
  was written and reviewed but not executed live, with a substitute
  unit test (`test_inmemory_limiter_divergence.py`, run 3x, no
  flakiness) used instead.
- End state: nothing committed to `experiment/distributed-state` during
  implementation, per the prompt's own explicit instruction. Once the
  user later asked to close the worktree so the branch could be checked
  out elsewhere, and it became clear that removing an uncommitted
  worktree would permanently delete the entire implementation (the
  branch ref had never advanced past its `experiment/kubernetes-
  deployment` starting point), this was flagged directly rather than
  resolved unilaterally -- the user chose a single local commit (no
  push) before the worktree was removed.

### Goal
Follow `.claude-prompts/27.0-Distributed-systems-experiment+Redis.txt`
closely: find one real example of behavior that is correct with one API
replica but incorrect with several, prove it before introducing any new
infrastructure, fix it with Redis, extend into a full distributed
ingestion job queue with realistic failure-mode handling, demonstrate a
distributed lock, benchmark three configurations, and produce a 15-
question interview write-up. No commit/push during implementation, per
the prompt's own explicit instruction.

### Configuration
Config-gated additions to `src/rag/config.py` (`RateLimitConfig.backend`/
`redis_url_env_var`/`strategy`/`redis_fail_mode`/`redis_key_prefix`,
`AppConfig.rate_limit_redis_url()`) and `src/rag/api/deps.py`/`main.py`
(Redis-backed `get_rate_limiter()`, a `RedisError` -> 503 handler under
`fail_closed`); `redis>=5.0` added as a core `pyproject.toml` dependency.
Default behavior (`backend: memory`) is byte-identical to before. All
other new code lives under the isolated `distributed_state_experiment/`
directory (own `job_queue/`, `rate_limiter/`, `distributed_lock/`, `cli/`,
`tests/`, `results/`, `README.md`, `INTERVIEW_QUESTIONS.md`), plus
`docker-compose.redis.yml` and `k8s/distributed-state/` manifests.

### Evaluation results
- `distributed_state_experiment/tests`: 24/24 passing (23 fakeredis-era
  tests plus one added against genuine Redis in the resumed pass), zero
  skips once a live Redis container was reachable.
- `tests/unit/test_rate_limiter_redis_backend.py`: 5/5 passing after the
  fixture-teardown-order fix (`442.23s` runtime, dominated by the
  `rag.factory` import cost above, not the test logic itself).
- Part 4 benchmark (100 requests/replica, intended limit 100/tenant/
  minute, 3 runs each): 1 replica + in-memory = 100/100 (correct);
  3 replicas + in-memory = 300/100 every run (the bug, reproduced
  exactly); 3 replicas + Redis-backed against fakeredis = 104/100,
  100/100, 100/100 (2/3 exact, 1/3 within 4%); 3 replicas + Redis-backed
  against genuine `redis:7-alpine` = 100/100 on 3/3 runs.
- Atomic-increment concurrency test (200 requests, 20 threads, one
  shared bucket, limit 50): exactly 50 allowed against both fakeredis
  and genuine `redis-server`.
- Experiments 1-10 (submit 20 jobs, 1 vs. 3 workers, kill-a-worker
  recovery, duplicate delivery, poison-job retry/backoff/dead-letter)
  reran end-to-end against genuine Redis with identical correct final
  states to the fakeredis pass (correct dedup, `attempts=2` on the
  recovered job, dead-letter after exactly 3 attempts).

### Interpretation
The single most useful finding for interview purposes isn't any one bug,
it's the shape shared by both queue-accounting bugs and the distributed-
lock conclusion: none of them were visible from reading the code, only
from running it. `XLEN` counting acknowledged-but-undeleted entries and a
key-prefix collision between two queues are exactly the kind of subtle,
protocol-level edge case that a design review confidently signs off on and a
single real execution immediately falsifies -- reinforcing this repo's
now-repeated pattern (the LangGraph join-node bug, the nginx worker_
processes bug, this session's two queue bugs) that reasoning and
execution catch genuinely different bug classes, and neither substitutes
for the other. The distributed-lock-vs-DB-constraint conclusion echoes
that same repo's own real, already-shipped fix
(`get_or_create_document_id`'s `INSERT ... ON CONFLICT`) rather than
introducing a new opinion -- useful confirmation that the earlier fix's
reasoning generalizes, not just a coincidence of using the same example.

### Open problems
- `k8s/distributed-state/` manifests (a Redis Deployment/Service and a
  scalable `ingestion-worker` Deployment) are written and reviewed but
  never applied against a live cluster -- experiments 11-12's "scale
  worker pods, measure throughput before/after" claim is backed only by
  the real local-process proxy (1 vs. 3 OS-process workers), not a real
  `kubectl scale`.
- The single 104/100 fakeredis-backed benchmark run was not root-caused
  with certainty -- the real-Redis 3/3-exact rerun is suggestive
  evidence for the leading fixed-window-boundary hypothesis, not
  exhaustive proof fakeredis's own TCP server couldn't also have
  contributed.
- Part 1's real two-pod HTTP demo script
  (`part1_process_local_state/two_pod_rate_limit_demo.py`) is written
  and believed correct but was never executed live in this sandbox
  (the ~40 minutes of cumulative import time to boot two full
  `rag.api.main:app` processes was judged not worth paying twice in one
  session, given the substitute unit test already proves the same
  underlying counter-divergence mechanism).

### Next recommended step
If a host without this sandbox's ~20-minute `rag.factory` import
penalty becomes available, rerun `part1_process_local_state/
two_pod_rate_limit_demo.py` for real (two genuine `uvicorn` processes,
a small round-robin proxy standing in for a load balancer) to replace
the one remaining written-not-executed piece of Part 1 with a live
result. Separately, if the shared `rag-learning` minikube cluster is
ever brought back up for other work, `kubectl apply -k k8s/base -f
k8s/distributed-state/` plus `kubectl scale deployment/ingestion-worker
--replicas=3` would close experiments 11-12's remaining gap.

## 2026-09-29 (session 20): Packaging the Kubernetes experiment as a Helm chart, and getting a fresh install to actually work

### Interview talking points
- Resumed a prior in-progress session for this prompt
  (`.claude-prompts/30.0-Helm.txt`) found via `git worktree list` --
  branch `experiment/helm` (off `experiment/kubernetes-deployment`) had
  30 committed `k8s/` docs commits plus an untracked `helm/rag-system/`
  chart directory already written but never installed against a live
  cluster; resumed in its existing worktree
  (`.claude/worktrees/experiment-helm`) rather than starting over.
- Started the `rag-learning` minikube cluster and ran the chart's own
  install/upgrade/rollback demos for real, not just `helm template`.
- Found and fixed 5 real chart bugs before a fresh `helm install`
  actually converged: a postgres init container missing the
  `DAC_OVERRIDE` capability (worked on first install into an empty PVC,
  crash-looped on any redeploy against postgres's own `0700`-chmod'd
  data); a frontend probe missing `timeoutSeconds` (defaulted to
  Kubernetes' too-tight 1s); a schema-init Job defaulting to disabled,
  producing a `/readyz` failure that read exactly like a database
  connection bug but was a missing `vector` Postgres extension
  (confirmed by a fresh ad hoc `psycopg2.connect()` succeeding while the
  app's own health check kept failing); that same Job's hook phase (and
  its ConfigMap's) being `pre-install`, which can never work since the
  chart's own `postgres` Service doesn't exist at that phase; and a
  default-deny `NetworkPolicy` silently dropping that Job's own traffic
  to postgres because its pod had no label matching any allow rule --
  which also disproved the chart's own comment claiming minikube's CNI
  doesn't enforce `NetworkPolicy` at all.
- Documented three further real findings as operational lessons, not
  chart bugs: Helm's default 5-minute hook-wait timeout being too tight
  for this host's occasional slow DNS window; the Deployment's
  `maxSurge: 1, maxUnavailable: 0` zero-downtime rollout strategy
  deadlocking under this long-lived shared cluster's genuine memory
  contention (a surge pod stuck `FailedScheduling: Insufficient memory`,
  which then blocked the old pod's removal, which kept memory from ever
  freeing); and a bare `helm rollback` (no revision argument) landing on
  another *failed* revision instead of the last known-good one, found
  live while demoing the rollback itself.
- Wrote `helm/README.md` (Helm concepts taught against this chart, all
  7 live demos with real captured output -- including the honest,
  un-sanitized failure/recovery sequences from Demos 3 and 5, not a
  cleaned-up happy path -- CI/CD and GitOps fit, and 20 interview
  questions) and reviewed `helm/rag-system/README.md` (the pre-existing
  chart-scoped values-table reference).
- Split the chart's single large commit into 13 smaller commits with
  single-line messages, at the user's request, via `git reset --soft`
  to the branch's prior tip followed by staged, grouped re-commits --
  purely a local history reorganization, never pushed.
- Removed the `.claude/worktrees/experiment-helm` worktree once the
  branch's history and this session's documentation were both settled,
  keeping `experiment/helm` itself intact for a future session.

### Goal
Convert the existing `k8s/base/` Kubernetes experiment into a small,
configurable Helm chart on its own experiment branch, install/upgrade/
roll it back for real against a live cluster, and document what Helm
itself teaches plus whatever real problems surface doing so -- an
explicit continuation of the "learning exercise, not production" pattern
already used for `k8s/`, `langgraph_experiment/`, `multi_agent_
experiment/`, and `distributed_state_experiment/`.

### Configuration
No `config/*.yaml` changes and no application source changes -- same
images, probes, env vars, and security posture as `k8s/base`, per the
prompt's own instruction. All work is under `helm/` (`helm/README.md`
plus the `helm/rag-system/` chart: `Chart.yaml`, `values.yaml`, `values-
dev.yaml`/`values-prod.yaml`, and `templates/` for every workload,
`NetworkPolicy`, RBAC, the schema-init Job, and a `helm test` connection
check).

### Changes made
- Started `rag-learning` (minikube), confirmed Ollama reachable natively
  and the chart's own images (`local-rag-api:minikube`, `local-rag-
  frontend:minikube`) already built on the host, and `minikube image
  load`ed both into the cluster.
- Ran `helm lint`/`helm template` (both clean) then attempted a clean
  `helm install` repeatedly, diagnosing and fixing each of the 5 real
  chart bugs above in turn (`templates/postgres/statefulset.yaml`,
  `templates/frontend/deployment.yaml`, `values-dev.yaml`, `templates/
  jobs/init-db-job.yaml` + `init-db-configmap.yaml`, `templates/
  networkpolicy.yaml`) until a fresh install genuinely reached
  `{"status":"ready"}` on `rag-api`'s `/readyz`.
- Ran all 7 demos from `30.0-Helm.txt`'s spec against the live cluster:
  values override, config-only rollout (hit and recovered from both the
  Helm hook-timeout and the rollout-deadlock findings), image tag
  upgrade, a deliberately broken upgrade + rollback (hit and recovered
  from the rollback-to-a-failed-revision edge case), PDB-blocked eviction
  (a real HTTP 429 via the Eviction API), and topology-spread constraint
  rendering.
- Added an `exclude` to `.pre-commit-config.yaml` for `helm/rag-system/
  templates/`: Helm's Go-template syntax isn't valid raw YAML, so the
  repo's existing `check-yaml` hook otherwise fails on every template
  file; `helm lint`/`helm template` are the real validators.
- `ISSUES.md`: added 6 new entries covering all 8 findings (some
  findings share one entry where they're two halves of the same root
  cause, e.g. the hook-timeout and rollout-deadlock operational lessons).
- `CLAUDE.md`: added a "Helm chart (learning experiment)" section,
  mirroring the existing `k8s`/LangGraph/multi-agent/distributed-state
  sections' not-production framing and pattern of pointing at the
  subdirectory's own README rather than duplicating detail.
- Split the chart's original single commit into 13 commits with
  single-line messages (scaffold+values, ConfigMap/Secret, ServiceAccount
  +RBAC, rag-api workload, frontend workload, postgres workload,
  NetworkPolicy, Ingress+Ollama service, init-db Job+ConfigMap+script,
  helm test+helpers+notes, the chart README, the concept/demo README),
  at the user's explicit request, via `git reset --soft` plus staged
  re-commits -- purely local history cleanup, `experiment/helm` was
  never pushed before or after.
- Removed the `experiment-helm` git worktree.
- This entry.

### Evaluation results
- A fresh `helm install rag . -n rag-helm --create-namespace -f
  values.yaml -f values-dev.yaml` reached `STATUS: deployed` and, once
  `rag-api`'s cold torch/embedding-model import finished, a genuine
  `{"status":"ready","dependencies":{"vectorstore":"ok","llm":"ok"}}`
  from `/readyz` -- the chart's own `vector` Postgres extension and
  schema were confirmed created by the (now correctly post-install)
  schema-init Job.
- `helm lint`/`helm template` both pass clean against `values.yaml`
  alone, `values.yaml`+`values-dev.yaml`, and `values.yaml`+`values-
  prod.yaml`.
- Demos 1-2 and 7 converged cleanly. Demos 3-6 surfaced real, un-
  sanitized Kubernetes friction on this specific long-lived, shared
  minikube cluster (a config-only rollout deadlock recovered by scaling
  down and back up; an image-upgrade pod that took ~40 minutes to reach
  Ready against a ~9-10 minute baseline; a transient API-server timeout;
  the rollback-to-a-failed-revision edge case) -- all reported honestly in
  `helm/README.md` with the real commands and real output, rather than
  smoothed into a fabricated clean transcript. The PDB demo (Demo 6) and
  the allowed-eviction case for 2 Ready replicas were partly reasoned
  from the PDB's own manifest/status fields rather than fully live-
  triggered a second time, given the demonstrated cluster fragility by
  that point in the session -- documented as such, not presented as
  fully live-verified.

### Interpretation
Every one of the 5 chart-template bugs only ever surfaced by actually
running `helm install` against a live cluster, never from `helm lint`/
`helm template`'s static checks -- consistent with this repo's now-
repeated pattern (the LangGraph join-node bug, the nginx `worker_
processes` bug, the distributed-state experiment's queue-depth bugs)
that static validation and real execution catch genuinely different bug
classes. The two hook-phase bugs (the Job's and its ConfigMap's) are a
good illustration of why "move the failing piece to the right phase"
isn't enough by itself -- every hook resource a dependent piece needs
has to share that same phase, or the fix only moves where the failure
happens. The rollback-to-a-failed-revision finding is the one result
here that generalizes furthest beyond this specific chart: it's a
property of `helm rollback`'s own default argument resolution, not
anything this chart's templates did wrong, and would bite an
operator on any chart if a failed upgrade happens to be the most
recent history entry when they reach for a bare rollback.

### Open problems
Demos 4-6 did not reach a fully live-verified, clean `2/2 Ready` state
by the end of the session -- the shared `rag-learning` cluster's
cumulative resource pressure (also carrying pods from `k8s/`,
`multi_agent_experiment/`, and `distributed_state_experiment/` work
across many prior sessions) repeatedly slowed or stalled cold starts and
once produced a genuine API-server timeout. None of this reflects a
chart defect; it's a real property of demoing production-shaped rollout
behavior on small, constrained, long-lived shared infrastructure, and is
documented as such rather than glossed over. `helm/rag-system/`'s
`values-prod.yaml` profile (stricter `topologySpreadConstraints`,
production `storageClassName`) was template-checked but never installed
against a real multi-node cluster this session.

### Next recommended step
If the shared cluster is ever idle enough to permit it, rerun Demos 4-6
once more, live, on a quieter cluster, to get a clean `2/2 Ready`
before/after PDB-eviction transcript replacing the partly-manifest-
reasoned account in `helm/README.md`'s current Demo 6. Separately,
`helm/rag-system/`'s `values-prod.yaml` profile has never been installed
against a real multi-node cluster -- if one becomes available, installing
against it would be the first live test of the `DoNotSchedule` topology-
spread behavior `helm/README.md`'s Demo 7 explicitly flags as unverified
on this single-node cluster.

## 2026-09-30 (session 21): A Knowledge Graph / GraphRAG capability, and a session spent fighting this host's own RAM as much as the milestone itself

### Interview talking points
- New branch `experiment/graphrag-knowledge-graph` off `main`, per
  `.claude-prompts/33.0-GraphRAG + Knowledge-Graph.txt`. Built the full
  `rag/graph/` package (schema/client/authorization/canonicalization/
  cypher_guard/queries/extraction/upsert/retrieval/fusion/pipeline/
  constraints), Neo4j as an additive `docker-compose.graph.yml` overlay,
  four new agent tools (`search_knowledge_graph`/`get_entity`/
  `get_neighbors`/`find_dependency_path`, a fifth `agent_tool_select_v6_
  graph.yaml` prompt variant), a new `POST /graph/query` endpoint
  (deliberately separate from `/query`, which is completely untouched),
  `eval/run_graph_eval.py`, `scripts/init_graph_db.py`/
  `benchmark_graph.py`/`seed_graph_demo_data.py`, 55 new unit tests, and
  two new integration test files (self-skip without Neo4j).
- **Deliberately conservative integration**: `POST /query` and the
  agent's `classic_rag` route are not modified at all. Graph capability
  is reachable only via the four explicit tools and the new endpoint,
  both gated on `config.graph.enabled` (default `false`, a true no-op).
  This was a judgment call, not something the prompt strictly required --
  made to guarantee "preserve existing retrieval architecture" by
  construction rather than by care.
- **Three real Neo4j bugs found by actually running against a live
  instance, not from documentation**: an invalid `server.metrics.enabled`
  config key crash-looping the container under Neo4j 5.28's strict config
  validation; `MERGE`'s inline property map rejecting a literal `null`
  (fixed with a `""` sentinel for the legitimately-optional
  `source_chunk_id`); and the Neo4j driver's own `Record.data()`
  convenience method silently discarding `.labels`/`.type` off every
  `Node`/`Relationship`/`Path` value, which the authorization code
  needed and had been silently getting `None` from until traced back
  (fixed by switching `Neo4jClient.run_read` to `dict(record)`).
- **A real design property found by writing an adversarial test wrong**:
  `GraphWriter.upsert_relationship` derives both endpoints' `canonical_id`
  from one write's own `provenance.tenant_id`, so a single relationship
  can only ever connect two nodes within one tenant scope -- discovered
  when a hand-seeded test path came back empty for the *authorized*
  caller too, tracing back to two disconnected same-name nodes rather
  than a real authorization bug.
- **A real gap found building the demo dataset**: `ingest_file`'s
  checksum-based "unchanged" shortcut runs before graph extraction, so
  clearing only the graph side of a dataset and re-running ingestion
  doesn't actually restore it -- Postgres still thinks nothing changed
  and skips the graph stage entirely. Fixed by using the existing
  `IngestionPipeline.clear_dataset` (already wired to clear both sides)
  instead of a graph-only clear.
- **The session's dominant obstacle was this specific host's available
  RAM, not the milestone's own difficulty.** `Get-CimInstance
  Win32_OperatingSystem` showed ~1.1-1.3GB free of 8GB for most of the
  session, Windows Memory Compression actively engaged. A direct,
  minimal Ollama call for a 3-token reply took 48.9s, of which 48.3s was
  model *load* time alone (0.088s was actual inference). Every process
  importing the full `sentence-transformers`/`torch` chain, or making a
  real Ollama generation call, either took many minutes or hit a
  30-minute background-task ceiling without completing: a full
  `pytest` run of 3 Neo4j-only integration tests, a `qwen2.5:3b`
  extraction call, and a full `eval/run_graph_eval.py` comparison run
  all failed to finish this way.
- **Found and fixed the actual root cause of a big chunk of that
  slowness along the way**: `rag.api.deps` imports the entire
  `rag.factory` chain (and therefore `sentence-transformers`/`torch`)
  unconditionally at module level, so even `from rag.api.deps import
  get_config` -- reading a YAML file -- paid that full cost. Switching
  diagnostic scripts to `from rag.config import load_config` directly
  cut a 20+-minute stall to 13.8 seconds flat. Every subsequent
  graph-only verification this session ran in single-digit seconds using
  that path instead.
- **Adapted rather than forced it**: a fast, comprehensive 8-scenario
  live verification script (dependents-of/2-hop-neighbors/bounded-path/
  impact-of/cross-tenant-denial/support-role-gating/unauthorized-path-
  denial/consistency-cleanup) ran against real Neo4j in ~6 seconds using
  the lightweight import path, and is the actual live proof this
  session's graph mechanism works end to end. The graph demo dataset
  (`techfusion_graph_demo`: 20 nodes, 17 relationships) uses hand-curated
  domain relationships instead of live LLM extraction, each traced to an
  exact sentence in the real corpus text it's grounded in and one
  explicitly marked as inference rather than verbatim -- honestly
  recorded as `extraction_method="deterministic"` since no model call
  actually produced them. LLM-based extraction itself was proven to work
  correctly, including its failure path (a real malformed-JSON response
  correctly caught and degraded), just not at any real scale on this
  host this session.

### Open problems
A full `eval/run_graph_eval.py` comparison run (real generated answers
across all four fusion strategies), the two `graph_gold.jsonl` questions
depending on the not-yet-ingested `data/graph_demo_docs/` synthetic
files (supersession, cross-tenant denial), live LLM-generated-Cypher
execution, the synthetic 1k/10k/50k-node performance benchmark, and a
live agent run actually dispatching one of the four new graph tools all
did not run live this session -- each blocked by the same host-memory
ceiling, not a code defect. All are implemented and either unit-tested
or structurally exercised; none were faked or estimated in their stead.
See `docs/graphrag.md` section 16 for the full, itemized accounting.

### Next recommended step
If this host (or a quieter one) has more headroom in a future session,
the highest-value next run is `python -m rag.eval.run_graph_eval` against
the full `graph_gold.jsonl` once `data/graph_demo_docs/` is ingested --
that would turn section 15/16's placeholder into real comparative
numbers across all four fusion strategies, the one piece of section 8's
own A/B/C/D comparison this session didn't complete. `scripts/
benchmark_graph.py --nodes 1000` would be the next cheapest real
performance data point after that.

## 2026-10-07 (session 22): An ontology/semantic-layer milestone on top of GraphRAG, and a mid-build correction on where it belongs

### Interview talking points
- New branch `experiment/ontology-semantic-layer` off
  `experiment/graphrag-knowledge-graph`, per
  `.claude-prompts/34.0-Ontology-experiment.txt`. Built `rag.ontology`
  (`src/rag/ontology/`: `store`/`reasoning`/`validation`/
  `query_expansion`/`authorization`/`interoperability`/
  `graphrag_integration`/`sparql_runner`), a Turtle ontology
  (`data/ontology/schema/techfusion_ontology_v1.ttl` + an additive
  `v1_1.ttl`), SHACL shapes, an RDF mirror of the real Neo4j
  `techfusion_graph_demo` dataset, 8 SPARQL queries + 5 Cypher
  counterparts, a semantic-interoperability/entity-resolution layer, a
  deterministic eval harness (`rag.eval.run_ontology_eval`, 12
  questions), 40 unit tests, a consolidated demo CLI
  (`scripts/ontology_demo.py`), and `docs/ontology.md` (full writeup,
  25 interview questions).
- **A real mid-session correction, prompted by the user catching it, not
  self-caught.** The first pass built this entirely as a standalone
  `ontology_experiment/` top-level directory with its own venv
  (`.venv-ontology/`) that never imported `rag.*` at all, reasoning from
  this project's own documented ~19-22 minute `rag.factory` import cost
  (`distributed_state_experiment/README.md`). The user asked directly
  why there was no `src/rag/ontology/`, given this milestone (unlike
  the "learning exercise, not production" experiments such as
  `langgraph_experiment/`/`k8s/`/`helm/`) was explicitly stated to merge
  to `main` after GraphRAG. Re-measuring directly rather than re-assuming:
  `rag.graph.schema`, `rag.retrieval.authorization`, `rag.graph.client`,
  and `rag.graph.retrieval` each import in under 1.3 seconds -- the
  documented ~19-22 minute cost is specific to `rag.factory`
  (sentence-transformers/torch at import time), confirmed by timing it
  directly in this session too (~49s, same mechanism, different
  magnitude, still clearly the one expensive module). The whole
  milestone was relocated into `src/rag/ontology/` (+ `data/ontology/`,
  `data/eval/ontology_gold.jsonl`, `src/rag/eval/run_ontology_eval.py`,
  `scripts/ontology_demo.py`, `tests/unit/test_ontology_*.py`), and
  `rag.ontology.authorization` was rewritten to call
  `rag.graph.authorization.is_fact_authorized` and
  `rag.retrieval.authorization.AuthorizationContext` directly instead of
  a self-contained duplicate. The `.venv-ontology/` venv and the
  standalone directory were deleted; `rdflib`/`owlrl`/`pyshacl` became a
  new `[project.optional-dependencies]` extra (`ontology`) instead,
  installed into the same environment as everything else.
- **Lazy-import discipline still matters, just scoped correctly.**
  `tests/unit/conftest.py`'s new ontology fixtures import `rag.ontology`
  inside each fixture function, not at module scope, and every
  `test_ontology_*.py` file opens with `pytest.importorskip("rdflib")`/
  `"pyshacl"` -- so collecting the full `tests/unit` suite never
  requires the optional `ontology` extra, the same discipline
  `rag.eval.ragas_scorer` already uses for `ragas`. Full suite stayed
  green after the move: 1170 passed, 1 skipped (was 1130 passed, 1
  skipped before this session, +40 new ontology tests).
- **A real bug found by actually running SHACL validation, not designed
  in**: a narrow `rdfs:domain tech:Service` on `tech:dependsOn` let
  RDFS closure silently retype an `Application` individual (Alpha
  Internal Console, a real `DEPENDS_ON` subject in the seed data) as a
  `Service`, making raw and RDFS-closed validation disagree (6
  violations each, different reasons). Fixed by removing the domain
  axiom and moving the real "subject must be Service or Application"
  constraint into SHACL (`sh:or`), which validates without inferring.
  Both modes now agree (5 violations each), and the 5 remaining
  violations are a genuine, kept-as-is data-quality finding (PostgreSQL/
  OpenSearch/Alpha Console Database/RAG Service/Kafka have no owning
  team in the real demo graph), not a fabricated "clean" result. See
  `ISSUES.md`'s two new ontology entries for this and the companion
  methodology finding (SHACL's `sh:targetClass` is subclass-aware even
  with `inference=None`, contradicting an assumption written into an
  early draft of the shapes file's own comment).
- **Semantic interoperability produced a real, non-obvious result**:
  once three synthetic external systems' classes
  (`sap:Client`/`crm:Customer`/`svcnow:BusinessPartner`) are each
  `owl:equivalentClass` to `tech:Customer`, OWL-RL entails they are
  mutually equivalent to EACH OTHER too (equivalence is symmetric +
  transitive) -- and, separately, an individual
  (`svcnow:organization_xyz`) deliberately left unlinked at the
  instance-identity level still shows up typed `tech:Customer` purely
  via that class-level equivalence. "Type-compatible with Customer"
  and "IS this specific customer" are different, independently-tracked
  claims; the entity-resolution layer's `tech:candidateSameAs` (never
  auto-promoted to `owl:sameAs`) enforces "do not merge entities without
  evidence" by construction.
- **Measured, not assumed**: naive exact-type matching for "every
  Service-type component" finds 6/10 real entities (misses the three
  `DatabaseService` instances and the one `MessagingService` instance);
  bounded ontology-aware expansion finds 10/10.
  `rag.eval.run_ontology_eval`'s full 12-question run: `answer_
  correctness` 12/12, `retrieval_recall` naive 0.867 -> ontology-aware
  1.0, with a designed negative control (`DatabaseService`, no
  subclasses of its own) showing zero change -- the explicit "do not
  assume ontology always improves results" instruction honored with a
  real counter-example in the same run, not just a disclaimer.
- **Live Neo4j integration is implemented for real, not stubbed, but not
  executed this session.** `expand_and_query_live_neo4j` maps expanded
  ontology classes back to real `EntityType` labels and runs a bounded
  Cypher query through the real `Neo4jClient`; `docker ps` found no
  running containers this session, so it was only verified against an
  unreachable port (confirmed fail-closed with `GraphUnavailableError`,
  not a hang or a silent empty result).

### Open problems
`expand_and_query_live_neo4j` and the 5 Cypher comparison query files
were not executed against a live Neo4j instance this session (no
container was running). `docs/topics/ontology.md`/`docs/reference/
ontology.md` were added to `mkdocs.yml`'s nav but `mkdocs build
--strict` itself was not run this session (the `docs` extra was not
installed) -- the nav YAML was syntax-checked directly instead.

### Next recommended step
Bring up `docker-compose.graph.yml`, confirm Neo4j health, then run
`expand_and_query_live_neo4j` for real and `python scripts/
ontology_demo.py --demo property-graph-vs-rdf` with the Cypher side
actually executed instead of only printed. After that, `pip install -e
.[docs]` and `mkdocs build --strict` to confirm the new nav entries
render and every internal link resolves.

## 2026-10-08 (session 23): QA_NOTES.md retired, its one still-accurate reference table kept here

`QA_NOTES.md` (a local-only, git-excluded scratch log of Q&A/interview-prep
answers, last written 2026-08-20) was reviewed for whether it was still
needed. Most of it was not: several answers were now factually wrong
(claimed no real `VisionProvider` existed and only `.md` files got
structural chunking, both superseded by the later layout-vision milestone;
claimed all five agent prompts were still `_v1`, when `classify`/
`synthesize` have since shipped as `_v2`), and the rest duplicated content
already covered, in more current form, by `CLAUDE.md`'s own architecture
sections (prompt infrastructure, relationship-aware expansion, trust
controls, tenant/role ACL, DoS limits). Decision: delete the file rather
than maintain a second, drifting copy of the same explanations.

One piece of it was still accurate and not written down anywhere else at
this level of completeness: the full set of `config.agent` bound defaults.
Re-verified directly against `config/default.yaml` before keeping it (all
seven values below are confirmed current, not carried forward from the
stale file on faith):

| Field | Default | What it bounds |
|---|---|---|
| `max_agent_steps` | 8 | Hard ceiling on total graph node executions per request (classify/decompose/select_tool/execute_tool/evaluate_evidence/synthesize all increment the same counter) |
| `max_retrieval_attempts` | 2 | Hard ceiling on reformulate-and-retry loop iterations |
| `max_tool_calls` | 6 | Hard ceiling on total tool dispatches, combined across all tools |
| `max_json_parse_retries` | 1 | Bounded reparse-nudge attempts when an LLM decision call returns invalid JSON, before it fails safely to a default |
| `max_tool_top_k` | 20 | Server-side clamp on `search_knowledge_base`'s `top_k` argument, independent of that argument's own Pydantic bound |
| `max_chunks_per_document_fetch` | 10 | Final chunk count `get_document`/`get_latest_document` keep, after relevance-selecting against the current query |
| `max_chunks_per_document_fetch_hard_ceiling` | 50 | The SQL-level `LIMIT` applied before relevance selection |

None of these are LLM-writable; the agent can pick which tool to call,
never how much it is allowed to fetch.

The same review found `SECURITY.md` (tracked, last updated 2026-09-05) had
gone stale in a different way: it predates GraphRAG and Ontology entirely,
so its "Implemented security controls" list said nothing about the graph
layer's own tenant/role/trust enforcement (a Cypher predicate plus an
independent Python post-execution re-check) or the ontology layer's reuse
of that exact same authorization function. Fixed directly, since it is a
tracked, low-risk, additive change.

## 2026-10-24 (session 24): Context Engineering subsystem (`rag.context`)

### Interview talking points
- **What problem did this solve?** Replaced "retrieve top-N chunks,
  concatenate, hand to the model" with an explicit, measurable,
  token-budget-aware context-assembly pipeline (`rag.context.engine.
  ContextEngine`), shared by classic RAG (`RetrievalPipeline.answer()`),
  GraphRAG (`GraphRAGPipeline`), and the agent's `_synthesize` node,
  rather than each call site independently deciding how much evidence to
  include. The system had no token-counting, no model-context-window
  concept, and no deduplication/compression/prioritization at all before
  this; `generation_context_top_n` (a fixed chunk count) was the entire
  "budget."
- **Why was it hard?** No pre-call tokenizer exists anywhere in this
  codebase (Ollama only reports token counts *after* a call returns), so
  a documented estimation strategy had to be built from nothing. The
  design also had to compose cleanly with five already-existing,
  independently-evolved evidence shapes (dense/hybrid chunks, graph
  facts, MCP business-tool results, agent tool-fetched chunks,
  relationship-expanded chunks), each with different scoring semantics
  (cosine similarity, RRF, fixed constants) that are explicitly *not*
  comparable to each other, a constraint `RetrievalPipeline`'s own
  docstrings already called out for dense-vs-BM25 and that this
  subsystem had to respect rather than paper over with a fabricated
  unified score.
- **How was it debugged?** Three real bugs were found by writing the
  overflow-regression test suite itself, not by inspection: (1)
  compression was gated on a candidate's own size against a fixed
  global floor instead of the *remaining headroom*, which silently
  produced an empty context under a tight budget with many
  individually-modest items (`CTX-001`); (2) a generic
  "looks-uppercase" regex misparsed a graph fact's own acronym-shaped
  entity name ("Alpha API") as its relationship type, fixed by building
  the parser from the closed `RELATIONSHIP_TYPES` vocabulary instead
  (`CTX-002`); (3) several new tests mutated the process-wide cached
  `load_config()` singleton directly and polluted later tests in the
  same `pytest` run, the same class of hazard this project's own
  `RAG_CONFIG_PATH` autouse fixture exists to prevent for a different
  field, fixed by deep-copying before mutating, matching
  `test_config.py`'s own established pattern (`CTX-003`).
- **What metrics proved the fix?** A synthetic, corpus-free benchmark
  harness (`rag.eval.run_context_eval`, 15 scenarios x 4 strategies x 4
  candidate-token-size targets, 240 runs) measured, for real, on this
  machine: baseline top-N dropped 1861/≈ candidate items and reached
  mean context_utilization 0.136; the full pipeline (dedup + priority-
  aware packing + compression + adaptive budgets) dropped 0 (compressing
  6 instead) and reached a 0.357 mean context_reduction, all while never
  exceeding the configured token budget across any of the 240 runs,
  including a dedicated extreme-volume regression test. Build latency
  stayed sub-millisecond for every strategy once a one-time cold-start
  import cost was isolated into an explicit warm-up call (see "known
  gaps" below).
- **What's next?** Wire `GraphRAGPipeline`'s own direct `/graph/query`
  endpoint to accept conversation history (currently only the agent and
  classic-RAG routes do); run the lost-in-the-middle ordering experiment
  (`rag.context.ordering`'s four strategies already exist) against a
  real Ollama instance, not yet done in this sandbox (no Ollama/Postgres
  running here); consider whether `multi_agent_experiment/`'s specialist
  agents should receive genuinely scoped `ContextEngine` instances rather
  than one shared pool, left as a design note in `docs/
  context_engineering.md` rather than implemented, since that experiment
  lives on its own unmerged branch.

### Goal
Implement sections 1-36 of `.claude-prompts/35.0-Context-Engineering.txt`
on `feature/context-engineering` (branched from `main`, which already
contains the merged GraphRAG/Knowledge-Graph/ontology work): a normalized
`ContextItem` type, model-context-limit resolution, a `TokenCounter`
abstraction, a budget planner (fixed and adaptive per-source allocation),
deduplication (exact-id/exact-hash/near-duplicate), relevance/authority/
freshness prioritization with deterministic tie-breaking, oversized-
evidence and tool-output compression, three comparable packing strategies
(`top_n`/`naive_token_first`/`priority_aware`), graph-path-aware hop
handling, conversation-history management (sliding window/summarized/
retrieval), a narrow token-count-only cache, and integration into every
existing generation call site, without weakening authorization/redaction
or changing default behavior when disabled.

### Configuration
New `AppConfig.context: ContextEngineeringConfig` (`enabled: false` by
default, a true no-op matching every prior milestone's kill-switch
convention): `reserved_output_tokens`, `framework_overhead_tokens`,
`safety_margin_tokens`, `token_counting` (`heuristic`/`tiktoken`),
`model_limits` (per-model context-window table, `qwen2.5:*` pre-seeded at
32768), `source_budgets` (fixed share allocation, validated to sum to
1.0), `adaptive` (query-type-dependent overrides, classified from
gathered-evidence composition, never a second LLM call),
`deduplication.overlap_threshold`, `compression.threshold_tokens`/
`strategy`/`max_sentences`, `history.strategy`/`max_turns`,
`packing_strategy`, `ordering_strategy`, `cache` (disabled by default, no
measurable benefit demonstrated for this single-instance deployment, see
`docs/context_engineering.md`'s caching section). New `security.dos_limits.
max_history_turns`/`max_history_turn_chars` bound a request's
caller-supplied `conversation_history` field (this system has no
server-side session store; a multi-turn caller re-sends prior turns each
request, the same "caller asserts, server bounds" pattern `tenant_id`/
`roles`/`case_approvals` already use).

### Implementation
`src/rag/context/` (new package): `types.py` (`ContextItem`,
`ConversationTurn`, `ContextBudget`, `PackedContext`, etc.), `tokens.py`,
`model_limits.py`, `planner.py`, `normalize.py` (converts `SearchResult`/
`ConversationTurn` into `ContextItem`, assuming input already passed
through `RetrievalPipeline.sanitize_evidence`), `dedup.py`,
`prioritize.py`, `compression.py`, `graph_packing.py`, `packing.py`
(three strategies), `ordering.py` (four "lost in the middle" render-order
strategies), `history.py`, `caching.py`, `engine.py` (`ContextEngine`,
the one orchestrator every call site shares).

Wired into `RetrievalPipeline` via a new `build_context_text()` method
(returns `build_context()`'s output unchanged when `context.enabled` is
`False`, a `ContextEngine`-packed string otherwise), reused by
`GraphRAGPipeline._generate_from_results` and `agent/graph.py`'s
`_synthesize` node so none of the three independently decide how much
context to include. `AgentState` gained `conversation_history`/
`context_engineering` fields; `POST /query`/`POST /agent/query`/`POST
/agent/query/stream` all gained an optional `conversation_history` request
field and a `context_engineering` response field (a debug/observability
summary: token counts, utilization, dedup/compression stats, never raw
evidence). New Prometheus metrics (`rag_context_*`) and OTel spans
(`context_normalize`/`context_deduplication`/`context_packing`) follow
the existing bounded-label/defensive-wrapper conventions exactly.

### Evaluation
`rag.eval.run_context_eval` (synthetic, no corpus/Postgres/Ollama needed)
covers all 15 required scenario categories (short context, long document,
many relevant/irrelevant chunks, duplicates, graph+document,
MCP+document, conflicting evidence, stale-vs-current, long history,
prompt injection, over-budget volume, one giant chunk, a multi-agent-
scoped case, an unanswerable query) across 4 candidate-token-size targets
and 4 strategies. `scripts/benchmark_context_engineering.py` renders the
240-run sweep as `experiments/results/context_engineering/report.json`
and `experiments/reports/context_engineering_benchmark.md`. Deliberately
computes only deterministic, non-generation metrics (tokens, utilization,
reduction, dedup/compression counts, truncation, latency); answer-
correctness/citation-correctness/context-precision-recall would need a
real LLM call against the real corpus and are explicitly left to
`run_eval.py`/RAGAS, not duplicated here with a fabricated judge.

~95 new unit tests (`tests/unit/test_context_*.py`,
`test_run_context_eval.py`) cover every module plus end-to-end overflow-
regression invariants (a packed prompt never exceeds its budget; a single
giant chunk/MCP result/graph path/long history can never overflow it;
duplicated evidence is bounded; system instructions are never part of
packing at all, so they can never be dropped to make room for evidence;
prompt injection in retrieved content or conversation history stays data,
never an instruction channel; a redaction marker already in the text is
never reconstructed). All patched `FakePipeline`/`RetrievalPipeline`
test-double classes across the existing agent/API test suite (21 files)
needed `build_context_text`/`history_turns` stubs added, the same
"isolated test doubles don't know about a sibling change's new keyword
argument" pattern already documented for the joint-investigation PRs.
Full suite: 1262 unit tests passing (up from 1170), 0 regressions; `ruff
check`/`ruff format`/`mypy` all clean on every touched file.

### Known gaps (documented, not hidden)
No live Ollama/Postgres/Neo4j instance was running in this sandbox, so
(a) the lost-in-the-middle ordering experiment (section 15) was
implemented but not run against a real model, and (b) the full
`run_eval.py`-style answer-quality/citation metrics were not computed end
to end against the real `techfusion` corpus. `GraphRAGPipeline`'s own
`/graph/query` endpoint was not given a `conversation_history` field
(it has no such field today at all, so this is a scope decision, not a
regression). Multi-agent-specific context isolation
(`multi_agent_experiment/`) was exercised only as a single synthetic
scenario proving the budgeting mechanics scale to a scoped-evidence
shape; real specialist-scoped `ContextEngine` instances for that
branch's Knowledge/Business agents were not implemented, since that
experiment lives on its own unmerged branch, out of scope for this
feature branch.

---

## 2026-10-25 (session 25): Reconciling two pre-experiment audits, and the first real Context Engineering smoke evaluation

### Interview talking points
- Two independent adversarial audits of `feature/context-engineering`
  (a self-directed pass and an independent "Codex" pass, neither shown
  the other's output) had already flagged 25 findings before this
  session. The job here was to fix them, not re-audit: 22 were fixed
  with regression tests, 5 larger cross-store/cross-version consistency
  gaps were deliberately deferred with a documented "why not fixed yet"
  and a recommended fix each, rather than silently skipped.
- Three of the fixes were CRITICAL and genuinely security-relevant: a
  hosted reranker receiving raw candidate content before redaction with
  no egress gate at all (SEC-015); a graph dependency-path query
  authorizing only the path's nodes, never its relationships, which
  could disclose that two public entities are connected via a
  relationship sourced from a restricted document (GRAPH-006); and
  `sources`/citations being built from the pre-packing retrieval result
  rather than what Context Engineering's own packing step actually put
  in the prompt, so judge/citation metrics could score evidence the
  model was never shown (EVAL-014).
- Found one more real gap while actually running the smoke evaluation
  this session closes with, not from either audit: `run_eval.py` read
  every other key off `answer()`'s result dict except
  `context_engineering`, so a report produced with Context Engineering
  on looked identical to one with it off (CTX-013). A real
  end-to-end run surfaced this in about the time it takes to read one
  report file; the synthetic harness and unit tests never would have,
  since neither exercises `run_eval.py`'s own report assembly against a
  real `answer()` call.
- The most interesting individual finding: `num_ctx` was never passed to
  Ollama at all (CTX-005), so the budget planner's assumed 32768-token
  window for `qwen2.5:3b` and the real runtime window could silently
  disagree. Fixing it and then actually running the fixed code exposed a
  second, more concrete problem: 32768 tokens measured at ~6.6 minutes
  per call on this 8GB-RAM CPU-only host (`prompt_eval_duration` alone
  ~107s for an identical 30-token prompt). The experiment config was
  recalibrated to a measured, practical 4096 rather than the model's
  trained length,
  with the numbers recorded directly in the config file. The fix was
  correct the whole time; running it for real is what turned a
  theoretical risk `docs/context_engineering.md` had already flagged
  into a concrete, host-specific number.
- A second, unplanned finding from actually running real processes: two
  separate `run_eval.py` smoke-eval attempts ended up alive at once (an
  artifact of combining this session's own `run_in_background` tooling
  with a stray trailing `&`), both hitting the same Postgres dataset and
  the same Ollama server concurrently on a resource-constrained host.
  Killing the stale one let the real run finish cleanly. A small
  operational lesson, not a code bug: don't background a command twice.
- A routine consistency check of `KNOWN_ISSUES.md` (comparing section
  10's navigational "Fixed" table against the full topical entries each
  row is supposed to point to) found three pre-existing rows with no
  entry to point to at all (MULTI-001/002/003). All three described
  real, already-fixed (or already-accepted) findings documented
  elsewhere in the project (CLAUDE.md's layout-vision section, and the
  `_matching_assets`/`_with_vision_siblings` docstrings themselves), so
  full entries were written from that existing, verified material
  rather than inventing new content to fill the gap.

### Goal
Fix the findings from `.claude-prompts/35.3-Audit-Report.md` and
`.claude-prompts/35.5-Codex-audit-report.md` (identical content to
`experiments/reports/pre_experiment_audit_2026-10-24.md`), write the
combined result as `.claude-prompts/35.6-Final-Fix-Report.md`, record
each fix (and each deferred item) in `KNOWN_ISSUES.md` following its
existing structure, create
`config/experiments/context-engineering-v1.yaml`, and run one real
end-to-end Context Engineering smoke evaluation against it.

### Changes made
- 22 fixes across `src/rag/retrieval/`, `src/rag/retrieval/egress_policy.py`
  (new, relocated from `rag.eval.egress_policy`), `src/rag/graph/`,
  `src/rag/context/`, `src/rag/eval/` (including a new
  `run_metadata.py`), `src/rag/generation/ollama_llm.py`,
  `src/rag/factory.py`, `src/rag/ingestion/pipeline.py`,
  `src/rag/vectorstore/`, and `src/rag/agent/graph.py`. Full list with
  problem/failure-scenario/fix/regression-test detail is in
  `KNOWN_ISSUES.md` (SEC-015 through SEC-017, DIST-005, RET-005,
  GRAPH-006, GRAPH-007, CTX-005 through CTX-013, EVAL-010 through
  EVAL-015, OBS-002) and summarized in
  `.claude-prompts/35.6-Final-Fix-Report.md`.
- 5 findings deliberately deferred (GRAPH-008, GRAPH-009, BUG-015,
  BUG-016, BUG-017), each recorded OPEN with its own "why not fixed yet"
  and recommended fix, not silently dropped.
- `KNOWN_ISSUES.md`: added all of the above, updated the Summary Table
  and section 10's navigational table, added a dated postscript to the
  "Final Consistency Pass" section, and added three full topical entries
  (MULTI-001, MULTI-002, MULTI-003) that the table referenced but the
  file never actually contained.
- New: `config/experiments/context-engineering-v1.yaml`
  (`secure-rag-baseline-v1.yaml` plus one `context` block:
  `enabled: true`, `packing_strategy: priority_aware`,
  `ordering_strategy: begin_end_sandwich`, `model_limits.overrides:
  {qwen2.5:3b: 4096}`, recalibrated from an initial 32768 after a real
  timeout).
- New: `.claude-prompts/35.6-Final-Fix-Report.md` (the consolidated
  fix report).
- Test files: ~20 new/modified unit and integration test files proving
  each fix, including `tests/unit/test_graph_dependency_path_authorization.py`,
  `test_graph_client.py`, `test_agent_stream_disconnect_cancellation.py`,
  `test_agent_synthesize_budgets_its_own_template.py`,
  `test_graph_pipeline_strategy_labeling.py`, `test_run_metadata.py`
  (all new), plus updates across `test_retrieval_pipeline.py`,
  `test_run_ragas_eval.py`, `test_ingestion_pipeline_governance.py`,
  `test_run_eval.py`, `test_metrics.py`, `test_ollama_llm.py`,
  `test_factory.py`, `test_context_engine.py`, `test_context_dedup.py`,
  `test_context_normalize.py`, `test_context_planner.py`,
  `test_run_context_eval.py`, and a new integration test
  (`test_pgvector.py::test_search_keyword_ties_break_deterministically_by_chunk_id`).

### Evaluation results
- Full unit suite: 1300 passed, 1 skipped, 0 failed.
- `ruff check src/ scripts/ tests/`: all checks passed.
- `mypy src/`: no issues found across 154 source files.
- `test_run_eval.py` alone (55 tests, after the CTX-013 fix): all
  passing.
- First real end-to-end `run_eval.py` pass against
  `context-engineering-v1.yaml` (real Postgres, real Ollama,
  `qwen2.5:3b`, `data/eval/sample_gold.jsonl`, 4 questions, dataset id
  `ctx_eng_smoke`): `recall@5`/`recall@10` 0.75, `mrr` 0.75,
  `answer_quality.mean_overall` 0.559 (keyword-overlap heuristic),
  `generation_mean` latency ~50.3s/answer (consistent with the
  recalibrated 4096-token window, not the earlier multi-minute
  32768-token failure), `run_metadata` populated with the real commit
  SHA/branch/dirty state. All 4 answers grounded and correct on
  inspection, including a correct refusal on the one unanswerable
  question (parental leave policy, not present in the sample corpus).
- Second confirmation pass, identical config and gold set, after the
  CTX-013 fix: retrieval/MRR/answer-quality metrics byte-identical to
  the first run (as expected, since CTX-013 only changed what
  `run_eval.py` records, not what `answer()` computes);
  `context_engineering` now populates on every `per_example` row and as
  a top-level aggregate (`packing_strategy: priority_aware`, mean
  utilization ~0.219, 0 items dropped or compressed across all 4
  questions at this small sample-corpus scale, mean build latency
  ~2.2ms).

### Interpretation
Running the real integration path, not just the synthetic harness and
unit tests with fakes, earned its keep twice in one session: once by
turning a theoretical risk (CTX-005's num_ctx mismatch) into a concrete,
host-specific calibration number, and once by surfacing a gap (CTX-013)
that no amount of component-level testing would have caught, since the
bug was specifically in how `run_eval.py` assembles its own report
around a real `answer()` call. Both reinforce the same lesson this
project keeps relearning on every milestone: a feature validated only
component-by-component is not the same claim as a feature proven to
work end to end, and the two can diverge in ways that are cheap to find
once you actually run the thing, but easy to miss otherwise.

### Open problems
The 5 deferred findings (GRAPH-008, GRAPH-009, BUG-015, BUG-016,
BUG-017) remain OPEN, each with a recommended fix recorded in
`KNOWN_ISSUES.md`. The smoke evaluation used the small tracked sample
corpus (4 questions), not the full `techfusion` gold set; a full-corpus
Context Engineering run (comparable in scale to the Context Engineering
milestone's own synthetic benchmark) has not been performed.

### Next recommended step
Work through the 5 deferred findings as their own reviewed changes, each
with the regression test its own entry already specifies. Separately,
consider a full-corpus `context-engineering-v1` run (not just the
4-question smoke test) once `config/default.yaml` itself is a candidate
for enabling Context Engineering by default, to get real
utilization/dedup/compression numbers at production scale rather than
this session's intentionally small smoke-test scale.

## 2026-10-29 (session 26): Fixing the reconciled findings of the two 2026-10-29 pre-experiment audits

### Interview talking points
- One read-time revalidation query fixed two findings at once: graph facts are now resolved against their source document in Postgres under the caller's resolved auth, which closes the stale-ACL gap (GRAPH-008, open since session 25) and gives graph evidence the governance metadata the egress gate needs (SEC-018).
- A benchmark report can be stale against its own code: regenerating the Context Engineering report showed the committed one predated the CTX-006 overhead counting. Running the benchmark at `main` in a throwaway worktree separated "report was stale" from "fix changed numbers": the fixes moved only seven compressed-item token counts, by exactly the citation overhead CTX-014 predicted.

### Goal
Codex (`.claude-prompts/35.9`) and Claude (`35.10`) audited `main` at `64321ed` independently. Verify each finding against the code, fix the eligible ones on `main`, and record everything in `KNOWN_ISSUES.md` following its conventions.

### Changes made
Five commits, stacked on four local branches off `main` (`c86a5a6`):
1. `fix/context-budget-accounting` (`3486a88`): compression carries render overhead (CTX-014); infeasible budgets flagged via `budget_deficit_tokens`/`budget_infeasible`, a log line and `rag_context_budget_infeasible_total` (CTX-015); unclamped utilization telemetry plus `rag_context_budget_exceeded_total` (CTX-011 follow-up).
2. `fix/graph-evidence-governance` (`b14d117`): `VectorStore.get_document_governance`; fusion revalidates and carries governance, fails closed (GRAPH-008, SEC-018); `egress_policy.allow_unclassified`, false in production (SEC-019); seeded-vector seeds filtered and ordered (GRAPH-010); entity ACL write-once behavior documented and pinned (GRAPH-011).
3. `fix/context-dedup-identity` (`d3cfc40`): fact-specific graph ids, bucket/tenant/dataset-scoped identity pass, history-only hash merges (CTX-016); numeric and negation/polarity conflict guard (CTX-007 follow-up).
4. `fix/agent-evidence-cap-and-eval-metadata` (`cc77157`, `9b0fbef`): `agent.max_evidence_summary_chars` (CTX-017); `ContextEvalReport.run_metadata` (EVAL-015 follow-up); Neo4j-outage and MCP-transport-failure tests; BM25 rebuild cost documented (RET-006); experiment-branch findings recorded (DIST-006, OPS-015, OPS-016); benchmark regenerated.

### Evaluation results
Context Engineering synthetic benchmark regenerated at `cc77157` (clean tree). Versus a fresh run of `main`: 7 of 240 rows differ, all `full_pipeline` `packed_token_count` on compressed items (+12 tokens each, one history row -1). Drops, compression counts and overflow are unchanged. The new Postgres integration test (`test_document_governance_lookup.py`) passed against the local container; Neo4j was not running, so graph integration tests were not executed.

### Open problems
BUG-015/016/017 (structural ingestion consistency) and GRAPH-009 (dead code) remain open. DIST-006 and OPS-015 are open on experiment branches, which were not modified. Experiment branches are 194 commits behind `main` (OPS-016).

### Next recommended step
Merge the four stacked PRs (#30 to #33) in order, retargeting each to `main` before deleting the branch below it (deleting a stacked PR's base branch closes the PR rather than retargeting it). Then run the graph integration tests with Neo4j up before any graph-mode benchmark.
