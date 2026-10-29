# Known Issues

This file tracks confirmed defects, security/reliability risks, architectural
limitations, production-hardening gaps, and known constraints discovered
through implementation, testing, evaluation, and audits of this local RAG
system. It is a reformatted, structured version of the project's original
interview-prep issue log; no issue was deleted, invented, or re-severitized
without basis. Detailed chronological session history remains in
`PROJECT_JOURNAL.md`.

## Status Legend

- **OPEN**: confirmed, reproducible, not yet fixed, no accepted-limitation decision made.
- **IN PROGRESS**: partially fixed; a known sub-gap remains, tracked explicitly below.
- **FIXED**: a concrete code/config/test change closed the issue; evidence is cited.
- **ACCEPTED LIMITATION**: understood, deliberately not fixed (scope, cost, or a documented tradeoff), not hidden.
- **NEEDS VERIFICATION**: evidence of a possible issue exists, but it has not been reproduced or proven.

## Severity Legend

- **CRITICAL**: silent data loss/corruption, an authorization/security bypass that could expose protected data, or a production-breaking failure with no safeguard.
- **MAJOR**: a real, reproducible correctness or security defect with measurable impact; required a structural fix.
- **MODERATE**: a real defect or gap with narrower scope, a measurement artifact that misled a decision, or an infrastructure/tooling issue with a workaround.
- **MINOR**: a tooling/methodology finding, a documented tradeoff, or an edge case with negligible practical impact.

## Summary Table

| ID | Severity | Status | Area | Title |
|---|---|---|---|---|
| RET-001 | MODERATE | ACCEPTED LIMITATION | Retrieval | Retrieval window too narrow for one document |
| RET-002 | MINOR | ACCEPTED LIMITATION | Retrieval | Embedding and chunk-size changes each traded one metric for another |
| RET-003 | MINOR | ACCEPTED LIMITATION | Retrieval | Hybrid retrieval's headline recall hid how each retriever contributed |
| RET-004 | MODERATE | ACCEPTED LIMITATION | Retrieval | A cross-encoder changed answers without changing retrieved evidence |
| RET-005 | MAJOR | FIXED | Retrieval | BM25 ties depended on unspecified SQL row order |
| BUG-009 | MAJOR | OPEN | Generation | Retrieval was correct, but generation fabricated the answer |
| BUG-010 | MODERATE | ACCEPTED LIMITATION | Generation | A larger generation model trades hallucination for over-refusal |
| EVAL-007 | MODERATE | ACCEPTED LIMITATION | Evaluation | A keyword-overlap answer metric penalized correct refusals |
| EVAL-009 | MINOR | ACCEPTED LIMITATION | Evaluation | Generation dominated latency, and the latency A/B could not be trusted |
| AGT-004 | MAJOR | IN PROGRESS | Agentic RAG | The tool worked, but the agent rarely chose it (specialized-tool selection) |
| AGT-005 | MODERATE | NEEDS VERIFICATION | Agentic RAG | Tool-selection prompt's chunk_id fix is unverified against live tool selection |
| SEC-007 | MODERATE | NEEDS VERIFICATION | Security | Upload atomicity fix: crash window between DB write and file install unverified |
| SEC-008 | MINOR | NEEDS VERIFICATION | Security | Cross-tenant upload isolation integration tests were written but never run |
| OBS-001 | MAJOR | IN PROGRESS | Observability | `/health` depends on a heavy pipeline dependency through a graph check |
| MULTI-002 | MINOR | ACCEPTED LIMITATION | Multimodal | A working vision provider did not improve answers, because the model was weak |
| GRAPH-005 | MODERATE | ACCEPTED LIMITATION | GraphRAG | LLM graph extraction worked correctly, but was impractical on a memory-starved host |
| PROD-001 | MODERATE | ACCEPTED LIMITATION | Infrastructure | Durable workflow state did not make the side effects it triggered durable |
| PROD-002 | MINOR | ACCEPTED LIMITATION | Data | Re-spelling the same ingestion root churns document identity |
| PROD-003 | MINOR | ACCEPTED LIMITATION | GraphRAG | Resetting only the graph side of a dataset needs a dataset-level clear, not a graph-only one |
| CTX-004 | MINOR | ACCEPTED LIMITATION | Context Engineering | Extractive compression needs sentence boundaries; graph-path hops are a best-effort text parse |
| BUG-015 | MAJOR | OPEN | Ingestion | A graph write failure after a successful vector commit has no durable retry |
| BUG-016 | MAJOR | OPEN | Ingestion | Replacing or removing a referenced image does not invalidate the document that references it |
| BUG-017 | MAJOR | OPEN | Ingestion | A staged upload can resolve vision assets against the wrong document version |
| GRAPH-008 | MAJOR | OPEN | GraphRAG | A stale graph fact is never revalidated against its source's current Postgres state |
| GRAPH-009 | MODERATE | OPEN | Ontology | A live-Neo4j ontology query helper has no authorization check at all (currently unreachable) |

*Fixed issues (the majority of this log, ~65 entries) are in [section 10](#10-fixed--closed-issues) and summarized separately there, not repeated above.*

---

## 1. Confirmed Bugs

### BUG-001: Two datasets shared one vector table and leaked into each other's results

**Status:** FIXED
**Severity:** MAJOR
**Area:** Data / retrieval isolation
**Affected code:** `VectorStore`, `dataset_id` scoping, `delete_document()`

**Problem**
A smoke-test collection (`sample_docs`) and the real knowledge base (`techfusion`) lived in the same Postgres table, separated only by file path, with no enforced namespace.

**Failure scenario**
An adversarial fixture with two near-identical documents, differing only in one fact value, ingested into two namespaces: an unfiltered query returned the wrong one with roughly equal probability. Integration tests also left orphaned rows, since the only cleanup method was designed for re-ingestion, not full teardown.

**Impact**
Silent cross-dataset answer contamination; no error, just a wrong but plausible-looking result.

**Fix applied**
`dataset_id` became required on every chunk and part of a composite document key, `(source, dataset_id)`. Every evaluation run injects its `dataset_id` filter unconditionally. A dedicated `delete_document()` handles teardown.

**Regression test**
The adversarial two-namespace fixture now passes; a teardown test confirms no rows remain after `delete_document()`. Recall@5 0.859→0.891 and MRR 0.830→0.847 after removing the contaminating dataset confirmed the leak was real, not just closed in theory.

---

### BUG-002: A connection leak that only a fresh, uninitialized database could trigger

**Status:** FIXED
**Severity:** MAJOR
**Area:** Database / connection pool
**Affected code:** Postgres connection-pool checkout helper, `register_vector(conn)`

**Problem**
CI's Docker-build job failed with `psycopg2.pool.PoolError: connection pool exhausted` on `/health`; the same check never failed locally.

**Failure scenario**
A helper checked a connection out of the pool, then called `register_vector(conn)`. The `try/finally` returning the connection wrapped only the code *after* the helper returned, so if `register_vector()` raised, the connection was never returned. CI's brand-new, schema-less Postgres container made `register_vector()` raise on every health check, leaking one connection each time until the 5-connection pool was exhausted.

**Impact**
CI-only, non-local-reproducible pool exhaustion; the underlying bug class ("something can throw between checkout and return") could recur anywhere a similar pattern is written.

**Fix applied**
A single context manager now wraps the whole checkout, including `register_vector()`; every pool user goes through it. The CI job also runs schema init first.

**Regression test**
A test that reverted the fix failed as expected (2 of 6 new pool tests); with the fix, all 6 passed. 15 health checks against an uninitialized DB returned `200 degraded` (not `PoolError`); 15 against an initialized DB returned `200 ok`.

---

### BUG-003: A new metadata column never reached the database

**Status:** FIXED
**Severity:** MODERATE
**Area:** Data / persistence
**Affected code:** `VectorStore` INSERT column list, read path, `page` metadata field

**Problem**
`page_localization_accuracy` read 0.0 on all 28 questions even though recall was 93% and the loader emitted correct page numbers.

**Failure scenario**
The field was set on the chunk and passed through the writer, but the store's INSERT column list and its read path never included it. Postgres silently stored NULL; nothing raised, since the column list and value tuple are both positional.

**Impact**
A schema field appeared to exist end-to-end in code but was never actually persisted; only a metric run against a real database caught it, since in-memory tests cannot see this class of bug.

**Fix applied**
Added the column to the INSERT, the conflict-update clause, and the shared read path. Re-ingested the corpus.

**Regression test**
`page_localization_accuracy` went 0.0 → 0.82; integration tests against real Postgres stayed green.

---

### BUG-004: A UNIQUE constraint on a nullable tenant column would not deduplicate anonymous submissions

**Status:** FIXED
**Severity:** MODERATE
**Area:** Data / feedback store
**Affected code:** Feedback dedup key, `tenant_id` column

**Problem**
Feedback is deduplicated per caller and per answer. The natural key includes `tenant_id`, which was nullable, giving a no-tenant caller a NULL key component.

**Failure scenario**
In SQL, every NULL is distinct, so a unique constraint never fires for two rows with a NULL key: two anonymous submissions for the same answer would each insert a separate row. Invisible in any test that always supplied a tenant.

**Impact**
Duplicate feedback rows for anonymous callers, undermining the dedup guarantee.

**Fix applied**
The column is `TEXT NOT NULL DEFAULT ''`; the application maps the empty string to `None` on read.

**Regression test**
Integration test against real Postgres: two anonymous submissions for different runs stay separate; a repeat from the same caller/run updates the existing row.

---

### BUG-005: Re-ingesting a changed file could leave a document permanently empty

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Data / ingestion transactions
**Affected code:** Store's checksum/chunk-replace write path, `get_or_create_document_id`

**Problem**
Re-ingesting a changed file ran three separate database transactions: update the checksum, delete the old chunks, insert the new chunks.

**Failure scenario**
A failure between transactions left the checksum updated but the chunks empty. The next run saw an unchanged checksum and skipped the file entirely, leaving the document permanently empty with no error. A related race: a first-time insert from two concurrent ingestions could raise a unique-violation error.

**Impact**
Silent, permanent data loss for an ingested document with no indication anything was wrong, the single most severe item in this log.

**Fix applied**
One store method now writes the checksum, deletes old chunks, and inserts new chunks inside a single transaction, so a mid-sequence failure rolls all three back. `document_id` creation uses `INSERT ... ON CONFLICT DO NOTHING` followed by a select, removing the race.

**Regression test**
A permanent fault-injecting test reproduces the mid-sequence failure and proves the transaction rolls back cleanly. This was one of five fix batches later merged into `main`.

---

### BUG-006: A decorator captured its rate limiter at import time, so tests could not swap it

**Status:** FIXED
**Severity:** MODERATE
**Area:** API / testability
**Affected code:** `@_limiter.limit(...)` route decorator

**Problem**
A rate-limit test set a 2-per-minute limit via config and a monkeypatched limiter, but the third request still returned 200, not 429.

**Failure scenario**
The route decorator binds to the limiter object that existed at module-import time; replacing the module attribute later never changes the already-decorated function. A decorator is not a live injection point.

**Fix applied**
Rate-limit config is read once at process start (same pattern as other singletons). Tests build a small app with the same limiter/middleware wiring rather than trying to reconfigure the production app's already-bound decorator.

**Regression test**
Two deterministic tests: one proves the mechanism, one proves the shipped default never blocks.

---

### BUG-007: `top_k=0` silently became the default, and a negative value reached SQL

**Status:** FIXED
**Severity:** MAJOR
**Area:** API / input validation
**Affected code:** `candidate_k or default` fallback, DoS-limit check

**Problem**
The API rejected `top_k` above the maximum, but not zero or negative values.

**Failure scenario**
`0` is falsy in Python, so `candidate_k or default` silently replaced an explicit `0` with the default value instead of raising. A negative `top_k` had no guard at all before reaching the SQL `LIMIT` clause.

**Impact**
A caller's explicit, valid "zero results" request was silently overridden; a negative value could have reached the database layer unvalidated.

**Fix applied**
The shared DoS check rejects any value below 1 with a 422 before retrieval runs. The internal `or default` fallback is kept for other callers that rely on it intentionally.

**Regression test**
Tests check both zero and negative values return 422 with zero pipeline calls made.

---

### BUG-008: A streaming endpoint validated requests after its 200 status was already sent

**Status:** FIXED
**Severity:** MAJOR
**Area:** API / streaming
**Affected code:** `POST /agent/query/stream`, `StreamingResponse`

**Problem**
`POST /agent/query/stream` ran its validation inside the async generator body.

**Failure scenario**
A generator's body only executes on its first pull, but `StreamingResponse` sends its 200 status *before* that first pull. An oversized request therefore got a committed 200 with an empty stream, while the equivalent non-streaming route correctly returned 422.

**Fix applied**
Validation moved into a dedicated FastAPI dependency, resolved before `StreamingResponse` is constructed, and declared *before* four heavy singleton dependencies (pipeline, store, embedder, LLM) in the route signature, so a rejected request resolves none of them.

**Regression test**
Oversized requests now return 422 with zero pipeline calls; a valid request still streams normally.

---

### BUG-013: Recall@10 was structurally unmeasurable

**Status:** FIXED
**Severity:** MODERATE
**Area:** Evaluation
**Affected code:** `retrieve()`'s truncation to `rerank_top_n`

**Problem**
Recall@10 always equalled Recall@5; a suspiciously clean result.

**Failure scenario**
`retrieve()` truncated its output to `rerank_top_n` (default 3) regardless of the `top_k` the caller actually requested. Evaluation asked for 10 and silently received 3.

**Fix applied**
Added an explicit override so evaluation could request a broader candidate set, while production `answer()` kept its real, narrower value.

**Regression test**
Recall@5 and Recall@10 became distinct (0.859 vs. 0.957 on the first real measured run).

---

### BUG-014: A RAGAS metric returned null for every question without raising an error

**Status:** FIXED
**Severity:** MODERATE
**Area:** Evaluation / RAGAS integration
**Affected code:** `ragas_scorer.py` result-column lookup

**Problem**
Two new RAGAS metrics (noise sensitivity, factual correctness) reported as computed with no failures, but every score was `null`.

**Failure scenario**
Score extraction assumed a metric's result column matched its bare name; ragas 0.3.9 bakes the metric's `mode` into the column name (e.g. `noise_sensitivity(mode=relevant)`). The lookup silently returned nothing, and the code treated a missing column the same as a genuinely missing metric.

**Fix applied**
Lookup now tries an exact match, then a prefix match on `<name>(`. A unit test uses the real suffixed column names (the earlier tests used exact-match fake data, which is why they'd passed).

**Regression test**
Both metrics now produce real scores; the re-run hit the judge cache on 55% of calls, so the earlier paid calls were not wasted.

---

### BUG-015: A graph write failure after a successful vector commit has no durable retry

**Status:** OPEN
**Severity:** MAJOR
**Area:** Ingestion / GraphRAG consistency
**Affected code:** `IngestionPipeline._run_graph_extraction`, `ingest_file`'s unchanged-document shortcut

**Problem**
Postgres commits before graph extraction runs; a graph-extraction failure is caught and logged, never re-raised, which correctly protects the vector write but leaves no record that this document's graph projection is now missing or stale. The checksum-unchanged shortcut then returns before extraction on every later re-ingest of the identical bytes, so there is no path back to repair it short of a full dataset clear.

**Failure scenario**
Ingest while Neo4j is unavailable, restore Neo4j, then re-ingest the identical file: the vector corpus looks complete and ingestion reports success, but graph evidence for that document stays permanently absent until someone notices and clears the whole dataset.

**Why not fixed yet**
Closing this needs a durable graph-projection-status column plus an independent retry path that doesn't depend on the vector checksum changing; a structural ingestion change, not a local patch, and out of scope for this pass.

**Recommended fix**
Track graph projection status/checksum durably (e.g. a `graph_projected_checksum` column alongside the existing vector checksum), and retry pending projection independently of whether the source bytes changed.

---

### BUG-016: Replacing or removing a referenced image does not invalidate the document that references it

**Status:** OPEN
**Severity:** MAJOR
**Area:** Ingestion / multimodal consistency
**Affected code:** `ingest_file`'s checksum gate, `Writer`'s vision-sibling-chunk step

**Problem**
The document checksum covers only the source file's own bytes; image checksums and the vision-description cache lookup only run *after* that unchanged-document gate, so they never run at all when the gate already returned early.

**Failure scenario**
Replace or remove an image referenced by an otherwise-unchanged Markdown file, then re-ingest the Markdown: `changed=False`, and the old vision description / image-derived evidence is retained regardless of what the image now actually is (or that it no longer exists).

**Why not fixed yet**
Including referenced-image state in the unchanged-check would require parsing the document to discover its images *before* the cheap checksum gate that exists specifically to avoid that cost on every re-ingest; a real tradeoff, not yet resolved.

**Recommended fix**
Fold referenced-asset digests (or at least existence) into the document's lineage/checksum, or add a separate, narrower invalidation path for image-derived chunks specifically.

---

### BUG-017: A staged upload can resolve vision assets against the wrong document version

**Status:** OPEN
**Severity:** MAJOR
**Area:** Ingestion / multimodal / upload atomicity
**Affected code:** `api/routers/ingest.py`'s staged-upload flow, `Writer`'s image-path resolution

**Problem**
PDF/DOCX image extraction resolves asset paths beside the *staging* file; `source_override` only renames the final persisted `source` string after `ingest_file` already ran, and staged assets are installed into the real `UPLOAD_DIR/assets/` only once `ingest_file` returns successfully.

**Failure scenario**
On a document's first upload, vision resolution can find no final-location asset yet and skip the image. On a same-filename replacement, resolution can instead find and describe the *previous* version's final asset while the new document's own checksum is being committed, mismatching vision-generated text to the wrong document version.

**Why not fixed yet**
Fixing this cleanly means separating physical asset resolution (which legitimately needs a real path to read bytes from) from stable citation identity more thoroughly than the current staging-directory-only fix (which already solved the adjacent temp-path-leak bug, see **SEC-006**); judged out of scope for this pass.

**Recommended fix**
Resolve and process vision assets entirely from the staged location during ingestion, independent of the final installed path, and reconcile only identity/citation strings afterward.

---

### MULTI-001: Shared asset folders paired the wrong images with the wrong documents

**Status:** FIXED
**Severity:** MAJOR
**Area:** Ingestion / multimodal
**Affected code:** `loaders/base.py:resolve_image_asset`, `_matching_assets`

**Problem**
A category folder can hold more than one source document, each contributing images to one shared sibling `assets/` directory. Pairing embedded images with that folder's files purely by position silently handed every document in the folder the *first* document's images.

**Failure scenario**
Three review documents under one `operations/` category, each with its own two images in a shared `operations/assets/`: every document's citations and vision descriptions pointed at the first document's images instead of its own.

**Fix applied**
`_matching_assets` narrows candidates to the ones whose filename shares a non-stopword word with the source document's own filename (this project's asset-naming convention), falling back to positional pairing only when no candidate matches any word.

**Regression test**
`test_resolve_image_asset_matches_by_shared_filename_word` and `test_resolve_image_asset_falls_back_to_all_candidates_when_no_word_matches` in `tests/unit/test_loaders.py`.

---

### MULTI-002: A working vision provider did not improve answers, because the model was weak

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Multimodal
**Affected code:** `vision.provider: ollama` (`moondream`), the A/B/C layout-vision evaluation

**Problem**
The B (layout-aware, no vision) to C (layout-aware plus vision) comparison against the 28-question layout/vision gold set showed vision descriptions correctly retrieved and cited (`visual_evidence_support_rate` 0.0 to 1.0) and fewer hallucinations on unanswerable-visual questions, but no improvement in `multimodal_answer_quality` on answerable vision-required questions (0.161 to 0.144).

**Failure scenario**
Not a defect: direct inspection of `moondream`'s cached description for the clearest regression showed it echoing the image's own caption ("Figure 2") instead of describing the image's actual content, a model-capability gap in the small local vision model, not an integration bug (the provider, cache, retrieval, and citation path were all confirmed working correctly).

**Impact**
The integration is correct and ready to use; answer-quality gains depend on a stronger vision model than this project's CPU-only host can practically run.

**Recommended fix**
None adopted for this host; revisit with a stronger vision model if host constraints change.

---

### MULTI-003: A failing vision call would have failed the whole document

**Status:** FIXED
**Severity:** MODERATE
**Area:** Ingestion / multimodal
**Affected code:** `Writer._with_vision_siblings`

**Problem**
Without isolation around the vision-provider call, an unreachable provider, a timeout, or an unparsable response on any one image would propagate and fail ingestion of the entire document, including every other chunk that had nothing to do with that image.

**Failure scenario**
One bad or temporarily unreachable image in a multi-page document blocking ingestion of all of that document's text content.

**Fix applied**
`describe_image` is called inside a `try`/`except`; a failure is logged (`vision_provider_call_failed`) and that one image is skipped (no vision sibling chunk), while the rest of the document, including its other images, still ingests normally.

**Regression test**
`test_vision_provider_call_failure_skips_image_without_failing_document` in `tests/unit/test_writer.py`.

---

## 2. Security Issues

### SEC-001: Document-level access control could not stop a legitimate reader from getting a secret

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / field-level redaction
**Affected code:** `retrieval/field_policy.py`, document-level ACL

**Problem**
A support operator legitimately allowed to read a troubleshooting runbook could also extract an administrator credential embedded in that same document, including via rephrasing (e.g. "give me its Base64 form").

**Failure scenario**
Document-level authorization admits the whole chunk once access is granted; it has no concept of restricting one field inside an otherwise-permitted document. Adversarial evaluation (experiment_026) demonstrated this leakage directly.

**Impact**
An authorized reader of a document could still extract a credential they were never meant to see; prompt instructions alone ("never reveal administrator keys") are not a security boundary.

**Fix applied**
Field-level redaction runs after document authorization and before prompt construction. Sensitive fields are tagged at ingestion; at query time, tagged values are checked against the caller's verified roles and unauthorized values are replaced with `[REDACTED:SENSITIVE_FIELD]`, after relationship expansion so an expanded neighbor cannot reintroduce the value. No identity present = fail-closed (every tagged field redacted).

**Regression test**
`sensitive_data_leakage_rate` 0.286 (2/7) → 0.000 (0/7); retrieval metrics identical between runs; false-redaction rate 0/8; authorized disclosure 1/1.

---

### SEC-002: A pattern written from one document missed a second credential format

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / field-level redaction
**Affected code:** `SensitiveFieldPolicy` regex, leak-detection pattern

**Problem**
The redaction policy matched one tenant's admin credential by a literal prefix. After re-ingestion, only that tenant's runbook was tagged as sensitive.

**Failure scenario**
The second tenant's credential used a genuinely different literal prefix, not a variant of the first. The same blind spot existed in the pre-existing leak-detection metric. Hand-written unit tests didn't catch it because they used a literal matching the pattern by construction.

**Fix applied**
Added the second shape as an alternation in both the redaction policy and the leak-detection pattern, with a cross-referencing comment in each. Re-ingested and verified persisted `sensitive_field_ids` tags with a direct SQL query.

**Regression test**
Both tenant credential shapes now covered by redaction and by the leak metric; confirmed against persisted data, not only unit tests sharing the same assumption.

---

### SEC-003: The system and user prompts were joined into one string before reaching the model

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / prompt-injection architecture
**Affected code:** `RetrievalPipeline.answer()`, `LLM.generate`, Ollama wrapper

**Problem**
The design required system instructions and retrieved evidence to stay structurally separate at the model boundary; the data model supported this, but the runtime code did not preserve it.

**Failure scenario**
`RetrievalPipeline.answer()` built `f"{system}\n\n{user}"` and handed one blob to `generate(prompt)`; the Ollama wrapper called the raw-completion endpoint, not the role-aware chat endpoint. The model had to infer the instruction/evidence boundary from formatting alone.

**Impact**
Weakened instruction/data separation, a structural prerequisite for injection resistance, despite the design intending otherwise.

**Fix applied**
`LLM.generate` now takes `(system, user)` across the base class and all implementations; the Ollama wrapper sends two role-tagged chat messages. Seven test doubles needed the new signature.

**Regression test**
Integration test runs against a live Ollama server proving system and evidence arrive as separate turns.

---

### SEC-004: A demo looked unsafe because the running config had every safety control turned off

**Status:** FIXED
**Severity:** MODERATE
**Area:** Security / operational visibility
**Affected code:** `GET /` feature-flags response

**Problem**
During a web UI demo, an adversarial prompt got the model to repeat and encode a credential, which looked like a redaction regression.

**Failure scenario**
The backend was running a config with authorization and field redaction both off by default; the redaction code itself was correct, but nothing in the UI indicated which config was active. The identical query against the secure config returned a paraphrase instead.

**Fix applied**
`GET /` returns a `features` block (booleans/provider names only, no connection/model identifiers), rendered in the UI as an always-visible pill bar.

**Regression test**
A test checks the response against a denylist of identifying values; live checks confirm safe vs. unsafe configs are now visually distinguishable.

---

### SEC-005: Uploads could become visible to other tenants

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / multi-tenancy
**Affected code:** `POST /ingest`, `resolve_ingest_tenant_id`

**Problem**
`POST /ingest` checked the caller's *role* but never bound the uploaded document to the caller's *tenant*.

**Failure scenario**
Documents without governance front matter were stored with `tenant_id = NULL`; the retrieval SQL predicate treats NULL as visible to every tenant. An authenticated tenant-alpha upload was therefore retrievable by tenant-beta under the secure config. Found by an independent audit, reproduced end to end with a plain text file.

**Impact**
Cross-tenant data exposure for any upload lacking governance metadata, the common case for PDF/DOCX uploads.

**Fix applied**
`resolve_ingest_tenant_id` stamps the caller's own tenant when none is given, allows an explicit same-tenant value, and allows a different tenant only for a configured cross-tenant support role; anything else returns 403 before any write, audit-logged.

**Regression test**
17 new tests passed with no regressions.

**Notes**
See **SEC-008** below: cross-tenant *integration* tests for this fix were written but never actually run (no local Postgres available at the time).

---

### SEC-006: Upload handling: writes landed before validation, and temp paths leaked into citations

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / upload atomicity
**Affected code:** Upload staging path, `_ingest_upload_atomically`, loader `path.stem` fallbacks

**Problem**
Three follow-up reviews of the SEC-005 fix found three further issues: the upload was written to its final path *before* validation, so a rejected re-upload destroyed a previously accepted file; and the temp filename leaked into a document's default title and into extracted-image citations, since loaders derive those from the input path.

**Failure scenario**
A same-filename re-upload that was later rejected (oversized, or governance denial) had already overwritten/truncated the previously accepted file on disk, with no DB-level trace. Separately, a randomly-named staged file leaked its temp name into stored, user-visible metadata.

**Fix applied**
Uploads are staged in a unique directory, keeping the destination's own filename (only the parent directory is randomized) so path-derived fields resolve correctly with zero loader changes. Install happens only after ingestion fully succeeds, via an atomic `os.replace`.

**Regression test**
New tests prove: a rejected upload leaves no final file; a rejected re-upload leaves prior bytes unchanged; no temp-path fragment survives in any persisted chunk.

**Notes**
Two residual gaps were explicitly recorded, not closed, see **SEC-007** below.

---

### SEC-009: Metadata-only sensitive values bypassed field-level redaction

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / field-level redaction
**Affected code:** Ingestion-time tagging, `redact_source_metadata`

**Problem**
Redaction ran only when a chunk's *body text* had been tagged at ingestion. A credential placed in a heading, filename, or source path was never tagged, so the entire redaction pass was skipped for it.

**Failure scenario**
An untagged sensitive value in metadata reached the prompt, citations, and the API response unredacted. Found by an independent review asking "does anything skip the gate."

**Fix applied**
Metadata is now scanned on every query regardless of tag state; tagging at ingestion covers the same fields. A single shared scannable-field list is used by tagging, redaction, and the duplicate-occurrence diagnostic.

**Regression test**
12 new tests cover the metadata paths including `source`; a regression test confirms redacting one result doesn't affect another.

---

### SEC-010: A signed token was accepted as an approval, not only as an identity

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / MCP approval trust boundary
**Affected code:** `resolve_case_action_approvals`, internal service token

**Problem**
A case-status change to "closed" requires human approval, carried as a claim on an internal token. The MCP-side check verified only the token's signature.

**Failure scenario**
Since any token signed with the shared key verifies successfully, and the role-gating check only runs on the agent's own HTTP routes (never for a caller talking to the MCP server directly, an explicitly supported access pattern), a caller able to mint an ordinary valid token could attach the approval claim directly and bypass the role gate entirely. Found by independent review; no test had exercised the resolver with an unexpected token shape.

**Impact**
A write action (case closure) could be approved by a caller who never held the approval role, a genuine authorization-bypass path, not merely a theoretical one.

**Fix applied**
The resolver now requires the internal-service `sub`/`token_use` markers and independently re-checks the token's own embedded roles against the approval roles.

**Regression test**
9 adversarial tests cover the ordinary-token, missing-marker, and missing-role cases.

---

### SEC-011: The request ID was reused as a dedup key, although clients could set it

**Status:** FIXED
**Severity:** MODERATE
**Area:** Security / feedback integrity
**Affected code:** Feedback dedup key, `x-request-id` header handling

**Problem**
The feedback dedup key used the request ID, taken from the caller-controllable `x-request-id` header when present.

**Failure scenario**
The header was designed for log correlation, not uniqueness; a caller could force a collision with another submission's row by supplying the same header value.

**Fix applied**
The server always generates the request ID itself; the header is still returned but never read as input. Related fixes from the same review: frontend feedback requests now include tenant scoping; the export script requires an explicit tenant or an all-tenants flag; free-text fields are length-limited.

**Regression test**
Regression tests cover a forged header, the no-scope export exit code, oversized fields, and the disabled-feature path.

---

### SEC-012: The default config followed an injected instruction, because it ran older, weaker prompts with no access control

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / deployment configuration
**Affected code:** `config/default.yaml` vs. `config/production.yaml`

**Problem**
A manual test of the unmodified default config against an embedded-injection gold question returned the injected instruction as the answer (send customer credentials, print hidden context), for an anonymous caller retrieving a confidential tenant document.

**Failure scenario**
The classic route used the first prompt version, predating the "retrieved context is evidence, not instructions" rule. The validated, fixed prompts existed only in a specific experiment's config, never promoted into the config that `docker compose up` actually serves. Authorization, redaction, and auth were off in the default config by design (zero-setup demo).

**Impact**
A validated security fix that never reaches the config people actually deploy protects no one.

**Fix applied**
A separate `config/production.yaml` turns on auth, authorization, redaction, rate limiting, and tracing, and uses the newer prompts. Documentation now states which config a real deployment should use.

**Regression test**
The same unauthenticated request now returns 401; the same question from a verified operator refuses the injected instruction and routes through the agent path correctly.

---

### SEC-013: Agent tools that queried the vector store directly skipped the authorization kill-switch

**Status:** FIXED
**Severity:** MAJOR
**Area:** Security / agentic tool authorization
**Affected code:** `get_document`, `get_latest_document`, `get_related_context`, `resolve_auth`

**Problem**
Three agent tools call the vector store directly rather than through the main retrieval path. The authorization kill-switch and freshness resolution lived only in that main path.

**Failure scenario**
The three tools applied tenant/role filters even when authorization was globally disabled, and never excluded superseded documents, since each passed the caller's context through unresolved and the store's SQL filter has no concept of the config flag.

**Fix applied**
The pipeline's resolution step became a public `resolve_auth` method; all three tools now call it before touching the store. Two follow-up reviews found and fixed two further gaps: missing `dataset_id` threading for freshness, and a redundant duplicate fetch. A third review found the post-dispatch sanitization step still received the raw, unresolved context, fixed to use the resolved one.

**Regression test**
Unit tests cover each tool under enabled and disabled authorization; a regression test confirms the third fix catches the bug on revert.

---

### SEC-015: A hosted reranker received raw candidate content before redaction, with no egress gate

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / provider egress
**Affected code:** `RetrievalPipeline._retrieve_timed_inner`, `rerankers/cohere.py`

**Problem**
`config.reranker.provider: cohere` sends every candidate's full chunk content to Cohere's hosted API before the field-redaction pass ran, and no provider-egress check applied to the reranker boundary at all (only `run_ragas_eval.py`'s judge call was gated).

**Failure scenario**
A caller the field policy or egress policy would otherwise restrict from seeing a value (or from having it leave the local environment) still had that raw content sent externally, because reranking happened strictly before redaction and the egress policy never knew the reranker boundary existed.

**Fix applied**
Field redaction now runs on the candidate pool before any reranker call, hosted or local. Moved `apply_egress_policy`/`EgressDecision` to `rag.retrieval.egress_policy` (re-exported from `rag.eval.egress_policy` for existing callers) so `RetrievalPipeline` can apply the same egress check to the Cohere boundary, dropping any blocked candidate before it reaches the reranker.

**Regression test**
`test_hosted_reranker_never_receives_unredacted_sensitive_content` and `test_hosted_reranker_never_receives_egress_blocked_candidates` in `tests/unit/test_retrieval_pipeline.py`.

---

### SEC-016: Egress policy dropped a blocked source's context but still sent the derived answer to the hosted judge

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / provider egress
**Affected code:** `run_ragas_eval._build_rows`, `run_agent_ragas_eval._build_rows`

**Problem**
When egress policy blocked one of a question's retrieved sources, only that source's `content` was dropped from `contexts`; the generated `answer` (which may itself quote the blocked content) was still sent to the hosted judge, along with the question/expected-answer.

**Failure scenario**
An authorized local answer quoting a restricted value: the context was correctly redacted before it left the machine, but the answer carrying the same value was not, defeating the egress gate's purpose.

**Fix applied**
The judge row, not the individual source, is now the egress unit: if any source for a row is blocked, the whole row is dropped from the judge payload. A new `num_skipped_egress_blocked` count is reported alongside the existing `num_skipped_missing_expected_answer`.

**Regression test**
`test_run_ragas_drops_whole_row_when_egress_policy_blocks_a_source` in `tests/unit/test_run_ragas_eval.py`.

---

### SEC-017: A same-named upload could silently replace another tenant's document

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Security / ingestion authorization
**Affected code:** `IngestionPipeline.ingest_file`, `governance.resolve_ingest_tenant_id`, `VectorStore.get_document_owner_tenant_id`

**Problem**
Document identity is `(source, dataset_id)`; `_safe_upload_path` strips the filename to its basename with no tenant/dataset namespacing. `resolve_ingest_tenant_id` only ever validated the *new* content's claimed tenant, never whether an *existing* document at that identity already belonged to someone else.

**Failure scenario**
Tenant A uploads `report.md` into a shared dataset. A different, independently ingest-authorized tenant B later uploads an unrelated file also named `report.md` into the same dataset, with no front matter at all (so the existing governance check saw nothing to object to). Both resolve to the same `document_id`; tenant B's own legitimate ingest permission silently replaces tenant A's chunks.

**Fix applied**
`ingest_file` now checks, before any changed-document write, whether the resolved `document_id` already has chunks owned by a different tenant (`VectorStore.get_document_owner_tenant_id`); a non-privileged caller is rejected with the same `IngestGovernanceError`/403 path the front-matter-mismatch case already used. A privileged (`cross_tenant_support_roles`) caller may still knowingly replace it.

**Regression test**
`test_non_privileged_caller_cannot_replace_a_document_already_owned_by_another_tenant` and `test_privileged_caller_may_replace_a_document_owned_by_another_tenant` in `tests/unit/test_ingestion_pipeline_governance.py`.

---

## 3. Reliability and Distributed Systems

### DIST-001: Reading a request-scoped context variable inside a stream would have raced its middleware

**Status:** FIXED
**Severity:** MODERATE
**Area:** Reliability / async context handling
**Affected code:** SSE streaming response, request-ID contextvar

**Problem**
The streaming response needed the request ID in its final frame; reading the context variable directly inside the generator was the obvious first approach.

**Failure scenario**
The middleware resets that context variable once its own awaited call returns, but the stream body runs *after* that point, so reading it inside the generator would sometimes return null depending on timing, a genuine race, caught by reasoning before it shipped.

**Fix applied**
The route reads the ID once, before the response object exists, and passes it down explicitly as an argument.

**Regression test**
The ID now appears correctly in both plain and streaming responses; the race was avoided by design rather than patched after manifesting.

---

### DIST-002: A LangGraph join node ran twice, because its parallel branches had unequal lengths

**Status:** FIXED
**Severity:** MAJOR
**Area:** Reliability / multi-agent workflow (learning exercise)
**Affected code:** LangGraph join node, `defer=True`

**Problem**
In a multi-agent experiment, a mixed query fanned out to two specialists; the join node ran twice for one query, duplicating an LLM call.

**Failure scenario**
One branch was one hop from the fan-out point, the other was three. A plain join edge fires as soon as *any* incoming branch delivers, not once *all* have. Reproduced with a minimal two-branch probe.

**Fix applied**
The join node was given `defer=True` (waits until every pending task finishes), confirmed against the pinned library's own docstring.

**Regression test**
LLM call count for the mixed query went from 2 to 1; the probe is kept permanently, covering both the bug and the fix, so a future library upgrade changing this behavior fails there first.

---

### DIST-003: A deliberate workflow pause would have been counted as a failure in metrics

**Status:** FIXED
**Severity:** MODERATE
**Area:** Reliability / observability wrapper correctness
**Affected code:** Node-level failure-counter wrapper, LangGraph interrupt exception hierarchy

**Problem**
Adding a failure counter around every graph node risked counting intentional human-in-the-loop pauses as failures, since the approval node pauses a run by raising an internal exception.

**Failure scenario**
The pause exception subclasses the framework's control-flow base class, itself an ordinary `Exception`; a generic `except Exception` cannot distinguish a deliberate pause from a real error. Caught by reading the pinned library's exception hierarchy before writing the wrapper, not after a bad metric shipped.

**Fix applied**
The wrapper re-raises the framework's control-flow base class first, before the generic handler; a separate counter records interrupts distinctly.

**Regression test**
The full approval flow passes with the failure counter at zero.

---

### DIST-004: A queue-depth metric built on XLEN could only grow

**Status:** FIXED
**Severity:** MODERATE
**Area:** Reliability / Redis Streams job queue (learning exercise)
**Affected code:** `queue_depth()`, Redis `XLEN`

**Problem**
`queue_depth()` was asserted to read zero after all jobs finished, but instead read 1, then 2, on later runs.

**Failure scenario**
Redis `XLEN` counts every entry ever added to a stream; acknowledging a message removes it only from a consumer group's pending list, the entry itself stays in the stream, since other consumer groups may still need to read it.

**Fix applied**
The original metric is kept, now documented as "total entries ever added." A separate backlog counter, incremented on submit and decremented on completion/dead-lettering, backs backpressure and the gauge instead.

**Regression test**
Both assertions pass; no consumer calls the old metric for remaining-work semantics anymore.

---

### DIST-005: An SSE disconnect was only noticed between queue items, never while waiting for one

**Status:** FIXED
**Severity:** MAJOR
**Area:** Reliability / agent streaming
**Affected code:** `agent_stream._stream_agent_query`

**Problem**
The SSE consumer loop checked `request.is_disconnected()` only after `await queue.get()` returned an item. A long-running node that emits no intermediate event (in practice, a single slow LLM call) left the queue empty for the whole call, during which a disconnect was never noticed and the worker's `cancel_event` was never set.

**Failure scenario**
A client disconnects while the agent worker thread is mid-generation; cooperative cancellation never fires, so the worker runs to completion regardless, consuming CPU/model time for a client no longer listening.

**Fix applied**
The queue read is now bounded by a short poll timeout (`_DISCONNECT_POLL_SECONDS`), checking `is_disconnected()` on every timeout rather than only after an item arrives. `cancel_event.set()` also runs unconditionally in the loop's `finally` block, covering generator teardown paths the explicit disconnect branch didn't reach.

**Regression test**
`test_disconnect_is_noticed_while_the_queue_is_empty` and `test_generator_teardown_also_sets_cancel_event` in `tests/unit/test_agent_stream_disconnect_cancellation.py`.

---

## 4. Retrieval, Graph, Ontology and Context

### RET-001: Retrieval window too narrow for one document (idempotency-key question)

**Status:** ACCEPTED LIMITATION
**Severity:** MODERATE
**Area:** Retrieval
**Affected code:** Production `top_k=5` retrieval window

**Problem**
One gold question ("How long are idempotency keys retained?") failed on every configuration tried; the correct document was in the knowledge base, but production retrieval never returned it.

**Failure scenario**
A cross-encoder reranker left the top-5 set unchanged (a reranker reorders, it cannot add). A different embedder ranked the document #1 at a broad top-10 fetch, but the production `top_k=5` call still missed it. A chunk-size change had no effect either. The bottleneck is the retrieval window itself.

**Impact**
This specific question is unanswerable under production config; `qwen2.5:3b` degrades this to a correct "I don't know" rather than `qwen2.5:1.5b`'s fabrication, but the underlying retrieval gap was never closed.

**Recommended fix**
Widen `top_k` for this class of query, or add a secondary identifier-exact-match retrieval path; not attempted, since no config lever tested moved the document into the window.

**Required regression test**
A gold-question regression confirming the document enters the top-k once a fix is applied.

**Notes**
Documented as an open, unresolved known limitation rather than a false "fixed" claim across experiments 002–005.

---

### RET-002: Embedding and chunk-size changes each traded one metric for another

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Retrieval
**Affected code:** Embedding model selection, chunk-size config

**Problem**
Two retrieval-side changes were tested against the current best config: BGE-small in place of `all-MiniLM-L6-v2`, and 300-character chunks in place of 500.

**Failure scenario**
Not a defect: BGE-small raised Recall@5/Hit Rate@5 but lowered Recall@10 and answer quality. 300-character chunks lowered MRR and answer quality with three new regressions. Every change improved something while harming something else.

**Impact**
Neither change is a net improvement under this project's own metric set.

**Recommended fix**
None; both are recorded as measurements, not adopted.

**Notes**
Preserved here as a documented tradeoff finding, not a bug.

---

### RET-003: Hybrid retrieval's headline recall hid how each retriever contributed

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Retrieval
**Affected code:** `retrieve_attribution()`

**Problem**
The fused hybrid ranking improved the overall score but gave no visibility into whether BM25 or dense retrieval caused each success.

**Failure scenario**
Not a defect: RRF fusion exposes only the combined ranking. A dedicated attribution pass found BM25 never independently found a document dense missed, and that hybrid's Recall@10 was actually *lower* than dense-only (RRF demoted correct dense top results on 6 questions) while still winning on Hit Rate@5, MRR, and supporting-context hit rate.

**Impact**
The aggregate recall number alone would have supported the wrong conclusion about what fusion is actually doing.

**Recommended fix**
None required; `retrieve_attribution()` now exists specifically to make this visible per-question going forward.

---

### RET-004: A cross-encoder changed answers without changing retrieved evidence

**Status:** ACCEPTED LIMITATION
**Severity:** MODERATE
**Area:** Retrieval / reranking
**Affected code:** `reranker.provider: cross_encoder` candidate

**Problem**
A controlled A/B of `reranker.provider: none` against `cross_encoder` moved aggregate metrics by less than noise.

**Failure scenario**
One question retained the identical reference-context bucket and identical two source documents in both runs; its answer quality still fell from 0.947 to 0.053 purely from a reordering of two equally relevant chunks, the model hedged heavily from the same evidence once chunk order changed. 15 questions improved by >0.05, 17 regressed by >0.05, netting to a misleadingly small positive average.

**Impact**
"Same evidence retrieved" does not guarantee "same answer produced"; prompt position can swing a local model's answer sharply, invisible to retrieval-side metrics alone.

**Recommended fix**
None adopted; the finding argued against enabling the reranker, so `reranker.provider: none` stayed the default.

---

### RET-005: BM25 ties depended on unspecified SQL row order

**Status:** FIXED
**Severity:** MAJOR
**Area:** Retrieval / determinism
**Affected code:** `PgVectorStore.search_keyword`

**Problem**
The BM25 candidate fetch had no `ORDER BY`, and the Python ranking sorted only by score; a tied score's relative order depended on whatever order Postgres happened to return rows in.

**Failure scenario**
Identical config and corpus could still produce different BM25 (and therefore fused RRF) rankings across runs or after a reload, for any pair of candidates with equal or zero-overlap scores, common in a small corpus.

**Fix applied**
Added `ORDER BY chunk_id` to the fetch, matching dense search's own existing `ORDER BY embedding <=> ..., chunk_id ASC` tie-break, plus an explicit `(-score, chunk_id)` sort key in Python so determinism doesn't rely solely on the SQL ordering being remembered.

**Regression test**
`test_search_keyword_ties_break_deterministically_by_chunk_id` in `tests/integration/test_pgvector.py` (returns identical rankings across repeated calls for three byte-identical-content chunks).

---

### GRAPH-001: Neo4j failed on one unknown config key and on a null inside a MERGE key

**Status:** FIXED
**Severity:** MAJOR
**Area:** GraphRAG
**Affected code:** Neo4j container config, `GraphWriter` MERGE pattern

**Problem**
The graph container crash-looped on startup, then writes failed with "cannot merge because of null property value."

**Failure scenario**
Neo4j 5.28 strictly validates environment-mapped setting names; a guessed telemetry key didn't exist, causing a boot failure. Separately, a relationship's `source_chunk_id` is legitimately null for document-level facts, and Neo4j rejects a null inside a MERGE property map outright.

**Fix applied**
Removed the unknown setting rather than guessing again. Coalesced the chunk id to an empty-string sentinel before it reaches the MERGE pattern.

**Regression test**
Container starts cleanly; both chunk-scoped and document-level facts merge correctly.

---

### GRAPH-002: The driver's convenience method stripped the node labels authorization code needed

**Status:** FIXED
**Severity:** MAJOR
**Area:** GraphRAG / authorization
**Affected code:** `Neo4jClient` result-row construction

**Problem**
Entity-type resolution and the graph authorization re-check read labels from result rows; unit tests passed against hand-built dicts, but real results had wrong or missing types.

**Failure scenario**
The driver's `Record.data()` recursively flattens nodes/relationships into plain dicts, discarding `.labels`/`.type`. `dict(record)` preserves the typed objects instead.

**Fix applied**
The client now builds rows with `dict(record)`; each query returns label information explicitly.

**Regression test**
Entity typing, path traversal, and the authorization re-check all read real labels; 8 chained retrieval/authorization scenarios pass against a live instance in ~6 seconds.

---

### GRAPH-003: A graph path test failed because one relationship write cannot span two tenant scopes

**Status:** FIXED
**Severity:** MINOR
**Area:** GraphRAG / test fixture correctness
**Affected code:** Adversarial path-authorization test fixture

**Problem**
A test for "an unauthorized hop is never partly shown" returned an empty path even for a caller who should see the whole chain.

**Failure scenario**
Node identity derives from each write's own tenant; the test wrote one hop with no tenant and the next with `tenant_alpha`, creating two disconnected nodes with the same name rather than one continuous path. The test fixture, not the authorization logic, was broken.

**Fix applied**
Rebuilt the test with two internally consistent chains: one fully single-tenant (matching real extraction behavior) and one open control chain.

**Regression test**
Denied, same-tenant-authorized, and open-control cases all pass, cleanly separating "denial" from "broken query."

---

### GRAPH-004: Resetting the graph alone left the graph empty, because change detection only looked at Postgres

**Status:** ACCEPTED LIMITATION (as a residual constraint; the demo-reset gap itself is FIXED)
**Severity:** MINOR
**Area:** GraphRAG / ingestion consistency
**Affected code:** Graph-reset seeding script, checksum-based change detection

**Problem**
A seeding script cleared the graph and re-ran ingestion; ingestion reported no documents had changed, so the graph stayed empty.

**Failure scenario**
Ingestion compares Postgres checksums before reaching graph extraction. A graph-only reset never touches that checksum, so the unchanged-checksum early return skips the graph stage entirely; the two stores are kept in sync by a one-directional check only.

**Fix applied (for the demo script)**
The script now calls the dataset-level clear, removing chunks and graph facts together.

**Notes**
Documented residual design property, not fully closed: *any* future recovery path that clears only the graph side will hit the identical gap, since the one-directional check is structural, not specific to this one script. See **PROD-003**.

---

### GRAPH-005: LLM graph extraction worked correctly, but was impractical on a memory-starved host

**Status:** ACCEPTED LIMITATION
**Severity:** MODERATE
**Area:** GraphRAG / resource constraints
**Affected code:** LLM-based entity extraction path

**Problem**
LLM-based entity extraction needed a real run against corpus text; the first call took 37.9s for one chunk and returned malformed JSON, and a larger model didn't finish within 30 minutes.

**Failure scenario**
Not a code bug: ~1.1GB of 8GB RAM was free with memory compression active. A 3-token call showed 48.3 of 48.9 seconds spent purely loading the model into memory, with only 88ms of actual inference.

**Impact**
Full-scale LLM graph extraction is not practical on this host's hardware.

**Recommended fix**
Run on hardware with sufficient free RAM to hold the model; not attempted here.

**Notes**
The demo graph instead uses hand-curated relationships, each traceable to a real corpus sentence, honestly labeled `extraction_method: "deterministic"`, not `"llm"`. The extraction code's failure path (malformed output → empty result, not a crash) was proven to work correctly.

---

### GRAPH-006: A relationship's own ACL was never checked on the dependency-path query

**Status:** FIXED
**Severity:** CRITICAL
**Area:** GraphRAG / authorization
**Affected code:** `GraphRetriever._search_dependency_path`, `graph/queries.py:dependency_path_query`

**Problem**
The shortest-path template returns a Neo4j `Path` carrying both nodes and relationships; the Python-side re-check only ever inspected `path.nodes`, never `path.relationships`, even though a relationship carries its own `allowed_roles`/`trust_level`/`source_document_id` (set at write time, see `graph/upsert.py`).

**Failure scenario**
Two public (authorized) entities connected only by a relationship sourced from a restricted document: a caller unauthorized for that relationship could pass every node check and still receive the path, disclosing a connection they had no right to know about. Permission to know two entities exist is not permission to know they're connected.

**Fix applied**
Every relationship on the returned path is now re-checked with the same `is_fact_authorized` call nodes already get; a path with any unauthorized hop (node or relationship) is never partially surfaced.

**Regression test**
`test_a_restricted_relationship_between_two_public_entities_is_not_disclosed` (plus two positive-path tests) in `tests/unit/test_graph_dependency_path_authorization.py`.

---

### GRAPH-007: Neo4j routing/timeout options silently became Cypher query parameters

**Status:** FIXED
**Severity:** MAJOR
**Area:** GraphRAG / reliability
**Affected code:** `Neo4jClient.run_read`, `run_write`

**Problem**
`execute_query` was called with `routing_control=`/`transaction_timeout=` kwargs. Verified directly against the installed `neo4j==5.28.7` driver: its real parameter is `routing_=`, and the timeout belongs on a `neo4j.Query(text, timeout=...)` object passed as the query itself, not a bare kwarg. The two unrecognized kwargs fell into `execute_query`'s own `**kwargs`, which it forwards into `Transaction.run(query, parameters, **kwargs)` as if they were Cypher query parameters.

**Failure scenario**
The configured server-side transaction timeout was never actually applied, and every read used the driver's default (write) routing regardless of the `RoutingControl.READ` argument intended.

**Fix applied**
Both methods now build a real `neo4j.Query(cypher, timeout=effective_timeout)` and pass `routing_=RoutingControl.READ`/`WRITE`.

**Regression test**
`tests/unit/test_graph_client.py` (a mocked driver, asserting the exact call shape against the real `Query`/`RoutingControl` types).

---

### GRAPH-008: A stale graph fact is never revalidated against its source's current Postgres state

**Status:** OPEN
**Severity:** MAJOR
**Area:** GraphRAG / cross-store consistency
**Affected code:** `graph/fusion.py`, `rag.graph.retrieval`

**Problem**
Graph retrieval authorizes a fact against the ACL copied onto its Neo4j nodes/relationships at write time; it never re-checks the fact's source document/chunk against Postgres's current, authoritative state.

**Failure scenario**
A document's ACL is narrowed, or the document is deleted, in Postgres, while the corresponding Neo4j fact survives (a real possibility given the already-documented graph-cleanup gaps, see **GRAPH-004**/**PROD-003**, and the lack of any durable graph-write retry ledger). The stale fact is still authorized and surfaced as prompt evidence, potentially after the access it reflects should have been revoked.

**Why not fixed yet**
Closing this fully requires batch-resolving every fact's supporting document/chunk against Postgres before it's ever converted to evidence (a real per-query cost), and failing closed on a mismatch; this is a structural change to the graph-fusion read path, not a local patch, and was judged out of scope for this pass.

**Recommended fix**
Resolve each fact's `source_document_id` against Postgres's live ACL/existence at read time (not just at graph-write time), failing closed on a checksum/ACL mismatch.

---

### GRAPH-009: A live-Neo4j ontology query helper has no authorization check at all

**Status:** OPEN
**Severity:** MODERATE (unreachable today; would be CRITICAL if ever wired into a live endpoint)
**Area:** Ontology / GraphRAG integration
**Affected code:** `rag.ontology.graphrag_integration.expand_and_query_live_neo4j`

**Problem**
This function queries live Neo4j, filtering only by `dataset_id` and label; it has no `AuthorizationContext` parameter and never calls `is_fact_authorized`/`is_entity_authorized`.

**Failure scenario**
Confirmed dead code today: `grep` across `src/rag/{api,agent,graph,retrieval}` finds zero callers, and the module's own docstring calls it "sketched (written, not executed)." The scenario is latent: if this is ever promoted into a real agent tool or endpoint without adding the same authorization gate every other graph-read path has, it would disclose unauthorized entities outright.

**Why not fixed yet**
No live caller exists yet to fix a live bug in; adding authorization now to dead code was judged lower priority than the two reachable gaps above.

**Recommended fix**
Add `is_entity_authorized` filtering, mirroring `rag.graph.retrieval`'s pattern, before this function is ever wired into anything live. Add a regression test now, preemptively, so the gate exists before the first real caller does.

---

### ONT-001: A narrow RDFS domain axiom silently reclassified an Application as a Service

**Status:** FIXED
**Severity:** MODERATE
**Area:** Ontology / semantic layer
**Affected code:** `tech:dependsOn` ontology property, SHACL shapes

**Problem**
SHACL validation run in two modes (raw, and after RDFS closure) on the same otherwise-valid data disagreed: both reported 6 violations, but for different entities and different reasons.

**Failure scenario**
`tech:dependsOn` declared `rdfs:domain tech:Service`, assuming only services depend on things. The real demo graph also has an `Application` as a `DEPENDS_ON` subject. RDFS closure entails a domain axiom's type for *any* subject of that property regardless of its existing type, silently retyping the Application as a `Service` too, moving it into a different SHACL shape's scope. A domain/range axiom is an inference trigger, not a validation constraint, it manufactures a new fact rather than only checking one.

**Fix applied**
Removed the domain declaration entirely; the real constraint now lives in a SHACL shape (`sh:or` over both classes), which checks without ever adding a triple.

**Regression test**
Raw and RDFS-closed validation now agree exactly (5 violations each, same 5 entities). A second, isolated toy-ontology reproduction of the same general mechanism is kept as its own demo script.

---

### ONT-002: SHACL was already subclass-aware, before any inference option was requested

**Status:** FIXED
**Severity:** MINOR
**Area:** Ontology / semantic layer / documentation correctness
**Affected code:** SHACL shapes file documentation, validation module docs

**Problem**
The ontology experiment's SHACL shapes file documented an incorrect assumption: that `sh:targetClass`/`sh:class` only match an individual with an *explicit* `rdf:type` triple, and that catching a subclass-typed individual required `pyshacl` with `inference="rdfs"` first.

**Failure scenario**
Running the validator against real data (PostgreSQL/OpenSearch, typed only `tech:DatabaseService`, never `tech:Service` directly) showed `ServiceShape` already flagging both under `inference=None`, pyshacl's default. The W3C SHACL Recommendation itself bakes `rdf:type`/`rdfs:subClassOf*` closure into core "SHACL instance" semantics, independent of any separate inference option.

**Fix applied**
Corrected the shapes file's and validation module's documentation; added a dedicated test proving subclass-typed individuals are caught under `inference=None`.

**Regression test**
The corrected test now guards against the wrong assumption silently returning.

---

### CTX-001: Compression threshold was gated on item size, not remaining headroom, silently starving graceful degradation under tight budgets

**Status:** FIXED
**Severity:** MODERATE
**Area:** Context Engineering
**Affected code:** `rag.context.packing.pack_priority_aware`, `CompressionConfig.threshold_tokens`

**Problem**
Under a very tight token budget with many mid-sized candidate items, the full Context Engineering pipeline (`priority_aware` packing, compression enabled) returned an *empty* rendered context, even though the budget had real headroom and the items were plain, splittable prose.

**Failure scenario**
`pack_priority_aware`'s compression gate originally read `item.token_count >= compression_threshold_tokens` (default 400): an item was only a compression candidate if it was, by itself, at or above a fixed global size. A regression test (`test_graceful_degradation_never_raises_when_evidence_vastly_exceeds_budget`) built 50 candidates of ~179 tokens each against a ~69-token per-bucket headroom; none of them individually reached the 400-token floor, so compression never ran for any of them, and every item was excluded outright, producing `rendered_context == ""` despite `truncated=True` and real, compressible content on hand.

**Impact**
"Retain highest-value evidence" (section 27's graceful-degradation requirement) silently failed whenever the budget was small relative to item size but the items themselves were individually modest, a realistic shape for a small local model's context window.

**Fix applied**
Redefined `threshold_tokens` as a floor on the *remaining headroom*, not the candidate's own size: `if compression_enabled and headroom >= compression_threshold_tokens`. An item that doesn't fit its headroom is now always a compression candidate regardless of its own size, as long as the headroom is large enough to compress into something usable. Lowered the config default from 400 to 30 to match this corrected meaning.

**Regression test**
`tests/unit/test_context_engine.py::test_graceful_degradation_never_raises_when_evidence_vastly_exceeds_budget` now asserts a non-empty, correctly-truncated result under the same tight-budget shape that originally exposed the gap.

---

### CTX-002: A generic uppercase-word regex misparsed a graph fact's own entity name as its relationship type

**Status:** FIXED
**Severity:** MINOR
**Area:** Context Engineering / GraphRAG integration
**Affected code:** `rag.context.graph_packing.parse_graph_fact_text`

**Problem**
Parsing a rendered `GraphFact.text` string (e.g. `"Alpha API DEPENDS_ON Alpha Database"`) back into structured `GraphHop`s picked the wrong token as the relationship when an entity name was itself an all-uppercase acronym.

**Failure scenario**
The original regex (`r"^(.+?)\s+([A-Z][A-Z_]{2,})\s+(.+)$"`) matched the *first* uppercase run of 3+ characters, which for `"Alpha API DEPENDS_ON Alpha Database"` is `"API"`, not `"DEPENDS_ON"`. Several real entity names in this corpus are themselves acronyms (e.g. "Alpha API"), making this a live risk, not a contrived test fixture.

**Impact**
Display/metadata-only: `ContextItem.graph_path` (used for the optional hop-aware compression shrink and future richer rendering) would carry a nonsensical hop. Never affects authorization, budgeting, or packing correctness, since `ContextItem.content` always carries the original, unparsed rendered text regardless of parse success.

**Fix applied**
Built the regex's relation alternation directly from the closed `rag.graph.schema.RELATIONSHIP_TYPES` vocabulary (longest names first), rather than a generic "looks uppercase" pattern.

**Regression test**
`tests/unit/test_context_normalize.py::test_graph_result_normalizes_to_graph_path_source_type` asserts the parsed relation is `"DEPENDS_ON"`, not `"API"`, for exactly this entity-name shape.

---

### CTX-003: Test doubles mutating the process-wide cached `load_config()` singleton silently polluted later tests in the same run

**Status:** FIXED
**Severity:** MINOR
**Area:** Testing methodology
**Affected code:** New `tests/unit/test_context_*.py` files

**Problem**
Several new Context Engineering unit tests passed in isolation but failed when the full `tests/unit` suite ran together, in both directions (a test expecting a small window read the real 32K default; a different test's model-override assumption silently applied to an unrelated later test).

**Failure scenario**
`AppConfig.load_config()` is `lru_cache`'d (see `config.py`): it returns the *same* mutable Pydantic instance to every caller in a process. Several new tests did `config = load_config()` and then mutated a nested field in place (`config.context.model_limits.default_context_window_tokens = 1000`), permanently changing the shared singleton for every test that ran afterward in the same pytest process, the exact same class of hazard this project's own `tests/unit/conftest.py` `RAG_CONFIG_PATH` autouse fixture was written to close for a different field.

**Impact**
Order-dependent test failures that passed under `pytest path/to/one_file.py` but failed under the full-suite run, the hardest kind of test bug to reproduce from a bug report alone.

**Fix applied**
Changed every mutating test to `load_config().model_copy(deep=True)` before mutating, the same pattern `tests/unit/test_config.py` already established for exactly this reason. Also discovered and fixed a second variant of the same hazard: `config.context.model_limits.default_context_window_tokens` is only consulted when the active `generation.model_name` has no entry in `model_limits.overrides`; several tests set the default without also clearing `overrides`, so the real default (32768, from the `qwen2.5:1.5b` override) silently won regardless of the test's intended window size.

**Regression test**
The corrected tests (`tests/unit/test_context_planner.py`, `test_context_model_limits.py`, `test_context_tokens.py`, `test_context_engine.py`) now pass identically whether run alone or as part of the full `tests/unit` suite; this was verified directly (not merely fixed and assumed), the same "flag the gap, don't imply more was checked than was" discipline this file follows elsewhere.

---

### CTX-004: Extractive compression cannot shrink text with no sentence boundaries, and graph-path hop structure is a best-effort text parse, not real structured data

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Context Engineering
**Affected code:** `rag.context.compression.compress_item`, `rag.context.graph_packing.parse_graph_fact_text`

**Problem**
Two related, deliberate scope boundaries in the Context Engineering subsystem, found and confirmed while writing its own regression tests, not designed in from the start.

**Failure scenario**
(1) `compress_item`'s sentence-boundary trimming (both `"sentence_trim"` and `"extractive"`) requires at least one `.`/`!`/`?` in the text; a prose blob with none (e.g. a bare, punctuation-free word list) is treated as one unsplittable unit and is either kept whole or dropped, never cut at an arbitrary character offset, per section 10's explicit prohibition on that. (2) `rag.graph.fusion.graph_fact_to_search_result` already collapses a `GraphFact`'s structured `entities`/`fact_type` into one rendered prose string before a `SearchResult` ever exists, so `rag.context.graph_packing.parse_graph_fact_text`'s hop reconstruction is a best-effort regex parse of that rendering, not a read of real structured data, and returns `[]` for any fact text it doesn't recognize.

**Impact**
(1) is rare in this corpus (real documents are prose/markdown/tables/code, all of which have the structural markers their respective compression paths expect) but is a real, reachable gap for unusual content. (2) only affects the optional hop-aware display/shrink enrichment; it never affects correctness, since `ContextItem.content` always carries the full original text regardless of whether parsing succeeds.

**Recommended fix**
(1) would require either a non-sentence-based fallback splitter (e.g. fixed-size word windows) or accepting that some evidence is genuinely atomic; not implemented, since inventing a splitter for content this corpus doesn't contain would be speculative. (2) would require `rag.graph.fusion` to preserve `GraphFact.entities`/`fact_type` through to `ContextItem` directly rather than re-deriving it from rendered text; not done this milestone, scoped out to avoid changing the existing, tested `rag.graph.fusion` contract for a display-only enrichment.

**Notes**
Both are documented, not hidden: `rag.context.compression`'s and `rag.context.graph_packing`'s own module docstrings state this directly, and `docs/context_engineering.md`'s "Known limitations" section repeats it for anyone evaluating the design.

---

### CTX-005: The context budget assumed a model window that nothing ever told Ollama about

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Context Engineering
**Affected code:** `OllamaLLM.generate`, `rag.factory.build_llm`

**Problem**
`config.context.model_limits` assumes a per-model context window (32768 for `qwen2.5:*`), but `OllamaLLM.generate()` never set `options.num_ctx`, so Ollama's own real runtime window (its default, independent of a model's trained length) could silently differ from what the budget planner assumed.

**Failure scenario**
A packed prompt reported as "within budget" against the assumed window could still be silently truncated by the real Ollama server running at a smaller actual window, with no error and no metric ever showing it.

**Fix applied**
`build_llm` now resolves `num_ctx` from the same `rag.context.model_limits` table the budget planner uses and passes it explicitly to `OllamaLLM`, which forwards it as `options.num_ctx`. The assumption and the real runtime setting can no longer silently disagree.

**Regression test**
`test_generate_passes_num_ctx_in_options_when_configured` in `tests/unit/test_ollama_llm.py`; `test_build_llm_resolves_num_ctx_from_the_context_engineering_model_limits_table` in `tests/unit/test_factory.py`.

**Notes**
Measuring this directly on this project's 8GB-RAM CPU-only host found the assumed 32768 window impractical for `qwen2.5:3b` (a single trivial call took ~6.6 minutes; see `config/experiments/context-engineering-v1.yaml`'s own comments for the numbers). The fix makes the real cost of a given window *visible* rather than removing the cost; calibrate `model_limits.overrides` per deployment.

---

### CTX-006: Packed token counts omitted rendered overhead, and agent synthesis budgeted the wrong template

**Status:** FIXED
**Severity:** MAJOR
**Area:** Context Engineering
**Affected code:** `ContextEngine.build`, `RetrievalPipeline.build_context_text`, `agent/graph.py:_synthesize`

**Problem**
An item's `token_count` was exactly `count(item.content)`, never including the `"[Source N: ...]"` label or the `"---"` join separator `render_context()` actually adds. Separately, the agent's `_synthesize` node renders with its own `agent_synthesize_v2.yaml` template but called `build_context_text` with no override, so the budget measured the *pipeline's* classic-RAG system template's token cost instead.

**Failure scenario**
A packing decision could admit (and the reported `packed_token_count` could under-report) real per-item rendered overhead; an agent-routed synthesis call's budget could be computed against an entirely different system prompt's length than the one actually sent.

**Fix applied**
Each item's `token_count` now also includes a representative render-overhead estimate (`_render_overhead_text`). `build_context_text` gained an optional `system_prompt_text` override; `_synthesize` passes its own template's `system_template` explicitly.

**Regression test**
`test_item_token_count_includes_citation_label_and_separator_overhead` in `tests/unit/test_context_engine.py`; `test_synthesize_passes_its_own_templates_system_template_for_budgeting` in `tests/unit/test_agent_synthesize_budgets_its_own_template.py`.

---

### CTX-007: Deduplication could merge conflicting facts, and could cross a tenant boundary

**Status:** FIXED
**Severity:** MAJOR
**Area:** Context Engineering
**Affected code:** `rag.context.dedup.deduplicate_items`

**Problem**
The near-duplicate pass merged on word-shingle overlap alone; two long, otherwise-identical passages differing only in one number (e.g. "timeout 60" vs. a current "timeout 30") could still exceed the overlap threshold and get merged, silently discarding whichever value lost. Separately, neither the exact-content-hash nor the near-duplicate pass checked `tenant_id`, so a legitimately multi-tenant candidate set (a `cross_tenant_support_roles` request) could merge one tenant's citation into another's `merged_from`.

**Failure scenario**
A stale and a current value for the same fact collapsing into one, with no signal for which survived; or a shared-boilerplate fact ending up attributed to the wrong tenant's document in the final citation list.

**Fix applied**
The near-duplicate pass now refuses to merge two items whose extracted numeric tokens differ. Both the exact-hash and near-duplicate passes now require matching `tenant_id`; `content_hash -> kept index` tracks a list per hash (not a single overwritable slot) so a later same-tenant duplicate still finds its own match after a different tenant's byte-identical item was correctly kept separate.

**Regression test**
`test_near_duplicate_paragraphs_with_a_conflicting_number_are_never_merged`, `test_exact_content_hash_duplicate_across_tenants_is_never_merged`, `test_near_duplicate_across_tenants_is_never_merged`, `test_exact_content_hash_matching_finds_the_right_tenants_entry_among_several` in `tests/unit/test_context_dedup.py`.

---

### CTX-008: A tokenizer failure during budget planning escaped the documented graceful fallback

**Status:** FIXED
**Severity:** MODERATE
**Area:** Context Engineering
**Affected code:** `rag.context.planner.plan_context_budget`

**Problem**
Evidence/history counting goes through `ContextEngine._count`'s try/except char-estimate fallback, but the planner called a raw `TokenCounter.count()` directly for the system/query text, bypassing that fallback entirely.

**Failure scenario**
A tokenizer failure during budget planning (as opposed to evidence counting) turned into an unhandled exception reaching the caller, even with zero evidence/history to blame it on.

**Fix applied**
`plan_context_budget` now takes a `count_tokens: Callable[[str], int]` instead of a raw `TokenCounter`; `ContextEngine.build` passes its own safe-counting wrapper, the same one evidence/history counting already uses.

**Regression test**
`test_build_degrades_gracefully_when_the_token_counter_always_raises` in `tests/unit/test_context_engine.py`.

---

### CTX-009: Conversation-history evidence never got injection-flagged

**Status:** FIXED
**Severity:** MODERATE
**Area:** Context Engineering / security telemetry
**Affected code:** `rag.context.normalize.normalize_conversation_turn`

**Problem**
Every other evidence type (document, graph, MCP, tool-fetched) carries `injection_suspected` through from `detect_injection`; history turns never ran that check at all, so the flag was structurally always `False` for them.

**Failure scenario**
The actual trust boundary (history renders as data, never as an instruction) held regardless, but `prompt_injection_success_rate`-style telemetry had zero visibility into history-borne injection attempts, and a forged/replayed turn got no reinforcement annotation a suspicious document chunk already gets.

**Fix applied**
`normalize_conversation_turn` now runs `detect_injection` on the turn's content, same as every other evidence path.

**Regression test**
`test_conversation_turn_flags_suspected_injection` in `tests/unit/test_context_normalize.py`.

---

### CTX-010: Adaptive budget classification counted post-dedup evidence, contrary to its own contract

**Status:** FIXED
**Severity:** MINOR
**Area:** Context Engineering
**Affected code:** `ContextEngine.build`, `rag.context.planner.classify_context_query_type`

**Problem**
`classify_context_query_type`'s own docstring says its input should be pre-dedup and exclude history; `ContextEngine.build` called it with the post-dedup `deduped` list instead.

**Failure scenario**
Harmless while history items aren't counted by the bucket logic, but duplicate graph/document/tool items collapsing before classification runs could flip a genuinely relationship-dominant request into reading as knowledge-only, applying the wrong adaptive budget split.

**Fix applied**
`ContextEngine.build` now passes the pre-dedup `evidence_items` to the budget planner, matching the documented contract.

**Regression test**
`test_adaptive_classification_uses_pre_dedup_evidence_composition` in `tests/unit/test_context_engine.py` (three duplicate graph facts plus two distinct documents, genuinely graph-dominant pre-dedup).

---

### CTX-011: The Context Engineering benchmark clamped utilization to 1.0, hiding real overflow

**Status:** FIXED
**Severity:** MODERATE
**Area:** Context Engineering / evaluation
**Affected code:** `rag.eval.run_context_eval.run_scenario`

**Problem**
`context_utilization` was clamped to `1.0` in the synthetic benchmark report, so a `top_n` baseline run overflowing its budget by 600% read identically to one that landed exactly at capacity.

**Failure scenario**
`docs/context_engineering.md`'s own benchmark table claimed "every run respected its computed budget exactly," which the project's own committed `report.json` directly contradicted (5/240 `top_n` rows overflowed at larger candidate sizes).

**Fix applied**
`context_utilization` is now reported unclamped; `context_overflow`/`context_excess_tokens` fields were added, and the benchmark report/table/doc text were corrected and regenerated.

**Regression test**
`test_top_n_baseline_genuinely_overflows_and_reports_it_honestly` (and the updated `test_full_pipeline_never_exceeds_budget...` now also asserting `context_overflow is False`) in `tests/unit/test_run_context_eval.py`.

---

### CTX-013: `run_eval.py` silently dropped `answer()`'s own `context_engineering` field

**Status:** FIXED
**Severity:** MINOR
**Area:** Context Engineering / evaluation
**Affected code:** `rag.eval.run_eval.evaluate`

**Problem**
`RetrievalPipeline.answer()` returns a `"context_engineering"` key (a `context_engineering_summary()` dict: utilization, reduction, dedup/compression counts, truncation, build latency) whenever `config.context.enabled` is true, but `run_eval.py`'s generation loop read every other key off that same result dict (`answer`, `sources`, `retrieval_ms`, `prompt_tokens`, ...) and never this one. A `run_eval.py` report produced with Context Engineering turned on looked identical to one with it off.

**Failure scenario**
A reviewer comparing a Context-Engineering-enabled report against a baseline would have no deterministic signal that Context Engineering was exercised at all, short of re-reading the config.

**Fix applied**
`evaluate()` now records each `answer()` call's `context_engineering` dict onto its `per_example` entry, and (when at least one example populated it) reports a `report["context_engineering"]` aggregate (`packing_strategy`, mean utilization/reduction/items-compressed/dropped-item-count, truncated-run count, mean build latency).

**Regression test**
Verified by unit tests exercising `evaluate()` with `config.context.enabled=True` and inspecting the resulting report's `context_engineering` key.

---

### EVAL-014: `sources`/citations could credit evidence Context Engineering had already dropped or compressed

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Evaluation / Context Engineering
**Affected code:** `RetrievalPipeline.build_context_text`/`answer`, `GraphRAGPipeline._generate_from_results`, `agent/graph.py:_synthesize`

**Problem**
`sources`/citations (and, downstream, RAGAS's judge `contexts`) were always built from the pre-packing retrieval result, never from what Context Engineering's packing step actually included in the rendered prompt.

**Failure scenario**
A packing decision dropping or compressing a chunk left `sources` (and the RAGAS judge context built from it) still crediting the original, un-packed evidence, so citation-accuracy and judge-faithfulness metrics could be scored against evidence the model was never actually shown.

**Fix applied**
`build_context_text` now also returns the subset of results that actually reached the rendered prompt (content-adjusted for any compressed item), and all three generation call sites build `sources`/citations from that value instead of the pre-packing list. Byte-identical to the old behavior when Context Engineering is disabled.

**Regression test**
`test_answer_sources_reflect_what_context_engineering_actually_packed` in `tests/unit/test_retrieval_pipeline.py`.

---

## 5. Agent, MCP and Workflow Issues

### AGT-001: Step budgets collided with the happy path, mislabeling most runs as max_steps

**Status:** FIXED
**Severity:** MODERATE
**Area:** Agentic RAG / bounded execution
**Affected code:** Step-budget check ordering vs. evidence-sufficiency check

**Problem**
In the first agentic baseline, 12 of 17 agent-routed questions ended with `termination_reason: "max_steps"`, even when answers were grounded and cited correctly.

**Failure scenario**
Step arithmetic: classify (1), decompose (2), then 3 steps per tool loop; two tool calls land exactly on the limit of 8. The step-bound check ran *before* the sufficiency check, so a run converging on its last allowed step was mislabeled as a budget stop rather than a successful convergence. The retry bound (2 attempts) also fires on the same step, so the two bounds always tied.

**Fix applied**
The sufficiency check now runs first on that step; a run still genuinely short of evidence at the bound still correctly reports `max_steps`.

**Regression test**
Two unit tests cover both cases (converged-at-bound vs. genuinely-insufficient-at-bound). The precedence between the two bounds when they tie was documented and deliberately left as-is, a semantic choice, not a bug.

---

### AGT-002: Citation metrics measured the wrong set and a model that rarely cites

**Status:** FIXED
**Severity:** MODERATE
**Area:** Agentic RAG / evaluation
**Affected code:** `citation_support_rate`

**Problem**
`citation_support_rate` read 0.111 for the agent route vs. 0.389 for classic RAG on the same 18 questions, suggesting (incorrectly, at face value) that the agent cited unsupported sources more often.

**Failure scenario**
The agent's citation list held every chunk gathered across *all* tool calls (typically 8–16); classic RAG listed only the 3 chunks actually shown to the model. Any unused, loosely related chunk failed the whole example under the old metric definition. Fixing that then exposed a second, real finding: 16 of 18 answers never wrote an explicit `(Source N)` citation at all.

**Fix applied**
The metric now scores only sources the answer actually cites by number; a keyword-overlap fallback for uncited answers is tracked separately as `inferred`, distinct from `explicit`.

**Regression test**
Post-fix, the explicit-citation sample was 2 of 18 at rate 1.0; the small sample size is reported alongside the rate rather than hidden.

---

### AGT-003: A classify fix exposed a RAGAS gap on the classic route

**Status:** FIXED
**Severity:** MODERATE
**Area:** Agentic RAG / RAGAS integration
**Affected code:** RAGAS context-building from agent evidence

**Problem**
After a routing fix sent more traffic through the classic path, 5 of 126 RAGAS judge jobs failed with "retrieved_contexts is missing."

**Failure scenario**
The RAGAS harness built contexts from the agent's tool-call evidence structure, which the classic route never populates. The underlying gap pre-dated the fix, but the old routing sent almost everything to the agent, so the classic branch rarely ran and the gap stayed latent.

**Fix applied**
The classic result now carries its own `sources` list, which the harness uses for classic-routed rows.

**Regression test**
3 of 5 failures fixed (`context_precision` 0.482→0.559, `context_recall` 0.481→0.537); the remaining 2 are intended egress-policy blocks on confidential sources, not bugs.

---

### AGT-004: The tool worked, but the agent rarely chose it (specialized-tool selection)

**Status:** IN PROGRESS
**Severity:** MAJOR
**Area:** Agentic RAG / tool routing and selection
**Affected code:** `_looks_like_case_mutation_request`, `_classify_query`, tool-selection prompt

**Problem**
A 14-question subset, each requiring exactly one specific tool, found that individual agent tools passed all deterministic unit tests, but in live evaluation the model rarely selected `get_document`/`get_latest_document`/`get_related_context`, using generic search instead. Two identical "close case" requests were also misclassified `simple` and routed to the classic path, which has no tool access at all.

**Failure scenario**
`routing_accuracy` read 0.857 (12/14) because of the two misrouted write-action requests. Separately, the three named local tools reached 0.0 reachability, the model answered with extra generic search calls every time instead.

**Impact**
A routing decision gating a security-relevant write path (`update_case_status`) depended on prompt compliance alone; a tool that is correct in isolation is useless if the model never selects it.

**Fix applied (routing sub-issue only)**
A deterministic regex check (`_looks_like_case_mutation_request`) now force-overrides the LLM's classification to `complex` for case-mutation-shaped queries, regardless of what the model itself decided.

**Result of fix applied**
`routing_accuracy` 0.857→1.000 (14/14); `update_case_status` reachability 0.5→0.75; `case_action_outcome_accuracy` 0.333→0.667; `denial_leakage_rate` 0.0 before and after; `tool_success_rate` 1.0 across 22 dispatches.

**What remains open**
The three local tools' 0.0 reachability was explicitly **not changed** by this fix. The model still prefers generic search over `get_document`/`get_latest_document`/`get_related_context` once correctly routed to the agent path. One denial row's argument mismatch also remains unconfirmed (raw tool arguments are not recorded in this eval).

**Required regression test**
A live re-run of the same 14-question subset measuring `get_document`/`get_latest_document`/`get_related_context` reachability specifically; currently no evidence this has moved off 0.0.

**Notes**
"The tool works when called" and "the model calls it when it should" are separate claims requiring separate tests, this entry is the canonical example in this codebase.

---

### AGT-005: Tool-selection prompt's chunk_id fix is unverified against live tool selection

**Status:** NEEDS VERIFICATION
**Severity:** MODERATE
**Area:** Agentic RAG / tool-selection prompt correctness
**Affected code:** `_summarize_evidence`, tool-select prompt template

**Problem**
The `get_related_context` tool requires a real `chunk_id`; the tool-select prompt told the model to use one from gathered evidence, but the evidence summary shown to the model omitted chunk ids entirely.

**Failure scenario**
The model had to invent a `chunk_id`; an invented id resolves to nothing, and the tool silently returns an empty result with no error, a silent failure mode.

**Fix applied**
The evidence summary now includes `chunk_id=<id>` on each line; a new prompt version tells the model to copy that value verbatim, never the bracketed display index.

**What needs verification**
A unit test runs a full agent flow and confirms the id *shown* to the model is the id *dispatched*, proving the mechanism works. However, the live 18-question re-run did not happen to select the local tools at all, so there is no live evidence this fix actually changes model behavior in practice, only that the plumbing is correct when exercised directly.

**Required verification**
A live run where the model actually selects `get_related_context`, confirming it copies the shown `chunk_id` rather than inventing one.

---

## 6. Evaluation and Observability Risks

### EVAL-001: A keyword-overlap answer metric penalized correct refusals

**Status:** ACCEPTED LIMITATION
**Severity:** MODERATE
**Area:** Evaluation
**Affected code:** `answer_quality` (`KeywordOverlapScorer`)

**Problem**
The placeholder `answer_quality` metric scored unanswerable questions far lower than answerable ones (0.243 vs. 0.466 in an early run).

**Failure scenario**
A correct refusal shares almost no keywords with the reference phrasing, so the gap is a ceiling of the heuristic itself, not a real quality difference. The same metric also rewarded wrong answers containing an incidental matching keyword.

**Recommended fix**
None patched; documented as an inherent limitation of a keyword-overlap heuristic. RAGAS's `faithfulness` metric was adopted as the trusted signal for hallucination checks instead.

**Notes**
Every report now carries an explicit caveat that `answer_quality` is unreliable specifically on unanswerable questions.

---

### EVAL-002: A 15-question pilot hid a failure that the full 62-question set exposed

**Status:** FIXED (process corrected)
**Severity:** MODERATE
**Area:** Evaluation / methodology
**Affected code:** RAGAS evaluation sampling practice

**Problem**
A stratified 15-question pilot's refusal sample had only 2 unanswerable questions, both correctly refused, a 100% correct-refusal rate, before paying for the full RAGAS run.

**Failure scenario**
Scaling to the full 62-question gold set (9 unanswerable questions) found one confidently fabricated answer; correct-refusal dropped from 100% (n=2) to 89% (n=9). Refusal behavior had not actually changed between runs, a sample of 2 was simply too small to contain a failure already present.

**Fix applied (process)**
Established as a standing rule for every later paid evaluation: validate tooling/cost/judge reliability on a small sample, then scale up before trusting the small sample's result.

**Notes**
On the full set, `faithfulness` agreed with manual human review 85% of the time, the most trustworthy RAGAS metric measured in this project.

---

### EVAL-003: A keyword heuristic produced false regression counts, and a test relied on it

**Status:** FIXED
**Severity:** MODERATE
**Area:** Evaluation / methodology
**Affected code:** Refusal-change keyword scanner, an integration test's assertion

**Problem**
A keyword scanner flagging refusal-vs-answer changes between two configs kept producing misleading headline counts.

**Failure scenario**
Reading flagged answers by hand changed the story every time: a correct answer was flagged for an incidental hedging clause; a genuine refusal was missed for lacking an exact literal phrase. Across at least 4 experiments, roughly 40–50% of first-pass "regressions" were not real. A pytest assertion built on the same heuristic failed on a correct answer phrased outside its literal phrase list.

**Fix applied**
Heuristics are kept only as triage aids; every claim built on one is checked against full answer text. The specific failing integration test was rewritten to assert a structural property (no fabricated specific figure) instead of a phrase match.

**Regression test**
The rewritten test asserts the harder-to-game structural claim directly.

---

### EVAL-004: Generation dominated latency, and the latency A/B could not be trusted

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Evaluation / benchmarking methodology
**Affected code:** Latency A/B/C experiment on a shared host

**Problem**
An A/B/C test changed `generation_context_top_n` and `max_tokens` one at a time to measure latency impact.

**Failure scenario**
Milliseconds-per-completion-token swung 4–5x across three back-to-back runs with identical model and token counts; decode cost per token should be constant on fixed hardware. The swing traced to host-level noise (a shared Windows machine), not either config knob. A later run's apparent 35% slowdown matched the same host-noise rate exactly, disproving the initially suspected cause.

**Impact**
Wall-clock latency comparisons on this shared host are not a reliable signal in isolation.

**Recommended fix**
Compare normalized rates (tokens/second) rather than wall-clock totals; no production config change made from this experiment. `max_tokens: 256` was confirmed safe (max observed completion 154 tokens) independent of the noisy latency question.

---

### EVAL-005: Safety metrics read as a perfect zero until the gold paths were checked

**Status:** FIXED
**Severity:** MAJOR
**Area:** Evaluation / gold-data correctness
**Affected code:** Gold file path-suffix matching, `security_evaluation/` directory layout

**Problem**
The first full run of new authorization metrics gave `0.0` for every one, suspicious for a brand-new control; Recall@5 on the 126-question set also dropped to 0.579.

**Failure scenario**
A hand-run query for a "forbidden" question returned the forbidden document in its top results, retrieval and authorization were actually working, the gold-file *matching* was failing. Four security categories were authored in the gold file as nested under `security_evaluation/<category>/`, but the actual corpus had those folders sitting flat, so path-suffix matching failed on every one of those ~40 gold rows.

**Fix applied**
Moved the four folders to match the authored nesting; re-ingested and verified the match directly before re-running evaluation.

**Regression test**
Affected rows now resolve correctly instead of failing closed on every question.

**Notes**
A perfect `0.0` on a brand-new security control is itself a signal to check the measurement, not celebrate the result.

---

### EVAL-006: `oversized_request_rejection_accuracy` read 0/3, because the probe token was rejected first

**Status:** FIXED
**Severity:** MODERATE
**Area:** Evaluation / API security probes
**Affected code:** DoS-probe token minting, PyJWT required-claims check

**Problem**
During a prompt A/B across v3/v4/v5, DoS-limit probes were never actually rejected with a 4xx; all three returned 401 instead.

**Failure scenario**
The probe token had `tenant_id: None`. PyJWT's `require` check treats a null claim the same as a missing one, so the probe was rejected for authentication before the DoS check ever ran. The same fix then exposed a second masked probe: wrong-issuer/wrong-audience payloads passed under the default config, since those claims are only validated when explicitly configured.

**Fix applied**
The probe token now carries a real tenant string; issuer/audience probes run only when those settings are actually configured.

**Regression test**
`oversized_request_rejection_accuracy` 0/3 → 3/3; `authentication_failure_acceptance_rate` correctly stayed at 0.

**Notes**
A security probe that fails for an unrelated reason gives a false pass; identical results across unrelated prompt variants was the tell.

---

### EVAL-010: The corpus-wide duplicate-secret scan was not scoped to dataset_id

**Status:** FIXED
**Severity:** MAJOR
**Area:** Evaluation / security-metric correctness
**Affected code:** `run_eval._fetch_chunks_for_duplicate_scan`

**Problem**
This was the one query in the module not scoped by `dataset_id`, contrary to the module's own stated invariant; it scanned every chunk in the table regardless of which dataset was under evaluation.

**Failure scenario**
If multiple datasets (e.g. `techfusion`, `sample_docs`, graph/ontology/context-eval corpora) ever share one Postgres instance for a consolidated benchmark run, `duplicate_sensitive_field_miss_rate` for one dataset would silently include findings from an unrelated one.

**Fix applied**
Added `WHERE dataset_id = %s`, threaded the parameter through from the one real caller.

**Regression test**
`test_duplicate_scan_query_is_scoped_to_dataset_id` in `tests/unit/test_run_eval.py`.

---

### EVAL-011: Relationship expansion could inflate Recall@K/Hit@K/MRR, and MRR had no rank cutoff

**Status:** FIXED
**Severity:** MAJOR
**Area:** Evaluation / retrieval-metric correctness
**Affected code:** `run_eval.evaluate`, `metrics.reciprocal_rank`

**Problem**
The primary ranking-quality population (`all_retrieved_sources`) included every result `retrieve()` returned, including `origin="expanded"` relationship-expansion filler appended after the primary ranked list. `reciprocal_rank` itself has no cutoff at all.

**Failure scenario**
Whenever a gold question's primary candidate pool was thinner than the fixed `RETRIEVAL_K=10`, expanded chunks filled the measured top-K slots (or, for MRR, contributed from any rank), inflating ranking metrics in a way a vector/hybrid-only baseline with expansion off would never produce for the identical primary ranking.

**Fix applied**
The population feeding Recall@K/Hit@K/MRR is now filtered to `origin == "retrieved"` only, leaving the broader (expansion-inclusive) list in place for the safety-leak checks that intentionally want it.

**Regression test**
`test_recall_and_mrr_exclude_expanded_origin_results_from_the_primary_population` in `tests/unit/test_run_eval.py`.

---

### EVAL-012: Duplicate relevant-document ids in gold data inflated Recall@K

**Status:** FIXED
**Severity:** MODERATE
**Area:** Evaluation / metric correctness
**Affected code:** `metrics.recall_at_k`

**Problem**
`relevant_ids` was never deduplicated; a gold row listing the same id twice counted it twice against an un-deduplicated denominator.

**Failure scenario**
`retrieved=["a"]`, `relevant=["a","a","b"]` computed `2/3` instead of the correct `1/2` over the distinct relevant set. Not currently firing on the 126-row `techfusion_gold.jsonl` (no duplicates present), but a real risk for any future or smaller gold file.

**Fix applied**
`relevant_ids` is now deduplicated (`dict.fromkeys`) before scoring, the standard definition of Recall@K over a *set* of relevant items.

**Regression test**
`test_recall_at_k_deduplicates_repeated_relevant_ids` in `tests/unit/test_metrics.py`.

---

### EVAL-013: A graph-strategy comparison could silently relabel every row as vector-only

**Status:** FIXED
**Severity:** CRITICAL
**Area:** Evaluation / GraphRAG
**Affected code:** `GraphRAGPipeline.answer`, `run_graph_eval.compare_strategies`/`evaluate_strategy`

**Problem**
`GraphRAGPipeline.answer()` silently substitutes plain vector retrieval whenever `graph.enabled=False`, with nothing in the returned dict indicating a substitution happened; `evaluate_strategy` labeled every result row with the *requested* strategy regardless of what actually ran.

**Failure scenario**
Running the graph evaluator's comparison under the shipped, graph-disabled default with all four strategies requested would produce four differently-labeled rows that all executed identical vector-only retrieval, a plausible-looking comparison table that could invalidate a Hybrid-vs-GraphRAG conclusion with no execution error at all.

**Fix applied**
`GraphRAGPipeline.answer()` now reports `graph_strategy_executed` on every path, including every internal fallback. `compare_strategies` raises `ValueError` up front if a non-`vector_only` strategy is requested while `graph.enabled=False`, rather than silently degrading; `evaluate_strategy` also records a per-row `degraded`/`degraded_run_rate`.

**Regression test**
`tests/unit/test_graph_pipeline_strategy_labeling.py` (three tests: the disabled-graph mislabel, the correct `vector_only` label, and `compare_strategies` raising).

---

### EVAL-015: No experiment record captured which git commit produced it

**Status:** FIXED
**Severity:** MAJOR
**Area:** Evaluation / reproducibility
**Affected code:** `rag.eval.run_metadata` (new), every `run_*.py` report builder, the three `scripts/record_*.py` scripts

**Problem**
Every experiment-recording path captures an extensive model/config/corpus field list, but none captured the exact code version (commit, branch, dirty state) that produced it.

**Failure scenario**
A benchmark run against uncommitted code becomes permanently unattributable to a specific commit the moment that code is later committed, amended, or rebased.

**Fix applied**
Added `rag.eval.run_metadata.capture_git_metadata()` (best-effort, never raises); wired a `run_metadata` field into every `run_eval`/`run_ragas_eval`/`run_agent_ragas_eval`/`run_agent_eval`/`run_graph_eval`/`retrieval_attribution` report, and flattened `git_commit_sha`/`git_branch`/`git_dirty` into the flat record schema in `record_experiment.py`/`record_agent_experiment.py` (the third, `record_attribution_experiment.py`, inherits it automatically since it spreads the whole raw report).

**Regression test**
`tests/unit/test_run_metadata.py` (a real-repo check plus a simulated git-unavailable degrade).

---

### OBS-001: `/health` depends on a heavy pipeline dependency through a graph check

**Status:** IN PROGRESS
**Severity:** MAJOR
**Area:** Observability / health-check design
**Affected code:** `GET /health`, `get_graphrag_pipeline` → `get_retrieval_pipeline()`

**Problem**
After GraphRAG was added, a concurrency test asserting `/health` stays responsive during a slow `/ingest` failed: `/health` took 7.2s against a 0.5s limit.

**Failure scenario**
The graph health check calls `get_graphrag_pipeline()`, which calls `get_retrieval_pipeline()`, which loads the embedding model on first use. The test had overridden other health dependencies but not this one, so the first `/health` call paid the full model-load cost (~5078ms observed). The event loop itself was not blocked, the test was measuring the wrong thing, but the underlying design issue is real.

**Fix applied (test only)**
The test now overrides `get_graphrag_pipeline` with a stub; product code is unchanged.

**What remains open**
In production, the first real `/health` call still loads the embedding model just to answer a graph check. A health route should depend only on cheap checks; the graph check should not reach through the full retrieval pipeline. **Not fixed yet.**

**Recommended fix**
Give the graph health check its own lightweight connectivity probe that does not transitively construct the embedding/retrieval singleton.

**Required regression test**
A `/health` latency assertion against a cold process, with no dependency pre-warmed, to catch this class of cost regression going forward.

---

### OBS-002: Shared mutable LLM token counters could leak across concurrent requests

**Status:** FIXED
**Severity:** MODERATE
**Area:** Observability / concurrency
**Affected code:** `OllamaLLM.last_prompt_tokens`/`last_completion_tokens`

**Problem**
`OllamaLLM` is a process-wide singleton; its token counts were plain instance attributes, written by `generate()` and read by the caller afterward.

**Failure scenario**
Two concurrent requests sharing the singleton: request A's `generate()` returns, but before A reads `last_prompt_tokens`, request B's concurrent `generate()` call overwrites it with B's own counts, so A reports B's usage.

**Fix applied**
Both counters are now backed by `threading.local()`, exposed as read-only properties with the same names, so each calling thread sees only its own most recent call's counts. No call-site change needed.

**Regression test**
`test_concurrent_calls_on_one_shared_instance_never_cross_token_counts` in `tests/unit/test_ollama_llm.py` (two threads, forced interleaving via `threading.Event`, each asserting its own counts).

---

## 7. Deployment and Operations

### OPS-001: A 504 at the frontend came from the host running out of memory

**Status:** FIXED
**Severity:** MODERATE
**Area:** Deployment / resource exhaustion
**Affected code:** nginx `proxy_read_timeout`, host memory/model lifecycle

**Problem**
After an unrelated fix, every classic query in the web UI returned 504, despite the fix itself being verified correct.

**Failure scenario**
Traced layer by layer: nginx's 60s read timeout expired waiting on the upstream; a direct request to the API hung for 3 minutes; a direct request to Ollama also hung, with CPU sampling showing it stalled, not busy. Free RAM was 0.25–0.4GB of 7.79GB with two models loaded at once. Stopping models through the normal command did not finish; killing the parent process left the actual inference engine running as an orphan holding ~885MB (Windows does not kill child processes when a parent is killed).

**Fix applied**
Killed the orphaned inference process directly to reclaim memory; raised the `/query` proxy read timeout to 300s to match the agent route (the backend was in fact succeeding, just past nginx's own timeout: confirmed 79s backend success against a 60s nginx cutoff in a later recurrence).

**Regression test**
A test query returned 200 in 39s after the timeout fix.

---

### OPS-002: A frontend panel stayed "unavailable" because the reverse proxy never routed two endpoints

**Status:** FIXED
**Severity:** MODERATE
**Area:** Deployment / reverse proxy configuration
**Affected code:** nginx config, Vite dev proxy, `/info` and `/feedback` routes

**Problem**
The runtime-info panel showed "unavailable" against a healthy, directly-verified backend.

**Failure scenario**
Neither the production nginx config nor the dev proxy listed `/info` or `/feedback`; requests fell through to the SPA's catch-all, returning HTML with a 200 the panel couldn't parse. Every frontend test mocks `fetch` directly, so none of them could ever observe this class of routing gap.

**Fix applied**
Both proxies now explicitly route both paths, mirroring every other proxied route.

**Regression test**
`/info` now returns real JSON; `/feedback` reaches the backend and returns its own 401 correctly.

**Notes**
Mocked HTTP clients cannot test reverse-proxy routing; this requires a check against the actual built container.

---

### OPS-003: The MCP SDK's built-in auth did not match this system's trust model

**Status:** FIXED
**Severity:** MAJOR
**Area:** MCP integration
**Affected code:** MCP server identity resolution, mount lifecycle

**Problem**
The MCP server had to guarantee an LLM-driven client could never supply `tenant_id`/`roles` itself; the SDK's built-in bearer-auth assumes an external OAuth server, while this project mints and verifies its own HS256 tokens.

**Failure scenario**
Reading the installed SDK source directly found: the built-in auth needs an issuer URL and OAuth discovery routes this project doesn't provide; a resolver-marked parameter is correctly excluded from the generated tool schema; a closure-based resolver fails under postponed annotation evaluation; and mounting with a plain `app.mount()` never starts the session manager's lifespan.

**Fix applied**
Skipped the SDK's OAuth machinery entirely; every tool resolves identity via the same `verify_jwt` the HTTP API uses, with the lifespan entered explicitly. Two further gaps closed later: a 307 redirect on the bare mount path (rewritten server-side) and silently-dropped unknown tool arguments (now rejected).

**Regression test**
An end-to-end test drives a real MCP client against a live server, proving injected identity fields have zero effect; Docker verification confirmed the mount and rejection behavior.

---

### OPS-004: Two JWT claims that looked symmetric behaved differently when unconfigured

**Status:** FIXED
**Severity:** MAJOR
**Area:** MCP integration / JWT handling
**Affected code:** Internal service token minting, `iss`/`aud` claims

**Problem**
The agent's internal MCP calls failed with "Invalid or expired token" whenever the audience setting was left at its default, despite a validly-signed token.

**Failure scenario**
PyJWT raises `InvalidAudienceError` for *any* `aud` claim once the verifier has no expected audience configured; it never inspects an `iss` claim under the equivalent condition. The token unconditionally minted a placeholder `aud`, so every default (audience-unconfigured) deployment would fail. A separate, related fault: the ASGI transport's host address tripped the SDK's own DNS-rebinding protection.

**Fix applied**
`iss` is always included (harmless either way); `aud` is included only when a real audience is configured. The ASGI transport now uses a loopback address with a non-default port.

**Regression test**
Tests cover both audience settings and confirm the asymmetry directly; end-to-end dispatch now succeeds on the default config.

**Notes**
A single smoke test with a *configured* audience would never have caught this; the dangerous case is specifically the default, unconfigured path.

---

### OPS-005: A committed config change broke 112 unit tests

**Status:** FIXED
**Severity:** MODERATE
**Area:** CI / configuration management
**Affected code:** `config/default.yaml` security/retrieval flags

**Problem**
One commit flipped four security/retrieval flags to `true` in the default config; the full unit suite had 112 failures in files the change never touched.

**Failure scenario**
Relationship expansion alone caused 83 of the 112 failures, test doubles lacked the two methods expansion now called unconditionally.

**Fix applied**
Confirmed intended defaults with the repo owner (only the new agent flag was meant to stay `true`); reverted the other three flags and restored eight test files from git exactly as committed, rather than hand-undoing edits.

**Regression test**
112 failing → 0; final suite 615 passed, 3 skipped (optional deps not installed).

---

### OPS-006: A CI integration test assumed a corpus that CI never has

**Status:** FIXED
**Severity:** MODERATE
**Area:** CI
**Affected code:** New integration test file, `require_techfusion_corpus` fixture

**Problem**
A new integration test file passed locally and failed on every CI run of the same commit.

**Failure scenario**
Six tests read the real `techfusion` corpus, which is gitignored; CI only applies the schema, so no documents exist there at all. The last green CI run predated the commit that added the file, pointing at an environment difference, not flakiness.

**Fix applied**
A fixture skips those six tests with an explicit message when the corpus is absent, mirroring the existing Postgres-unavailable skip pattern.

**Regression test**
The six tests now skip cleanly in CI and run unchanged locally, with an honest skip reason rather than a false pass or a flaky failure.

---

### OPS-007: A CI retrieval gate could not reuse the documented baseline, because CI does not have the corpus

**Status:** FIXED
**Severity:** MODERATE
**Area:** CI
**Affected code:** `config/experiments/ci_eval.yaml`, `scripts/check_eval_gates.py`

**Problem**
A PR-blocking retrieval gate needed real floors; the obvious source was the already-recorded baseline for the main corpus.

**Failure scenario**
That corpus and its gold files are gitignored; only a small sample corpus/gold file are tracked. Reusing the main-corpus numbers against the tracked sample would either always fail or silently describe a different corpus entirely.

**Fix applied**
A dedicated, frozen CI config measures its own baseline directly against the tracked sample corpus, retrieval-only, no model call; floors are recorded with an explicit rationale per metric. Security invariants needing the private corpus are enforced as deterministic pytest assertions instead of eval-gate thresholds.

**Regression test**
The retrieval gate runs end to end in under a minute using only CI's existing Postgres service container.

---

### OPS-008: Five independently green branches broke each other when merged

**Status:** FIXED
**Severity:** MODERATE
**Area:** CI / merge integration
**Affected code:** Test doubles in two of five fix branches

**Problem**
Five fix branches each passed CI independently; after merging into `main`, six tests failed, none of which had failed on any individual branch.

**Failure scenario**
Two branches each changed a signature that a test double from a *different* branch still matched exactly; each branch's own tests were internally consistent, the break only appeared once the sibling change actually landed alongside it.

**Fix applied**
Widened the affected test doubles to accept the newer signatures loosely; only test code changed.

**Regression test**
The combined suite passes on `main`.

**Notes**
Green CI on separate branches never proves the union is green; always run the full suite again after a multi-branch merge.

---

### OPS-009: Four Docker problems that only appeared when the container ran

**Status:** FIXED
**Severity:** MODERATE
**Area:** Deployment / containerization
**Affected code:** `Dockerfile`, healthcheck, `docker-compose.prod.yml`

**Problem**
Containerizing the FastAPI service introduced four failures, none visible from reading the Dockerfile or Compose file, none failing loudly.

**Failure scenario**
(1) Image was 5.6GB because `sentence-transformers` pulled a CUDA `torch` wheel even on a CPU-only config. (2) Container crashed on config load: a `Path(__file__)`-based root calculation only worked in an editable install. (3) Healthcheck stuck "starting" because `localhost` resolved to `::1` and the minimal image has no IPv6 loopback. (4) Dev bind mounts leaked into the production override, since Compose merges `volumes`/`ports` by target across `-f` files rather than replacing them.

**Fix applied**
Install the CPU-only torch wheel first; set `PYTHONPATH=/app/src`; use `127.0.0.1` in the healthcheck; tag overridden keys with `!override`.

**Regression test**
Image dropped 5.6GB→1.6GB; health reached `healthy`; merged production config had one volume mount (not three) and no published Postgres port; a full smoke test passed on both dev and production-shaped stacks.

---

### OPS-010: A Kubernetes readiness probe could not see a database outage, because health was only in the response body

**Status:** FIXED
**Severity:** MAJOR
**Area:** Deployment / Kubernetes probes (learning exercise)
**Affected code:** `/health`, `/livez`, `/readyz`

**Problem**
Stopping Postgres during a Kubernetes exercise did not remove the API pod from service endpoints; the pod stayed in rotation while every real query would fail.

**Failure scenario**
`/health` reports degraded dependencies in its JSON body but always returns HTTP 200; an HTTP probe only reads the status code, so body-only degradation is structurally invisible to Kubernetes.

**Fix applied**
Added `/livez` (never checks dependencies) and `/readyz` (503 when a required dependency is down); `/health` kept its original contract unchanged. Startup, liveness, and readiness probes each wired to the matching route.

**Regression test**
On minikube with a slow first start, the old single-endpoint design crash-looped (liveness killed the container mid-startup); with separate probes, a startup probe absorbs the slow start and the rollout completes with zero restarts.

---

### OPS-011: A local Kubernetes tool hung at container creation on one host, and a different runtime build avoided it

**Status:** FIXED (worked around; root cause unproven)
**Severity:** MODERATE
**Area:** Deployment / local Kubernetes tooling (learning exercise)
**Affected code:** `kind` cluster creation on this specific host/kernel

**Problem**
`kind` never produced a running cluster on one development host; every attempt stalled at control-plane startup.

**Failure scenario**
Static pods were accepted for creation and never progressed, no error, no timeout. One-variable-at-a-time testing across cgroup version, memory, node image, `kind` release, node count, and orchestration path all reproduced the identical hang, matching a known `runc`-on-nested-overlay-filesystem hang class with newer WSL kernels. Multiple prebuilt WSL kernels were tried; none resolved it.

**Fix applied**
Switched orchestration tool, not the kernel: `minikube` with the Docker driver runs its node in a container using a different, newer containerd build, which does not hit the hang on the identical host/kernel.

**Regression test**
Full end-to-end validation on minikube: all `kube-system` pods Running, a throwaway deployment pulled/started/answered over its cluster service name from a second pod.

**Notes**
The cause is recorded as strong correlation, not a proven root cause, since the `kind` runtime build itself was never pinned down exactly. `kind` is kept documented as a reproducible negative result for this host.

---

### OPS-012: An nginx image worked in Compose and crash-looped in Kubernetes, because three opt-in settings were off

**Status:** FIXED
**Severity:** MODERATE
**Area:** Deployment / Kubernetes (learning exercise)
**Affected code:** nginx resolver config, worker-count setting

**Problem**
The frontend pod crash-looped on its Kubernetes probes; a manual request inside the pod hit the probe timeout, so nginx itself was genuinely slow, not merely misconfigured probes.

**Failure scenario**
Three stacked issues: (1) nginx's resolver was hardcoded to Docker's embedded DNS address, which doesn't exist in a Kubernetes pod. (2) The official image can derive that address from the container's own resolver config, but only behind an opt-in environment variable. (3) nginx's default worker count reads the *host* CPU count, not the container's CPU limit; cgroup stats showed throttling in 14% of scheduling periods (25 of 177).

**Fix applied**
Templated the nginx config to render resolver/upstream from the environment at startup; enabled the two opt-in variables; rewrote worker count from `auto` to `1` to match the 200m CPU limit.

**Regression test**
Frontend pod stayed Ready with zero further restarts on minikube; cgroup throttling stopped; Docker Compose unaffected, since the two opt-ins are harmless without a CPU limit present.

---

### OPS-013: A Helm first install failed on hook ordering and a network policy

**Status:** FIXED
**Severity:** MODERATE
**Area:** Deployment / Helm (learning exercise)
**Affected code:** Schema-init Job hook phase, default-deny `NetworkPolicy`

**Problem**
On a fresh install, the schema-init Job hung, then failed with a DNS error resolving the Postgres service name.

**Failure scenario**
Helm runs pre-install hooks *before* the chart's normal resources exist, so the Postgres service wasn't up yet. Moving only the Job to post-install exposed the identical issue in its ConfigMap (a pre-install delete policy removed it too early). A further "connection timed out" traced to a default-deny network policy allowing only the API's own label; the Job's pod carried no such label, so its packets were dropped silently, disproving an existing comment claiming the cluster didn't enforce policies.

**Fix applied**
Moved the Job and its ConfigMap to the same post-install phase together; gave the Job's pod its own label that the network policy allows.

**Regression test**
A fresh install now runs the schema job only after the Postgres service genuinely exists.

---

### OPS-014: A docs-only CI gate silently never skipped, because of a reversed reading of one option

**Status:** FIXED
**Severity:** MINOR
**Area:** CI / GitHub Actions
**Affected code:** `.github/workflows/ci.yml`'s `changes` job (`dorny/paths-filter`)

**Problem**
A gate meant to skip the five expensive CI jobs (`unit-tests`, `integration-tests`, `security-regression-gate`, `retrieval-eval-gate`, `docker-build`) on documentation-only pushes never skipped them, on any push, including single-file README/KNOWN_ISSUES.md-only changes.

**Failure scenario**
The filter listed five positive doc patterns (`*.md`, `**/*.md`, `docs/**`, `mkdocs.yml`, `LICENSE`) under `predicate-quantifier: every`, read (incorrectly) as "every changed file must match one of these patterns." The option actually means the opposite kind of "every": a single file only counts as a match for the filter if it satisfies *every pattern in the list simultaneously*, intended for combining one positive scope pattern with negative exclusions in a single filter (the action's own documented example). Five unrelated positive patterns under `every` can never all match one file at once (a file can't be named `*.md` and `mkdocs.yml` and `LICENSE` at the same time), so the filter's match count was zero on every run regardless of content, confirmed directly against two real CI runs (a one-file README.md change, a three-file docs-only change) before being traced to the action's own README rather than guessed at again.

**Fix applied**
Replaced the five positive patterns with their negated form under a filter named `code`, still using `predicate-quantifier: every`: a file now matches only if it satisfies every negation, i.e. it is *not* one of the doc patterns. `code` is true iff at least one real non-documentation file changed. The five expensive jobs gate on `needs.changes.outputs.code == 'true'` instead of the original (always-false) `docs_only != 'true'`.

**Regression test**
Verified live: a push touching only `.github/workflows/ci.yml` itself (a real file) still ran the full suite; a genuinely docs-only follow-up push is the actual regression check for the skip path.

---

## 8. Production-Hardening Gaps

These are not bugs; they are scope boundaries and process-local designs that are acceptable for this project's current demo/learning-exercise posture but would need real work before a genuine multi-replica or enterprise production deployment.

### PROD-001: Durable workflow state did not make the side effects it triggered durable

**Status:** ACCEPTED LIMITATION
**Severity:** MODERATE
**Area:** Production-hardening / LangGraph experiment
**Affected code:** LangGraph checkpointer vs. in-memory business-case store

**Problem**
A LangGraph experiment paused a case-update request for human approval and resumed correctly across separate process runs for a single mutation; a two-step scenario did not, the first mutation reported success, but the next process still saw the old status.

**Failure scenario**
The checkpoint itself was correct (`get_state` reliably reloaded the paused workflow). The business store is a plain in-memory dict; each CLI invocation re-imports the module and re-seeds data, so the first process's mutation was lost the moment it exited. This is correct behavior for the real deployment shape (one long-running server), just not for a cross-process CLI demo.

**Impact**
Durable execution only ever makes the *workflow's own declared state* durable; any external side effect (like this synthetic store) needs its own persistence layer, which this learning exercise deliberately didn't build.

**Why not fixed**
Out of scope for the experiment; the two-step scenario now simply runs both mutations inside one process to demonstrate the sequence, while the single-mutation demo remains the legitimate cross-process example.

**Notes**
Documented explicitly rather than hidden; this is the interview-ready "durability of what you call into vs. durability of the workflow itself" distinction.

---

### PROD-002: Re-spelling the same ingestion root churns document identity

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Production-hardening / data identity
**Affected code:** `document_id` keyed on literal `source` string

**Problem**
Document identity in this system is keyed on the literal `source` string everywhere; re-discovering the exact same physical file under a differently-spelled root (relative vs. absolute, or a different OS path-separator convention) is treated as a different document.

**Failure scenario**
The old entry is deleted and a new one created under the new spelling, rather than recognized as unchanged, churning identity without duplicating or losing content.

**Why not fixed**
Normalizing every discovered source to a canonical resolved form would force a one-time mass delete-and-recreate across every already-ingested document in every existing dataset on its very next re-ingestion, disproportionate to the actual bug this was found alongside (see **BUG** entry for incremental-ingestion deleted-document detection, folded into this limitation).

**Notes**
Verified directly with two end-to-end tests that this churn never duplicates or loses content, only re-creates identity.

---

### PROD-003: Resetting only the graph side of a dataset doesn't actually clear it

**Status:** ACCEPTED LIMITATION
**Severity:** MINOR
**Area:** Production-hardening / GraphRAG consistency
**Affected code:** Checksum-based change detection (Postgres-only)

**Problem**
Any future recovery path that clears *only* the Neo4j graph side of a dataset (not just the one demo script already fixed, see **GRAPH-004**) will hit the identical gap: ingestion's checksum-based "unchanged" shortcut lives entirely in Postgres and has no visibility into the graph store at all.

**Why not fixed generally**
A fully bidirectional consistency check was judged out of scope; the dataset-level clear (which resets both stores together) is the documented correct recovery path instead.

**Recommended mitigation**
Always use the dataset-level clear, never a graph-only reset, when recovering from any graph-side data loss.

---

## 9. Needs Verification

### SEC-007: Upload atomicity fix: crash window between DB write and file install unverified

**Status:** NEEDS VERIFICATION
**Severity:** MODERATE
**Area:** Security / upload atomicity
**Affected code:** `_ingest_upload_atomically`, post-DB-write file install step

**What needs to be verified**
Whether a process crash occurring specifically between the successful database write and the final atomic file install could leave a DB row pointing at a file that was never actually installed. This was explicitly recorded as a residual gap during the **SEC-006** fix, not reproduced, not closed, and not proven to be a real, triggerable window in practice.

**Recommended verification approach**
A fault-injection test killing the process between the two steps, inspecting whether the resulting DB/filesystem state is detectable and recoverable.

---

### SEC-008: Cross-tenant upload isolation integration tests were written but never run

**Status:** NEEDS VERIFICATION
**Severity:** MINOR
**Area:** Security / multi-tenancy test coverage
**Affected code:** Cross-tenant upload integration tests (associated with **SEC-005**)

**What needs to be verified**
The cross-tenant upload integration tests for the **SEC-005** fix were written but never executed, no local Postgres instance was available at the time. The unit-level fix is tested and believed correct, but end-to-end cross-tenant behavior against a real database has not been confirmed.

**Recommended verification approach**
Run the existing (already-written) integration test suite against a real Postgres instance.

---

## 10. Fixed / Closed Issues

The following issues are resolved, with a concrete fix and regression test cited inline under each entry above. This section is a navigational summary; full detail (problem, failure scenario, fix, regression test) lives with each issue's entry in its topical section above, not duplicated here.

| ID | Area | Title |
|---|---|---|
| BUG-001 | Data | Two datasets shared one vector table and leaked into each other's results |
| BUG-002 | Data | A connection leak that only a fresh, uninitialized database could trigger |
| BUG-003 | Data | A new metadata column never reached the database |
| BUG-004 | Data | A UNIQUE constraint on a nullable tenant column would not deduplicate anonymous submissions |
| BUG-005 | Data | Re-ingesting a changed file could leave a document permanently empty |
| BUG-006 | API | A decorator captured its rate limiter at import time |
| BUG-007 | API | `top_k=0` silently became the default, and a negative value reached SQL |
| BUG-008 | API | A streaming endpoint validated requests after its 200 status was sent |
| BUG-013 | Evaluation | Recall@10 was structurally unmeasurable |
| BUG-014 | Evaluation | A RAGAS metric returned null for every question without raising an error |
| SEC-001 | Security | Document-level access control could not stop a legitimate reader from getting a secret |
| SEC-002 | Security | A pattern written from one document missed a second credential format |
| SEC-003 | Security | The system and user prompts were joined into one string before reaching the model |
| SEC-004 | Security | A demo looked unsafe because the running config had every safety control turned off |
| SEC-005 | Security | Uploads could become visible to other tenants |
| SEC-006 | Security | Upload handling: writes landed before validation, and temp paths leaked into citations |
| SEC-009 | Security | Metadata-only sensitive values bypassed field-level redaction |
| SEC-010 | Security | A signed token was accepted as an approval, not only as an identity |
| SEC-011 | Security | The request ID was reused as a dedup key, although clients could set it |
| SEC-012 | Security | The default config followed an injected instruction, running older prompts with no access control |
| SEC-013 | Security | Agent tools that queried the vector store directly skipped the authorization kill-switch |
| SEC-015 | Security | A hosted reranker received raw candidate content before redaction, with no egress gate |
| SEC-016 | Security | Egress policy dropped a blocked source's context but still sent the derived answer to the hosted judge |
| SEC-017 | Security | A same-named upload could silently replace another tenant's document |
| DIST-001 | Reliability | Reading a request-scoped context variable inside a stream would have raced its middleware |
| DIST-002 | Reliability | A LangGraph join node ran twice, because its parallel branches had unequal lengths |
| DIST-003 | Reliability | A deliberate workflow pause would have been counted as a failure in metrics |
| DIST-004 | Reliability | A queue-depth metric built on XLEN could only grow |
| DIST-005 | Reliability | An SSE disconnect was only noticed between queue items, never while waiting for one |
| RET-005 | Retrieval | BM25 ties depended on unspecified SQL row order |
| GRAPH-001 | GraphRAG | Neo4j failed on one unknown config key and on a null inside a MERGE key |
| GRAPH-002 | GraphRAG | The driver's convenience method stripped the node labels authorization code needed |
| GRAPH-003 | GraphRAG | A graph path test failed because one relationship write cannot span two tenant scopes |
| GRAPH-004 | GraphRAG | Resetting the graph alone left the graph empty (demo-script fix; see PROD-003 for the general residual) |
| GRAPH-006 | GraphRAG | A relationship's own ACL was never checked on the dependency-path query |
| GRAPH-007 | GraphRAG | Neo4j routing/timeout options silently became Cypher query parameters |
| ONT-001 | Ontology | A narrow RDFS domain axiom silently reclassified an Application as a Service |
| ONT-002 | Ontology | SHACL was already subclass-aware, before any inference option was requested |
| AGT-001 | Agentic RAG | Step budgets collided with the happy path, mislabeling most runs as max_steps |
| AGT-002 | Agentic RAG | Citation metrics measured the wrong set and a model that rarely cites |
| AGT-003 | Agentic RAG | A classify fix exposed a RAGAS gap on the classic route |
| EVAL-002 | Evaluation | A 15-question pilot hid a failure that the full 62-question set exposed |
| EVAL-003 | Evaluation | A keyword heuristic produced false regression counts, and a test relied on it |
| EVAL-005 | Evaluation | Safety metrics read as a perfect zero until the gold paths were checked |
| EVAL-006 | Evaluation | `oversized_request_rejection_accuracy` read 0/3, because the probe token was rejected first |
| EVAL-010 | Evaluation | The corpus-wide duplicate-secret scan was not scoped to dataset_id |
| EVAL-011 | Evaluation | Relationship expansion could inflate Recall@K/Hit@K/MRR, and MRR had no rank cutoff |
| EVAL-012 | Evaluation | Duplicate relevant-document ids in gold data inflated Recall@K |
| EVAL-013 | Evaluation | A graph-strategy comparison could silently relabel every row as vector-only |
| EVAL-015 | Evaluation | No experiment record captured which git commit produced it |
| OBS-002 | Observability | Shared mutable LLM token counters could leak across concurrent requests |
| OPS-001 | Deployment | A 504 at the frontend came from the host running out of memory |
| OPS-002 | Deployment | A frontend panel stayed "unavailable" because the reverse proxy never routed two endpoints |
| OPS-003 | MCP | The MCP SDK's built-in auth did not match this system's trust model |
| OPS-004 | MCP | Two JWT claims that looked symmetric behaved differently when unconfigured |
| OPS-005 | CI | A committed config change broke 112 unit tests |
| OPS-006 | CI | A CI integration test assumed a corpus that CI never has |
| OPS-007 | CI | A CI retrieval gate could not reuse the documented baseline |
| OPS-008 | CI | Five independently green branches broke each other when merged |
| OPS-009 | Deployment | Four Docker problems that only appeared when the container ran |
| OPS-010 | Deployment | A Kubernetes readiness probe could not see a database outage |
| OPS-011 | Deployment | A local Kubernetes tool hung at container creation on one host |
| OPS-012 | Deployment | An nginx image worked in Compose and crash-looped in Kubernetes |
| OPS-013 | Deployment | A Helm first install failed on hook ordering and a network policy |
| OPS-014 | CI | A docs-only CI gate silently never skipped, because of a reversed reading of one option |
| MULTI-001 | Multimodal | Shared asset folders paired the wrong images with the wrong documents |
| MULTI-003 | Multimodal | A failing vision call would have failed the whole document |
| CTX-001 | Context Engineering | Compression threshold was gated on item size, not remaining headroom |
| CTX-002 | Context Engineering | A generic uppercase-word regex misparsed an entity name as a relationship type |
| CTX-003 | Testing methodology | Test doubles mutating the cached `load_config()` singleton polluted later tests |
| CTX-005 | Context Engineering | The context budget assumed a model window that nothing ever told Ollama about |
| CTX-006 | Context Engineering | Packed token counts omitted rendered overhead, and agent synthesis budgeted the wrong template |
| CTX-007 | Context Engineering | Deduplication could merge conflicting facts, and could cross a tenant boundary |
| CTX-008 | Context Engineering | A tokenizer failure during budget planning escaped the documented graceful fallback |
| CTX-009 | Context Engineering | Conversation-history evidence never got injection-flagged |
| CTX-010 | Context Engineering | Adaptive budget classification counted post-dedup evidence, contrary to its own contract |
| CTX-011 | Context Engineering | The Context Engineering benchmark clamped utilization to 1.0, hiding real overflow |
| CTX-013 | Context Engineering | `run_eval.py` silently dropped `answer()`'s own `context_engineering` field |
| EVAL-014 | Evaluation / Context Engineering | `sources`/citations could credit evidence Context Engineering had already dropped or compressed |

---

## 11. Entries Preserved As Non-Bug Findings (Informational, No Fix Applicable)

A handful of original entries describe a measured tradeoff or a confirmed-correct validation, not a defect. Preserved here verbatim in substance rather than forced into a bug template, per the instruction not to invent severity/status that doesn't fit:

- **"Structure-aware chunking needed measured thresholds, not row-count assumptions"**: a design assumption was disproven by measurement *before* shipping, not a shipped bug. Table/code/chart atomicity now uses measured thresholds (`table_row_group_size: 20`, `max_atomic_block_chars: 2000`). No open issue.
- **"Context expansion was tested to confirm it never changed the ranking"**: a validation test confirming a feature's additivity guarantee held (bit-for-bit identical Recall/MRR with expansion on vs. off). Not a bug; the test exists as a permanent regression guard.
- **"A reranker configuration parameter was doing two jobs"**: filed as **BUG-011** conceptually (a config parameter silently did double duty), fixed by splitting `retrieval.rerank_top_n` into `candidate_k`/`reranker.top_n`/`generation_context_top_n`. Listed here for cross-reference since it is retrieval-specific; treat as FIXED, MAJOR, Area: Retrieval.
- **"Retrieval metrics and generation metrics moved in different directions"**: a measurement-interpretation finding (Recall ≠ precision ≠ faithfulness ≠ correctness), resolved by measuring each separately via `reference_contexts`. Treat as **EVAL-008**, FIXED, MODERATE, Area: Evaluation.
- **"An injected claim led the answer, although the right value was retrieved"**: filed as **SEC-014** conceptually (an injection/trust-conflict resolution gap), FIXED via an explicit synthesis-prompt rule plus evidence reordering. Area: Security / Agentic RAG boundary.
- **"A working vision provider did not improve answers, because the model was weak"**: **MULTI-002** in the summary table above; ACCEPTED LIMITATION, MINOR. A model-capability finding, not an integration defect; the integration itself (provider, cache, retrieval, citation) was confirmed working correctly.

---

## Final Consistency Pass

Performed after restructuring:

- Every original entry from the source file accounted for above, either as a numbered issue or in section 11's informational list. None deleted.
- No issue's meaning was changed; problem/diagnosis/solution/result content was reorganized into the new template fields, not rewritten to a different conclusion.
- No issue already described as unresolved was relabeled FIXED, and no issue with a concrete, tested fix was left as OPEN.
- Two issues (**AGT-004**, **OBS-001**) are explicitly **IN PROGRESS** rather than FIXED or OPEN, because the source material itself describes a partial fix with a clearly-stated remaining gap; collapsing either into a single status would have lost real information.
- Duplicate entries: none merged. Every original entry maps to exactly one numbered issue or informational item; no two source entries described the same underlying defect.
- Severity is newly assigned (the source file had none); it follows the definitions in the Severity Legend consistently, not from any hidden prior labeling.

### Report

1. **OPEN issues:** 0 (see note below)
2. **IN PROGRESS issues:** 2 (AGT-004, OBS-001)
3. **FIXED issues:** 51
4. **ACCEPTED LIMITATION items:** 13 (RET-001, RET-002, RET-003, RET-004, BUG-010 equivalent, EVAL-001, EVAL-004, GRAPH-004, GRAPH-005, PROD-001, PROD-002, PROD-003, MULTI-002)
5. **NEEDS VERIFICATION items:** 3 (AGT-005, SEC-007, SEC-008)
6. **Counts by severity:** CRITICAL: 4 (BUG-005, SEC-001, SEC-005, SEC-010) · MAJOR: 20 · MODERATE: 28 · MINOR: 10
7. **Duplicate entries merged:** none; every source entry maps 1:1 to one issue or informational item.
8. **Ambiguous classifications, preserved conservatively:**
   - "Retrieval was correct, but generation fabricated the answer" (source: Generation and grounding) has no concrete fix and no explicit accepted-limitation decision recorded in the source text; assigned **OPEN** conceptually, but since the Summary Table's Status Legend requires a definitive call and the entry's own wording ("stayed unsolved across four configurations") reads as an active, unresolved gap rather than a deliberate non-fix decision, it is listed as **BUG-009, OPEN, MAJOR** in the Summary Table above rather than folded into section 11. (The report count in item 1 above reflects this as the one genuine OPEN item; the "0" stated there is corrected to **1**.)
   - "Durable workflow state did not make the side effects it triggered durable" and "Re-spelling the same ingestion root churns document identity" both describe a *design boundary* rather than a defect anyone is tracking toward a fix; classified as ACCEPTED LIMITATION rather than OPEN on that basis, since the source text explicitly frames each as an intentional, documented, out-of-scope boundary.
   - Several "Testing, CI and build" and "Infrastructure learning exercises" entries (e.g. OPS-005, OPS-008) are process/CI defects rather than product defects; kept in the Deployment/Operations and CI areas per the new structure's own category guidance rather than forced into "Confirmed Bugs," since they concern build/merge/environment behavior specifically.

**Postscript (context-engineering milestone):** this Report section is a point-in-time audit of the original restructuring and is left as originally written, not retroactively recomputed. Four new entries (CTX-001 through CTX-004) were added afterward in section 4 and are reflected in the Summary Table and section 10's table above; they are not folded into this Report's counts.

**Postscript (pre-experiment audit, ingestion/graph gaps):** an adversarial audit of the ingestion and graph-consistency paths found three new ingestion-consistency gaps (BUG-015, BUG-016, BUG-017) and two graph-authorization gaps (GRAPH-008, GRAPH-009), all deliberately left OPEN rather than fixed in this pass, each with its own "why not fixed yet" note on its entry; they are structural changes, not local patches. These are not folded into this Report's original counts above. Separately, this pass's own cross-check of section 10's table against the full topical entries found three pre-existing table rows (MULTI-001, MULTI-002, MULTI-003) with no corresponding topical-section entry; all three were real, already-fixed (or, for MULTI-002, already-accepted) findings already described elsewhere in the project (CLAUDE.md, source docstrings), so full entries were added rather than leaving the table's own citation dangling.
