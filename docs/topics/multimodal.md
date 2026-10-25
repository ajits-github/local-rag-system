# Multimodal & Layout-Aware Ingestion

PDF and DOCX documents keep their real structure (headings, tables,
code/config blocks, images, page numbers) instead of being flattened to
plain text: both loaders serialize what they extract into the same
Markdown-equivalent syntax the `structured_markdown` chunker already
parses, rather than introducing a second, parallel element model.

An optional local vision provider (`config.vision.provider: "ollama"`,
default `"none"`) can describe an image with a small offline model
(`moondream` by default). A vision description is always a second,
sibling chunk alongside the caption/alt-text chunk, never overwriting it.

Full design writeups:

- [Multimodal + Relationship-Aware Ingestion](../architecture.md#multimodal-relationship-aware-ingestion)
- [Layout-Aware Document Ingestion and Vision](../architecture.md#layout-aware-document-ingestion-and-vision)
- API reference: [Vision](../reference/vision.md), [Loaders](../reference/loaders.md), [Chunkers](../reference/chunkers.md)

## Multimodal ingestion: images, relationships, and gold evaluation

`data/knowledge_base` includes real images (dashboards, diagrams, charts,
image-tables) and the canonical gold file is
`data/eval/techfusion_gold.jsonl` (**84 questions**, not
`techfusion_gold_old.jsonl`/`techfusion_gold_ragas_sample15.jsonl`, both
older-schema artifacts kept for history, never edited).

- **Image chunks.** `structured_markdown` treats a standalone
  `![alt](path)` markdown line (on its own line, not embedded mid-
  paragraph) as its own `content_type="image"` `ChunkSpan`, at the same
  detection priority as tables/fences. An immediately-following emphasis-
  wrapped caption paragraph (`*Figure 1: ...*`) folds into that same
  span's text (mirrors the pre-existing chart-fence-plus-caption pattern, both share `structured_markdown.py`'s `_peek_caption` helper), a
  caption is never a separate, independently-retrievable element. An
  image link *embedded inline* within ordinary prose is still just
  attachment tagging (`attachment_name`/`source_anchor`), unaffected.
- **Relationships.** `ChunkMetadata.parent_chunk_id` (nullable, no FK)
  links a table/code/configuration/chart/image chunk to the nearest
  *preceding prose chunk in the same `section_path`*, computed in
  `Writer.write`'s existing single pass over a document's chunk spans
  (heuristic, not a real parse tree, `None` if no such prose exists,
  e.g. a section opening with a table). Prose chunks always get
  `parent_chunk_id=None`. `VectorStore.get_chunks_by_ids`/
  `get_chunks_by_section` are the two read primitives built on top of it.
- **Relationship-aware retrieval expansion.**
  `config.retrieval.relationship_expansion` (`enabled: false` by default, a true no-op unless turned on) is a post-*rerank* step in
  `RetrievalPipeline.retrieve()`: for each already-ranked result, its
  `parent_chunk_id` chunk and/or immediate chunk_index neighbors within
  the same section are appended (deduped, capped at
  `max_related_elements`) with `origin="expanded"` /
  `expanded_from=<id>` on `SearchResult` (`origin` defaults
  `"retrieved"`). Expanded results are *appended after* the ranked list,
  never interleaved and never given a freshly computed `.score` (it's
  inherited from the originating result), this is why enabling expansion
  never moves Recall@k/MRR (see `experiment_011` vs `experiment_012` in
  README's Benchmarks). Scoped to the originating chunk's own
  `document_id`, so it can never cross a `dataset_id` boundary.
  `retrieval/pipeline.py:_source_label` labels each prompt source with
  section/content-type/vision-provenance/expansion-origin, so the model
  is never handed grounds to imply it visually inspected an image when
  only caption/alt-text was available.
- **Vision.** `config.vision.provider` is `"none"` (default, no-op) or
  `"ollama"` (local, offline, see "Layout-aware document ingestion and
  vision" below for the full design; that milestone added the real
  provider, not this one). `factory.build_vision_provider`,
  `Writer._with_vision_siblings`, and a Postgres-backed
  `image_description_cache` table (composite-keyed on checksum + provider
  + model + prompt_version) are wired end-to-end. A vision description is
  always a *second, sibling* chunk (`vision_generated=True`,
  `vision_description=<text>`) alongside, never overwriting, the
  caption/alt-text chunk, regardless of provider.
- **Gold schema.** `GoldExample` (`eval/gold_schema.py`) has 6 fields
  beyond the original schema, all optional/defaulted so pre-existing gold
  files still parse: `content_type` (an *authored* ground-truth question
  category, e.g. `image_only`/`relationship_aware`/`caption_answerable`, distinct from `eval/content_type.py`'s *derived* document buckets,
  never conflated), `reference_contexts` (verbatim-excerpt textual
  ground truth), `reference_visual_contexts` (ground truth for facts
  visually present but intentionally absent from indexed text),
  `relevant_images`, `requires_vision`, `requires_relationship_expansion`.
  **`reference_contexts`/`reference_visual_contexts` are read exclusively
  by `eval/*.py`**, `ingestion`/`retrieval`/`generation` never import
  `rag.eval`, and gold data is never written into `chunks`; a static
  import-boundary test plus a runtime marker-string test
  (`tests/unit/test_gold_data_isolation.py`) both guard this directly,
  don't just assume it.
- **New deterministic eval metrics** (`eval/run_eval.py`, all from the
  same broad top-10 retrieval already made for Recall@10, no extra
  calls, fully deterministic): `content_type_breakdown` (gold-authored
  buckets), `reference_context_analysis` (A/B/C supporting-context-hit
  split via `reference_context_is_supported`'s whitespace-normalized
  substring match, documented under-count risk, not a semantic-
  similarity claim), `relevant_image_hit_rate`,
  `relationship_expansion_contribution_rate`, and (generation runs only)
  `vision_behavior_breakdown`, a small literal refusal-phrase heuristic,
  explicitly a triage aid, not a correctness judge (see
  `PROJECT_JOURNAL.md`'s 2026-08-11 entry for a real example of this
  heuristic's false-negative gap, caught by hand-inspection). Generation
  runs additionally report `token_usage` (mean prompt/completion tokens,
  read off `OllamaLLM.last_prompt_tokens`/`last_completion_tokens`),
  `refusal_behavior` (the same `_looks_like_refusal` phrase matcher as
  `vision_behavior_breakdown`, but applied to every `unanswerable=true`
  example, not just `requires_vision=true` ones), and
  `relationship_expansion_utilization`, distinct from
  `relationship_expansion_contribution_rate`: this one checks, for every
  question where an `origin="expanded"` chunk reached the prompt (any
  question, not just gold-annotated `requires_relationship_expansion`
  ones), whether the generated answer's wording actually echoes
  expanded-only content (`_expansion_utilization`'s keyword-overlap
  heuristic), see `PROJECT_JOURNAL.md`'s 2026-08-14 entry.
- **Prompt v2 experiment baseline.** `rag_answer_v2` (stricter grounding/
  refusal rules) is not `config/default.yaml`'s active prompt (still
  `v1`); exercised instead via `config/experiments/multimodal-v2-text-only.yaml`
  / `multimodal-v2-relationship.yaml` (recorded as `experiment_011`/
  `experiment_012` in README's Benchmarks table), which also pin
  `retrieval.provider=hybrid` (`rrf_k=60`)/`reranker=none`, not
  `config/default.yaml`'s own `dense`/`none`. A deliberate, documented
  discrepancy between the committed default and the recorded experiment
  configs (see `docs/milestones.md`'s "Known deviations" section and
  `docs/architecture.md`).
- **RAGAS validation.** Scored against both a stratified 15-question
  sample (`experiment_013`) and the full 84-question gold set
  (`experiment_015`); the `qwen2.5:3b` + prompt v2 + hybrid+RRF +
  relationship-expansion combination reached this project's highest
  recorded faithfulness (0.898) and answer_correctness (0.513). A bigger
  local model (`qwen2.5:3b` over `qwen2.5:1.5b`, `experiment_014`), not a
  new prompt version, resolved the one hallucination RAGAS had caught on
  the idempotency-key question; `v3` is deliberately not written for
  that reason. See `PROJECT_JOURNAL.md`'s 2026-08-14/2026-08-21 entries
  for the full metric tables and judge-cost accounting.


## Layout-aware document ingestion and vision (layout-vision milestone)

Extends ingestion beyond Markdown so PDF/DOCX files preserve real
structure (headings, tables, code/config blocks, images, page numbers,
prose-image relationships) instead of being flattened to plain
concatenated text, and adds the first real (local, offline)
`VisionProvider`. Deliberately does not redesign the established
classic/agentic RAG architecture: extracted structure is serialized into
the *same* Markdown-equivalent syntax `structured_markdown.py` already
parses, not a second parallel element/chunk model.

- **`loaders/pdf_loader.py`/`loaders/docx_loader.py`** (rewritten,
  `.pdf`/`.docx` already registered in `loaders/registry.py`): both still
  return a `RawDocument` with a single `content: str`, the interface is
  unchanged, only what fills that string changed. `PDFLoader` uses
  `pdfplumber` (`config.ingestion.layout_parsing.pdf_parser`, the only
  parser implemented so far, chosen over PyMuPDF for its permissive MIT
  license and over plain `pypdf` for its word-level font-size/table/
  position signal; `pypdf` is kept for document-info metadata and as the
  raw-byte source for the image-extraction fallback path) to detect
  headings via a font-size-ratio heuristic (two tiers:
  `heading_font_size_ratio`/`title_font_size_ratio` -> `##`/`#`),
  monospace-font-name runs as code/config blocks (`loaders/
  markdown_render.py`'s `sniff_code_language` gives a light JSON/YAML/XML
  shape-guess for an otherwise-untagged fence), tables via
  `page.find_tables()` -> pipe-table syntax, and interleaves paragraphs/
  tables/images in reading order by vertical position. `DocxLoader` walks
  `document.element.body`'s children in one linear pass (DOCX paragraphs/
  tables are already ordered, unlike a PDF's geometry), page numbers
  come from explicit manual page breaks (`<w:br w:type="page"/>`, the
  only reliable DOCX page signal; a DOCX with no manual breaks stays
  `page=1` throughout, an honest limitation, not a guess), code/config
  from run-level monospace font name (no dedicated Word style exists in
  this corpus). A shared `<!--page:N-->` sentinel line (both loaders emit
  it, `structured_markdown.py` parses and strips it, never visible in
  chunk text) is the one addition to the chunker itself:
  `ChunkSpan.page`/`ChunkMetadata.page` (`int | None`, `None` for
  Markdown/text/HTML) is tagged from it. `structured_markdown.split()`'s
  gating changed from `source_type == "markdown"` to `source_type in
  {"markdown", "pdf", "docx"}`, zero other changes to its table/heading/
  image/caption/relationship-expansion logic.
- **Image-asset resolution** (`loaders/base.py`'s `resolve_image_asset`,
  shared by both loaders): prefers an existing sibling `assets/` folder
  next to the source document (this project's eval-corpus convention for
  pre-supplied figures), narrowing a *shared* `assets/` folder's
  candidates to the ones whose filename shares a non-stopword word with
  the source document's own filename (`_matching_assets`), a real bug
  found via integration testing: naive positional pairing across a folder
  shared by multiple sibling documents handed every document the first
  document's images (see `KNOWN_ISSUES.md`). Falls back to extracting and
  writing the image's own bytes (`<document-stem>-figure-{n:02d}<ext>`)
  when no pre-existing `assets/` folder exists at all, so a PDF/DOCX with
  no pre-supplied assets still ingests successfully.
- **`vision/ollama_vision.py`'s `OllamaVisionProvider`** is the first real
  `VisionProvider` (`config.vision.provider: "none" | "ollama"`, `"none"`
  stays the default no-op): a local, offline Ollama chat call (default
  model `moondream`, `config.vision.ollama.model_name`/`prompt_version`),
  reusing `config.ollama_base_url()`, same native-host server
  `OllamaLLM` uses, just a different model, so no `provider_egress_policy`
  involvement at all. `VisionProvider` gained an abstract `prompt_version`
  property; `image_description_cache`'s primary key changed from
  `image_checksum` alone to a composite `(image_checksum, provider,
  model_name, prompt_version)` (`scripts/init_db.py`'s idempotent
  `DROP CONSTRAINT`/`ADD CONSTRAINT` migration, matching the project's
  existing migration idiom), closes a real pre-existing gap: switching
  provider/model/prompt used to silently replay a description generated
  under different instructions. `Writer.vision_stats`/`IngestionStats.
  vision_stats` (`schemas.VisionCallStats`: images processed/cache hits/
  cache misses/total latency) is an ingestion-time concern the CLI
  prints, deliberately separate from `run_eval.py`'s retrieval/generation
  report.
- **Agent `content_type` filtering**: `SearchKnowledgeBaseArgs` gained an
  optional `content_type: Literal["prose", "table", "code",
  "configuration", "image", "chart"] | None` field, a closed set, not a
  free string, merged server-side (`tools.search_knowledge_base`) into
  the caller-supplied `filters` dict, already validated against
  `VectorStore.ALLOWED_FILTER_FIELDS`; the LLM can select *which* allowed
  value to filter by, never name an arbitrary field. `agent_tool_select_v2.
  yaml` describes the new argument (v1 stays on disk unused, per the
  established prompt-versioning convention; `config.agent.
  tool_select_prompt_path` now points at v2).
- **API metadata**: `api.routers.query.SourceItem` and `agent.state.
  Citation` both gained `content_type`/`section_path`/`page`/
  `attachment_name`/`source_anchor`/`vision_generated`, all
  already-computed, non-sensitive structural metadata (`source_dict()` in
  `retrieval/pipeline.py` already produced everything but `page`, the one
  field genuinely missing from that dict). `source_label()` (the
  `[Source N: ...]` provenance string shown to the generation LLM) now
  includes `page N` when set.
- **New deterministic eval metrics** (`eval/run_eval.py`'s `layout_vision`
  report section, all from the same broad top-10 retrieval already made
  for Recall@10, no extra calls): `table_retrieval_hit_rate`,
  `visual_retrieval_hit_rate`, `page_localization_accuracy`,
  `section_localization_accuracy` (substring match against
  `section_path`, not exact equality, a gold-authored short label like
  "Review Purpose" is expected to match a full breadcrumb like "DocuFlow
  ... > 1. Review Purpose"), `visual_evidence_support_rate` (reads `0.0`
  by construction whenever `vision.provider="none"`, the metric the
  later text-vs-vision A/B/C comparison is meant to isolate on), and
  (generation runs) `multimodal_answer_quality` (`answer_quality`
  restricted to `requires_vision`/`requires_layout_awareness`/
  `requires_relationship_expansion` examples). `GoldExample` gained the
  new gold schema's additive-optional fields (`requires_layout_awareness`,
  `expected_content_types`, `relevant_pages`, `visual_question_type`,
  `evidence_mode`, `source_format`), same pattern as every prior
  milestone's gold-schema extension. `record_experiment.py`/
  `mlflow_logger.py` extended with `layout_parser`/`vision_provider`/
  `vision_model`/`vision_prompt_version` params and the six new metrics.
- **Fixed evaluation corpus**: `data/eval/
  techfusion_layout_vision_extension_gold.jsonl` (28 questions) references
  its 8 new PDF/DOCX + 1 Markdown source documents (plus their
  pre-supplied image assets) under `data/knowledge_base/
  layout_vision_extension/<category>/`, a real directory-nesting
  mismatch was found and fixed before implementation (the files initially
  sat flat under the existing category folders, not nested), the same
  failure mode already documented for `security_evaluation/` above;
  confirmed these 9 files are referenced by no other gold set before
  moving them, so no other gold file's path-suffix matching was affected.
- **Controlled A/B/C evaluation: done (RAGAS still deferred).** The
  deterministic A (text-only, pre-milestone `pypdf`/
  `python-docx` flat extraction) vs. B (layout-aware, `vision.provider:
  none`) vs. C (layout-aware + `vision.provider: ollama`/`moondream`)
  comparison against the fixed 28-question gold set is recorded as
  `experiment_033`/`034`/`035`. A -> B was an unambiguous win (Recall@5
  0.946 -> 1.000, `page_localization_accuracy` 0.0 -> 0.964,
  `visual_retrieval_hit_rate` 0.0 -> 1.0). B -> C was mixed: vision
  descriptions were correctly retrieved/cited (`visual_evidence_support_rate`
  0.0 -> 1.0) and reduced hallucination on unanswerable-visual questions
  (3 -> 1 -> 0 across A/B/C, correct refusals 0 -> 2 -> 3), but did not
  improve `multimodal_answer_quality` on answerable vision-required
  questions (0.161 -> 0.144), direct inspection of `moondream`'s cached
  description for the clearest regression showed it echoing the image's
  own caption ("Figure 2") instead of reading the image's content, a
  model-capability gap, not an integration bug. See
  `PROJECT_JOURNAL.md`'s 2026-08-21 evaluation entry and `KNOWN_ISSUES.md` for
  the full writeup. RAGAS was deliberately not run yet, per the
  evaluation's own sequencing (deterministic metrics validated first).
  `config/default.yaml` keeps `vision.provider: none`, unchanged.


## Vision provider status

`vision/base.py` defines the `VisionProvider` ABC (`describe_image` +
`provider_name`/`model_name`/`prompt_version` properties for cache
provenance). `"none"` (text-only image handling: alt text/caption only,
no image bytes ever read) is `config.vision.provider`'s default and the
only no-op; `"ollama"` (see "Layout-aware document ingestion and vision"
above) is the first real, locally-running provider. No hosted-API-calling
provider exists; if one is ever added, `factory.build_vision_provider`
gains another branch, `config.security.egress_policy` gates it, and its
output becomes a sibling chunk exactly like `"ollama"`'s already does,
never overwriting the caption/alt-text chunk.
