# local-rag-system

A modular, config-driven local RAG system: sentence-transformers embeddings,
Postgres+pgvector storage, Ollama generation, FastAPI serving. Every infra
choice (embedding model, vector backend, chunker, reranker, LLM) is
swappable via `config/default.yaml`: nothing is hardcoded, so comparative
experiments can change one axis without touching pipeline code. Runs fully
offline on a CPU-only, 8GB-RAM machine; no API keys required by default.

See [`CLAUDE.md`](CLAUDE.md) for the architecture map and module conventions.

## Prerequisites

- Python 3.11+
- [Docker](https://www.docker.com/) (for the Postgres+pgvector container)
- [Ollama](https://ollama.com/) installed, **running**, and with
  `qwen2.5:1.5b` pulled:
  ```
  ollama pull qwen2.5:1.5b
  ```
  Ollama usually starts automatically after install; verify it's actually
  serving with `ollama list` (or `curl http://localhost:11434/api/version`).
  If it's installed but not on your PATH, invoke it by its full path (e.g.
  on Windows: `C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe`).
- **`make`**: used for the `up`/`ingest`/`query`/`test` shortcuts below.
  **Not installed by default on Windows** (neither Git Bash nor PowerShell
  ship it). Install it via `choco install make`, `scoop install make`, or
  WSL, *or* skip it entirely and run the underlying command shown next to
  each `make` target below; every target is a one-liner.

## Setup

1. Create a virtualenv and install the project:
   ```
   python -m venv .venv
   .venv/Scripts/activate        # Windows
   pip install -e ".[dev]"
   ```
2. Copy `.env.example` to `.env` and adjust if needed. `DATABASE_URL` points
   at Postgres on **host port 15987** (not the default 5432) so it won't
   collide with any other local Postgres instance:
   ```
   cp .env.example .env
   ```
   Already have a Postgres+pgvector container running locally (e.g. from
   earlier work)? Point `DATABASE_URL` at it directly instead of bringing up
   the bundled `docker-compose.yml` in step 3; just make sure `POSTGRES_DB`
   in `.env` matches its actual database name, and still run
   `python scripts/init_db.py` against it (idempotent, safe to run against
   a database that already has the tables) to create/migrate the
   `documents`/`chunks` tables.
3. Bring up Postgres+pgvector and initialize the schema:
   ```
   make up
   ```
   Without `make`, run the two steps it wraps:
   ```
   docker compose up -d
   python scripts/init_db.py
   ```
4. Ingest a knowledge base - any file or directory works, nothing is
   hardcoded. The bundled smoke-test set:
   ```
   make ingest FILE=data/sample_docs
   ```
   Without `make`:
   ```
   python -m rag.ingestion.pipeline data/sample_docs
   ```
   This repo doesn't ship a large real-world knowledge base. One
   (internally called "TechFusion") was used against this pipeline during
   development, with a folder-per-category layout (`engineering/`,
   `security/`, `hr/`, etc.) preserved as filterable `category` metadata,
   but neither that knowledge base nor its gold eval set are committed here
   (see `data/README.md` / `data/VALIDATION.md` for how it was structured).
   Point the same ingestion command at your own directory to reproduce
   that; the path is always a CLI argument, never hardcoded.

   Re-running ingestion on unchanged files is a no-op (checksum-based, no
   duplicate chunks); edited files are detected and re-chunked in place
   under the same `document_id`.
5. Start the API:
   ```
   uvicorn rag.api.main:app --reload
   ```
   Then hit `GET /health`, `POST /ingest`, `POST /query`, or browse
   `/docs` / `/redoc`.
6. Query via the Makefile once the API is running:
   ```
   make query Q="What are the deployment windows for production releases?"
   ```
   Without `make`:
   ```
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What are the deployment windows for production releases?"}'
   ```
   Filter retrieval by the `category` metadata preserved from folder
   structure; only meaningful once you've ingested a dataset with
   subfolders (`data/sample_docs/` is flat, so this is illustrative rather
   than directly runnable against it):
   ```
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"query": "What MFA methods are approved?", "filters": {"category": "security"}}'
   ```

## Configuration

All provider choices and tunables live in `config/default.yaml`:
embedding model, chunk size/overlap, reranker (`none` / `cross_encoder` /
`cohere`), LLM, retrieval `top_k`. Point at an alternate config with
`--config path/to/other.yaml` on the ingestion/eval CLIs, or by editing the
default file directly for local experiments.

Generation prompts are versioned YAML files under
`src/rag/prompts/templates/` (`rag_answer_v1.yaml`, `rag_answer_v2.yaml`,
...), loaded and validated by `src/rag/prompts/loader.py`. The active
version is selected by `config/default.yaml`'s `generation.prompt` block
(`id`, `version`, `path`); `RetrievalPipeline` loads it once at
construction and rejects the file if its own declared `prompt_id`/`version`
don't match what's configured. `rag_answer_v2.yaml` ships as an example of
a stricter grounding/citation prompt but isn't active by default.

## Metadata & filtering

Every chunk carries: `document_id, chunk_id, source, source_type, title,
author, url, created_at, last_modified, language, category`. For Markdown
sources with a YAML front-matter block (`title`/`owner`/`last_reviewed`,
the convention used by the private TechFusion knowledge base mentioned
above), `title`/`author`/`last_modified` are parsed from it rather than
falling back to the filename/filesystem timestamp, and the front-matter
block itself is stripped before chunking.
`category` is the file's folder path relative to whatever root you ingested
(e.g. `security`, or `security/subteam` for nested folders); `None` for a
single ingested file or an API upload, which have no folder context.
`POST /query`'s `filters` field can restrict retrieval by any of
`document_id, source, source_type, title, author, url, language, category`
(see the `curl` example above).

## Evaluation

A gold file is a JSONL file, one labeled question per line:
`question, expected_answer, relevant_documents, question_type, difficulty,
unanswerable`. `relevant_documents` are paths *relative to wherever the
knowledge base root is* (e.g. `"knowledge_base/security/access-control-policy.md"`)
rather than `document_id`s, since ids are only assigned at ingestion time.
Matching a retrieved chunk's stored `source` against these paths is done by
path-suffix, not exact equality or a hardcoded root, so the same gold file
and the same `run_eval` command work regardless of what directory you
actually ingested from.

`data/eval/sample_gold.jsonl` is a tiny smoke-test gold file bundled with
this repo, matching `data/sample_docs/`. A larger 46-question gold set
(`question_type`s `single_document`/`multi_hop`, plus unanswerable
questions) was used against the private TechFusion knowledge base
mentioned above during development. That gold file isn't committed here
either, for the same reason; point `--gold` at your own to reproduce that
kind of evaluation.

```
python -m rag.eval.run_eval --gold data/eval/sample_gold.jsonl --dataset-id sample_docs
```
`--dataset-id` is mandatory, it's injected as a filter on every retrieval
the runner makes, so an evaluation can never silently score chunks from a
different dataset. Point `--gold`/`--dataset-id` at any gold file and
dataset; nothing about the runner is tied to a particular one. Reports,
for the configured pipeline:
- **Recall@5 / Recall@10** and **Hit Rate@5 / Hit Rate@10**: recall
  averages the fraction of *all* relevant documents found per question;
  hit rate is binary (was *any* relevant document found), so the two
  diverge on multi-document questions.
- **MRR** (mean reciprocal rank).
- **Retrieval / generation / total latency** (mean, ms), measured from the
  pipeline's actual configured `rerank_top_n` (not the broader top-10 fetch
  used for the Recall@10 calculation above, which would otherwise inflate
  the latency number).
- **Answer quality**: a keyword-overlap heuristic scored against
  `expected_answer` (see the caveat in its own output; it's a placeholder,
  not a correctness judge — see the RAGAS section below for the real one).

Add `--verbose` to include full per-question detail (retrieved sources,
generated answer, individual scores) in the JSON output, or
`--skip-generation` to get only the retrieval metrics quickly, without
waiting on LLM calls for every question.

### RAGAS generation-quality evaluation (optional)

An opt-in, additive layer on top of the metrics above: faithfulness,
answer relevancy, context precision, context recall, and (if supported
cleanly by the installed RAGAS version) answer correctness — each scored
by an independently-configured LLM judge, not by `qwen2.5:1.5b`/`3b` (the
models already used for generation). Install with:
```
pip install .[ragas]         # local (ollama) judge
pip install .[ragas,anthropic]  # + hosted anthropic judge
```
The judge is selected via `config/default.yaml`'s `judge:` block
(`provider: openai | anthropic | ollama`, each with its own model/API-key
settings, resolved from `*_env_var` environment variables — never
hardcoded). `ollama` is offered for free local experimentation only; the
docs and RAGAS's own guidance both note small local judges give less
reliable scores than a hosted model.

```
python -m rag.eval.run_ragas_eval --gold data/eval/sample_gold.jsonl \
    --dataset-id sample_docs --verbose > /tmp/ragas_report.json
```
`--sample-size` (default **15**) caps how many gold questions get judged —
validate cost, latency, and score quality on a small subset before
scaling up to the full 46-question TechFusion gold set. The report
records per-question and aggregate scores, the judge's provider/model and
estimated API usage, `prompt_id`/`prompt_version`, and the RAG config that
produced it — feed it into `scripts/record_experiment.py` exactly like a
normal `run_eval.py` report; the ragas fields are captured alongside the
rest.

**RAGAS scores are not validated until reviewed against real human
judgment.** Two scripts help with that:
```
python scripts/generate_manual_review.py --eval-output /tmp/ragas_report.json --num-rows 10
# ... fill in the human_faithful/human_correct/human_relevant/human_correct_refusal
# fields in the generated JSONL by hand ...
python scripts/compare_ragas_manual.py --ragas-output /tmp/ragas_report.json \
    --manual-review data/eval/manual_review/sample_docs_manual_review.jsonl
```
The comparison report shows where the judge agrees or disagrees with the
human labels — do not treat RAGAS scores as ground truth until you've run
this and reviewed the result yourself.

## Benchmarks

`experiments/` tracks comparable eval runs over time: `configs/` holds a
snapshot of `config/default.yaml` per experiment, `results/` holds a flat
JSON metrics record per experiment (schema below), and `reports/` holds
the generated comparison table. Only aggregate metrics and config values
are stored here, never corpus content or generated answers, so all three
are safe to commit even though the underlying dataset isn't (per dataset
above).

The table below is generated, not hand-written; don't edit it directly,
regenerate it instead (see "Recording a new experiment").

<!-- EXPERIMENTS_TABLE_START -->
| # | Label | Generation model | Embedder | Reranker | Recall@5 | Recall@10 | Hit Rate@10 | MRR | Answer quality | RAGAS Faithful | RAGAS Correct | Total latency | Dataset | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TechFusion baseline (qwen2.5:1.5b, no reranker) | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.432 | - | - | 3.7s | techfusion | 2026-08-05 |
| 2 | qwen2.5:3b (candidate generation model) | qwen2.5:3b | all-MiniLM-L6-v2 | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.453 | - | - | 12.1s | techfusion | 2026-08-05 |
| 3 | qwen2.5:3b + cross_encoder reranker | qwen2.5:3b | all-MiniLM-L6-v2 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.891 | 0.967 | 0.978 | 0.822 | 0.507 | - | - | 10.1s | techfusion | 2026-08-05 |
| 4 | qwen2.5:3b + cross_encoder + BGE-small embedder | qwen2.5:3b | bge-small-en-v1.5 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.902 | 0.935 | 0.957 | 0.825 | 0.483 | - | - | 10.3s | techfusion-bge-small | 2026-08-05 |
| 5 | qwen2.5:3b + cross_encoder + BGE-small + chunk_size=300 | qwen2.5:3b | bge-small-en-v1.5 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.902 | 0.946 | 0.957 | 0.808 | 0.427 | - | - | 9.9s | techfusion-bge-small-chunk300 | 2026-08-05 |
| 6 | structured_markdown chunker (62-question gold set, tables/code/config/chart) | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.919 | 0.952 | 0.968 | 0.865 | 0.387 | - | - | 2.8s | techfusion | 2026-08-06 |
| 7 | RAGAS full evaluation (OpenAI gpt-4o-mini judge, all 62 questions) | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.919 | 0.952 | 0.968 | 0.865 | 0.413 | 0.786 | 0.469 | 4.4s | techfusion | 2026-08-06 |
<!-- EXPERIMENTS_TABLE_END -->

*Total latency is the mean of retrieval+generation per question, at the
config's production `rerank_top_n` (not the broader top-10 fetch used for
Recall@10). Every row measured against `dataset_id`-isolated retrieval
(see Metadata & filtering above), so results are never contaminated by a
different dataset in the same vector store.*

### Recording a new experiment

1. Change one thing in `config/default.yaml` (reranker, model, chunk size,
   prompt version, ...).
2. Run the eval and save its full report (or `rag.eval.run_ragas_eval` for
   the RAGAS-scored variant, see above):
   ```
   python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl \
     --dataset-id techfusion --verbose > /tmp/eval_report.json
   ```
3. Register it as an experiment (never re-runs eval, just records the
   report above):
   ```
   python scripts/record_experiment.py --eval-output /tmp/eval_report.json \
     --experiment-id experiment_002 --label "cross-encoder reranker" \
     --config config/default.yaml
   ```
   Every recorded experiment also captures `prompt_id`/`prompt_version` and
   a `prompt_file_checksum`, the same way it captures
   `embedding_model`/`chunk_size`/`reranker_model` — a prompt wording
   change is its own comparable experiment axis, tracked exactly like any
   other config change. If the report came from `run_ragas_eval`, the
   RAGAS aggregate scores and judge provider/model are captured too.
4. Regenerate the comparison table (updates
   `experiments/reports/comparison.md` and this README section in place):
   ```
   python scripts/compare_experiments.py
   ```

The eval report and per-question detail behind each experiment aren't
committed (see Evaluation above); the flat metrics record and config
snapshot under `experiments/` are what make the comparison reproducible
without needing that raw file.

## Testing

```
make test
```
Without `make`: `pytest tests/ -v`.

Unit tests (`tests/unit`) mock all external I/O and always run. Integration
tests (`tests/integration`) assume Postgres (`make up`) and a local Ollama
with `qwen2.5:1.5b` are already running, and skip with a clear message if
either is unreachable.

## Roadmap

Deferred for now, tracked here rather than left as empty scaffolding:

- **Hybrid Search**: combine vector similarity with keyword/BM25 search.
- **Cross Encoder**: mature the scaffolded `cross_encoder` reranker
  (`rerankers/cross_encoder.py`) with real benchmarking/tuning.
- **Cohere Reranking**: mature the scaffolded optional `cohere` provider
  (`rerankers/cohere.py`) once there's a use case needing a hosted reranker.
- **RAGAS scaling**: validate the judge against human labels
  (`scripts/compare_ragas_manual.py`) and, once trusted, scale from the
  15-question default sample to the full 46-question TechFusion gold set.
- **LangGraph**: move retrieval/generation orchestration to a graph for
  multi-step or agentic query handling.
- **MCP**: expose ingest/query as MCP tools for use from other agents.
- **Multi-agent workflows**: e.g. a query-planning agent in front of
  retrieval, or specialized agents per document source.
- **Kubernetes deployment**: containerize the API and vector store for a
  non-local deployment target.
- **Observability**: OpenTelemetry tracing/metrics on top of the current
  structured JSON logging.
