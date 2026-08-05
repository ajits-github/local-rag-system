# local-rag-system

A modular, config-driven local RAG system: sentence-transformers embeddings,
Postgres+pgvector storage, Ollama generation, FastAPI serving. Every infra
choice (embedding model, vector backend, chunker, reranker, LLM) is
swappable via `config/default.yaml` — nothing is hardcoded — so comparative
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
- **`make`** — used for the `up`/`ingest`/`query`/`test` shortcuts below.
  **Not installed by default on Windows** (neither Git Bash nor PowerShell
  ship it). Install it via `choco install make`, `scoop install make`, or
  WSL, *or* skip it entirely and run the underlying command shown next to
  each `make` target below — every target is a one-liner.

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
   the bundled `docker-compose.yml` in step 3 — just make sure `POSTGRES_DB`
   in `.env` matches its actual database name, and still run
   `python scripts/init_db.py` against it (idempotent — safe to run against
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
   hardcoded. To ingest the bundled TechFusion sample knowledge base
   (recursively, preserving `engineering/`, `security/`, `hr/`, etc. as
   filterable `category` metadata):
   ```
   make ingest FILE=data/knowledge_base
   ```
   Without `make`:
   ```
   python -m rag.ingestion.pipeline data/knowledge_base
   ```
   Re-running ingestion on unchanged files is a no-op (checksum-based, no
   duplicate chunks); edited files are detected and re-chunked in place
   under the same `document_id`. Point the same command at any other
   directory or single file to ingest a different dataset — the path is
   always a CLI argument, never hardcoded.
5. Start the API:
   ```
   uvicorn rag.api.main:app --reload
   ```
   Then hit `GET /health`, `POST /ingest`, `POST /query`, or browse
   `/docs` / `/redoc`.
6. Query via the Makefile once the API is running:
   ```
   make query Q="What is DocuFlow?"
   ```
   Without `make`:
   ```
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is DocuFlow?"}'
   ```
   Filter retrieval by the `category` metadata preserved from the folder
   structure (e.g. only search the security docs):
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

## Metadata & filtering

Every chunk carries: `document_id, chunk_id, source, source_type, title,
author, url, created_at, last_modified, language, category`. For Markdown
sources with a YAML front-matter block (`title`/`owner`/`last_reviewed`,
as used throughout `data/knowledge_base/`), `title`/`author`/`last_modified`
are parsed from it rather than falling back to the filename/filesystem
timestamp, and the front-matter block itself is stripped before chunking.
`category` is the file's folder path relative to whatever root you ingested
(e.g. `security`, or `security/subteam` for nested folders) — `None` for a
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
path-suffix, not exact equality or a hardcoded root — so the same gold file
and the same `run_eval` command work regardless of what directory you
actually ingested from.

Two gold files are included: `data/eval/techfusion_gold.jsonl` (46
questions against the TechFusion knowledge base — single-document,
multi-hop, and unanswerable) and `data/eval/sample_gold.jsonl` (a tiny
smoke-test companion to `data/sample_docs/`).

```
python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl
```
Point `--gold` at any other file to evaluate a different dataset; nothing
about the runner is TechFusion-specific. Reports, for the configured
pipeline:
- **Recall@5 / Recall@10** and **Hit Rate@5 / Hit Rate@10** — recall
  averages the fraction of *all* relevant documents found per question;
  hit rate is binary (was *any* relevant document found), so the two
  diverge on multi-document questions.
- **MRR** (mean reciprocal rank).
- **Retrieval / generation / total latency** (mean, ms) — measured from the
  pipeline's actual configured `rerank_top_n` (not the broader top-10 fetch
  used for the Recall@10 calculation above, which would otherwise inflate
  the latency number).
- **Answer quality** — a keyword-overlap heuristic scored against
  `expected_answer` (see the caveat in its own output; it's a placeholder,
  not a correctness judge — RAGAS is the real fix, see Roadmap).

Add `--verbose` to include full per-question detail (retrieved sources,
generated answer, individual scores) in the JSON output, or
`--skip-generation` to get only the retrieval metrics quickly, without
waiting on LLM calls for all 46 questions.

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

- **Hybrid Search** — combine vector similarity with keyword/BM25 search.
- **Cross Encoder** — mature the scaffolded `cross_encoder` reranker
  (`rerankers/cross_encoder.py`) with real benchmarking/tuning.
- **Cohere Reranking** — mature the scaffolded optional `cohere` provider
  (`rerankers/cohere.py`) once there's a use case needing a hosted reranker.
- **RAGAS** — plug a standard answer-quality/faithfulness metric suite into
  `eval/answer_quality.py`, replacing the keyword-overlap placeholder.
- **LangGraph** — move retrieval/generation orchestration to a graph for
  multi-step or agentic query handling.
- **MCP** — expose ingest/query as MCP tools for use from other agents.
- **Multi-agent workflows** — e.g. a query-planning agent in front of
  retrieval, or specialized agents per document source.
- **Kubernetes deployment** — containerize the API and vector store for a
  non-local deployment target.
- **Observability** — OpenTelemetry tracing/metrics on top of the current
  structured JSON logging.
