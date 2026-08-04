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
- [Ollama](https://ollama.com/) installed and running locally, with
  `qwen2.5:1.5b` pulled:
  ```
  ollama pull qwen2.5:1.5b
  ```
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
   earlier work)? Point `DATABASE_URL` at it directly instead of step 3 —
   just make sure `POSTGRES_DB` in `.env` matches its actual database name,
   and still run `python scripts/init_db.py` (step 3b) against it to create
   the `documents`/`chunks` tables if they don't already exist.
3. Bring up Postgres+pgvector and initialize the schema:
   ```
   make up
   ```
   Without `make`, run the two steps it wraps:
   ```
   docker compose up -d
   python scripts/init_db.py
   ```
4. Ingest something:
   ```
   make ingest FILE=data/sample_docs
   ```
   Without `make`:
   ```
   python -m rag.ingestion.pipeline data/sample_docs
   ```
5. Start the API:
   ```
   uvicorn rag.api.main:app --reload
   ```
   Then hit `GET /health`, `POST /ingest`, `POST /query`, or browse
   `/docs` / `/redoc`.
6. Or query via the Makefile once the API is running:
   ```
   make query Q="what is pgvector?"
   ```
   Without `make`:
   ```
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "what is pgvector?"}'
   ```

## Configuration

All provider choices and tunables live in `config/default.yaml`:
embedding model, chunk size/overlap, reranker (`none` / `cross_encoder` /
`cohere`), LLM, retrieval `top_k`. Point at an alternate config with
`--config path/to/other.yaml` on the ingestion/eval CLIs, or by editing the
default file directly for local experiments.

## Evaluation

`data/eval/gold.jsonl` is a JSONL file of labeled queries
(`query_id, query, relevant_document_ids, relevant_chunk_ids, question_type,
difficulty, unanswerable`). Run:
```
python -m rag.eval.run_eval --gold data/eval/gold.jsonl
```
to get recall@k and MRR, computed at the document level (stable across
chunking experiments) and at the chunk level when chunk-level labels are
present.

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
