# local-rag-system

[![CI](https://github.com/ajits-github/local-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ajits-github/local-rag-system/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/github/license/ajits-github/local-rag-system)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](#documentation)
![Offline](https://img.shields.io/badge/inference-offline%20%2F%20CPU--only-informational)

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![pgvector](https://img.shields.io/badge/pgvector-0891B2?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com)

[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-425CC7?logo=opentelemetry&logoColor=white)](https://opentelemetry.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)](https://grafana.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)

A modular, config-driven Retrieval-Augmented Generation system that runs
fully offline on a CPU-only machine: sentence-transformers embeddings,
Postgres+pgvector storage, Ollama generation, FastAPI serving. Every
infrastructure choice, embedding model, vector backend, chunker,
reranker, LLM, is a swappable "provider" selected in
`config/default.yaml`; nothing is hardcoded, so comparative experiments
can change one axis without touching pipeline code.

- Fully offline, CPU-only, runs on 8GB RAM; no API keys required by default
- Configurable retrieval (dense or hybrid dense+BM25), chunking, and reranking
- Optional, independently toggleable: JWT auth + tenant/role authorization,
  field-level redaction, rate limiting, a bounded agentic (tool-calling)
  workflow, and OpenTelemetry/Prometheus/Grafana observability
- Deterministic + RAGAS evaluation with an experiment-tracking table (see
  [Benchmarks](#benchmarks))
- A React web UI (`frontend/`) for interactive testing of both RAG paths

The fastest path is: install Python dependencies, start Postgres, ingest
`data/sample_docs`, start the API, and call `/query` (see
[Setup](#setup)).

## Table of contents

- [Architecture](#architecture)
- [Multimodal & Layout-Aware Ingestion](#multimodal--layout-aware-ingestion)
- [Agentic RAG](#agentic-rag)
- [MCP](#mcp)
- [Security](#security)
- [Observability](#observability)
- [Web UI](#web-ui)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Containerized development](#containerized-development)
- [Configuration](#configuration)
- [Metadata & filtering](#metadata--filtering)
- [Evaluation](#evaluation)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Documentation](#documentation)
- [Roadmap](#roadmap)

## Architecture

<!-- --8<-- [start:docs-architecture-diagram] -->
```mermaid
flowchart TD
    Docs["Documents / Knowledge Sources"] --> Ingestion["Ingestion Pipeline"]
    Ingestion --> DB[("PostgreSQL + pgvector")]

    User(("User")) --> API["FastAPI"]
    API --> Retrieval["Retrieval"]
    DB --> Retrieval
    Retrieval --> Prompt["Prompt Builder"]
    Prompt --> LLM["LLM"]
    LLM -->|"Answer + Sources"| User

    Retrieval --> Evaluation
    LLM --> Evaluation["Evaluation<br/>Recall at k / MRR / RAGAS"]
```
<!-- --8<-- [end:docs-architecture-diagram] -->

For the detailed system view, see [`docs/architecture.md`](docs/architecture.md).

| Component | Current implementation | Configurable / future |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | swappable |
| Vector DB | PostgreSQL + pgvector | swappable |
| Chunker | structured Markdown | swappable |
| Retrieval | dense or hybrid search | configurable |
| Reranker | optional cross-encoder | Cohere / none |
| LLM | Qwen2.5 via Ollama | swappable |
| Prompt | versioned YAML templates | configurable |
| Evaluation | Recall@k, MRR, RAGAS | extensible |
| Security | JWT auth, tenant/role ACL, field redaction, rate limiting | independently toggleable, off by default |
| Agent | bounded tool-calling workflow (`POST /agent/query`) | extensible |
| MCP | Streamable HTTP server, 4 agentic-RAG tools + 2 synthetic business tools | agent-as-client (Stage 2) |

## Multimodal & Layout-Aware Ingestion

PDF and DOCX documents keep their real structure (headings, tables,
code/config blocks, images, page numbers) instead of being flattened to
plain text; both loaders serialize into the same Markdown-equivalent
syntax the default chunker already parses. Layout-aware ingestion is
production-quality: `experiment_034` vs. `experiment_033` in the
[Benchmarks](#benchmarks) table below shows a clear win from preserving
structure (Recall@5 0.946 → 1.000, answer quality 0.264 → 0.310).

An optional local vision provider (`vision.provider: ollama`, default
`none`) can describe images with a small offline model (`moondream`), as
a second chunk alongside the caption/alt-text, never replacing it.
**Vision is experimental and off by default**: `experiment_035` (vision
on) shows no answer-quality improvement over `experiment_034`
(layout-aware text alone, vision off) on the same 28-question set. See
[Multimodal & Layout](docs/topics/multimodal.md) for the full design and
evaluation.

<!-- --8<-- [start:docs-agentic-rag] -->
## Agentic RAG

`POST /agent/query` is an optional bounded workflow above the classic
`POST /query` path. It can route simple questions to classic RAG and use a
small tool loop for complex or multi-hop questions. The graph is bounded
by `max_agent_steps`, `max_retrieval_attempts`, and `max_tool_calls`, and
uses the same retrieval authorization, freshness, redaction, and injection
checks as the classic path.
<!-- --8<-- [end:docs-agentic-rag] -->

The details live in [`docs/architecture.md`](docs/architecture.md). A
baseline is evaluated end-to-end; see
[Agentic RAG benchmarks](#agentic-rag-benchmarks) below.

<!-- --8<-- [start:docs-mcp] -->
## MCP

A secure MCP (Model Context Protocol) server, mounted into the same
`rag-api` process, exposes the four agentic-RAG tools
(`search_knowledge_base`/`get_document`/`get_latest_document`/
`get_related_context`) over Streamable HTTP for external MCP clients
(Claude Desktop, another agent runtime). It wraps the exact same,
already-hardened `rag.agent.tools.*` functions the in-process agent
calls, unmodified: identity is resolved from the transport (a verified
JWT, never a tool argument -- structurally excluded from the tool schema,
not just rejected by validation), every result passes through the same
sanitization choke point as the agent path, and an unknown or injected
tool argument now fails loudly rather than being silently dropped.

The same server also exposes two synthetic business-backend tools
(`get_customer_case`/`get_case_status`, Stage 1B), demonstrating MCP as
an integration layer to a separate backend system, not just another
transport for this deployment's own RAG tools. They read from a small,
in-memory synthetic customer-support-case dataset with its own tenant/
role authorization (same rule, same config allow-list document ACL
already uses for cross-tenant access) -- structurally independent of the
RAG tools' chunk-level sanitization, since a case has no chunk/field-
redaction concept. Server-only for now; making the in-process agent an
MCP client (Stage 2) is still deferred until there's a concrete need for
it.
<!-- --8<-- [end:docs-mcp] -->

Off by default (`mcp.enabled: false`, a true no-op); full design and the
two hardening fixes: [MCP](docs/topics/mcp.md).

<!-- --8<-- [start:docs-security] -->
## Security

Retrieval-time tenant/role authorization, document-version freshness,
field-level sensitive-data redaction, prompt-injection detection, and an
authenticated API boundary (JWT verification, DoS limits, rate limiting)
all sit above both the classic and agentic RAG paths, each independently
toggleable and off by default (`config.security.*`). Authentication
happens at the API boundary; authorization stays enforced entirely at
retrieval, as two structurally separate modules.
<!-- --8<-- [end:docs-security] -->

Full control-by-control breakdown: [Security](docs/topics/security.md).

<!-- --8<-- [start:docs-observability] -->
## Observability

Operational telemetry on top of the agentic workflow: per-node timing
(with a real-LLM-inference-vs-overhead split), OpenTelemetry traces,
Prometheus metrics, a local Grafana dashboard, and a safe live-progress
SSE stream (`POST /agent/query/stream`). Distinct from
[MLflow tracking](#mlflow-tracking) below, which tracks experiment runs,
not live requests.

`/metrics` and the SSE stream are on by default (both true no-ops with
nothing scraping/consuming them by default); OpenTelemetry tracing is off
by default (`observability.tracing.enabled: false`) since it needs a real
OTLP endpoint to be useful. To see traces and dashboards locally:

```bash
# set observability.tracing.enabled: true in config/default.yaml first
make observability-up
```

This brings up Jaeger (traces, `http://localhost:16686`), Prometheus
(`http://localhost:9090`), and Grafana (`http://localhost:3000`,
pre-provisioned with a dashboard) alongside the base stack, layered via
`docker compose -f docker-compose.yml -f docker-compose.observability.yml
up -d`, never brought up by plain `make up`. Teardown: `make
observability-down`.
<!-- --8<-- [end:docs-observability] -->

Full design writeup: [Observability](docs/topics/observability.md).

<!-- --8<-- [start:docs-web-ui] -->
## Web UI

A small React/Vite/TypeScript chat interface (`frontend/`) sits over the
existing API: choose Classic or Agentic RAG, watch live agent progress,
inspect sources/citations, and see an always-visible summary of the
connected backend's active security/agent feature flags (`GET /`'s
`features` block, booleans and provider names only, never a secret). It
reuses the backend as-is; no RAG logic lives in the frontend.

```bash
make up             # backend only (Postgres + rag-api)
make frontend-up    # backend + frontend, http://localhost:3001
```

or for local frontend development against a native `uvicorn`/Docker
backend: `cd frontend && npm install && npm run dev`
(`http://localhost:5173`). See [`frontend/README.md`](frontend/README.md)
for the full setup, authentication behavior, and known limitations.
<!-- --8<-- [end:docs-web-ui] -->

<!-- --8<-- [start:docs-prereq-setup] -->
## Prerequisites

- Python 3.11+
- [Docker](https://www.docker.com/) (for the Postgres+pgvector container)
- [Ollama](https://ollama.com/) installed, **running**, and with
  `qwen2.5:1.5b` pulled:
  ```bash
  ollama pull qwen2.5:1.5b
  ```
  Ollama usually starts automatically after install; verify it's actually
  serving with `ollama list` (or `curl http://localhost:11434/api/version`).
  If it's installed but not on your PATH, invoke it by its full path (e.g.
  on Windows: `C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe`).
  Optional: pull `moondream` (`ollama pull moondream`) only if you plan to
  set `vision.provider: ollama` in config. The default `vision.provider:
  none` needs no extra model.
- **`make`**: optional shortcut for `up`/`ingest`/`query`/`test`.
  **Not installed by default on Windows** (neither Git Bash nor PowerShell
  ship it). Install it via `choco install make`, `scoop install make`, or
  WSL, *or* skip it entirely and run the underlying command shown next to
  each `make` target below. The Makefile uses `bash`, so native
  PowerShell users should prefer the explicit commands.

## Setup

1. Create a virtualenv and install the project.
   PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e ".[dev]"
   ```
   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Copy `.env.example` to `.env` and adjust if needed. `DATABASE_URL` points
   at Postgres on **host port 15987** (not the default 5432) so it won't
   collide with any other local Postgres instance:
   ```bash
   cp .env.example .env
   ```
   Already have a Postgres+pgvector container running locally? Point
   `DATABASE_URL` at it directly instead of step 3's `docker-compose.yml`
   (matching its real `POSTGRES_DB` name), then still run
   `python scripts/init_db.py` (idempotent) to create/migrate the schema.
3. Bring up Postgres+pgvector and initialize the schema:
   ```bash
   make up
   ```
   Without `make`, run the two steps it wraps:
   ```bash
   docker compose up -d
   python scripts/init_db.py
   ```
   If `init_db.py` cannot connect immediately, wait until the
   `local-rag-postgres` container is healthy and run it again.
4. Ingest the bundled smoke-test documents.
   ```bash
   make ingest FILE=data/sample_docs DATASET_ID=sample_docs
   ```
   Without `make`:
   ```bash
   python -m rag.ingestion.pipeline data/sample_docs --dataset-id sample_docs
   ```
   Any file or directory can be ingested. Use a different `--dataset-id`
   when you ingest a separate corpus.

   Re-running ingestion on unchanged files is a no-op (checksum-based, no
   duplicate chunks); edited files are detected and re-chunked in place
   under the same `document_id`.
5. Start the API:
   ```bash
   uvicorn rag.api.main:app --reload
   ```
   Then hit `GET /health`, `POST /ingest`, `POST /query`, or browse
   `/docs` / `/redoc`.
6. Query via the Makefile once the API is running:
   ```bash
   make query Q="What are the deployment windows for production releases?"
   ```
   Without `make`:
   ```bash
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What are the deployment windows for production releases?"}'
   ```
   Response shape (`answer` text, `score`, and timing values vary by run;
   the `deployment-process.md` source is what to expect from this exact
   query against the sample corpus):
   ```json
   {
     "answer": "Standard deployments are only permitted on Tuesdays and Thursdays, between 10:00 and 14:00 UTC. Deployments outside this window require sign-off from the on-call engineering manager. Emergency hotfixes for active incidents are exempt from the window restriction.",
     "sources": [
       {
         "chunk_id": "3f1a...2_1",
         "document_id": "3f1a...2",
         "source": "deployment-process.md",
         "category": null,
         "score": 0.83,
         "content_type": "prose",
         "section_path": "Deployment Windows"
       }
     ],
     "retrieval_ms": 42.1,
     "generation_ms": 3190.4,
     "total_ms": 3232.5
   }
   ```
   Filter retrieval by the `category` metadata preserved from folder
   structure; only meaningful once you've ingested a dataset with
   subfolders (`data/sample_docs/` is flat, so this is illustrative rather
   than directly runnable against it):
   ```bash
   curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"query": "What MFA methods are approved?", "filters": {"category": "security"}}'
   ```
<!-- --8<-- [end:docs-prereq-setup] -->

<!-- --8<-- [start:docs-containerized-dev] -->
## Containerized development

Postgres+pgvector and the API can both run in containers. Ollama usually
stays native on the host, and the API container reaches it through
`host.docker.internal`. Requires **Docker Compose V2**.

```bash
docker compose up -d --build          # first build pulls ~1-2GB of ML deps (PyTorch); later ones are cached
python scripts/init_db.py             # against the containerized Postgres, via DATABASE_URL in .env
curl http://localhost:8000/health     # {"vectorstore":"ok","llm":"ok"} confirms it reached both deps
scripts/smoke_test_containers.sh      # full ingest -> query round trip, one shot
docker compose down                   # add -v to also drop the pgdata volume
```
Ingest/query work exactly as in native dev, same port/endpoints. `src/`
and `config/` are bind-mounted with `--reload`, so Python edits need no
rebuild. Only `pyproject.toml` changes require rebuilding the image.

**Production-shaped run**: `docker-compose.prod.yml` drops the bind
mounts and `--reload`, stops publishing Postgres's port, and tightens
health-check timing (same-host "productionized," not a Kubernetes
substitute):
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

**Running the API against an alternate config file**: `RAG_CONFIG_PATH`
(unset by default) overrides which YAML the API loads instead of
`config/default.yaml`. `config/` is already bind-mounted into the
container at `/app/config`, so any file under `config/experiments/` is
reachable by its in-container path. Set it in the host shell before
bringing the stack up (`docker-compose.yml` forwards it through
unchanged; it is never hardcoded there):
```bash
RAG_CONFIG_PATH=/app/config/experiments/agentic-rag-baseline-v1.yaml docker compose up -d
```
An unset or empty value preserves the default behavior exactly. A path
that doesn't exist, or isn't valid YAML, fails loudly at config-load time
rather than silently falling back to `config/default.yaml`.

### Windows-specific notes

- `host.docker.internal` (used to reach native Ollama) is a Docker Desktop
  feature; `docker-compose.yml` also adds `extra_hosts:
  host.docker.internal:host-gateway` so it resolves on native Linux too.
- Docker Desktop's Windows file sharing doesn't enforce host-side POSIX
  permissions, so the non-root `rag` user can read/write the bind-mounted
  volumes without any `chown`/`chmod` step (native Linux would need the
  container UID to match the host user's).
- The embedding model downloads from Hugging Face Hub on first use inside
  a fresh container/volume if not already cached. It needs outbound internet
  that one time, same as native.
<!-- --8<-- [end:docs-containerized-dev] -->

## Configuration

All provider choices and tunables live in `config/default.yaml`:
embedding model, chunk size/overlap, reranker (`none` / `cross_encoder` /
`cohere`), LLM, retrieval `candidate_k`, generation context size, security
toggles, and agent bounds. Point at an alternate config with
`--config path/to/other.yaml` on the ingestion/eval CLIs. The running API
(not a CLI) instead reads the `RAG_CONFIG_PATH` environment variable,
unset by default; see "Running the API against an alternate config file"
above. Comparable configs for experiments live under
`config/experiments/`, each a full standalone copy rather than a partial
override.

Generation prompts are versioned YAML files under
`src/rag/prompts/templates/` (`rag_answer_v1.yaml`, `rag_answer_v2.yaml`,
...), loaded and validated by `src/rag/prompts/loader.py`. The active
version is selected by `config/default.yaml`'s `generation.prompt` block
(`id`, `version`, `path`); `RetrievalPipeline` loads it once at
construction and rejects the file if its own declared `prompt_id`/`version`
don't match what's configured. API reference:
[Configuration & factory](docs/reference/config.md).

## Metadata & filtering

Every chunk carries source metadata such as `document_id`, `chunk_id`,
`source`, `source_type`, timestamps, language, `category`, governance
fields, and structural fields. For directory ingestion, `category` is the
file's path relative to the ingested root, such as `security` or
`security/subteam`. See [Ingestion & Chunking](docs/topics/ingestion.md)
for the full structural-metadata model.

`POST /query` accepts exact-match filters for an explicit allow-list of
metadata fields (never an arbitrary column). The most common filter is
`category`.

## Evaluation

`data/eval/sample_gold.jsonl` is a small smoke-test gold file for
`data/sample_docs`. Run it after the sample dataset is ingested:

```bash
python -m rag.eval.run_eval --gold data/eval/sample_gold.jsonl --dataset-id sample_docs
```
`--dataset-id` is mandatory and is injected as a filter on every retrieval.
Reports include Recall@5/10, Hit Rate@5/10, MRR, latency, and a simple
keyword-overlap answer-quality score. Add `--verbose` for per-question
details or `--skip-generation` for retrieval-only metrics. See
[Evaluation & RAGAS](docs/topics/evaluation.md) and the
[metrics reference](docs/metrics.md) for the full deterministic-metric
breakdown, including the multimodal/security-specific ones not shown here.

### RAGAS generation-quality evaluation (optional)

RAGAS scoring is optional and uses a separately configured judge model.
Install the extra only if you plan to run it:
```bash
pip install .[ragas]           # local (ollama) judge
pip install .[ragas,anthropic] # + hosted anthropic judge
```
The judge is selected in `config/default.yaml` under `judge:`. Hosted
judges read API keys from environment variables; no key is hardcoded.

```bash
python -m rag.eval.run_ragas_eval --gold data/eval/sample_gold.jsonl \
    --dataset-id sample_docs --verbose > /tmp/ragas_report.json
```
`--sample-size` defaults to 15 so hosted judging can be tested cheaply
before scaling up. Judge calls are cached under `.cache/ragas` by default.

RAGAS scores should be checked against human labels before being treated
as a quality gate:
```bash
python scripts/generate_manual_review.py --eval-output /tmp/ragas_report.json --num-rows 10
# fill in the human_faithful/human_correct/human_relevant/human_correct_refusal
# fields in the generated JSONL by hand, then:
python scripts/compare_ragas_manual.py --ragas-output /tmp/ragas_report.json \
    --manual-review data/eval/manual_review/sample_docs_manual_review.jsonl
```
The comparison report shows where the judge agrees or disagrees with the
human labels.

<!-- --8<-- [start:docs-benchmarks] -->
## Benchmarks

`experiments/` tracks comparable eval runs over time: `config_snapshots/`
holds a snapshot of the config used per experiment, `results/` holds a
flat JSON metrics record per experiment, and `reports/` holds the
generated comparison table. Only aggregate metrics and config values are
stored here, never corpus content or generated answers, so all three are
safe to commit even though the underlying corpus (`data/knowledge_base/`)
is gitignored (only `data/sample_docs/` ships in the repo).

The table below is generated, not hand-written; don't edit it directly,
regenerate it instead (see "Recording a new experiment"). It shows only
the most recent 7 experiments; the full, ever-growing history always
lives in `experiments/reports/comparison.md`.

<!-- EXPERIMENTS_TABLE_START -->
| Experiment | Label | Retrieval | Generation model | Embedder | Reranker | Prompt | Rel.Exp | Recall@5 | Recall@10 | Hit Rate@10 | MRR | Answer quality | Supp.Ctx Hit | Img Hit | RAGAS Faithful | RAGAS Correct | Total latency | Dataset | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| experiment_033 | layout_vision_a_text_only | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v2 | on | 0.946 | 1.000 | 1.000 | 0.896 | 0.264 | 0.938 | 0.000 | - | - | 7.9s | layout_vision_text_baseline | 2026-08-21 |
| experiment_034 | layout_vision_b_structured | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v2 | on | 1.000 | 1.000 | 1.000 | 0.936 | 0.310 | 0.938 | 0.958 | - | - | 8.8s | layout_vision_structured | 2026-08-21 |
| experiment_035 | layout_vision_c_vision | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v2 | on | 1.000 | 1.000 | 1.000 | 0.948 | 0.298 | 0.938 | 0.958 | - | - | 6.8s | layout_vision_structured_vision | 2026-08-21 |
| experiment_036 | rag_answer_v4_prompt_ab_vs_v3_control | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v4 | on | 0.788 | 0.832 | 0.865 | 0.753 | 0.440 | 0.781 | 0.842 | - | - | 21.4s | techfusion | 2026-08-22 |
| experiment_037 | rag_answer_v5_prompt_ab_vs_v3_control | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v5 | on | 0.788 | 0.832 | 0.865 | 0.753 | 0.433 | 0.781 | 0.842 | - | - | 29.2s | techfusion | 2026-08-22 |
| experiment_038 | rag_answer_v3_control_ragas_25q_prompt_ab_sample | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v3 | on | 0.440 | 0.480 | 0.480 | 0.407 | 0.389 | 0.833 | - | 0.674 | 0.439 | 10.3s | techfusion | 2026-08-22 |
| experiment_039 | rag_answer_v5_ragas_25q_prompt_ab_sample | hybrid | qwen2.5:3b | all-MiniLM-L6-v2 | none | v5 | on | 0.440 | 0.480 | 0.480 | 0.407 | 0.361 | 0.833 | - | 0.707 | 0.379 | 12.6s | techfusion | 2026-08-22 |
<!-- EXPERIMENTS_TABLE_END -->

*Total latency is the mean retrieval plus generation time per question.
Every row is measured with `dataset_id`-isolated retrieval, so results are
not mixed across datasets in the same vector store.*
<!-- --8<-- [end:docs-benchmarks] -->

Detailed experiment writeups live in
[`experiments/reports/`](experiments/reports/) and
[`docs/architecture.md`](docs/architecture.md). The README keeps the
comparison table so changes remain visible at a glance without turning
this file into an experiment log.

### Agentic RAG benchmarks

The table above is classic-RAG only (Recall/MRR/answer_quality). The
agentic evaluation (`rag.eval.run_agent_eval`, an 18-question gold set)
reports a different metric family, so it gets its own small table here
instead of being forced into incompatible columns. Recorded by hand from
`experiments/results/agentic/*.json`; not wired into
`scripts/compare_experiments.py`, which targets the classic-RAG schema
only.

| Experiment | Label | Classify prompt | Synthesize prompt | Routing accuracy | Unnecessary agent rate | Citation support | Answer correctness | RAGAS Faithful | RAGAS Correct | Mean latency | Dataset | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| experiment_029 | agentic_rag_baseline_v1 | v1 | v1 | 0.833 | 1.000 | 0.111 | 0.449 | 0.443 | 0.584 | 138.5s | techfusion | 2026-08-21 |
| experiment_032 | agentic_rag_baseline_v2_fixed | v2 | v2 | 0.889 | 0.000 | 1.000 | 0.463 | 0.511 | 0.531 | 47.4s | techfusion | 2026-08-21 |

`experiment_032` fixed the two issues `experiment_029` found: a revised
classify prompt makes the simple/complex routing boundary explicit (cut
the unnecessary-agent rate to 0.0 and mean latency 2.9x, purely via
routing composition), and a revised synthesize prompt plus a code-level
evidence-ordering fix makes an authoritative source always take priority
over an untrusted, conflicting one at synthesis time (raised citation
support 0.111 to 1.000). Full before/after numbers and per-question trace
validation:
[`experiments/reports/agentic_rag_baseline_v2_fixed.md`](experiments/reports/agentic_rag_baseline_v2_fixed.md).

### Recording a new experiment

1. Change one thing in `config/default.yaml` (reranker, model, chunk size,
   prompt version, ...).
2. Run the eval and save its full report (or `rag.eval.run_ragas_eval` for
   the RAGAS-scored variant, see above):
   ```bash
   python -m rag.eval.run_eval --gold data/eval/sample_gold.jsonl \
     --dataset-id sample_docs --verbose > /tmp/eval_report.json
   ```
3. Register it as an experiment (never re-runs eval, just records the
   report above):
   ```bash
   python scripts/record_experiment.py --eval-output /tmp/eval_report.json \
     --experiment-id experiment_002 --label "cross-encoder reranker" \
     --config config/default.yaml
   ```
   Captures every config axis (embedding model, chunk size, reranker,
   prompt version + checksum, and RAGAS scores if present) and also logs
   an MLflow run (see "MLflow tracking" below), unless `mlflow.enabled:
   false`.
4. Regenerate the comparison table (writes the full table to
   `experiments/reports/comparison.md` and splices the most recent 7
   experiments into this README section):
   ```bash
   python scripts/compare_experiments.py
   ```
   Pass `--exclude experiment_id[,experiment_id...]` to leave a
   non-comparable record (e.g. a small pilot run at a different sample
   size) out of the table.

### MLflow tracking

Every `scripts/record_experiment.py` call also logs an MLflow run (config
as params, metrics, and the eval-output/record/config files as artifacts),
under a readable run name like `experiment_015_qwen2-5-3b_v2_hybrid_rel-exp`
and tagged with `experiment_id`/`label`/`generation_model`/`prompt_version`/
`retrieval_provider`/`reranker_provider`/`relationship_expansion`/
`dataset_id` for fast filtering in the MLflow UI. Requires the `mlflow`
extra (`pip install .[mlflow]`); fails loudly rather than silently
skipping if enabled but not installed. Local by default, no server
required (`mlflow.tracking_uri: sqlite:///mlflow.db`):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
`scripts/backfill_mlflow.py` re-logs every `experiments/results/*.json`
into MLflow in one pass, useful if `mlflow.db`/`mlruns/` is ever deleted.

## Testing

```bash
make test
```
Without `make`: `pytest tests/ -v`.

Unit tests (`tests/unit`) mock all external I/O and always run. Integration
tests (`tests/integration`) assume Postgres (`make up`) and a local Ollama
with `qwen2.5:1.5b` are already running, and skip with a clear message if
either is unreachable.

`scripts/smoke_test_containers.sh` is a separate check for the
containerized setup specifically: it assumes `docker compose up -d` (or
the prod-override form) is already running, and proves the *containerized*
API can reach Postgres+pgvector and native Ollama, then does a real
ingest -> query round trip against them (see
[Containerized development](#containerized-development) above).

The frontend has its own test suites (unit, integration, and Playwright
e2e); see [`frontend/README.md`](frontend/README.md).

## Continuous Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every
PR/push to `main` (and via manual `workflow_dispatch`), in four parallel
jobs:

- **code-quality**: `pre-commit run --all-files` (ruff, ruff-format, mypy,
  hygiene hooks; the same checks `pre-commit install` runs locally).
- **unit-tests**: the full `tests/unit` suite; no external services.
- **integration-tests**: `tests/integration` against a `pgvector/pgvector:pg16`
  service container spun up by the workflow itself (not your local `make up`
  stack). Tests gated on a local Ollama (`require_ollama` in
  `tests/integration/conftest.py`) skip cleanly, since CI doesn't run Ollama.
- **docker-build**: builds the production image from `Dockerfile`, runs it
  against the same Postgres service container, and checks `GET /health`
  reports `vectorstore: ok` (Ollama is expected `unreachable` here, for the
  same reason as above). No image is pushed anywhere yet.

What CI intentionally leaves out: RAGAS scoring, hosted-judge (OpenAI/
Anthropic) calls, MLflow experiment logging, and full private-corpus
evaluation. These need either a local Ollama, hosted API keys, or
minutes-long LLM calls that don't belong in per-PR feedback. They stay
manual, run locally via `scripts/record_experiment.py` /
`eval/run_ragas_eval.py` as today.

## Documentation

A generated docs site (MkDocs + Material + mkdocstrings) sits over this
README, `docs/architecture.md`, `docs/metrics.md`, and every module's
NumPy-style docstrings, without duplicating any of them:

```bash
pip install -e ".[docs]"
make docs-serve   # live-reloading local server, http://127.0.0.1:8000
make docs-build   # strict build to site/; fails on broken links/anchors/nav references
```

Without `make`: `mkdocs serve` / `mkdocs build --strict`. Nav covers
architecture, ingestion/chunking, retrieval, multimodal/layout, security,
Agentic RAG, MCP, evaluation/RAGAS, observability, the web UI, deployment/
runtime, and a full API reference generated from `src/rag/**`. Not
published anywhere yet (`mkdocs build --strict` is the local/CI gate for
now).

## Roadmap

Deferred for now, tracked here rather than left as empty scaffolding:

- **Cross Encoder**: mature the scaffolded `cross_encoder` reranker
  (`rerankers/cross_encoder.py`) with real benchmarking/tuning.
- **Cohere Reranking**: mature the scaffolded optional `cohere` provider
  (`rerankers/cohere.py`) once there's a use case needing a hosted reranker.
- **RAGAS trust**: judge-vs-human agreement has been spot-checked
  (`scripts/compare_ragas_manual.py`) on several samples, but scores still
  aren't validated against human labels by default. See the caveat in
  every RAGAS report.
- **Vision**: mature the experimental `ollama`/`moondream` vision provider
  (see [Multimodal & Layout-Aware Ingestion](#multimodal--layout-aware-ingestion)
  above) once description quality on real images has been evaluated
  further; current evidence doesn't show an answer-quality improvement.
- **MCP Stage 2**: making the in-process agent itself an MCP client,
  deferred until there's a concrete need for it. See [MCP](#mcp) above
  for what's already shipped (Stage 1A: the four agentic-RAG tools;
  Stage 1B: the two synthetic business-case tools).
- **LangGraph**: revisit only if a future need appears for checkpoint/
  resume across requests, human-in-the-loop approval, or substantially
  more complex branching than today's bounded agentic workflow (see
  [`docs/architecture.md`](docs/architecture.md)'s "Agentic RAG" section)
  before adopting it.
- **Kubernetes deployment**: containerize the API and vector store for a
  non-local deployment target.

Recently shipped features (observability, layout-aware ingestion +
vision, security/auth, the web UI, this documentation site) are covered
in their own sections above, not repeated here. See the
[documentation site](#documentation) for the full design writeups behind
everything already shipped.
