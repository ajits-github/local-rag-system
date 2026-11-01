# local-rag-system

A modular, config-driven local RAG system: sentence-transformers embeddings,
Postgres+pgvector storage, Ollama generation, FastAPI serving. Every infra
choice (embedding model, vector backend, chunker, reranker, LLM) is
swappable via `config/default.yaml`, so comparative experiments can change
one axis without touching pipeline code. The core serving path runs fully
offline on a CPU-only, 8GB-RAM machine; no API keys required by default.

This site is generated from the same Markdown files and Python docstrings
that live in the repository (`README.md`, `docs/architecture.md`,
`docs/metrics.md`, and `src/rag/**`), so it stays in sync with the code
without duplicating it.

![Local RAG System architecture: Web UI, REST API and documentation entry points; classic RAG and bounded agent as alternative runtime paths with three agent tool families; retrieval stack and knowledge graph; context engineering; Ollama generation; cross-cutting security, observability and evaluation; Docker Compose deployment](assets/architecture-diagram.png)

Where this diagram differs from the code: prompt-injection detection only flags
content (telemetry and prompt reinforcement), it filters nothing. Ollama runs
natively on the host, not in Compose, and Neo4j and monitoring are opt-in Compose
overlays. The Web UI has no graph explorer. OpenTelemetry carries traces only
(metrics are Prometheus, logs are structured JSON). Trust and freshness select
document versions and filter by trust level; they do not validate content. Graph
tools are search, entity, neighbors and dependency path, with no analytics. The MCP
business tools use one in-process server and a synthetic, in-memory case store.
Hybrid retrieval, reranking and Context Engineering are opt-in; the defaults are
dense retrieval, no reranker, and Context Engineering off.

## Start here

| Goal | Page |
|---|---|
| Get the system running locally | [Deployment & Runtime](topics/deployment.md) |
| Understand how documents become chunks | [Ingestion & Chunking](topics/ingestion.md) |
| Understand how a query becomes an answer | [Retrieval](topics/retrieval.md) |
| See how PDFs, DOCX, and images are handled | [Multimodal & Layout](topics/multimodal.md) |
| Understand tenant/role authorization and redaction | [Security](topics/security.md) |
| See the bounded agent workflow above classic RAG | [Agentic RAG](topics/agentic-rag.md) |
| Call the agentic-RAG tools from an external MCP client | [MCP](topics/mcp.md) |
| Check retrieval/generation quality metrics or RAGAS | [Evaluation & RAGAS](topics/evaluation.md) |
| See tracing, Prometheus metrics, and dashboards | [Observability](topics/observability.md) |
| See how answer ratings are captured and exported | [Feedback loop](topics/feedback.md) |
| Read function/class-level API docs | [API Reference](reference/index.md) |
| Read the full system design writeup | [Architecture](architecture.md) |

## Component summary

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
