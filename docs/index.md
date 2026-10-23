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

![Local RAG System architecture: users and clients through authentication
and access control, query processing, routing to classic RAG or the
bounded agent harness, evidence sources (documents, retrieval stack,
graph/semantic layer, MCP tools), context engineering, Ollama
generation, response generation, evaluation/feedback, observability, and
deployment/infrastructure](assets/architecture-diagram.png)

A few pieces of this diagram are ahead of what's implemented today, not
built capabilities: **Context Engineering** (the whole token-budget
stage) is planned next, today context sizing is three explicit cutoffs;
see [Retrieval](topics/retrieval.md), not a dedicated
collect/filter/score/pack pipeline. **OCR for scanned documents** doesn't
exist; PDF/DOCX parsing extracts existing text/tables/structure, it
doesn't read text out of a scanned image with no text layer.
**Alerting** isn't wired up either, Prometheus/Grafana exist, but
nothing pages on a threshold today (see [Metrics](metrics.md)). The
"Integrations (Confluence, Jira, etc.)" and "Custom tools (search,
calculators, workflows)" boxes describe the *shape* the tool layer could
extend to, not systems this project actually connects to; MCP's real
tools today are the four RAG tools plus a synthetic, in-memory
customer-support case store (see [MCP](topics/mcp.md)).

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
