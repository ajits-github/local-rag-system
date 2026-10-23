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

![Local RAG System architecture: users and clients through security and
request context, the classic/agent RAG runtime, context engineering,
Ollama generation, evidence sources (retrieval stack, graph/semantic
layer, tools, conversation history), evaluation and observability, and
the separate experiment/learning branches](assets/architecture-diagram.png)

Two pieces of this diagram are ahead of what's implemented today, not
built capabilities: **Context Engineering** is planned next (today,
context sizing is three explicit cutoffs; see
[Retrieval](topics/retrieval.md)), and **Conversation / history** /
**conversation continuity** don't exist yet either, every `/query` and
`/agent/query` call is stateless. **NDCG** also isn't an implemented
metric; Recall@k, hit-rate, MRR, and RAGAS are what `eval/run_eval.py`
actually computes (see [Metrics](metrics.md)).

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
