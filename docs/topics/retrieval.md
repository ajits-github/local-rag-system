# Retrieval

`config.retrieval.provider` selects `dense` (pure vector similarity) or
`hybrid` (dense + BM25 keyword search, fused via Reciprocal Rank Fusion
before reranking). Fusion goes through rank position rather than
combining raw scores directly, since a cosine-similarity score and an
unbounded BM25 score are never on comparable scales.

Retrieval has three independent, explicitly-named cutoffs rather than one
overloaded "top k": `candidate_k` (how many candidates each branch fetches
from the vector store), `reranker_top_n` (how many a real reranker keeps
after rescoring), and `generation_context_top_n` (how many ranked chunks
actually reach the generation prompt). See
[Retrieval Cutoff Semantics](../architecture.md#retrieval-cutoff-semantics)
for why that split exists and what broke before it did.

An optional post-rerank relationship-expansion step and field-level
redaction pass both run inside the same retrieval pipeline; see
[Security](security.md) and
[Multimodal + Relationship-Aware Ingestion](../architecture.md#multimodal-relationship-aware-ingestion)
for those.

API reference: [Retrieval](../reference/retrieval.md), [Vector store](../reference/vectorstore.md).
