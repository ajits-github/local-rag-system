| # | Label | Retrieval | Generation model | Embedder | Reranker | Recall@5 | Recall@10 | Hit Rate@10 | MRR | Answer quality | RAGAS Faithful | RAGAS Correct | Total latency | Dataset | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TechFusion baseline (qwen2.5:1.5b, no reranker) | dense | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.432 | - | - | 3.7s | techfusion | 2026-08-05 |
| 2 | qwen2.5:3b (candidate generation model) | dense | qwen2.5:3b | all-MiniLM-L6-v2 | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.453 | - | - | 12.1s | techfusion | 2026-08-05 |
| 3 | qwen2.5:3b + cross_encoder reranker | dense | qwen2.5:3b | all-MiniLM-L6-v2 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.891 | 0.967 | 0.978 | 0.822 | 0.507 | - | - | 10.1s | techfusion | 2026-08-05 |
| 4 | qwen2.5:3b + cross_encoder + BGE-small embedder | dense | qwen2.5:3b | bge-small-en-v1.5 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.902 | 0.935 | 0.957 | 0.825 | 0.483 | - | - | 10.3s | techfusion-bge-small | 2026-08-05 |
| 5 | qwen2.5:3b + cross_encoder + BGE-small + chunk_size=300 | dense | qwen2.5:3b | bge-small-en-v1.5 | cross_encoder (ms-marco-MiniLM-L-6-v2) | 0.902 | 0.946 | 0.957 | 0.808 | 0.427 | - | - | 9.9s | techfusion-bge-small-chunk300 | 2026-08-05 |
| 6 | structured_markdown chunker (62-question gold set, tables/code/config/chart) | dense | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.919 | 0.952 | 0.968 | 0.865 | 0.387 | - | - | 2.8s | techfusion | 2026-08-06 |
| 7 | RAGAS full evaluation (OpenAI gpt-4o-mini judge, all 62 questions) | dense | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.919 | 0.952 | 0.968 | 0.865 | 0.413 | 0.786 | 0.469 | 4.4s | techfusion | 2026-08-06 |
| 8 | hybrid retrieval (BM25+dense, RRF k=60, punctuation-aware tokenizer) full 62-question vs. experiment_008 dense baseline | hybrid | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.903 | 0.944 | 0.952 | 0.865 | 0.391 | 0.841 | 0.504 | 2.1s | techfusion | 2026-08-07 |
