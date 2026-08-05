| # | Label | Generation model | Reranker | Recall@5 | Recall@10 | Hit Rate@10 | MRR | Answer quality | Total latency | Dataset | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TechFusion baseline (qwen2.5:1.5b, no reranker) | qwen2.5:1.5b | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.432 | 3.7s | techfusion | 2026-08-05 |
| 2 | qwen2.5:3b (candidate generation model) | qwen2.5:3b | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.453 | 12.1s | techfusion | 2026-08-05 |
