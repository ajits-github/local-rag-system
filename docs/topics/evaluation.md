# Evaluation & RAGAS

`eval/run_eval.py` has no RAGAS/MLflow dependency: it always computes
Recall@5/@10, hit-rate, MRR, a keyword-overlap answer-quality score, and
the deterministic multimodal/relationship/safety metrics against a
mandatory `--dataset-id`-scoped gold set. RAGAS (LLM-judge scoring of
faithfulness, answer relevancy, context precision/recall, answer
correctness, and more) and MLflow experiment tracking are both
independently optional layers on top, gated behind their own extras
(`pip install .[ragas]` / `.[mlflow]`) and never required for the offline
serving path.

See [Metrics reference](../metrics.md) for the full metric-by-metric
breakdown (what each one measures, its known limitations, and which
report section it lives in).

--8<-- "README.md:docs-benchmarks"

API reference: [Evaluation](../reference/eval.md).
