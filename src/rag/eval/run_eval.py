"""Run retrieval evaluation (recall@k, MRR) against a gold JSONL dataset.

Usage:
    python -m rag.eval.run_eval --gold data/eval/gold.jsonl [--k 5] [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import load_config
from rag.eval.gold_schema import load_gold_jsonl
from rag.eval.metrics import mean_recall_at_k, mean_reciprocal_rank
from rag.retrieval.pipeline import RetrievalPipeline


def run(gold_path: Path, config_path: str | None, k: int) -> dict:
    config = load_config(config_path) if config_path else load_config()
    pipeline = RetrievalPipeline(config)

    examples = load_gold_jsonl(gold_path)
    doc_retrieved, doc_relevant = [], []
    chunk_retrieved, chunk_relevant = [], []

    for example in examples:
        if example.unanswerable:
            continue
        results = pipeline.retrieve(example.query)
        retrieved_doc_ids = [r.chunk.metadata.document_id for r in results]
        retrieved_chunk_ids = [r.chunk.metadata.chunk_id for r in results]

        if example.relevant_document_ids:
            doc_retrieved.append(retrieved_doc_ids)
            doc_relevant.append(set(example.relevant_document_ids))
        if example.relevant_chunk_ids:
            chunk_retrieved.append(retrieved_chunk_ids)
            chunk_relevant.append(set(example.relevant_chunk_ids))

    report: dict = {"num_examples": len(examples), "k": k}
    if doc_retrieved:
        report["document_level"] = {
            f"recall@{k}": mean_recall_at_k(doc_retrieved, doc_relevant, k),
            "mrr": mean_reciprocal_rank(doc_retrieved, doc_relevant),
        }
    if chunk_retrieved:
        report["chunk_level"] = {
            f"recall@{k}": mean_recall_at_k(chunk_retrieved, chunk_relevant, k),
            "mrr": mean_reciprocal_rank(chunk_retrieved, chunk_relevant),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold.jsonl file")
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    report = run(Path(args.gold), args.config, args.k)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
