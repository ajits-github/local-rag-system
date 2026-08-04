from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldExample(BaseModel):
    """One row of data/eval/gold.jsonl.

    Document-level labels (`relevant_document_ids`) are primary since
    chunk_ids shift whenever the chunking strategy changes; chunk-level
    labels are used when present for finer-grained scoring.
    """

    query_id: str
    query: str
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    question_type: str | None = None
    difficulty: str | None = None
    unanswerable: bool = False


def load_gold_jsonl(path: str | Path) -> list[GoldExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldExample.model_validate(json.loads(line)))
    return examples
