"""Gold-data schema and path-suffix matching for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field


class GoldExample(BaseModel):
    """One row of a gold JSONL file (e.g. data/eval/techfusion_gold.jsonl).

    `relevant_documents` are paths *relative to wherever the knowledge base
    root is* (e.g. "knowledge_base/security/access-control-policy.md"), not
    document_id UUIDs — ids are assigned at ingestion time and can't be
    known ahead of authoring gold data. Matching against a retrieved
    chunk's stored `source` is therefore done by path-suffix (see
    `source_matches_relevant`), which works regardless of what root path a
    particular ingestion run used — nothing here hardcodes "data/" or
    "knowledge_base/".
    """

    question: str
    expected_answer: str | None = None
    relevant_documents: list[str] = Field(default_factory=list)
    question_type: str | None = None
    difficulty: str | None = None
    unanswerable: bool = False


def load_gold_jsonl(path: str | Path) -> list[GoldExample]:
    """Parse a gold JSONL file into `GoldExample` rows, skipping blank lines.

    Parameters
    ----------
    path : str | Path
        Path to the gold JSONL file.

    Returns
    -------
    list[GoldExample]
        One `GoldExample` per non-blank line.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldExample.model_validate(json.loads(line)))
    return examples


def _path_parts(path: str) -> tuple[str, ...]:
    """Split a possibly-backslash path into POSIX-style path segments."""
    return PurePosixPath(path.replace("\\", "/")).parts


def source_matches_relevant(retrieved_source: str, relevant_path: str) -> bool:
    """Check whether `relevant_path`'s segments trail `retrieved_source`'s segments.

    E.g. gold's "knowledge_base/security/x.md" matches a stored source of
    "data/knowledge_base/security/x.md" regardless of the "data/" root,
    without either side needing to hardcode that root.

    Parameters
    ----------
    retrieved_source : str
        A chunk's stored `source` path.
    relevant_path : str
        A gold example's relative `relevant_documents` entry.

    Returns
    -------
    bool
        True if `relevant_path` is a trailing subsequence of `retrieved_source`.
    """
    retrieved_parts = _path_parts(retrieved_source)
    relevant_parts = _path_parts(relevant_path)
    if not relevant_parts or len(relevant_parts) > len(retrieved_parts):
        return False
    return retrieved_parts[-len(relevant_parts) :] == relevant_parts
