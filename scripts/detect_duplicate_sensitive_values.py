"""Diagnostic: find sensitive literals duplicated across chunks, or missing their ingestion tag.

Auth-boundary milestone (requirement 7: duplicate-secret/alternate-copy
protection). Reads every chunk's `content`/`sensitive_field_ids` straight
from Postgres and runs `field_policy.find_duplicate_sensitive_occurrences`
against them -- a diagnostic/validation tool over the real corpus, not
part of the query-time enforcement path (see that function's docstring).
Never prints a raw secret value; only sha256 hashes and chunk/document ids.

Usage:
    python scripts/detect_duplicate_sensitive_values.py [--config config/default.yaml]

Exit code is nonzero if any finding is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.retrieval.field_policy import find_duplicate_sensitive_occurrences  # noqa: E402


@dataclass
class _ChunkMetadataStub:
    """Minimal duck-typed stand-in for `ChunkMetadata` -- only what the detector reads."""

    sensitive_field_ids: list[str] | None


@dataclass
class _ChunkStub:
    """Minimal duck-typed stand-in for `Chunk` -- only what the detector reads."""

    id: str
    content: str
    metadata: _ChunkMetadataStub


def _load_chunks(config) -> list[_ChunkStub]:  # noqa: ANN001
    """Read every chunk's id/content/sensitive_field_ids directly from Postgres.

    A direct SQL read rather than going through `VectorStore.search` --
    there is no existing "fetch every chunk" primitive on that interface,
    and adding one solely for this one-off diagnostic script would be
    exactly the kind of speculative abstraction this codebase avoids.
    """
    table = config.vectorstore.chunks_table
    conn = psycopg2.connect(config.database_url())
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT chunk_id, content, sensitive_field_ids FROM {table}")  # noqa: S608
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        _ChunkStub(id=chunk_id, content=content, metadata=_ChunkMetadataStub(sensitive_field_ids))
        for chunk_id, content, sensitive_field_ids in rows
    ]


def main() -> None:
    """CLI entrypoint: load chunks, run the detector, print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    chunks = _load_chunks(config)
    findings = find_duplicate_sensitive_occurrences(chunks)

    report = {
        "chunks_scanned": len(chunks),
        "findings_count": len(findings),
        "findings": [f.model_dump() for f in findings],
    }
    print(json.dumps(report, indent=2))
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
