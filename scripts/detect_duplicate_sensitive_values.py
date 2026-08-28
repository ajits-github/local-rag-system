"""Diagnostic: find sensitive literals duplicated across chunks, or missing their ingestion tag.

Reads every chunk's `content`, scannable metadata, and
`sensitive_field_ids` straight from Postgres, then runs
`field_policy.find_duplicate_sensitive_occurrences` against them. A
diagnostic/validation tool over the real corpus, not part of the
query-time enforcement path. Never prints a raw secret value; only sha256
hashes and chunk/document ids.

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
    """Minimal duck-typed stand-in for the metadata fields the detector reads."""

    sensitive_field_ids: list[str] | None
    source: str | None
    section_path: str | None
    attachment_name: str | None
    source_anchor: str | None


@dataclass
class _ChunkStub:
    """Minimal duck-typed stand-in for the chunk fields the detector reads."""

    id: str
    content: str
    metadata: _ChunkMetadataStub


def _load_chunks(config) -> list[_ChunkStub]:  # noqa: ANN001
    """Read chunk text and sensitive-scannable metadata directly from Postgres.

    Uses direct SQL because `VectorStore.search` is query-shaped and
    top_k-limited, and the interface has no "fetch every chunk" primitive.
    Fetches the same fields `find_duplicate_sensitive_occurrences` scans
    (see `field_policy.SCANNABLE_METADATA_FIELDS`), not just `content`, so
    a sensitive literal living only in metadata is caught by this
    diagnostic too.
    """
    table = config.vectorstore.chunks_table
    conn = psycopg2.connect(config.database_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id, content, sensitive_field_ids, "  # noqa: S608
                f"source, section_path, attachment_name, source_anchor FROM {table}"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        _ChunkStub(
            id=row[0],
            content=row[1],
            metadata=_ChunkMetadataStub(
                sensitive_field_ids=row[2],
                source=row[3],
                section_path=row[4],
                attachment_name=row[5],
                source_anchor=row[6],
            ),
        )
        for row in rows
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
