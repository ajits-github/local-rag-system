"""Initialize the Postgres schema: pgvector extension, documents + chunks tables.

Python instead of a static .sql file so schema setup can grow conditional
logic later (e.g. vector dimension driven by config, future migrations)
without switching tooling. Safe to re-run — every statement is idempotent.

Usage:
    python scripts/init_db.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def build_schema_sql(*, documents_table: str, chunks_table: str, dimension: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS {documents_table} (
        document_id UUID PRIMARY KEY,
        source TEXT NOT NULL UNIQUE,
        checksum TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_modified TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS {chunks_table} (
        chunk_id TEXT PRIMARY KEY,
        document_id UUID NOT NULL REFERENCES {documents_table}(document_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR({dimension}) NOT NULL,
        source TEXT NOT NULL,
        source_type TEXT NOT NULL,
        title TEXT,
        author TEXT,
        url TEXT,
        created_at TIMESTAMPTZ NOT NULL,
        last_modified TIMESTAMPTZ NOT NULL,
        language TEXT
    );

    CREATE INDEX IF NOT EXISTS {chunks_table}_document_id_idx
        ON {chunks_table} (document_id);

    CREATE INDEX IF NOT EXISTS {chunks_table}_embedding_hnsw_idx
        ON {chunks_table} USING hnsw (embedding vector_cosine_ops);
    """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    sql = build_schema_sql(
        documents_table=config.vectorstore.documents_table,
        chunks_table=config.vectorstore.chunks_table,
        dimension=config.embedding.dimension,
    )

    conn = psycopg2.connect(config.database_url())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print(
            f"Initialized '{config.vectorstore.documents_table}' and "
            f"'{config.vectorstore.chunks_table}' (dim={config.embedding.dimension}) "
            f"using {DEFAULT_CONFIG_PATH.name if args.config == str(DEFAULT_CONFIG_PATH) else args.config}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
