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
    """Build the idempotent DDL that creates/migrates the documents and chunks tables.

    Parameters
    ----------
    documents_table : str
        Name of the documents table.
    chunks_table : str
        Name of the chunks table.
    dimension : int
        Embedding vector dimension, used for the chunks table's `VECTOR(n)` column.

    Returns
    -------
    str
        A multi-statement SQL script, safe to re-run.
    """
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS {documents_table} (
        document_id UUID PRIMARY KEY,
        source TEXT NOT NULL,
        dataset_id TEXT,
        checksum TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_modified TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (source, dataset_id)
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
        language TEXT,
        category TEXT,
        dataset_id TEXT,
        content_type TEXT,
        section_path TEXT,
        code_language TEXT,
        table_headers TEXT[],
        attachment_name TEXT,
        source_anchor TEXT
    );

    -- Migrations for tables created before `category`/`dataset_id` existed.
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS category TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS dataset_id TEXT;
    ALTER TABLE {documents_table} ADD COLUMN IF NOT EXISTS dataset_id TEXT;

    -- Migrations for tables created before the structured-content metadata
    -- fields (content_type/section_path/code_language/table_headers/
    -- attachment_name/source_anchor) existed.
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS content_type TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS section_path TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS code_language TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS table_headers TEXT[];
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS attachment_name TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS source_anchor TEXT;

    -- documents.source used to be UNIQUE on its own; identity is now scoped
    -- per (source, dataset_id) so the same relative path can exist in two
    -- different datasets without colliding.
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = '{documents_table}_source_key'
        ) THEN
            EXECUTE 'ALTER TABLE {documents_table} DROP CONSTRAINT {documents_table}_source_key';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = '{documents_table}_source_dataset_id_key'
        ) THEN
            EXECUTE 'ALTER TABLE {documents_table} ADD CONSTRAINT '
                '{documents_table}_source_dataset_id_key UNIQUE (source, dataset_id)';
        END IF;
    END $$;

    CREATE INDEX IF NOT EXISTS {chunks_table}_document_id_idx
        ON {chunks_table} (document_id);

    CREATE INDEX IF NOT EXISTS {chunks_table}_category_idx
        ON {chunks_table} (category);

    CREATE INDEX IF NOT EXISTS {chunks_table}_content_type_idx
        ON {chunks_table} (content_type);

    CREATE INDEX IF NOT EXISTS {chunks_table}_dataset_id_idx
        ON {chunks_table} (dataset_id);

    CREATE INDEX IF NOT EXISTS {documents_table}_dataset_id_idx
        ON {documents_table} (dataset_id);

    CREATE INDEX IF NOT EXISTS {chunks_table}_embedding_hnsw_idx
        ON {chunks_table} USING hnsw (embedding vector_cosine_ops);
    """


def main() -> None:
    """CLI entrypoint: build the schema SQL from config and run it."""
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
        used_config = (
            DEFAULT_CONFIG_PATH.name if args.config == str(DEFAULT_CONFIG_PATH) else args.config
        )
        print(
            f"Initialized '{config.vectorstore.documents_table}' and "
            f"'{config.vectorstore.chunks_table}' (dim={config.embedding.dimension}) "
            f"using {used_config}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
