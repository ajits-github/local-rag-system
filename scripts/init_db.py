"""Initialize the Postgres schema: pgvector extension, documents + chunks tables.

Python instead of a static .sql file so schema setup can grow conditional
logic later (e.g. vector dimension driven by config, future migrations)
without switching tooling. Safe to re-run. Every statement is idempotent.

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

    -- parent_chunk_id links a table/code/config/chart/image chunk to the
    -- nearest preceding prose chunk in its section. No FK: parent and
    -- child chunks land in the same insert batch, and deletion is always
    -- whole-document via ON DELETE CASCADE, so a self-referential FK's
    -- insert-ordering fragility isn't worth taking on. vision_generated/
    -- vision_description hold a VisionProvider's output, kept on a
    -- separate sibling chunk from the image's original caption/alt-text.
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS vision_description TEXT;
    ALTER TABLE {chunks_table}
        ADD COLUMN IF NOT EXISTS vision_generated BOOLEAN NOT NULL DEFAULT FALSE;

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

    CREATE INDEX IF NOT EXISTS {chunks_table}_parent_chunk_id_idx
        ON {chunks_table} (parent_chunk_id);

    -- Document governance fields (tenant/role/classification/status/
    -- version/trust), copied onto every chunk of a document like
    -- category/dataset_id. doc_source_type is deliberately not named
    -- source_type a second time: that column already means "markdown"/
    -- "text" (the loader's file-type tag), a different concept from front
    -- matter's source_type (a trust-provenance label). No FK on tenant/
    -- allowed_roles; these are plain metadata columns. supersedes_source
    -- is a raw filename string, matched by path suffix at query time, not
    -- resolved to a document_id at ingestion time.
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS tenant_id TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS allowed_roles TEXT[];
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS classification TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS status TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS document_version TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS effective_from DATE;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS trust_level TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS doc_source_type TEXT;
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS supersedes_source TEXT;

    CREATE INDEX IF NOT EXISTS {chunks_table}_tenant_id_idx
        ON {chunks_table} (tenant_id);

    CREATE INDEX IF NOT EXISTS {chunks_table}_status_idx
        ON {chunks_table} (status);

    CREATE INDEX IF NOT EXISTS {chunks_table}_trust_level_idx
        ON {chunks_table} (trust_level);

    -- Ingestion-time tagging only: which SensitiveFieldPolicy.field_id
    -- patterns this chunk's text matches. No role decision is stored
    -- here; RetrievalPipeline uses this as a cheap "nothing to redact"
    -- skip and re-runs the actual role-aware redaction at query time.
    -- Not indexed: never used in a WHERE clause, only read back per-row.
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS sensitive_field_ids TEXT[];

    -- 1-indexed source page number (PDF: extracted page; DOCX: running
    -- count of manual page breaks). NULL for source types with no fixed
    -- pagination (Markdown/text/HTML).
    ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS page INTEGER;

    -- Caches a VisionProvider's generated description per image, keyed by
    -- the image file's own sha256 checksum *plus* provider/model/prompt
    -- version, so an unchanged image is never reprocessed by a
    -- (cost-incurring) call across documents or ingestion runs -- but
    -- switching provider, model, or prompt wording always misses rather
    -- than silently replaying a description generated under different
    -- instructions. Postgres-backed rather than a separate cache service;
    -- no concrete use case for Redis here.
    CREATE TABLE IF NOT EXISTS image_description_cache (
        image_checksum TEXT NOT NULL,
        source_path TEXT NOT NULL,
        provider TEXT NOT NULL,
        model_name TEXT NOT NULL,
        prompt_version TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- image_description_cache.image_checksum used to be PRIMARY KEY on its
    -- own; the cache key now also covers provider/model/prompt_version, so
    -- switching any of those no longer silently reuses a stale description.
    ALTER TABLE image_description_cache
        ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT '';
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'image_description_cache_pkey'
        ) THEN
            EXECUTE 'ALTER TABLE image_description_cache '
                'DROP CONSTRAINT image_description_cache_pkey';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'image_description_cache_key_pkey'
        ) THEN
            EXECUTE 'ALTER TABLE image_description_cache ADD CONSTRAINT '
                'image_description_cache_key_pkey '
                'PRIMARY KEY (image_checksum, provider, model_name, prompt_version)';
        END IF;
    END $$;
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
