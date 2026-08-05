from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

from rag.schemas import Chunk, ChunkMetadata, SearchResult
from rag.vectorstore.base import ALLOWED_FILTER_FIELDS, VectorStore

_DISTANCE_OPERATORS = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}


class PgVectorStore(VectorStore):
    """v1 VectorStore backend: Postgres + the pgvector extension."""

    def __init__(
        self,
        dsn: str,
        documents_table: str = "documents",
        chunks_table: str = "chunks",
        distance_metric: str = "cosine",
        minconn: int = 1,
        maxconn: int = 5,
    ) -> None:
        self._documents_table = documents_table
        self._chunks_table = chunks_table
        self._distance_op = _DISTANCE_OPERATORS[distance_metric]
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)

    def _conn(self):
        conn = self._pool.getconn()
        register_vector(conn)
        return conn

    def _putconn(self, conn) -> None:
        self._pool.putconn(conn)

    def health_check(self) -> bool:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception:
            return False
        finally:
            self._putconn(conn)

    def get_or_create_document_id(self, source: str, checksum: str) -> tuple[str, bool]:
        conn = self._conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT document_id, checksum FROM {self._documents_table} WHERE source = %s",
                        (source,),
                    )
                    row = cur.fetchone()
                    now = datetime.now(timezone.utc)

                    if row is None:
                        document_id = str(uuid.uuid4())
                        cur.execute(
                            f"""INSERT INTO {self._documents_table}
                                (document_id, source, checksum, created_at, last_modified)
                                VALUES (%s, %s, %s, %s, %s)""",
                            (document_id, source, checksum, now, now),
                        )
                        return document_id, True

                    document_id, existing_checksum = row
                    changed = existing_checksum != checksum
                    if changed:
                        cur.execute(
                            f"""UPDATE {self._documents_table}
                                SET checksum = %s, last_modified = %s
                                WHERE document_id = %s""",
                            (checksum, now, document_id),
                        )
                    return str(document_id), changed
        finally:
            self._putconn(conn)

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        conn = self._conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self._chunks_table} WHERE document_id = %s",
                        (document_id,),
                    )
        finally:
            self._putconn(conn)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        conn = self._conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    rows = [
                        (
                            c.metadata.chunk_id,
                            c.metadata.document_id,
                            c.metadata.chunk_index,
                            c.content,
                            c.embedding,
                            c.metadata.source,
                            c.metadata.source_type,
                            c.metadata.title,
                            c.metadata.author,
                            c.metadata.url,
                            c.metadata.created_at,
                            c.metadata.last_modified,
                            c.metadata.language,
                            c.metadata.category,
                        )
                        for c in chunks
                    ]
                    psycopg2.extras.execute_values(
                        cur,
                        f"""INSERT INTO {self._chunks_table}
                            (chunk_id, document_id, chunk_index, content, embedding,
                             source, source_type, title, author, url,
                             created_at, last_modified, language, category)
                            VALUES %s
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                last_modified = EXCLUDED.last_modified,
                                category = EXCLUDED.category""",
                        rows,
                    )
        finally:
            self._putconn(conn)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        where_clauses = []
        params: list[Any] = [query_embedding]

        if filters:
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_FIELDS:
                    raise ValueError(f"Filtering on '{key}' is not allowed")
                where_clauses.append(f"{key} = %s")
                params.append(value)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([query_embedding, top_k])

        sql = f"""
            SELECT chunk_id, document_id, chunk_index, content,
                   source, source_type, title, author, url,
                   created_at, last_modified, language, category,
                   embedding {self._distance_op} %s::vector AS distance
            FROM {self._chunks_table}
            {where_sql}
            ORDER BY embedding {self._distance_op} %s::vector
            LIMIT %s
        """

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._putconn(conn)

        results = []
        for row in rows:
            (
                chunk_id, document_id, chunk_index, content,
                source, source_type, title, author, url,
                created_at, last_modified, language, category, distance,
            ) = row
            metadata = ChunkMetadata(
                document_id=str(document_id),
                chunk_id=chunk_id,
                source=source,
                source_type=source_type,
                title=title,
                author=author,
                url=url,
                created_at=created_at,
                last_modified=last_modified,
                language=language,
                chunk_index=chunk_index,
                category=category,
            )
            chunk = Chunk(id=chunk_id, content=content, metadata=metadata)
            score = 1.0 - distance if self._distance_op == "<=>" else -distance
            results.append(SearchResult(chunk=chunk, score=score))
        return results
