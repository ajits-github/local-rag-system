"""pgvector `VectorStore` backend: Postgres + the pgvector extension."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import ThreadedConnectionPool
from rank_bm25 import BM25Okapi

from rag.schemas import Chunk, ChunkMetadata, SearchResult
from rag.vectorstore.base import ALLOWED_FILTER_FIELDS, VectorStore

_DISTANCE_OPERATORS = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}
_TOKEN_RE = re.compile(r"\w+")

# Columns shared by search()'s and search_keyword()'s SELECTs (everything
# except the dense-only `embedding`/`distance` and keyword-only ranking).
_METADATA_COLUMNS = """chunk_id, document_id, chunk_index, content,
    source, source_type, title, author, url,
    created_at, last_modified, language, category, dataset_id,
    content_type, section_path, code_language, table_headers,
    attachment_name, source_anchor"""


def _build_where_clause(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Build a `WHERE ...` SQL fragment and its bound params from a filters dict.

    Module-level (not a method) so it's directly unit-testable without
    constructing a `PgVectorStore`/opening a DB connection. Shared by
    `PgVectorStore.search` and `.search_keyword` -- the only piece of
    their query construction that's actually identical; their
    ranking/limiting SQL diverges (`search` ranks+limits in SQL via
    `ORDER BY ... LIMIT`, `search_keyword` fetches the full filtered set
    unranked and ranks in Python via BM25).

    Parameters
    ----------
    filters : dict[str, Any] | None
        Exact-match metadata filters; keys must be in `ALLOWED_FILTER_FIELDS`.

    Returns
    -------
    tuple[str, list[Any]]
        ``(where_sql, params)`` -- `where_sql` is ``""`` or
        ``"WHERE key = %s AND ..."``; `params` holds the bound values in
        the same order as the `%s` placeholders.

    Raises
    ------
    ValueError
        If `filters` contains a key not in `ALLOWED_FILTER_FIELDS`.
    """
    where_clauses = []
    params: list[Any] = []
    if filters:
        for key, value in filters.items():
            if key not in ALLOWED_FILTER_FIELDS:
                raise ValueError(f"Filtering on '{key}' is not allowed")
            where_clauses.append(f"{key} = %s")
            params.append(value)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


def _row_to_metadata(row: tuple[Any, ...]) -> tuple[str, str, ChunkMetadata]:
    """Unpack one `_METADATA_COLUMNS`-shaped row into (chunk_id, content, ChunkMetadata)."""
    (
        chunk_id,
        document_id,
        chunk_index,
        content,
        source,
        source_type,
        title,
        author,
        url,
        created_at,
        last_modified,
        language,
        category,
        dataset_id,
        content_type,
        section_path,
        code_language,
        table_headers,
        attachment_name,
        source_anchor,
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
        dataset_id=dataset_id,
        content_type=content_type,
        section_path=section_path,
        code_language=code_language,
        table_headers=list(table_headers) if table_headers is not None else None,
        attachment_name=attachment_name,
        source_anchor=source_anchor,
    )
    return chunk_id, content, metadata


def _tokenize(text: str) -> list[str]:
    r"""Split `text` into lowercase word tokens for BM25.

    Uses `\\w+` (word characters: letters, digits, underscore) rather
    than a plain whitespace split, so punctuation attached to a token
    (e.g. JSON's `"maximum_wait_minutes":`, or a trailing period in
    prose) doesn't prevent it from matching a plain-word query term.
    Still no stemming/lemmatization -- only exact (post-punctuation-
    stripping) token matches.

    Parameters
    ----------
    text : str
        Raw text to tokenize.

    Returns
    -------
    list[str]
        Lowercase word tokens, in order.
    """
    return _TOKEN_RE.findall(text.lower())


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
        """Open a threaded connection pool against `dsn`.

        Parameters
        ----------
        dsn : str
            Postgres connection string.
        documents_table : str, optional
            Name of the documents table, by default ``"documents"``.
        chunks_table : str, optional
            Name of the chunks table, by default ``"chunks"``.
        distance_metric : str, optional
            One of ``"cosine"``, ``"l2"``, ``"inner_product"``, by default
            ``"cosine"``.
        minconn : int, optional
            Minimum pooled connections, by default 1.
        maxconn : int, optional
            Maximum pooled connections, by default 5.
        """
        self._documents_table = documents_table
        self._chunks_table = chunks_table
        self._distance_op = _DISTANCE_OPERATORS[distance_metric]
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)

    @contextmanager
    def _connection(self) -> Iterator[PgConnection]:
        """Check out a pooled connection, guaranteeing it's always returned.

        `register_vector(conn)` runs *inside* the try/finally, not before
        it -- a connection checked out via `pool.getconn()` is only ever
        "ours" for as long as we hold a reference to it, so anything that
        can raise between checkout and use (registering the vector type
        adapter included) must already be inside the block that returns
        it, or the connection leaks out of the pool forever. This was the
        root cause of a real `PoolError: connection pool exhausted`: the
        previous `_conn()` helper called `pool.getconn()` then
        `register_vector(conn)` as two separate statements *before*
        returning to the caller's own `try/finally`, so a `register_vector`
        failure (e.g. the `vector` extension not yet installed on a fresh
        database) checked out a connection that no `finally` block ever
        saw, let alone returned.

        Yields
        ------
        PgConnection
            A pooled connection with the pgvector type adapter registered.
        """
        conn = self._pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def health_check(self) -> bool:
        """See `VectorStore.health_check`."""
        try:
            with self._connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception:
            return False

    def get_or_create_document_id(
        self, source: str, checksum: str, dataset_id: str
    ) -> tuple[str, bool]:
        """See `VectorStore.get_or_create_document_id`."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"""SELECT document_id, checksum FROM {self._documents_table}
                        WHERE source = %s AND dataset_id = %s""",
                    (source, dataset_id),
                )
                row = cur.fetchone()
                now = datetime.now(UTC)

                if row is None:
                    document_id = str(uuid.uuid4())
                    cur.execute(
                        f"""INSERT INTO {self._documents_table}
                            (document_id, source, dataset_id, checksum,
                             created_at, last_modified)
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                        (document_id, source, dataset_id, checksum, now, now),
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

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """See `VectorStore.delete_chunks_by_document_id`."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._chunks_table} WHERE document_id = %s",
                    (document_id,),
                )

    def delete_document(self, document_id: str) -> None:
        """See `VectorStore.delete_document`."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._documents_table} WHERE document_id = %s",
                    (document_id,),
                )

    def delete_dataset(self, dataset_id: str) -> None:
        """See `VectorStore.delete_dataset`."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._documents_table} WHERE dataset_id = %s",
                    (dataset_id,),
                )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """See `VectorStore.add_chunks`."""
        if not chunks:
            return
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
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
                        c.metadata.dataset_id,
                        c.metadata.content_type,
                        c.metadata.section_path,
                        c.metadata.code_language,
                        c.metadata.table_headers,
                        c.metadata.attachment_name,
                        c.metadata.source_anchor,
                    )
                    for c in chunks
                ]
                psycopg2.extras.execute_values(
                    cur,
                    f"""INSERT INTO {self._chunks_table}
                        (chunk_id, document_id, chunk_index, content, embedding,
                         source, source_type, title, author, url,
                         created_at, last_modified, language, category, dataset_id,
                         content_type, section_path, code_language, table_headers,
                         attachment_name, source_anchor)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            last_modified = EXCLUDED.last_modified,
                            category = EXCLUDED.category,
                            dataset_id = EXCLUDED.dataset_id,
                            content_type = EXCLUDED.content_type,
                            section_path = EXCLUDED.section_path,
                            code_language = EXCLUDED.code_language,
                            table_headers = EXCLUDED.table_headers,
                            attachment_name = EXCLUDED.attachment_name,
                            source_anchor = EXCLUDED.source_anchor""",
                    rows,
                )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """See `VectorStore.search`."""
        where_sql, filter_params = _build_where_clause(filters)
        params: list[Any] = [query_embedding, *filter_params, query_embedding, top_k]

        sql = f"""
            SELECT {_METADATA_COLUMNS},
                   embedding {self._distance_op} %s::vector AS distance
            FROM {self._chunks_table}
            {where_sql}
            ORDER BY embedding {self._distance_op} %s::vector
            LIMIT %s
        """

        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            *metadata_row, distance = row
            chunk_id, content, metadata = _row_to_metadata(tuple(metadata_row))
            chunk = Chunk(id=chunk_id, content=content, metadata=metadata)
            score = 1.0 - distance if self._distance_op == "<=>" else -distance
            results.append(SearchResult(chunk=chunk, score=score))
        return results

    def search_keyword(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        r"""See `VectorStore.search_keyword`.

        Builds a fresh in-memory `rank_bm25.BM25Okapi` index per call from
        the (optionally filtered) chunk content fetched via SQL -- no
        persistent BM25 index or caching. At this project's current scale
        (a few hundred chunks per dataset), rebuilding per query is
        sub-second and avoids the invalidation complexity of caching an
        index that would need to track re-ingestion; revisit (e.g. a
        persistent index, or Postgres `tsvector`/`ts_rank` full-text
        search) if corpus size ever grows enough to make this measurably
        slow. Tokenization (`_tokenize`) is `\\w+`-based -- lowercase word
        tokens with surrounding punctuation stripped, so e.g. JSON's
        `"maximum_wait_minutes":` still matches a plain query token
        `maximum_wait_minutes`. Still no stemming/lemmatization, so this
        only finds exact (post-punctuation-stripping) token matches.
        """
        where_sql, filter_params = _build_where_clause(filters)
        sql = f"SELECT {_METADATA_COLUMNS} FROM {self._chunks_table} {where_sql}"

        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, filter_params)
            rows = cur.fetchall()

        if not rows:
            return []

        unpacked = [_row_to_metadata(row) for row in rows]
        tokenized_corpus = [_tokenize(content) for _, content, _ in unpacked]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked_indices = sorted(range(len(unpacked)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_indices[:top_k]:
            chunk_id, content, metadata = unpacked[i]
            chunk = Chunk(id=chunk_id, content=content, metadata=metadata)
            results.append(SearchResult(chunk=chunk, score=float(scores[i])))
        return results
