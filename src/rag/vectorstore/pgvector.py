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

from rag.retrieval.authorization import AuthorizationContext
from rag.schemas import Chunk, ChunkMetadata, DocumentVersionInfo, SearchResult
from rag.vectorstore.base import ALLOWED_FILTER_FIELDS, VectorStore

_DISTANCE_OPERATORS = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}
_TOKEN_RE = re.compile(r"\w+")

# Columns shared by search()'s and search_keyword()'s SELECTs (everything
# except the dense-only `embedding`/`distance` and keyword-only ranking).
_METADATA_COLUMNS = """chunk_id, document_id, chunk_index, content,
    source, source_type, title, author, url,
    created_at, last_modified, language, category, dataset_id,
    content_type, section_path, code_language, table_headers,
    attachment_name, source_anchor, parent_chunk_id,
    vision_generated, vision_description,
    tenant_id, allowed_roles, classification, status, document_version,
    effective_from, trust_level, doc_source_type, supersedes_source"""


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


def build_authorization_where_clause(
    auth: AuthorizationContext | None, cross_tenant_support_roles: list[str]
) -> tuple[str, list[Any]]:
    """Build the authorization/freshness `AND (...)` SQL fragment for one query.

    Module-level (unit-testable without a DB connection), structurally
    separate from `_build_where_clause`: this predicate is never built from
    caller-suppliable input (see `retrieval/authorization.py`'s docstring),
    only from an `AuthorizationContext` a pipeline constructs itself, and it
    is always ANDed on top of whatever `_build_where_clause` produces so it
    can only narrow results, never broaden them.

    Enforcement rule (matches the TechFusion authorization matrix):
    a chunk is visible when its `tenant_id IS NULL` (untenanted legacy
    content -- never gated), OR the caller's tenant matches, OR the caller
    holds a configured cross-tenant support role that is *also* listed in
    this specific chunk's own `allowed_roles` (a document-specific grant,
    not a blanket one) -- AND, independently, when `allowed_roles` is set
    at all, the caller must hold at least one of them (this is what makes a
    same-tenant request still subject to role checks, e.g. a plain
    `employee` role denied a `restricted` document within their own
    tenant). `resolved_excluded_document_ids` (populated by
    `RetrievalPipeline.retrieve()` from `retrieval/freshness.py`) is ANDed
    in as a further, independent exclusion, and so is
    `require_trust_level` when set (section 7: a query that requires an
    authoritative source excludes chunks whose `trust_level` doesn't
    match, while leaving untenanted/untagged content -- `trust_level IS
    NULL` -- unaffected).

    Parameters
    ----------
    auth : AuthorizationContext | None
        `None` produces an empty (unrestricted) fragment.
    cross_tenant_support_roles : list[str]
        `config.security.authorization.cross_tenant_support_roles` --
        server policy, never caller-supplied.

    Returns
    -------
    tuple[str, list[Any]]
        ``(sql_fragment, params)`` -- `sql_fragment` is `""` (unrestricted)
        or a bare, unprefixed boolean expression (no leading `WHERE`/`AND`);
        see `_combine_where_clauses` for how it's joined with
        `_build_where_clause`'s own fragment. Params are in placeholder order.
    """
    if auth is None:
        return "", []

    caller_support_roles = [role for role in auth.roles if role in cross_tenant_support_roles]
    clauses = [
        """(
            tenant_id IS NULL
            OR tenant_id = %s
            OR (allowed_roles IS NOT NULL AND allowed_roles && %s::text[])
        )
        AND (
            allowed_roles IS NULL
            OR allowed_roles && %s::text[]
        )"""
    ]
    params: list[Any] = [auth.tenant_id, caller_support_roles, auth.roles]

    if auth.resolved_excluded_document_ids:
        clauses.append("NOT (document_id::text = ANY(%s))")
        params.append(auth.resolved_excluded_document_ids)

    if auth.require_trust_level:
        clauses.append("(trust_level IS NULL OR trust_level = %s)")
        params.append(auth.require_trust_level)

    return " AND ".join(clauses), params


def _combine_where_clauses(where_sql: str, auth_sql: str) -> str:
    """Join `_build_where_clause`'s and `build_authorization_where_clause`'s fragments.

    Parameters
    ----------
    where_sql : str
        `""` or a `"WHERE ..."`-prefixed fragment (caller-suppliable filters).
    auth_sql : str
        `""` or a bare boolean expression (authorization/freshness).

    Returns
    -------
    str
        `""`, `"WHERE ..."`, or `"WHERE (...) AND (...)"`.
    """
    if where_sql and auth_sql:
        return f"{where_sql} AND ({auth_sql})"
    if auth_sql:
        return f"WHERE ({auth_sql})"
    return where_sql


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
        parent_chunk_id,
        vision_generated,
        vision_description,
        tenant_id,
        allowed_roles,
        classification,
        status,
        document_version,
        effective_from,
        trust_level,
        doc_source_type,
        supersedes_source,
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
        parent_chunk_id=parent_chunk_id,
        vision_generated=bool(vision_generated),
        vision_description=vision_description,
        tenant_id=tenant_id,
        allowed_roles=list(allowed_roles) if allowed_roles is not None else None,
        classification=classification,
        status=status,
        document_version=document_version,
        effective_from=effective_from,
        trust_level=trust_level,
        doc_source_type=doc_source_type,
        supersedes_source=supersedes_source,
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
        cross_tenant_support_roles: list[str] | None = None,
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
        cross_tenant_support_roles : list[str] | None, optional
            `config.security.authorization.cross_tenant_support_roles` --
            server policy for `build_authorization_where_clause`, by default
            `["techfusion_support"]` (matching `AuthorizationConfig`'s
            default) when omitted.
        """
        self._documents_table = documents_table
        self._chunks_table = chunks_table
        self._distance_op = _DISTANCE_OPERATORS[distance_metric]
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn)
        self._cross_tenant_support_roles = (
            cross_tenant_support_roles
            if cross_tenant_support_roles is not None
            else ["techfusion_support"]
        )

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

    def list_document_sources(self, dataset_id: str) -> list[str]:
        """See `VectorStore.list_document_sources`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT source FROM {self._documents_table} WHERE dataset_id = %s",
                (dataset_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def delete_documents_by_source(self, dataset_id: str, sources: list[str]) -> int:
        """See `VectorStore.delete_documents_by_source`."""
        if not sources:
            return 0
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"""DELETE FROM {self._documents_table}
                        WHERE dataset_id = %s AND source = ANY(%s)""",
                    (dataset_id, sources),
                )
                return cur.rowcount

    def count_chunks_by_document(self, dataset_id: str) -> dict[str, int]:
        """See `VectorStore.count_chunks_by_document`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT document_id, COUNT(*) FROM {self._chunks_table}
                    WHERE dataset_id = %s GROUP BY document_id""",
                (dataset_id,),
            )
            return {str(document_id): count for document_id, count in cur.fetchall()}

    def list_document_versions(self, dataset_id: str) -> list[DocumentVersionInfo]:
        """See `VectorStore.list_document_versions`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT DISTINCT document_id, source, status, document_version,
                        effective_from, supersedes_source, tenant_id
                    FROM {self._chunks_table}
                    WHERE dataset_id = %s""",
                (dataset_id,),
            )
            rows = cur.fetchall()
        return [
            DocumentVersionInfo(
                document_id=str(document_id),
                source=source,
                status=status,
                document_version=document_version,
                effective_from=effective_from,
                supersedes_source=supersedes_source,
                tenant_id=tenant_id,
            )
            for (
                document_id,
                source,
                status,
                document_version,
                effective_from,
                supersedes_source,
                tenant_id,
            ) in rows
        ]

    def count_chunks_by_content_type(self, dataset_id: str) -> dict[str, int]:
        """See `VectorStore.count_chunks_by_content_type`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT content_type, COUNT(*) FROM {self._chunks_table}
                    WHERE dataset_id = %s GROUP BY content_type""",
                (dataset_id,),
            )
            return {(content_type or "prose"): count for content_type, count in cur.fetchall()}

    def get_document_checksums(self, dataset_id: str) -> dict[str, str]:
        """See `VectorStore.get_document_checksums`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT source, checksum FROM {self._documents_table} WHERE dataset_id = %s",
                (dataset_id,),
            )
            return dict(cur.fetchall())

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
                        c.metadata.parent_chunk_id,
                        c.metadata.vision_generated,
                        c.metadata.vision_description,
                        c.metadata.tenant_id,
                        c.metadata.allowed_roles,
                        c.metadata.classification,
                        c.metadata.status,
                        c.metadata.document_version,
                        c.metadata.effective_from,
                        c.metadata.trust_level,
                        c.metadata.doc_source_type,
                        c.metadata.supersedes_source,
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
                         attachment_name, source_anchor, parent_chunk_id,
                         vision_generated, vision_description,
                         tenant_id, allowed_roles, classification, status,
                         document_version, effective_from, trust_level,
                         doc_source_type, supersedes_source)
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
                            source_anchor = EXCLUDED.source_anchor,
                            parent_chunk_id = EXCLUDED.parent_chunk_id,
                            vision_generated = EXCLUDED.vision_generated,
                            vision_description = EXCLUDED.vision_description,
                            tenant_id = EXCLUDED.tenant_id,
                            allowed_roles = EXCLUDED.allowed_roles,
                            classification = EXCLUDED.classification,
                            status = EXCLUDED.status,
                            document_version = EXCLUDED.document_version,
                            effective_from = EXCLUDED.effective_from,
                            trust_level = EXCLUDED.trust_level,
                            doc_source_type = EXCLUDED.doc_source_type,
                            supersedes_source = EXCLUDED.supersedes_source""",
                    rows,
                )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        auth: AuthorizationContext | None = None,
    ) -> list[SearchResult]:
        """See `VectorStore.search`."""
        where_sql, filter_params = _build_where_clause(filters)
        auth_sql, auth_params = build_authorization_where_clause(
            auth, self._cross_tenant_support_roles
        )
        where_sql = _combine_where_clauses(where_sql, auth_sql)
        params: list[Any] = [
            query_embedding,
            *filter_params,
            *auth_params,
            query_embedding,
            top_k,
        ]

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
        auth: AuthorizationContext | None = None,
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
        auth_sql, auth_params = build_authorization_where_clause(
            auth, self._cross_tenant_support_roles
        )
        where_sql = _combine_where_clauses(where_sql, auth_sql)
        sql = f"SELECT {_METADATA_COLUMNS} FROM {self._chunks_table} {where_sql}"

        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [*filter_params, *auth_params])
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

    def get_chunks_by_ids(
        self, chunk_ids: list[str], auth: AuthorizationContext | None = None
    ) -> list[Chunk]:
        """See `VectorStore.get_chunks_by_ids`."""
        if not chunk_ids:
            return []
        auth_sql, auth_params = build_authorization_where_clause(
            auth, self._cross_tenant_support_roles
        )
        where_sql = _combine_where_clauses("WHERE chunk_id = ANY(%s)", auth_sql)
        sql = f"SELECT {_METADATA_COLUMNS} FROM {self._chunks_table} {where_sql}"
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [chunk_ids, *auth_params])
            rows = cur.fetchall()
        return [
            Chunk(id=chunk_id, content=content, metadata=metadata)
            for chunk_id, content, metadata in (_row_to_metadata(row) for row in rows)
        ]

    def get_chunks_by_section(
        self,
        document_id: str,
        section_path: str | None,
        auth: AuthorizationContext | None = None,
    ) -> list[Chunk]:
        """See `VectorStore.get_chunks_by_section`."""
        section_clause = "section_path = %s" if section_path is not None else "section_path IS NULL"
        base_sql = f"document_id = %s AND {section_clause}"
        auth_sql, auth_params = build_authorization_where_clause(
            auth, self._cross_tenant_support_roles
        )
        where_sql = _combine_where_clauses(f"WHERE {base_sql}", auth_sql)
        sql = f"""SELECT {_METADATA_COLUMNS} FROM {self._chunks_table}
            {where_sql}
            ORDER BY chunk_index"""
        params: list[Any] = (
            [document_id, section_path] if section_path is not None else [document_id]
        )
        params.extend(auth_params)
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            Chunk(id=chunk_id, content=content, metadata=metadata)
            for chunk_id, content, metadata in (_row_to_metadata(row) for row in rows)
        ]

    def get_cached_image_description(self, image_checksum: str) -> str | None:
        """See `VectorStore.get_cached_image_description`."""
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT description FROM image_description_cache WHERE image_checksum = %s",
                (image_checksum,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def cache_image_description(
        self,
        image_checksum: str,
        source_path: str,
        provider: str,
        model_name: str,
        description: str,
    ) -> None:
        """See `VectorStore.cache_image_description`."""
        with self._connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO image_description_cache
                        (image_checksum, source_path, provider, model_name, description)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (image_checksum) DO UPDATE SET
                            source_path = EXCLUDED.source_path,
                            provider = EXCLUDED.provider,
                            model_name = EXCLUDED.model_name,
                            description = EXCLUDED.description""",
                    (image_checksum, source_path, provider, model_name, description),
                )
