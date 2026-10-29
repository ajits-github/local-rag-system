"""Wrapper around the `neo4j` driver: the only module in `rag.graph` that imports it.

Every query runs with a server-side transaction timeout, and the driver's
connection pool is bounded by config.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, Query, RoutingControl
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from rag.config import AppConfig

logger = logging.getLogger(__name__)


class GraphUnavailableError(Exception):
    """Raised when Neo4j is unreachable or a query fails.

    Callers catch this to degrade: retrieval falls back to vector-only
    results, and ingestion skips graph extraction for the document.
    """


class Neo4jClient:
    """Owns one pooled `neo4j.Driver` and exposes bounded read/write query helpers."""

    def __init__(self, config: AppConfig) -> None:
        """Construct the driver from `config.graph.neo4j`.

        Parameters
        ----------
        config : AppConfig
            Application configuration; `config.graph.neo4j` supplies every
            connection setting, and `config.neo4j_uri`/`neo4j_user`/
            `neo4j_password` resolve the actual connection target from the
            environment.

        Raises
        ------
        RuntimeError
            If the configured password environment variable is unset (see
            `AppConfig.neo4j_password`). Never constructs a driver with an
            empty credential.
        """
        neo4j_cfg = config.graph.neo4j
        self._database = neo4j_cfg.database
        self._query_timeout_seconds = config.graph.traversal.query_timeout_seconds
        self._driver = GraphDatabase.driver(
            config.neo4j_uri(),
            auth=(config.neo4j_user(), config.neo4j_password()),
            max_connection_pool_size=neo4j_cfg.max_connection_pool_size,
            connection_acquisition_timeout=neo4j_cfg.connection_acquisition_timeout_seconds,
            connection_timeout=neo4j_cfg.connection_timeout_seconds,
            max_transaction_retry_time=neo4j_cfg.max_transaction_retry_time_seconds,
        )

    def close(self) -> None:
        """Close the underlying driver and its connection pool."""
        self._driver.close()

    def health_check(self) -> bool:
        """Cheap connectivity check, mirroring `VectorStore.health_check`'s contract.

        Returns
        -------
        bool
            `True` if the Neo4j instance is reachable and authenticated;
            `False` on any connectivity, auth, or protocol failure. Never
            raises.
        """
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            logger.warning("Neo4j health check failed", exc_info=True)
            return False

    def run_read(
        self, cypher: str, parameters: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Run a read-only Cypher query and return every row as a dict.

        Transient errors are retried by the driver. Callers need no retry loop.

        Parameters
        ----------
        cypher : str
            The query text. The caller must ensure it is read-only and
            bounded. This method does no validation.
        parameters : dict[str, Any] | None, optional
            Query parameters. Caller-influenced values must go here, never into
            `cypher`.
        timeout : float | None, optional
            Server-side transaction timeout in seconds. Defaults to
            `config.graph.traversal.query_timeout_seconds`.

        Returns
        -------
        list[dict[str, Any]]
            One dict per row, built with `dict(record)`. Unlike `Record.data()`,
            this keeps `Node`/`Relationship` values typed, so `.labels` and
            `.type` remain available to callers.

        Raises
        ------
        GraphUnavailableError
            If Neo4j is unreachable or the query fails, including on timeout.
        """
        try:
            effective_timeout = timeout if timeout is not None else self._query_timeout_seconds
            result = self._driver.execute_query(
                Query(cypher, timeout=effective_timeout),
                parameters or {},
                database_=self._database,
                routing_=RoutingControl.READ,
            )
            return [dict(record) for record in result.records]
        except (ServiceUnavailable, Neo4jError) as exc:
            raise GraphUnavailableError(f"Graph read query failed: {exc}") from exc

    def run_write(
        self, cypher: str, parameters: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, int]:
        """Run a write Cypher query (a `GraphWriter` upsert/delete) and return its counters.

        Parameters
        ----------
        cypher : str
            The query text. Callers pass only fixed, hand-written statements
            from `rag.graph.upsert`. Generated Cypher is never written
            through this method.
        parameters : dict[str, Any] | None, optional
            Query parameters.
        timeout : float | None, optional
            Server-side per-transaction timeout, in seconds; defaults to
            `config.graph.traversal.query_timeout_seconds`.

        Returns
        -------
        dict[str, int]
            A bounded summary of the write (`nodes_created`,
            `relationships_created`, `properties_set`, `nodes_deleted`,
            `relationships_deleted`), read from the query's own
            `ResultSummary.counters`.

        Raises
        ------
        GraphUnavailableError
            If Neo4j is unreachable, or the write fails for any reason.
        """
        try:
            effective_timeout = timeout if timeout is not None else self._query_timeout_seconds
            result = self._driver.execute_query(
                Query(cypher, timeout=effective_timeout),
                parameters or {},
                database_=self._database,
                routing_=RoutingControl.WRITE,
            )
            counters = result.summary.counters
            return {
                "nodes_created": counters.nodes_created,
                "relationships_created": counters.relationships_created,
                "properties_set": counters.properties_set,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_deleted": counters.relationships_deleted,
            }
        except (ServiceUnavailable, Neo4jError) as exc:
            raise GraphUnavailableError(f"Graph write query failed: {exc}") from exc
