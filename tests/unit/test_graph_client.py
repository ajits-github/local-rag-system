"""Unit tests for `Neo4jClient`'s use of the real `neo4j` driver's `execute_query` contract.

Regression for GRAPH-007: `execute_query` was called with `routing_control=`
and a bare `transaction_timeout=` kwarg, neither of which the installed
`neo4j` driver's `Driver.execute_query` actually accepts (its real
parameters are `routing_=` and a `neo4j.Query(text, timeout=...)` object
passed as the query itself). Those two unrecognized kwargs silently fell
into `execute_query`'s own `**kwargs`, which it forwards to
`Transaction.run(query, parameters, **kwargs)` as if they were *Cypher
query parameters*, not driver options; the server-side timeout was never
actually set, and reads always used the driver's default (write) routing.

Uses `unittest.mock.MagicMock` in place of the real `neo4j.Driver`, not a
live Neo4j instance, and constructs `Neo4jClient` by bypassing `__init__`
(no real connection/credentials needed) so these tests assert the exact
call shape against the real installed `neo4j` package's `Query`/
`RoutingControl` types, not a hand-rolled substitute for them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from neo4j import Query, RoutingControl

from rag.graph.client import Neo4jClient


def _fake_client(driver: MagicMock, *, query_timeout_seconds: float = 10.0) -> Neo4jClient:
    """Build a `Neo4jClient` with an injected mock driver, skipping `__init__`."""
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = driver
    client._database = "neo4j"
    client._query_timeout_seconds = query_timeout_seconds
    return client


def test_run_read_uses_routing_keyword_and_a_real_query_object_with_timeout():
    """run_read must pass routing_=READ and a Query carrying the real timeout."""
    driver = MagicMock()
    driver.execute_query.return_value.records = []
    client = _fake_client(driver, query_timeout_seconds=7.5)

    client.run_read("MATCH (n) RETURN n", {"a": 1})

    _, kwargs = driver.execute_query.call_args
    call_args = driver.execute_query.call_args.args
    query_arg = call_args[0]
    assert isinstance(query_arg, Query)
    assert query_arg.text == "MATCH (n) RETURN n"
    assert query_arg.timeout == 7.5
    assert kwargs["routing_"] == RoutingControl.READ
    assert "routing_control" not in kwargs
    assert "transaction_timeout" not in kwargs


def test_run_read_honors_an_explicit_per_call_timeout_override():
    """An explicit `timeout=` argument overrides the configured default."""
    driver = MagicMock()
    driver.execute_query.return_value.records = []
    client = _fake_client(driver, query_timeout_seconds=10.0)

    client.run_read("MATCH (n) RETURN n", timeout=2.0)

    query_arg = driver.execute_query.call_args.args[0]
    assert query_arg.timeout == 2.0


def test_run_write_uses_routing_keyword_and_a_real_query_object_with_timeout():
    """run_write must pass routing_=WRITE and a Query carrying the real timeout."""
    driver = MagicMock()
    driver.execute_query.return_value.summary.counters.nodes_created = 1
    driver.execute_query.return_value.summary.counters.relationships_created = 0
    driver.execute_query.return_value.summary.counters.properties_set = 2
    driver.execute_query.return_value.summary.counters.nodes_deleted = 0
    driver.execute_query.return_value.summary.counters.relationships_deleted = 0
    client = _fake_client(driver, query_timeout_seconds=5.0)

    client.run_write("MERGE (n:X {id: $id})", {"id": "1"})

    _, kwargs = driver.execute_query.call_args
    query_arg = driver.execute_query.call_args.args[0]
    assert isinstance(query_arg, Query)
    assert query_arg.timeout == 5.0
    assert kwargs["routing_"] == RoutingControl.WRITE
    assert "routing_control" not in kwargs
    assert "transaction_timeout" not in kwargs
