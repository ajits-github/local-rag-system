"""Shared fixtures for the distributed-state experiment's own test suite.

`fake_redis` uses `fakeredis` (with its `lupa`-backed Lua scripting
support, so `EVAL`/`register_script` work) for every test that does not
specifically need a real network round trip. `real_redis` self-skips
cleanly when no live Redis is reachable, mirroring
`tests/integration/conftest.py`'s `require_postgres`/`require_ollama`
pattern in the main `rag` test suite.
"""

from __future__ import annotations

import pytest
import redis


@pytest.fixture
def fake_redis():
    """A fresh in-process fake Redis server per test (via `fakeredis`)."""
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    return fakeredis.FakeStrictRedis(server=server)


@pytest.fixture
def real_redis():
    """A real Redis client, skipping the test cleanly if Redis is unreachable.

    Looks at `RATE_LIMIT_REDIS_URL` first (matching the production config
    field name), then falls back to `redis://localhost:6379/0`.
    """
    import os

    url = os.environ.get("RATE_LIMIT_REDIS_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(url)
    try:
        client.ping()
    except redis.exceptions.RedisError:
        pytest.skip(f"No reachable Redis at {url!r}; start one to run this test (see README.md).")
    yield client
    client.flushdb()
