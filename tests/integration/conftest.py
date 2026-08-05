"""Shared fixtures for integration tests.

Integration tests assume Postgres+pgvector (`make up`) and Ollama are
already running locally. These fixtures probe connectivity once per
session and skip with a clear message rather than failing hard when a
dependency isn't up.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from rag.config import load_config


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return whether a TCP connection to (host, port) succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def config():
    """Session-wide AppConfig, loaded once for the whole integration run."""
    return load_config()


@pytest.fixture(scope="session")
def postgres_available(config) -> bool:
    """Whether Postgres is reachable at `config.database_url()`, checked once per session."""
    parsed = urlparse(config.database_url())
    return _tcp_reachable(parsed.hostname or "localhost", parsed.port or 5432)


@pytest.fixture(scope="session")
def ollama_available(config) -> bool:
    """Whether Ollama is reachable at `config.ollama_base_url()`, checked once per session."""
    parsed = urlparse(config.ollama_base_url())
    return _tcp_reachable(parsed.hostname or "localhost", parsed.port or 11434)


@pytest.fixture
def require_postgres(postgres_available):
    """Skip the test if Postgres/pgvector isn't reachable."""
    if not postgres_available:
        pytest.skip("Postgres/pgvector is not reachable at DATABASE_URL — run `make up` first.")


@pytest.fixture
def require_ollama(ollama_available):
    """Skip the test if Ollama isn't reachable."""
    if not ollama_available:
        pytest.skip(
            "Ollama is not reachable at OLLAMA_BASE_URL — "
            "start it and `ollama pull qwen2.5:1.5b` first."
        )
