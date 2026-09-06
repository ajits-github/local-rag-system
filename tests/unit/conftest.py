"""Shared fixtures for `tests/unit`.

See `ISSUES.md`'s "A container-only env var, once loaded into a local
shell, quietly broke the full unit suite hours after the test that set it
had finished" for the full diagnosis this fixture closes.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_host_invalid_config_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete `RAG_CONFIG_PATH` before every unit test, unless the test sets it itself.

    `rag.config.load_config()` calls `load_dotenv(override=False)`, so a
    developer's own `.env` -- which may point `RAG_CONFIG_PATH` at a
    container-only path like `/app/config/production.yaml` for real
    Docker runs -- gets baked into `os.environ` the first time any test
    (or even module import during collection) triggers a config load,
    and stays there for the rest of the process. Combined with any test
    that clears `rag.api.deps.get_config`'s `lru_cache` (e.g.
    `test_api_config_override.py`), the next unrelated test to build a
    fresh `get_config()` would fail trying to load a path that only
    exists inside a container.

    Unit tests must never depend on a host machine's own `.env`; a test
    that wants to exercise `RAG_CONFIG_PATH` sets it explicitly via its
    own `monkeypatch.setenv(...)`, which layers on top of (and is
    reverted together with) this fixture's `delenv` at the end of that
    same test -- so this is a true default, never a test-order hazard.
    """
    monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
