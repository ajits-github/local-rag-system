"""`security.rate_limit.backend: redis` wiring in `rag.api.deps.get_rate_limiter`.

Part of the distributed-state-experiment milestone (see
`distributed_state_experiment/README.md`): these tests prove the
*construction-time wiring* is correct (the right `Limiter` kwargs for
each `redis_fail_mode`) without needing a reachable Redis -- `slowapi`'s
`Limiter.__init__` stores `storage_uri` but does not eagerly connect
(confirmed directly: `limits.storage.redis.RedisStorage.__init__` calls
`register_script`, which only creates a `redis.commands.core.Script`
object referencing the client, never a network round trip -- the actual
connection only happens on the first real `hit()`/`EVAL` call). The
Redis-backed limiter's actual atomic behavior under concurrency is
proven separately, and more directly, by
`distributed_state_experiment/tests/test_redis_fixed_window_limiter.py`
(the teaching-reference implementation) using `fakeredis`; this file's
job is only to prove `rag.api.deps` builds the *production* `Limiter`
object with the arguments its own docstring promises.

A note on this file's own test-run cost in this sandboxed environment:
importing `rag.api.deps` (via `rag.factory` -> `sentence_transformers` ->
`torch`) was measured at **1122 seconds** (~19 minutes) for a single
cold import on this machine -- see
`distributed_state_experiment/README.md`'s "What was executed vs.
simulated" section. This file was written and reviewed against that same
measured cost; a full pytest run including this module inherits it,
same as every other `tests/unit` module that already imports
`rag.api.deps` (e.g. `test_rate_limiting.py`, unmodified by this
milestone).
"""

from __future__ import annotations

import pytest

import rag.api.deps as deps
from rag.config import RateLimitConfig

# Captured once, at import time, before any test's `monkeypatch.setattr(deps,
# "get_config", ...)` replaces the module attribute with a plain lambda (which
# has no `.cache_clear()`). Fixture teardown order in this file runs this
# fixture's post-yield code *before* `monkeypatch` itself reverts that
# attribute back to the real `lru_cache`d function (confirmed by hitting
# exactly this `AttributeError` in a real run), so clearing the cache through
# this captured reference -- rather than through `deps.get_config`, which may
# still be a test's patched lambda at that point -- is what makes teardown
# order-independent.
_real_get_config = deps.get_config


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Ensure `get_rate_limiter()`/`get_config()` see only this test's own config.

    Clears both `lru_cache`s before and after every test in this file.
    """
    _real_get_config.cache_clear()
    deps.get_rate_limiter.cache_clear()
    yield
    _real_get_config.cache_clear()
    deps.get_rate_limiter.cache_clear()


def _patch_config(monkeypatch: pytest.MonkeyPatch, rate_limit: RateLimitConfig) -> None:
    """Patch `deps.get_config()` to return a fake exposing only `security.rate_limit`."""
    from types import SimpleNamespace

    fake_config = SimpleNamespace(
        security=SimpleNamespace(rate_limit=rate_limit),
        rate_limit_redis_url=lambda: "redis://unreachable-in-this-test:6379/0",
    )
    monkeypatch.setattr(deps, "get_config", lambda: fake_config)


def test_memory_backend_is_unchanged_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """`backend="memory"` (the default) builds a plain `Limiter`, no `storage_uri` at all."""
    _patch_config(monkeypatch, RateLimitConfig(enabled=True, backend="memory"))
    limiter = deps.get_rate_limiter()
    assert limiter._storage_uri is None
    assert limiter.enabled is True


def test_redis_backend_fail_open_enables_in_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`redis_fail_mode="fail_open"` enables an in-memory fallback limiter.

    Per `RateLimitConfig.redis_fail_mode`'s docstring: on a Redis error,
    the request degrades to a process-local in-memory limiter for the
    outage's duration rather than being silently let through unbounded
    (which plain `swallow_errors=True` alone would do) or hard-failing.
    """
    _patch_config(
        monkeypatch,
        RateLimitConfig(
            enabled=True, backend="redis", redis_fail_mode="fail_open", requests_per_minute=42
        ),
    )
    limiter = deps.get_rate_limiter()
    assert limiter._storage_uri == "redis://unreachable-in-this-test:6379/0"
    assert limiter._swallow_errors is False
    assert limiter._in_memory_fallback_enabled is True
    assert limiter._in_memory_fallback, "an explicit fallback limit string must be configured"


def test_redis_backend_fail_closed_has_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """`redis_fail_mode="fail_closed"` sets `swallow_errors=False` and no in-memory fallback.

    A Redis error must propagate (caught by `rag.api.main`'s explicit
    `redis.exceptions.RedisError` handler -> 503), never silently
    downgrade to weaker enforcement.
    """
    _patch_config(
        monkeypatch,
        RateLimitConfig(enabled=True, backend="redis", redis_fail_mode="fail_closed"),
    )
    limiter = deps.get_rate_limiter()
    assert limiter._swallow_errors is False
    assert limiter._in_memory_fallback_enabled is False
    assert limiter._in_memory_fallback == []


def test_redis_backend_passes_through_strategy_and_key_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured `strategy`/`redis_key_prefix` reach the constructed `Limiter` unchanged."""
    _patch_config(
        monkeypatch,
        RateLimitConfig(
            enabled=True,
            backend="redis",
            strategy="moving-window",
            redis_key_prefix="custom_prefix",
        ),
    )
    limiter = deps.get_rate_limiter()
    assert limiter._strategy == "moving-window"
    assert limiter._key_prefix == "custom_prefix"


def test_disabled_redis_limiter_still_builds_without_a_reachable_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`enabled=False` + `backend="redis"` must not itself require Redis connectivity to construct.

    Mirrors the memory backend's own "constructing a disabled Limiter
    never touches the network" property -- a coarse kill-switch should
    never have a stricter startup requirement than the feature it gates.
    """
    _patch_config(monkeypatch, RateLimitConfig(enabled=False, backend="redis"))
    limiter = deps.get_rate_limiter()
    assert limiter.enabled is False
