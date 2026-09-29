"""Part 2: the Redis-backed fixed-window limiter is atomic and shared across "processes"."""

from __future__ import annotations

import redis

from distributed_state_experiment.rate_limiter.redis_fixed_window_limiter import (
    RedisFixedWindowLimiter,
    RedisUnavailableError,
)


def test_single_limiter_enforces_limit(fake_redis) -> None:
    limiter = RedisFixedWindowLimiter(fake_redis, window_seconds=60, key_prefix="test_rl")
    allowed_count = sum(1 for _ in range(140) if limiter.allow("tenant:acme", limit=100).allowed)
    assert allowed_count == 100


def test_two_limiter_instances_sharing_redis_enforce_one_global_limit(fake_redis) -> None:
    """Two independent `RedisFixedWindowLimiter` objects (modeling Pod A / Pod B) pointed at the
    same Redis instance must together allow exactly 100, not 140 -- the fix for
    `test_inmemory_limiter_divergence.py`'s demonstrated bug.
    """
    pod_a = RedisFixedWindowLimiter(fake_redis, window_seconds=60, key_prefix="test_rl")
    pod_b = RedisFixedWindowLimiter(fake_redis, window_seconds=60, key_prefix="test_rl")

    allowed_a = sum(1 for _ in range(70) if pod_a.allow("tenant:acme", limit=100).allowed)
    allowed_b = sum(1 for _ in range(70) if pod_b.allow("tenant:acme", limit=100).allowed)

    assert allowed_a + allowed_b == 100


def test_concurrent_hits_from_multiple_threads_never_exceed_the_limit(fake_redis) -> None:
    """The atomic Lua check-and-increment holds even under real thread concurrency.

    Fires 200 concurrent `allow()` calls (limit 50) from a thread pool
    against one shared fake Redis instance and asserts the allowed count
    is exactly 50 -- proving there is no lost-update window between the
    increment and the limit check, the same property a naive
    GET-then-INCR implementation would not have.
    """
    from concurrent.futures import ThreadPoolExecutor

    limiter = RedisFixedWindowLimiter(
        fake_redis, window_seconds=60, key_prefix="test_rl_concurrent"
    )
    with ThreadPoolExecutor(max_workers=20) as pool:
        decisions = list(
            pool.map(lambda _: limiter.allow("tenant:acme", limit=50).allowed, range(200))
        )
    assert sum(decisions) == 50


def test_different_buckets_are_independent(fake_redis) -> None:
    limiter = RedisFixedWindowLimiter(fake_redis, window_seconds=60, key_prefix="test_rl")
    for _ in range(50):
        limiter.allow("tenant:acme", limit=50)
    # A different tenant's bucket starts fresh, unaffected by acme's count.
    decision = limiter.allow("tenant:other", limit=50)
    assert decision.allowed
    assert decision.count == 1


def test_fail_open_allows_request_when_redis_unreachable(monkeypatch, fake_redis) -> None:
    limiter = RedisFixedWindowLimiter(fake_redis, key_prefix="test_rl", on_error="fail_open")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise redis.exceptions.ConnectionError("simulated outage")

    monkeypatch.setattr(limiter, "_script", _boom)
    decision = limiter.allow("tenant:acme", limit=10)
    assert decision.allowed is True
    assert decision.backend == "fallback_open"


def test_fail_closed_raises_when_redis_unreachable(monkeypatch, fake_redis) -> None:
    limiter = RedisFixedWindowLimiter(fake_redis, key_prefix="test_rl", on_error="fail_closed")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise redis.exceptions.ConnectionError("simulated outage")

    monkeypatch.setattr(limiter, "_script", _boom)
    try:
        limiter.allow("tenant:acme", limit=10)
        raised = False
    except RedisUnavailableError:
        raised = True
    assert raised, "fail_closed must raise rather than silently allow or deny"


def test_concurrent_hits_against_real_redis_server_never_exceed_the_limit(real_redis) -> None:
    """The same atomicity property, verified against a genuine `redis-server` binary, not `fakeredis`.

    Every other test in this file runs against `fakeredis` for speed and
    determinism in CI/local dev with no infrastructure required. This one
    self-skips cleanly (see `conftest.py::real_redis`) unless a real Redis
    is reachable at `RATE_LIMIT_REDIS_URL`/`redis://localhost:6379/0` --
    e.g. via `docker compose -f docker-compose.yml -f docker-compose.redis.yml
    up -d redis` -- and exists specifically to close the gap `fakeredis`
    cannot: proving the atomic Lua check-and-increment genuinely holds
    against Redis's real single-threaded command execution guarantee, not
    just against a pure-Python simulation of it.
    """
    from concurrent.futures import ThreadPoolExecutor

    limiter = RedisFixedWindowLimiter(
        real_redis, window_seconds=60, key_prefix="real_redis_atomic_test"
    )
    with ThreadPoolExecutor(max_workers=20) as pool:
        decisions = list(
            pool.map(lambda _: limiter.allow("tenant:acme", limit=50).allowed, range(200))
        )
    assert sum(decisions) == 50


def test_no_jwt_or_token_ever_used_as_a_key_fragment(fake_redis) -> None:
    """Bucket keys are only ever the caller-supplied bucket string, never a token/secret.

    This test asserts the *contract*: the limiter never derives a key
    from anything except the `bucket_key` argument a caller passes in --
    it is the caller's job (mirroring `rag.api.deps._rate_limit_key`) to
    pass `f"tenant:{tenant_id}"` / `f"ip:{addr}"`, never a raw JWT.
    """
    limiter = RedisFixedWindowLimiter(fake_redis, key_prefix="test_rl")
    limiter.allow("tenant:acme", limit=10)
    keys = [k.decode() if isinstance(k, bytes) else k for k in fake_redis.keys("test_rl:*")]
    assert len(keys) == 1
    assert "tenant:acme" in keys[0]
    assert "eyJ" not in keys[0]  # a JWT's base64url header always starts this way
