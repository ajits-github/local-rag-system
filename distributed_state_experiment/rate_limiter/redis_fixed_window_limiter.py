"""A small, explicit, atomic Redis fixed-window rate limiter (teaching reference).

Loads and runs `lua/fixed_window.lua` via `redis.Redis.register_script`,
so the check-and-increment happens as one atomic operation on the Redis
server, regardless of how many client processes ("pods") call `allow()`
concurrently. See that script's own comments for why a plain
GET-then-INCR from the client side is not safe under concurrency.

This module is intentionally independent of `slowapi`/`limits` (the
libraries the production fix in `rag.api.deps.get_rate_limiter` actually
reuses) so the atomicity story is visible end to end in one small file
for this experiment's demos, benchmark, and tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import redis

_LUA_PATH = Path(__file__).parent / "lua" / "fixed_window.lua"


@dataclass(frozen=True)
class LimitDecision:
    """The outcome of one `RedisFixedWindowLimiter.allow()` call.

    Attributes
    ----------
    allowed : bool
        Whether the request is under the limit.
    count : int
        The bucket's count after this request was recorded.
    limit : int
        The limit that was checked against.
    backend : {"redis", "fallback_open", "fallback_closed"}
        `"redis"` for a normal atomic check. `"fallback_open"` /
        `"fallback_closed"` only appear when `on_error="fail_open"` /
        `"fail_closed"` actually fired because Redis was unreachable --
        see `RedisFixedWindowLimiter.allow`'s docstring.
    """

    allowed: bool
    count: int
    limit: int
    backend: Literal["redis", "fallback_open", "fallback_closed"] = "redis"


class RedisUnavailableError(RuntimeError):
    """Raised by `allow()` when Redis is unreachable and `on_error="fail_closed"`."""


class RedisFixedWindowLimiter:
    """An atomic, Redis-backed fixed-window limiter shared across processes.

    Parameters
    ----------
    redis_client : redis.Redis
        A connected `redis-py` client. The same client instance may be
        shared by many rate-limit buckets; nothing here is per-bucket
        state on the client side, so multiple `RedisFixedWindowLimiter`
        instances in *different* processes pointed at the same Redis
        server and the same `key_prefix` share their counters correctly.
        That sharing is the entire point of this class.
    window_seconds : int
        Fixed-window length in seconds.
    key_prefix : str
        Prefix for every key this limiter creates, kept small and fixed
        (never a raw query, tenant name, or token) to bound cardinality
        to "number of distinct buckets times number of windows in the
        retention period," the same bound the in-memory limiter already
        has.
    on_error : {"fail_open", "fail_closed"}
        What `allow()` does when the Redis call itself raises
        `redis.exceptions.RedisError` (connection refused, timeout,
        etc.). `"fail_open"` (the default) returns an allowed decision
        with `backend="fallback_open"`: availability wins over strict
        enforcement during an outage, matching the production default in
        `RateLimitConfig.redis_fail_mode`. `"fail_closed"` re-raises as
        `RedisUnavailableError` instead of silently allowing or denying,
        so the caller must decide explicitly (in the real API this is a
        503, not a silent pass-through or a silent block).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        window_seconds: int = 60,
        key_prefix: str = "edu_rl",
        on_error: Literal["fail_open", "fail_closed"] = "fail_open",
    ) -> None:
        self._client = redis_client
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self.on_error = on_error
        self._script = redis_client.register_script(_LUA_PATH.read_text())

    def _bucket_key(self, bucket_key: str) -> str:
        """Build the windowed Redis key for `bucket_key`.

        Bucketing the window itself into the key (rather than relying
        purely on TTL) makes the fixed-window boundary explicit and
        avoids any ambiguity from clock drift between what set the TTL
        and what's reading it later. Every request in the same
        `window_seconds`-wide slice of wall-clock time computes the exact
        same key.
        """
        window_index = int(time.time() // self.window_seconds)
        return f"{self.key_prefix}:{bucket_key}:{window_index}"

    def allow(self, bucket_key: str, limit: int) -> LimitDecision:
        """Atomically record and check one request against `bucket_key`.

        Parameters
        ----------
        bucket_key : str
            The rate-limit bucket, e.g. `"tenant:acme"` or `"ip:1.2.3.4"`.
            Never a raw token, query string, or full request path.
        limit : int
            Requests allowed per window for this bucket.

        Returns
        -------
        LimitDecision

        Raises
        ------
        RedisUnavailableError
            If Redis is unreachable and `on_error="fail_closed"`.
        """
        key = self._bucket_key(bucket_key)
        try:
            allowed_flag, count = self._script(keys=[key], args=[self.window_seconds, limit])
        except redis.exceptions.RedisError:
            if self.on_error == "fail_open":
                return LimitDecision(allowed=True, count=-1, limit=limit, backend="fallback_open")
            raise RedisUnavailableError(
                f"Redis unreachable while checking rate limit for bucket {bucket_key!r}"
            ) from None
        return LimitDecision(allowed=bool(allowed_flag), count=int(count), limit=limit)
