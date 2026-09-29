"""A minimal in-memory fixed-window limiter, deliberately reproducing the bug.

This is a stand-in for `slowapi`'s real `MemoryStorage` (which backs
`rag.api.deps.get_rate_limiter()` whenever `security.rate_limit.backend`
stays at its default, `"memory"`) -- simplified to a plain Python `dict`
so Part 1's demo and tests can construct two independent instances (one
per simulated pod) and show their counters diverge under identical
traffic, without needing to boot two full FastAPI processes just to
prove the mechanism.

The real `slowapi.Limiter` has the identical process-locality property
for the identical reason: its counters live in this process's memory,
full stop. Nothing here is a simplification that hides the bug; it is
the same bug with the surrounding framework code removed so the counter
divergence is easy to see and assert on directly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class InMemoryFixedWindowLimiter:
    """A per-process fixed-window counter, exactly as `slowapi.MemoryStorage` is.

    Attributes
    ----------
    limit : int
        Maximum requests allowed per bucket per window.
    window_seconds : int
        Window length in seconds.
    pod_name : str
        A label for which simulated replica owns this instance (display
        only; never affects behavior). Real `slowapi` instances have no
        such label at all -- one process, one limiter, no way to even ask
        "which pod is this."
    """

    limit: int
    window_seconds: int = 60
    pod_name: str = "unnamed"
    _counts: dict[str, int] = field(default_factory=dict)
    _window_started: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def hit(self, bucket_key: str) -> tuple[bool, int]:
        """Record one request against `bucket_key` and report whether it was allowed.

        Parameters
        ----------
        bucket_key : str
            The rate-limit bucket, e.g. `"tenant:acme"`.

        Returns
        -------
        tuple[bool, int]
            `(allowed, count_after_this_request)`, counted only in this
            process's own memory.
        """
        now = time.monotonic()
        with self._lock:
            started = self._window_started.get(bucket_key)
            if started is None or now - started >= self.window_seconds:
                self._window_started[bucket_key] = now
                self._counts[bucket_key] = 0
            self._counts[bucket_key] += 1
            count = self._counts[bucket_key]
            return count <= self.limit, count

    def current_count(self, bucket_key: str) -> int:
        """Return the current window's count for `bucket_key` (0 if unseen)."""
        with self._lock:
            return self._counts.get(bucket_key, 0)
