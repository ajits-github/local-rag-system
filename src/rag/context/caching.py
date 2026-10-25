"""Safe, narrow caching for the Context Engineering pipeline.

Section 20 of the design asks what is safe to cache. The answer here is
deliberately small: **only a text's estimated token count**, never raw
evidence, never a normalized `ContextItem`, never an assembled
`PackedContext`. Those all depend on authorization/redaction state that
can legitimately differ per caller even for byte-identical underlying
chunk text (a chunk's `sensitive_field_ids` redaction depends on the
caller's `roles`), so caching anything past "how many tokens is this
exact string" would risk serving one caller's authorized view to
another. A token count has no such dependency: it is a pure function of
`(text, token-counting strategy)`.

- **Cache unit**: one `(tenant_id, text, token-counter strategy)` ->
  token count.
- **Cache key**: a sha256 hash of the text (never the raw text itself,
  so the cache's own memory can't be read back as a content leak) plus
  `tenant_id` and the counter's `strategy_name`, so (a) no entry can
  ever be served across a tenant boundary even though the value itself
  carries no tenant-specific information today (defense in depth for a
  future, more context-dependent counting strategy), and (b) switching
  `context.token_counting.strategy` can never silently reuse a stale
  estimate computed by a different estimator.
- **Invalidation**: none needed beyond LRU eviction. The underlying
  text a `chunk_id` maps to never changes in place (re-ingestion assigns
  a new `chunk_id`, see `CLAUDE.md`'s "Document identity" section), so a
  cached token count never goes stale.
- **TTL**: none. A pure function's cached result does not expire; the
  LRU's `max_entries` bound is the only lifetime control, which bounds
  memory, not staleness.
- **Why not Redis**: `config.context.cache.enabled` defaults to `False`
  specifically because no measurable latency benefit was demonstrated
  for this single-instance, CPU-bound deployment (the heuristic token
  counter is already a single `len(text) / ratio` division; even
  `tiktoken` encoding a few KB of text is sub-millisecond). Adding Redis
  here would be the exact "infrastructure without a demonstrated
  requirement" this codebase has already declined twice (for the agent
  layer and for rate limiting pre-distributed-state-experiment). See
  `docs/context_engineering.md`'s caching section for the measurement.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from rag.context.tokens import TokenCounter


class TokenCountCache:
    """A small in-process LRU cache from `(tenant_id, text, strategy)` to token count."""

    def __init__(self, max_entries: int = 2048) -> None:
        """Construct the cache.

        Parameters
        ----------
        max_entries : int, optional
            LRU size cap, by default 2048.
        """
        self._max_entries = max_entries
        self._store: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(tenant_id: str | None, text: str, strategy_name: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{tenant_id or '_'}:{strategy_name}:{digest}"

    def get_or_count(self, tenant_id: str | None, text: str, counter: TokenCounter) -> int:
        """Return the cached token count for `text`, computing and storing it if absent.

        Parameters
        ----------
        tenant_id : str | None
            The caller's tenant, part of the cache key (see module
            docstring).
        text : str
            Text to count.
        counter : TokenCounter
            Used to compute the count on a cache miss.

        Returns
        -------
        int
            The token count.
        """
        key = self._key(tenant_id, text, counter.strategy_name)
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = counter.count(text)
        self._store[key] = value
        if len(self._store) > self._max_entries:
            self._store.popitem(last=False)
        return value

    def __len__(self) -> int:
        """Return the current number of cached entries."""
        return len(self._store)
