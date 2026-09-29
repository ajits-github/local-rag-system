"""Teaching-reference rate limiters for the distributed-state experiment.

`in_memory.py` reproduces the exact process-local bug this experiment
demonstrates; `redis_fixed_window_limiter.py` is the atomic, Redis-backed
fix. Neither module is imported by `src/rag/`; the production-shaped fix
lives in `rag.api.deps.get_rate_limiter` and reuses the `limits` package
instead of this module (see that function's docstring and this
directory's `README.md` section "Two limiter implementations, on
purpose" for why both exist).
"""
