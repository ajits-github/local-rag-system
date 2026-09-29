-- Atomic fixed-window rate limit check-and-increment.
--
-- This is the teaching reference implementation for
-- `distributed_state_experiment/rate_limiter/redis_fixed_window_limiter.py`.
-- The production-shaped fix actually wired into `src/rag/api/deps.py`
-- (`security.rate_limit.backend: redis`) instead reuses the `limits`
-- package's own `RedisStorage`, whose `incr_expire.lua` script does the
-- same INCR+EXPIRE atomically for the fixed-window strategy (confirmed
-- directly against the installed `limits==5.8.0` source). This script
-- exists to make that same atomicity visible and explicit for the demo
-- and benchmark scripts in this directory, and to embed the limit check
-- itself in the same round trip rather than comparing client-side.
--
-- KEYS[1] = the bucket key, e.g. "edu_rl:tenant:acme:2026-09-29T12:07"
-- ARGV[1] = window length in seconds (used only on the first INCR of a
--           fresh key, to set its TTL)
-- ARGV[2] = the limit (max requests allowed inside the window)
--
-- Returns a two-element array: {allowed (1/0), current_count}.
--
-- Why this must be one Lua script and not two round trips
-- ---------------------------------------------------------
-- A naive client-side implementation does:
--   count = GET key
--   if count is None or count < limit:
--       INCR key
--       if count is None: EXPIRE key, window
--       allow
--   else:
--       deny
--
-- Between the GET and the INCR, a second replica's own GET can read the
-- same pre-increment value and also decide to allow -- both replicas
-- then increment, and the bucket ends up over its limit by however many
-- replicas raced through that window. Redis is single-threaded for
-- command execution, and a Lua script runs as one atomic unit with no
-- other command interleaved inside it, so folding "increment" and "is
-- this over the limit" into one EVAL call removes that race entirely,
-- regardless of how many replicas call it concurrently.

local current = redis.call("INCR", KEYS[1])
if tonumber(current) == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end

if tonumber(current) > tonumber(ARGV[2]) then
    return {0, current}
end
return {1, current}
