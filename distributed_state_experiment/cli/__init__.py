"""Runnable entry points for the distributed-state experiment.

Every script here takes `--redis-url` (default `redis://127.0.0.1:6399/0`,
matching `fake_redis_server.py`'s default port -- deliberately not 6379,
Redis's real default, so this never collides with a genuine local Redis
instance someone might also have running) and is safe to run against
either `fake_redis_server.py` (this session's stand-in; see
`README.md`'s "What was executed vs. simulated" section for why) or a
real Redis started via `docker-compose.redis.yml`.
"""
