# Distributed-state experiment (branch: `experiment/distributed-state`)

You're viewing the `experiment/distributed-state` branch, a
self-contained learning exercise, not the production system. For the
production `local-rag-system` README, see
[`main`](https://github.com/ajits-github/local-rag-system/blob/main/README.md).

![Distributed rate limiting and shared state with Redis: an in-memory
limiter enforces its limit correctly with one API replica but fails
globally once there are several, each keeping its own counter; a
Redis-backed limiter with atomic counter updates enforces the same
limit correctly across all replicas](distributed_state_experiment/docs/assets/architecture-diagram.png)

A self-contained learning exercise that finds behavior in this codebase
that is correct with one API replica and silently incorrect with more
than one, reproduces it, and fixes it with shared distributed state
(Redis). It then builds a real distributed background-processing
workflow (a Redis Streams-backed ingestion job queue and worker fleet)
on top of the same Redis instance.

See [`distributed_state_experiment/README.md`](distributed_state_experiment/README.md)
for the full writeup: all 5 parts and 12 experiments, the benchmark
table, and the concurrency concept explanations, and
[`distributed_state_experiment/INTERVIEW_QUESTIONS.md`](distributed_state_experiment/INTERVIEW_QUESTIONS.md)
for 15 interview Q&A grounded in what was actually built.
