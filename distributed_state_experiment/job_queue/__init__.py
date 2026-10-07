"""Distributed ingestion job queue built on Redis Streams (Part 2B of the experiment).

Not imported by `src/rag/` or `rag.api`. See this repository's
`distributed_state_experiment/README.md` for the full design (producer/
consumer shape, retry/backoff, idempotency, dead-letter handling,
visibility semantics) and for why this stays a separate, isolated
learning exercise rather than a change to `POST /ingest`'s current
request-lifecycle-bound behavior.
"""
