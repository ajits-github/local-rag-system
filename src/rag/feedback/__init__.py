"""User-feedback-loop persistence: `POST /feedback` -> Postgres -> export for eval curation.

Structurally independent of `rag.retrieval`/`rag.agent`: this package only
persists and reads back caller-supplied ratings of an answer already
produced elsewhere; it never influences retrieval, generation, or
authorization.
"""
