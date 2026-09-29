"""Distributed-lock demonstration (Part 3/EXPERIMENTS' "one distributed-lock scenario").

Reproduces a race-prone document-identity race deliberately modeled on
this repository's own, already-fixed `get_or_create_document_id` history
(see `src/rag/vectorstore/pgvector.py` and the root `CLAUDE.md`'s
"joint-investigation backlog" section: it used to be a racy
SELECT-then-INSERT, now an atomic `INSERT ... ON CONFLICT`). `demo_lock.py`
shows three variants of the same race -- unprotected, protected by a
Redis `SET NX PX` distributed lock, and protected by a DB-style unique
constraint -- and why the third is what this codebase actually shipped.
"""
