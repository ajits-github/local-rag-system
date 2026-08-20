"""Bounded agentic RAG workflow: routing, tool dispatch, and evidence synthesis.

Sits above the classic `RetrievalPipeline` and reuses it (and the
vectorstore/freshness/field-redaction/injection-detection modules it
already composes) as tools, rather than reimplementing retrieval or
security logic. See `docs/architecture.md`'s "Agentic RAG" section for the
full design and its invariants.
"""

from __future__ import annotations
