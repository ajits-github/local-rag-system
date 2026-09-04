"""Bounded agentic RAG workflow: routing, tool dispatch, and evidence synthesis.

Sits above the classic `RetrievalPipeline` and reuses it (and the
vectorstore/freshness/field-redaction/injection-detection modules it
already composes) as tools, rather than reimplementing retrieval or
security logic.
"""

from __future__ import annotations
