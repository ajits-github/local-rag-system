"""Synthetic business-case backend exposed over MCP.

Demonstrates MCP as an integration layer to a separate backend system,
not just another transport for this deployment's own RAG tools.
`store.py` is a small, in-memory, mostly read-only case dataset with its
own tenant/role authorization, independent of
`rag.retrieval.authorization.AuthorizationContext` (no document/
freshness/trust concept applies here) and of
`RetrievalPipeline.sanitize_evidence` (a case is either authorized
whole or not returned at all, never partially redacted).
"""

from __future__ import annotations
