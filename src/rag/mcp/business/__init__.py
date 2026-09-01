"""Synthetic business-backend example exposed over MCP (Stage 1B).

Demonstrates MCP as an integration layer to a separate backend/business
system, not just another transport for this deployment's own RAG tools
(`rag.agent.tools`, Stage 1A). `store.py` is a small, in-memory,
read-only synthetic customer-support-case dataset with its own
tenant/role authorization, deliberately independent of
`rag.retrieval.authorization.AuthorizationContext` (document/freshness/
trust concepts that don't apply to this resource type) and of
`rag.retrieval.pipeline.RetrievalPipeline.sanitize_evidence` (chunk/
field-redaction machinery that doesn't apply either -- a case is either
authorized whole or not returned at all, never partially redacted).
See `rag.mcp.server` for the two tool registrations
(`get_customer_case`/`get_case_status`).
"""

from __future__ import annotations
