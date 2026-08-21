"""Validated argument schemas for the agent's four tools.

Every model is `extra="forbid"`. A deliberate deviation from this
codebase's usual `extra="ignore"` default (see `QueryRequest`,
`GoldExample`, `AuthorizationContext`). So an LLM attempting to smuggle a
`tenant_id`/`roles`/`auth`-shaped key into a tool call produces a loud,
auditable `ValidationError` rather than silently being dropped. Every
LLM-writable numeric field also carries a hard `Field(ge=..., le=...)`
range; no schema exposes a chunk-count/limit field at all, since those are
entirely server-controlled (see `rag.config.AgentConfig` and
`rag.agent.tools`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchKnowledgeBaseArgs(BaseModel):
    """Arguments for the `search_knowledge_base` tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class GetDocumentArgs(BaseModel):
    """Arguments for the `get_document` tool."""

    model_config = ConfigDict(extra="forbid")

    source: str


class GetLatestDocumentArgs(BaseModel):
    """Arguments for the `get_latest_document` tool."""

    model_config = ConfigDict(extra="forbid")

    source: str


class GetRelatedContextArgs(BaseModel):
    """Arguments for the `get_related_context` tool."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_knowledge_base": SearchKnowledgeBaseArgs,
    "get_document": GetDocumentArgs,
    "get_latest_document": GetLatestDocumentArgs,
    "get_related_context": GetRelatedContextArgs,
}
