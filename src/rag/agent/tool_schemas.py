"""Validated argument schemas for the agent's six tools.

Every model is `extra="forbid"`, a deliberate deviation from this
codebase's usual `extra="ignore"` default (`QueryRequest`, `GoldExample`,
`AuthorizationContext`): an LLM attempting to smuggle a
`tenant_id`/`roles`/`auth`-shaped key into a tool call produces a loud,
auditable `ValidationError` rather than being silently dropped. Every
LLM-writable numeric field also carries a hard `Field(ge=..., le=...)`
range; no schema exposes a chunk-count/limit field at all, since those are
entirely server-controlled (see `rag.config.AgentConfig` and
`rag.agent.tools`).

Four schemas (`SearchKnowledgeBaseArgs`/`GetDocumentArgs`/
`GetLatestDocumentArgs`/`GetRelatedContextArgs`) back the local,
in-process tools in `rag.agent.tools`. Two (`GetCustomerCaseArgs`/
`GetCaseStatusArgs`, see `REMOTE_MCP_TOOL_NAMES` below) back the remote
MCP business tools dispatched via `rag.agent.mcp_client`, with the same
`extra="forbid"` treatment: the "never let the LLM supply identity" rule
applies regardless of which side of the process boundary a tool executes
on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContentTypeFilter = Literal["prose", "table", "code", "configuration", "image", "chart"]


class SearchKnowledgeBaseArgs(BaseModel):
    """Arguments for the `search_knowledge_base` tool.

    `content_type` lets the model intentionally search for a specific
    structural kind of evidence (e.g. "find the table with these
    numbers"). It's a closed `Literal` set, not a free string, merged
    into the retrieval `filters` dict server-side against the same
    `VectorStore.ALLOWED_FILTER_FIELDS` whitelist every other filter
    uses: the LLM can select *which* allowed value to filter by, never
    name an arbitrary field.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    content_type: ContentTypeFilter | None = None


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


class GetCustomerCaseArgs(BaseModel):
    """Arguments for the remote `get_customer_case` MCP tool.

    Only `case_id` is LLM-writable; tenant/role identity is resolved
    server-side from the internal service token `rag.agent.mcp_client`
    mints, never from anything the model produces.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str


class GetCaseStatusArgs(BaseModel):
    """Arguments for the remote `get_case_status` MCP tool (see `rag.agent.mcp_client`)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_knowledge_base": SearchKnowledgeBaseArgs,
    "get_document": GetDocumentArgs,
    "get_latest_document": GetLatestDocumentArgs,
    "get_related_context": GetRelatedContextArgs,
    "get_customer_case": GetCustomerCaseArgs,
    "get_case_status": GetCaseStatusArgs,
}

# Tool names dispatched via a real MCP client call (rag.agent.mcp_client), rather
# than a direct in-process function call. Every other name in TOOL_ARG_MODELS is
# local. See rag.agent.graph._execute_tool for the dispatch branch this drives.
REMOTE_MCP_TOOL_NAMES: frozenset[str] = frozenset({"get_customer_case", "get_case_status"})
