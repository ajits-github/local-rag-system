"""Argument models for the agent's eleven tools.

Every model is `extra="forbid"`, so an LLM-supplied `tenant_id`, `roles`,
or `auth` key fails validation instead of being dropped silently. String
and numeric fields have explicit bounds. Limits such as chunk counts are
set by server config, not by the model.

Local tools:

- Four RAG tools in `rag.agent.tools`: `SearchKnowledgeBaseArgs`,
  `GetDocumentArgs`, `GetLatestDocumentArgs`, `GetRelatedContextArgs`.
- Four graph tools, offered only when `config.graph.enabled`.

Remote MCP tools (`REMOTE_MCP_TOOL_NAMES`), dispatched through
`rag.agent.mcp_client`: `GetCustomerCaseArgs`, `GetCaseStatusArgs`,
`UpdateCaseStatusArgs`. `UpdateCaseStatusArgs` has no approval field.
Approval comes from a signed internal token, never from a tool argument.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.graph.schema import EntityType, RelationshipType
from rag.mcp.business.schemas import CaseStatus

ContentTypeFilter = Literal["prose", "table", "code", "configuration", "image", "chart"]

# Generous bound for a free-text field, matching DoSLimitsConfig.max_query_length's
# existing precedent for a caller-supplied query string elsewhere in this codebase.
_MAX_QUERY_LENGTH = 2000
# Tight bound for an identifier-shaped field (a document path, chunk id, or case
# id), matching the convention already used in rag.api.routers.feedback.
_MAX_IDENTIFIER_LENGTH = 200


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

    query: str = Field(max_length=_MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    content_type: ContentTypeFilter | None = None


class GetDocumentArgs(BaseModel):
    """Arguments for the `get_document` tool."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


class GetLatestDocumentArgs(BaseModel):
    """Arguments for the `get_latest_document` tool."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


class GetRelatedContextArgs(BaseModel):
    """Arguments for the `get_related_context` tool."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


class GetCustomerCaseArgs(BaseModel):
    """Arguments for the remote `get_customer_case` MCP tool.

    Only `case_id` is LLM-writable; tenant/role identity is resolved
    server-side from the internal service token `rag.agent.mcp_client`
    mints, never from anything the model produces.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


class GetCaseStatusArgs(BaseModel):
    """Arguments for the remote `get_case_status` MCP tool (see `rag.agent.mcp_client`)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


class UpdateCaseStatusArgs(BaseModel):
    """Arguments for the remote `update_case_status` MCP tool.

    Only `case_id` and `new_status` are LLM-writable. Whether the
    transition is valid, and whether it needs prior approval, is decided
    entirely server-side (`rag.mcp.business.store.update_case_status`);
    no approval flag or token exists on this model.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)
    new_status: CaseStatus


class SearchKnowledgeGraphArgs(BaseModel):
    """Arguments for the `search_knowledge_graph` tool.

    A free-text question the graph retriever classifies into a bounded
    intent (dependents/runbooks/owners/policy/impact/path) and answers via
    entity linking + traversal (see `rag.graph.retrieval.GraphRetriever.search`).
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=_MAX_QUERY_LENGTH)


class GetEntityArgs(BaseModel):
    """Arguments for the `get_entity` tool: resolve a named entity and its direct neighbors."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)
    entity_type: EntityType | None = None


class GetNeighborsArgs(BaseModel):
    """Arguments for the `get_neighbors` tool: one-hop traversal from a named entity.

    `max_hops` is bounded to `1..3` here. `GraphRetriever.get_neighbors` also
    clamps it to `config.graph.traversal.max_traversal_depth`.
    """

    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)
    relationship_type: RelationshipType | None = None
    max_hops: int = Field(default=1, ge=1, le=3)


class FindDependencyPathArgs(BaseModel):
    """Arguments for the `find_dependency_path` tool: bounded shortest path between two entities."""

    model_config = ConfigDict(extra="forbid")

    source_entity_name: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)
    target_entity_name: str = Field(max_length=_MAX_IDENTIFIER_LENGTH)


# Graph tools. They run in-process, not through MCP. They are offered to the
# model only when config.graph.enabled. A call made while the graph is disabled
# fails in _execute_tool.
GRAPH_TOOL_NAMES: frozenset[str] = frozenset(
    {"search_knowledge_graph", "get_entity", "get_neighbors", "find_dependency_path"}
)

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_knowledge_base": SearchKnowledgeBaseArgs,
    "get_document": GetDocumentArgs,
    "get_latest_document": GetLatestDocumentArgs,
    "get_related_context": GetRelatedContextArgs,
    "get_customer_case": GetCustomerCaseArgs,
    "get_case_status": GetCaseStatusArgs,
    "update_case_status": UpdateCaseStatusArgs,
    "search_knowledge_graph": SearchKnowledgeGraphArgs,
    "get_entity": GetEntityArgs,
    "get_neighbors": GetNeighborsArgs,
    "find_dependency_path": FindDependencyPathArgs,
}

# Tool names dispatched via a real MCP client call (rag.agent.mcp_client), rather
# than a direct in-process function call. Every other name in TOOL_ARG_MODELS is
# local. See rag.agent.graph._execute_tool for the dispatch branch this drives.
REMOTE_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {"get_customer_case", "get_case_status", "update_case_status"}
)

# The subset of REMOTE_MCP_TOOL_NAMES that mutates state, gated by its own
# mcp.business_actions.enabled kill-switch and treated as the last tool call
# of a run (see rag.agent.graph's tool-call loop): a write-action attempt is
# never retried within the same run, regardless of its outcome.
WRITE_ACTION_TOOL_NAMES: frozenset[str] = frozenset({"update_case_status"})
