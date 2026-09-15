"""MCP (Model Context Protocol) server integration.

Exposes the four `rag.agent.tools` functions plus three synthetic
business-case tools (`rag.mcp.business`) over MCP, without duplicating
retrieval or authorization logic. See `rag.mcp.server` for the tool
registrations and `rag.mcp.identity` for how caller identity is resolved
from the transport, never from a tool call's arguments.
"""

from __future__ import annotations
