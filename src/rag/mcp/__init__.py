"""MCP (Model Context Protocol) server integration.

Exposes the existing agent tools (`rag.agent.tools`) over MCP so any
MCP-speaking client can call them, without duplicating retrieval or
authorization logic. See `rag.mcp.server` for the tool registrations and
`rag.mcp.identity` for how caller identity is resolved from the transport,
never from a tool call's arguments.
"""

from __future__ import annotations
