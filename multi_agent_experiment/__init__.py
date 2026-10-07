"""Experimental LangGraph *multi*-agent workflow: coordinator + specialists.

A second, isolated learning exercise, sibling to `langgraph_experiment/`
(the single-agent LangGraph experiment) and `rag.agent.graph.run_agent`
(the production custom harness). Neither existing agent is modified by
this package. Nothing here is imported by `src/rag/`, `rag.api`, or
`langgraph_experiment/`. This package imports *them*, not the other way
around.

This package exists to answer one question the single-agent LangGraph
experiment could not: what does LangGraph buy you once there is more than
one agent, each with genuinely different tool access and responsibility,
that needs to be routed to, coordinated, and merged? See `README.md` in
this directory for the full write-up (architecture, run commands,
required scenarios, three-way comparison against both existing agents,
and an interview Q&A).

Runs in the **same** `.venv-langgraph/` virtual environment
`langgraph_experiment/` already uses (see that package's README for why a
separate venv is needed at all: `langgraph>=1.2` pins
`langchain-core>=1.4,<2`, incompatible with this repo's pinned
`langchain>=0.3,<0.4`). No new venv, no new dependency. This package
imports `langgraph_experiment` directly (its business-branch node
functions, wiring, checkpointer, and demo identity are reused wholesale,
not reimplemented) alongside `rag.*`.
"""
