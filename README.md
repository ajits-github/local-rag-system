# Multi-agent LangGraph experiment (branch: `experiment/multi-agent`)

You're viewing the `experiment/multi-agent` branch, a self-contained
learning exercise, not the production system. For the production
`local-rag-system` README, see
[`main`](https://github.com/ajits-github/local-rag-system/blob/main/README.md).

![Multi-agent LangGraph experiment: a Coordinator/Router dispatches to a
Knowledge Agent, a Business Agent, or both in parallel for a mixed query,
merges their results, and synthesizes one final answer with citations and
business context](multi_agent_experiment/docs/assets/architecture-diagram.png)

A focused, isolated learning exercise: build a small **multi**-agent
LangGraph workflow (a Coordinator/Router plus two genuinely distinct
specialist agents) on top of this repository's RAG/agentic-RAG system,
and compare it honestly against the production custom harness
(`src/rag/agent/graph.py`) and the single-agent LangGraph experiment
(`langgraph_experiment/`). Neither is modified by this package.

See [`multi_agent_experiment/README.md`](multi_agent_experiment/README.md)
for the full writeup: concepts demonstrated, the real LangGraph join-node
bug found and fixed while building this, the deterministic evaluation
harness, a three-way comparison against the other two agent
implementations, and an interview-question reference.
