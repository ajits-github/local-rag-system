# LangGraph experiment (branch: `experiment/langgraph-agent`)

You're viewing the `experiment/langgraph-agent` branch, a self-contained
learning exercise, not the production system. For the production
`local-rag-system` README, see
[`main`](https://github.com/ajits-github/local-rag-system/blob/main/README.md).

![LangGraph single-agent experiment: classify, route to a read-only
retrieve/synthesize path or a tool-selection path, with a human-in-the-loop
approval interrupt before any business-write action, plus Postgres-backed
checkpointing for pause/resume/restart](langgraph_experiment/docs/assets/architecture-diagram.png)

A focused, isolated learning exercise: build a second, parallel agent
workflow for this repository's RAG/agentic-RAG system using LangGraph,
specifically to find out when LangGraph earns its keep over the existing
custom bounded agent harness (`src/rag/agent/graph.py`, left completely
unmodified by this package). This is **not** a migration; the production
agent is unaffected.

See [`langgraph_experiment/README.md`](langgraph_experiment/README.md) for
the full writeup: concepts demonstrated, run commands, a worked
human-in-the-loop approve/reject/restart-resume demo, a reasoned
comparison against the production harness, and an interview-question
reference.
