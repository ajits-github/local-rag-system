"""Experimental LangGraph agent workflow, isolated from the production custom harness.

This package exists to learn when LangGraph earns its keep over a plain
Python bounded loop (`rag.agent.graph.run_agent`, untouched). It is not a
migration and is not wired into `rag.api`/`rag.agent` in any way. See
`README.md` in this directory for the full write-up: architecture,
run commands, demo sequence, custom-harness comparison, and an interview
Q&A.

Runs in its own virtual environment (`.venv-langgraph/`, see README) rather
than the project's main `.venv`: `langgraph>=1.2` pins
`langchain-core>=1.4,<2`, which is incompatible with this repository's
pinned `langchain>=0.3,<0.4` (a real, confirmed dependency conflict, not a
style preference -- see README's "Dependency isolation" section). Code
here imports `rag.*` modules directly (the package is installed
editable, `--no-deps`, into that separate venv) but never the two modules
in `rag` that import `langchain` (`rag.chunkers.recursive_chunker`,
`rag.eval.ragas_adapters`); neither ingestion nor RAGAS evaluation is
exercised by this experiment.
"""
