# Knowledge Graph and GraphRAG

--8<-- "README.md:docs-graphrag"

The graph uses the same security checks as the rest of the system. Tenant
and role checks are applied in the Cypher query and again in Python on every
returned fact. See [Security](security.md) for the other controls, and
[Agentic RAG](agentic-rag.md) for how the graph tools fit into the agent loop.

The full design, schema, query guardrails, fusion strategies, failure
behavior, and verification status are in [`docs/graphrag.md`](../graphrag.md).

API reference: [Knowledge Graph / GraphRAG](../reference/graph.md).
