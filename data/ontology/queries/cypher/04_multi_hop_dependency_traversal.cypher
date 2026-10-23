// Cypher counterpart of ../sparql/04_multi_hop_dependency_traversal.rq
// This is the real template rag.graph.queries uses for multi-hop
// dependency/impact questions (see _DEPENDENCY_REL_TYPES in
// src/rag/graph/queries.py): a bounded variable-length pattern,
// *1..N, with N a code-controlled constant, never caller input.
// Contrast with the SPARQL file's `+` property path, which is
// UNBOUNDED in hop count by default. Cypher's variable-length match
// requires an explicit upper bound to stay safe; SPARQL's `+` does
// not force one (though `{1,3}` bounded-repeat syntax is available
// there too). Neither language is inherently safer here: the
// discipline of bounding hop count has to be applied by the query
// author either way.
MATCH (root {name: "RAG Service", dataset_id: $dataset_id})-[:DEPENDS_ON*1..3]->(transitiveDependency)
RETURN DISTINCT transitiveDependency.name AS dependencyName
LIMIT 50
