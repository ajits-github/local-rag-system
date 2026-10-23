// Cypher counterpart of ../sparql/02_dependencies_of_service.rq
MATCH (service {name: "RAG Service", dataset_id: $dataset_id})-[:DEPENDS_ON]->(dependency)
RETURN dependency.name AS depName, labels(dependency) AS depLabels
LIMIT 50
