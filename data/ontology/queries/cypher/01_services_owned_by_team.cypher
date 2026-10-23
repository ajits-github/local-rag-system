// Cypher counterpart of ../sparql/01_services_owned_by_team.rq
// Note what Cypher does NOT need to express here that SPARQL does not
// either in this particular query: no type-hierarchy reasoning, since
// "owned by Workflow" needs no subclass knowledge. Compare query 04
// below, where the two really diverge.
MATCH (service:Service)-[:OWNED_BY]->(team:Team)
WHERE service.dataset_id = $dataset_id AND team.name = "Workflow"
RETURN service.name AS serviceName
ORDER BY serviceName
LIMIT 50
