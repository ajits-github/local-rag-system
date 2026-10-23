// Cypher counterpart of ../sparql/03_policies_applying_to_service.rq
// Cypher has no owl:inverseOf axiom: GOVERNED_BY and the schema's own
// separate APPLIES_TO relationship type (rag.graph.schema.RelationshipType)
// are two independently-written edges. A caller who wants "either
// direction" must match both relationship types explicitly, every
// time, in every query that needs this. There is nothing declarative
// to fall back on the way there is in the ontology (see the SPARQL
// file's own comment on why its UNION becomes redundant post-inference).
MATCH (service)-[:GOVERNED_BY|APPLIES_TO]-(policy:Policy)
WHERE service.dataset_id = $dataset_id
RETURN service.name AS serviceName, policy.name AS policyName
LIMIT 50
