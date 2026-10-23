// Cypher counterpart of ../sparql/05_runbooks_for_affected_services.rq
MATCH (incident:Incident)-[:AFFECTS]->(service)-[:HAS_RUNBOOK]->(runbook:Runbook)
WHERE incident.dataset_id = $dataset_id
RETURN incident.name AS incidentName, service.name AS serviceName, runbook.name AS runbookName
LIMIT 50
