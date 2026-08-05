SHELL := bash

.PHONY: up down ingest query test

# Bring up Postgres+pgvector and initialize the schema.
up:
	docker compose up -d
	@echo "Waiting for Postgres to become healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' local-rag-postgres 2>/dev/null)" = "healthy" ]; do sleep 1; done
	python scripts/init_db.py

# Companion teardown to `up`.
down:
	docker compose down

# Ingest a file or directory: make ingest FILE=data/sample_docs DATASET_ID=sample_docs
# Add CLEAR=1 to wipe that dataset_id's existing documents/chunks first.
ingest:
	python -m rag.ingestion.pipeline $(FILE) --dataset-id $(DATASET_ID) $(if $(CLEAR),--clear,)

# Query the running API: make query Q="what is pgvector?"
query:
	curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "$(Q)"}'

# Unit tests always run; integration tests self-skip if Postgres/Ollama aren't up.
test:
	pytest tests/ -v
