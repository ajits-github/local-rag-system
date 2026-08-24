SHELL := bash

.PHONY: up down ingest query test observability-up observability-down frontend-up frontend-down docs-serve docs-build

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

# Bring up Prometheus + Grafana + Jaeger alongside the base stack (opt-in;
# never brought up by plain `make up`). Enable tracing by setting
# observability.tracing.enabled: true in config/default.yaml first.
observability-up:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
	@echo "Prometheus: http://localhost:9090  Grafana: http://localhost:3000  Jaeger: http://localhost:16686"

# Companion teardown to observability-up.
observability-down:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml down

# Bring up the web UI alongside the base stack (opt-in; never brought up by
# plain `make up`).
frontend-up:
	docker compose -f docker-compose.yml -f docker-compose.frontend.yml up -d --build
	@echo "Frontend: http://localhost:3001"

# Companion teardown to frontend-up.
frontend-down:
	docker compose -f docker-compose.yml -f docker-compose.frontend.yml down

# Local docs server with live reload: http://127.0.0.1:8000. Requires
# `pip install -e .[docs]`.
docs-serve:
	mkdocs serve

# Strict docs build to site/; fails on broken nav references, missing
# mkdocstrings targets, or broken internal links/anchors (htmlproofer).
docs-build:
	mkdocs build --strict
