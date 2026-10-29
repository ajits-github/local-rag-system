# Deployment & Runtime

Postgres+pgvector and the API are containerized; Ollama stays native on
the host (no GPU passthrough from Docker Desktop for Windows to a Linux
container in this setup) and the containerized API reaches it through
`host.docker.internal`. Ollama is a soft dependency (the container starts
and reports degraded health without it); Postgres is a hard dependency.

--8<-- "README.md:docs-prereq-setup"

--8<-- "README.md:docs-containerized-dev"

## Building the documentation site

Two local commands are available (see the repository root `Makefile`):

```
make docs-serve   # live-reloading local docs server at http://127.0.0.1:8000
make docs-build   # strict build to site/, failing on broken links/anchors/nav references
```

Without `make`:

```
mkdocs serve
mkdocs build --strict
```

`mkdocs build --strict` is also the documented way to detect broken
internal links or missing `mkdocstrings` references before publishing;
see [API Reference](../reference/index.md) for what's covered.

## Containerized runtime

- Postgres+pgvector and `rag-api` run in containers; Ollama stays **native
  on the Windows host** (not containerized). Docker Desktop for Windows
  has no GPU passthrough to a Linux container in this setup, and native
  Ollama is what the rest of local dev already assumes. The API reaches
  it via `host.docker.internal:11434`, which `docker-compose.yml` maps
  from `CONTAINER_OLLAMA_BASE_URL`. Ollama is a *soft* dependency: the
  container starts without it and `/health` reports degraded status
  rather than refusing to boot. Postgres is a *hard* dependency (`depends_on:
  condition: service_healthy`).
- `Dockerfile` is multi-stage: a `builder` stage installs the CPU-only
  torch wheel (`--index-url https://download.pytorch.org/whl/cpu`) *before*
  `pip install .`, otherwise `sentence-transformers`' transitive torch
  dependency pulls the ~4GB CUDA/cuDNN/NCCL build even though
  `config/default.yaml`'s `embedding.device: cpu` never uses a GPU (this
  dropped the image from 5.6GB to 1.6GB). The `runtime` stage copies only
  the venv + `src/` + `config/`, runs as a non-root `rag` user, and sets
  `PYTHONPATH=/app/src`, required because `pip install .` (non-editable)
  copies `rag` into site-packages, which breaks `rag.config.REPO_ROOT`'s
  `Path(__file__).resolve().parents[2]` computation unless the `src/`
  copy shadows it on `sys.path`. The `HEALTHCHECK` hits
  `http://127.0.0.1:8000/health`, not `localhost`, this minimal image
  has no IPv6 loopback, and `localhost` resolving to `::1` first makes
  the healthcheck fail even though the app is fine.
- `docker-compose.yml` is the dev stack (bind-mounts `src/`/`config/` for
  `uvicorn --reload`, Postgres published on host port 15987).
  `docker-compose.prod.yml` layers on top (`-f docker-compose.yml -f
  docker-compose.prod.yml`) for an immutable-image, no-bind-mount,
  DB-not-published-to-host run, still same-host, not a Kubernetes
  substitute. Its `volumes`/`ports` overrides use the Compose 2.24+
  `!override` YAML tag deliberately: Compose merges list-type keys like
  `volumes`/`ports` across `-f` files *by target* rather than replacing
  them, so a plain `ports: []` would silently leave the dev file's
  `15987:5432` publish in place. Verify any future override with `docker
  compose -f ... -f ... config` (prints the fully-merged YAML) rather than
  assuming the override took effect.
- `make up` brings up **both** `postgres` and `rag-api` (it's a bare
  `docker compose up -d`), then waits on Postgres's healthcheck and runs
  `scripts/init_db.py`. `scripts/smoke_test_containers.sh` is the
  end-to-end check against a running stack: waits for `/health`, ingests
  `data/sample_docs/password-policy.md` under a timestamped `dataset_id`,
  queries it, and asserts a non-empty answer.
- RAGAS, MLflow, and the OpenAI/Anthropic/Cohere SDKs' extras are
  deliberately **not** installed in the image (`pip install .` with no
  extras), those are eval/experiment-tooling concerns (see "Evaluation
  and experiment tracking" below), not runtime serving concerns. Redis and
  a separate ingestion worker were considered and deliberately left out
  (no concrete use case yet); don't add them speculatively.
- Known gap: the embedding model downloads from Hugging Face Hub on first
  use inside a *fresh* container (no pre-baked model cache), so the
  containerized path needs outbound internet once, unlike native dev
  where the model is usually already cached under `~/.cache/huggingface`.
  See `PROJECT_JOURNAL.md`'s containerization entry for the full
  bug list (image size, `REPO_ROOT`/`PYTHONPATH`, healthcheck IPv6,
  Compose merge semantics) and evidence for each.
