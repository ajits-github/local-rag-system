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
make docs-build   # strict build to site/ -- fails on broken links/anchors/nav references
```

Without `make`:

```
mkdocs serve
mkdocs build --strict
```

`mkdocs build --strict` is also the documented way to detect broken
internal links or missing `mkdocstrings` references before publishing;
see [API Reference](../reference/index.md) for what's covered.
