# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder — resolve and install runtime dependencies into a venv.
# Kept separate from the runtime stage so build tooling (compilers, pip
# cache) never ships in the final image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential covers any transitive dependency that doesn't ship a
# manylinux wheel; harmless here since this whole stage is discarded after
# the venv is copied out below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Only what's needed to resolve + install the `rag` package: pyproject.toml
# for dependency metadata, src/ for the package itself. No tests/, scripts/,
# data/, experiments/ — those never influence `pip install .`.
COPY pyproject.toml ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# sentence-transformers pulls in torch as a transitive dependency; PyPI's
# default torch wheel bundles CUDA/cuDNN/NCCL (~4 GB) that config.yaml's
# embedding.device: cpu never uses. Installing the CPU-only build from
# PyTorch's own index *first* satisfies that same requirement before pip
# ever considers the GPU build, without changing anything about how the
# app runs (it was CPU-only already) -- see PROJECT_JOURNAL.md for the
# image-size finding this fixes.
#
# Plain `pip install .` after that — no [dev]/[ragas]/[mlflow]/[anthropic]/
# [cohere] extras. Those pull in pytest/ruff/mypy, RAGAS+datasets+openai,
# mlflow, and the Cohere SDK respectively; none are needed to serve the API.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: runtime — the image that actually ships.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user: least-privilege execution inside the container.
RUN groupadd -r rag && useradd -r -g rag -d /app -s /usr/sbin/nologin rag

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app \
    PYTHONPATH=/app/src

# App code + the config template. `pip install .` in the builder stage
# copies `rag` into site-packages (non-editable), which would make
# rag.config.REPO_ROOT (derived from __file__) resolve to somewhere under
# /opt/venv instead of /app -- breaking config/default.yaml and prompt-
# template path resolution. PYTHONPATH=/app/src makes this src/ copy
# shadow that site-packages one on sys.path, so REPO_ROOT resolves to
# /app here exactly as it resolves to the repo root under local dev's
# `pip install -e .` (see Setup step 1 in README.md).
COPY --chown=rag:rag src ./src
COPY --chown=rag:rag config ./config

# POST /ingest writes uploads here (see rag/api/routers/ingest.py). Whole
# tree (not just data/) is chowned since sentence-transformers downloads
# and caches the embedding model under $HOME/.cache/huggingface on first
# use — see PROJECT_JOURNAL.md for the offline-model-caching tradeoff this
# implies.
RUN mkdir -p /app/data/uploads && chown -R rag:rag /app

USER rag

EXPOSE 8000

# No curl in python:slim; urllib avoids adding a package just for this.
# 127.0.0.1, not localhost: this image's minimal network stack doesn't have
# IPv6 loopback configured, and "localhost" resolves to ::1 first, which
# then fails with "Cannot assign requested address" rather than falling
# back to IPv4.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
