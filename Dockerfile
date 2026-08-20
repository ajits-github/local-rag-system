# syntax=docker/dockerfile:1

# Build runtime dependencies in a separate stage.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only torch before sentence-transformers resolves it.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Runtime image.
FROM python:3.11-slim AS runtime

RUN groupadd -r rag && useradd -r -g rag -d /app -s /usr/sbin/nologin rag

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app \
    PYTHONPATH=/app/src

# Keep app source and config under /app so repo-root-relative paths resolve.
COPY --chown=rag:rag src ./src
COPY --chown=rag:rag config ./config

# POST /ingest writes uploads here; model caches also live under $HOME.
RUN mkdir -p /app/data/uploads && chown -R rag:rag /app

USER rag

EXPOSE 8000

# Use Python for healthcheck so the runtime image does not need curl.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
