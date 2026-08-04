from __future__ import annotations

from fastapi import FastAPI

from rag.api.deps import get_config
from rag.api.middleware import RequestIDMiddleware
from rag.api.routers import health, ingest, query
from rag.logging_config import configure_logging

_config = get_config()
configure_logging(_config.app.log_level)

app = FastAPI(
    title=_config.app.name,
    description="Modular, config-driven local RAG system.",
)
app.add_middleware(RequestIDMiddleware)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
