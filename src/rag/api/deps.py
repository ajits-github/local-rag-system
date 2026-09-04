"""Builds pipeline objects once from config and shares them across requests.

Every getter is `lru_cache`d with no arguments, so each is a process-wide
singleton. Important on an 8GB-RAM CPU-only box where we don't want two
copies of the embedding model or two separate DB connection pools.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.applications import Starlette

from rag.api.auth import AuthenticationError, VerifiedIdentity, verify_jwt
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_llm, build_reranker, build_vectorstore
from rag.generation.base import LLM
from rag.ingestion.pipeline import IngestionPipeline
from rag.mcp.asgi import build_mcp_asgi_app
from rag.rerankers.base import Reranker
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore


@lru_cache
def get_config() -> AppConfig:
    """Return the process-wide `AppConfig` singleton.

    Loads `config/default.yaml` unless the `RAG_CONFIG_PATH` environment
    variable is set to a non-empty value, in which case that path is
    loaded instead. `load_config` itself fails loudly (`RuntimeError`) on
    a missing or unparseable override path; there is no fallback to
    `default.yaml` in that case.
    """
    override_path = os.environ.get("RAG_CONFIG_PATH")
    if override_path:
        return load_config(override_path)
    return load_config()


@lru_cache
def get_embedder() -> Embedder:
    """Return the process-wide `Embedder` singleton."""
    return build_embedder(get_config())


@lru_cache
def get_vectorstore() -> VectorStore:
    """Return the process-wide `VectorStore` singleton."""
    return build_vectorstore(get_config())


@lru_cache
def get_reranker() -> Reranker:
    """Return the process-wide `Reranker` singleton."""
    return build_reranker(get_config())


@lru_cache
def get_llm() -> LLM:
    """Return the process-wide `LLM` singleton."""
    return build_llm(get_config())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    """Return the process-wide `IngestionPipeline` singleton."""
    return IngestionPipeline(get_config(), vectorstore=get_vectorstore(), embedder=get_embedder())


@lru_cache
def get_retrieval_pipeline() -> RetrievalPipeline:
    """Return the process-wide `RetrievalPipeline` singleton."""
    return RetrievalPipeline(
        get_config(),
        vectorstore=get_vectorstore(),
        embedder=get_embedder(),
        reranker=get_reranker(),
        llm=get_llm(),
    )


@lru_cache
def get_mcp_asgi_app() -> Starlette | None:
    """Return the process-wide MCP ASGI app singleton, or `None` when `mcp.enabled=False`.

    The single source of truth for this object: `rag.api.main` mounts it
    (via `rag.mcp.asgi.mount_mcp_app`) and the agent's own MCP client
    (`rag.agent.mcp_client`, used for `get_customer_case`/
    `get_case_status` when `mcp.client.transport="asgi"`) binds an
    in-process ASGI transport directly to this same object, not a second,
    independently-constructed `MCPServer` whose session-manager lifespan
    `main.py`'s own lifespan context manager would never enter.
    """
    config = get_config()
    if not config.mcp.enabled:
        return None
    return build_mcp_asgi_app(config, get_retrieval_pipeline(), get_vectorstore(), get_embedder())


def get_current_identity(
    request: Request, config: AppConfig = Depends(get_config)
) -> VerifiedIdentity | None:
    """Resolve the caller's verified identity from the `Authorization` header.

    Returns `None` when `security.auth.enabled` is `False`. When enabled, a valid
    `Authorization: Bearer <jwt>` is required unless
    `security.auth.insecure_dev_mode` is `True` *and* no `Authorization`
    header was supplied at all; an invalid/expired/malformed/signature-
    mismatched token is always rejected with 401 regardless of that flag;
    there is never a silent fallback to unrestricted retrieval.

    Sets `request.state.identity` as a side effect so a later-evaluated
    rate-limit key function (see `get_rate_limiter`) can read it.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    config : AppConfig
        Application configuration.

    Returns
    -------
    VerifiedIdentity | None
        The verified caller identity, or `None` when auth is disabled.

    Raises
    ------
    HTTPException
        401, when a token is required but missing/invalid.
    """
    if not config.security.auth.enabled:
        request.state.identity = None
        return None

    header = request.headers.get("authorization")
    if header is None:
        if config.security.auth.insecure_dev_mode:
            request.state.identity = None
            return None
        log_audit_event("auth_failure", reason="missing_token")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        log_audit_event("auth_failure", reason="malformed")
        raise HTTPException(status_code=401, detail="Malformed Authorization header")

    try:
        identity = verify_jwt(token, config)
    except AuthenticationError as exc:
        log_audit_event("auth_failure", reason=exc.reason)
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    log_audit_event(
        "auth_success",
        subject=pseudonymous_subject(identity.subject),
        tenant_id=identity.tenant_id,
    )
    request.state.identity = identity
    return identity


def _rate_limit_key(request: Request) -> str:
    """Bucket rate-limit counters by tenant (if identified), else client IP."""
    identity: VerifiedIdentity | None = getattr(request.state, "identity", None)
    if identity is not None and identity.tenant_id is not None:
        return f"tenant:{identity.tenant_id}"
    return f"ip:{get_remote_address(request)}"


@lru_cache
def get_rate_limiter() -> Limiter:
    """Return the process-wide `slowapi.Limiter` singleton (in-memory backend).

    Bucketed per tenant when a verified identity is present, else per
    client IP. `enabled` is
    read from config once at construction time (no hot-reload, matching
    every other config-derived singleton in this module).
    """
    return Limiter(key_func=_rate_limit_key, enabled=get_config().security.rate_limit.enabled)
