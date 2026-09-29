"""`GET /health`, `GET /livez`, `GET /readyz`: dependency and process health.

Three distinct questions, not one endpoint wearing three hats:

- `/health`: detailed diagnostics for a human or dashboard. Always 200;
  degradation is reported in the response body, never the status code.
- `/livez`: is this process alive and able to handle a request at all?
  Never checks downstream dependencies -- a Postgres/Ollama outage must
  never cause a liveness failure, since that would make Kubernetes
  restart a container that isn't actually broken, turning one
  dependency's outage into a restart storm of otherwise-healthy pods.
- `/readyz`: can this Pod serve a normal request right now? A normal
  `/query` request needs both the vectorstore (retrieval) and the LLM
  (generation) to be reachable, so either one being down means this Pod
  should stop receiving traffic (non-2xx) until it recovers -- without
  the container itself being restarted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response

from rag.api.deps import get_llm, get_vectorstore
from rag.generation.base import LLM
from rag.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/health")
def health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLM = Depends(get_llm),
) -> dict[str, Any]:
    """Report overall status plus each dependency's individual reachability.

    Parameters
    ----------
    vectorstore : VectorStore
        Injected vector store singleton.
    llm : LLM
        Injected LLM singleton.

    Returns
    -------
    dict[str, Any]
        ``{"status": "ok" | "degraded", "dependencies": {...}}``.
    """
    db_ok = vectorstore.health_check()
    llm_ok = llm.health_check()
    return {
        "status": "ok" if (db_ok and llm_ok) else "degraded",
        "dependencies": {
            "vectorstore": "ok" if db_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
        },
    }


@router.get("/livez")
def livez() -> dict[str, str]:
    """Report whether this process is alive, independent of any dependency.

    Deliberately takes no dependency injection at all -- there is nothing
    here that a downstream outage could make fail. Kubernetes should use
    this for `startupProbe`/`livenessProbe`: a failure here means the
    process itself is unresponsive and a restart may help, which is not
    true of a Postgres or Ollama outage.

    Returns
    -------
    dict[str, str]
        ``{"status": "alive"}``, always, with a 200 status code.
    """
    return {"status": "alive"}


@router.get("/readyz")
def readyz(
    response: Response,
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLM = Depends(get_llm),
) -> dict[str, Any]:
    """Report whether this Pod should currently receive normal request traffic.

    Both the vectorstore and the LLM are required to serve a normal
    `/query` request end to end (retrieval, then generation), so either
    being unreachable sets a non-2xx status code -- Kubernetes should use
    this for `readinessProbe`, which removes the Pod from Service traffic
    without restarting the container.

    Parameters
    ----------
    response : Response
        Injected by FastAPI; used to set a non-2xx status code without
        raising, so the body still reports which dependency is down.
    vectorstore : VectorStore
        Injected vector store singleton.
    llm : LLM
        Injected LLM singleton.

    Returns
    -------
    dict[str, Any]
        ``{"status": "ready" | "not_ready", "dependencies": {...}}``, with
        `response.status_code` set to 503 whenever not ready.
    """
    db_ok = vectorstore.health_check()
    llm_ok = llm.health_check()
    ready = db_ok and llm_ok
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "not_ready",
        "dependencies": {
            "vectorstore": "ok" if db_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
        },
    }
