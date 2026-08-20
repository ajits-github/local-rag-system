"""`GET /metrics`: Prometheus text exposition of every metric in `rag.observability.metrics`."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from rag.api.deps import get_config
from rag.config import AppConfig
from rag.observability.metrics import CONTENT_TYPE_LATEST, render_metrics

router = APIRouter()


@router.get("/metrics")
def metrics(config: AppConfig = Depends(get_config)) -> Response:
    """Return Prometheus text exposition, or 404 when metrics are disabled.

    Parameters
    ----------
    config : AppConfig
        Application configuration; checked at request time (not baked
        into route registration) so tests can flip
        `observability.metrics.enabled` via `dependency_overrides`
        without rebuilding the app, matching this codebase's DoS-limits
        router-level-check convention.

    Returns
    -------
    Response
        Prometheus exposition-format text, or a 404 when disabled.
    """
    if not config.observability.metrics.enabled:
        raise HTTPException(status_code=404, detail="Metrics are disabled")
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
