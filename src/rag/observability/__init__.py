"""Operational telemetry: OpenTelemetry tracing + Prometheus metrics.

Request-scoped and disabled-by-default where it matters (tracing), unlike
`rag.eval.mlflow_logger`'s experiment-run-scoped tracking; neither replaces
the other.
"""

from __future__ import annotations
