"""Operational telemetry: OpenTelemetry tracing + Prometheus metrics.

Deliberately separate from `rag.eval.mlflow_logger`: this package is
request-scoped, always-on-by-construction (defensive, disabled-by-default)
operational telemetry; MLflow is experiment-run-scoped tracking invoked
explicitly by `scripts/record_experiment.py`/`record_agent_experiment.py`.
Neither replaces the other. See `docs/architecture.md`'s "Observability"
section.
"""

from __future__ import annotations
