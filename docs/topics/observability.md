# Observability

--8<-- "README.md:docs-observability"

Role split: **OpenTelemetry** is per-request distributed tracing,
**Prometheus** is aggregate time-series metrics, **Grafana** visualizes
both, and **MLflow** (a separate, unrelated concern) tracks experiment
runs rather than live requests.

Full design writeup: [Observability](../architecture.md#observability).

API reference: [Observability](../reference/observability.md).
