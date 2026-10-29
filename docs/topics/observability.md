# Observability

--8<-- "README.md:docs-observability"

Role split: **OpenTelemetry** is per-request distributed tracing,
**Prometheus** is aggregate time-series metrics, **Grafana** visualizes
both, and **MLflow** (a separate, unrelated concern) tracks experiment
runs rather than live requests.

Full design writeup: [Observability](../architecture.md#observability).

API reference: [Observability](../reference/observability.md).

## Observability (observability milestone)

Adds operational telemetry above the bounded agentic workflow: per-node
timing (with a real-LLM-inference-vs-node-overhead split), OpenTelemetry
traces, Prometheus metrics, a local Grafana dashboard, and a safe
live-progress SSE stream. Deliberately does not change agent decision
logic, prompts, retrieval parameters, or model config. See
`docs/architecture.md`'s "Observability" section for the full design;
this is the condensed version. Role split: **OpenTelemetry** = per-request
distributed tracing; **Prometheus** = aggregate time-series metrics;
**Grafana** = visualization of both; **MLflow** (unchanged) = experiment-run
tracking, none of the four replaces another.

- **`src/rag/observability/`** (new package): `tracing.py`
  (`configure_tracing`/`start_span`/`set_attributes`, OTel SDK-touching
  code confined to this one module; disabled by default, the
  OpenTelemetry API's own no-op `TracerProvider` makes every `start_span`
  call free with no "is it enabled" branching needed at call sites) and
  `metrics.py` (a dedicated `CollectorRegistry`, never
  `prometheus_client`'s process-wide default, which would otherwise raise
  "Duplicated timeseries" on `pytest`'s routine module re-imports,
  holding every `rag_*` Counter/Histogram, each `observe_*`/`inc_*`
  function wrapped so a broken metric object can never fail a request).
- **Node timing wraps calls, doesn't touch node logic.** `_TimingLLM`
  (`agent/graph.py`) wraps the injected `LLM` once per run, accumulating
  `total_llm_ms` across every `generate()` call (including JSON-parse
  retries) without changing what's generated. `_call_node` wraps every
  node call in `run_agent()`'s dispatch, timing it, opening an OTel span,
  and, when a `_TimingLLM` is passed, splitting `llm_ms` (real
  inference) from `overhead_ms` (parsing/validation/template rendering).
  `AgentState.node_timings_ms: dict[str, list[NodeInvocationTiming]]`
  records every invocation (not just the latest, `tool_select`/
  `tool_execute`/`evidence_sufficiency` can run more than once per bounded
  retry loop); `AgentRunResult.node_timings_ms: dict[str, NodeTimingStats]`
  is the per-node-type aggregate. `AgentRunResult`'s pre-existing
  `retrieval_ms`/`generation_ms`/`total_ms` fields and formulas are
  byte-identical to before this milestone, purely additive, so
  `run_agent_eval.py`/`mlflow_logger.py`/`agent_query.py`'s response model
  needed zero changes (verified: every `run_agent()` call site uses
  keyword args).
- **OpenTelemetry spans**: one HTTP root span per request
  (`api/middleware.py:RequestIDMiddleware`, the same place that already
  computed `duration_ms` for the `request_handled` log line, no second
  middleware), an `agent_query` root span opened in `run_agent()`, one
  span per node (from the same `_call_node` wrapper that does timing),
  one span per tool name (nested under `tool_execute`, opened in
  `_execute_tool`), and a `retrieval` span inside
  `RetrievalPipeline._retrieve_timed` (covering both the classic `/query`
  path and the agent's `search_knowledge_base` tool, since both call
  `retrieve()` -> this method). `_retrieve_timed` itself was split into a
  thin span-opening wrapper plus `_retrieve_timed_inner` (the original
  logic, moved verbatim, not rewritten). Never attached to any span:
  JWTs, credentials, retrieved chunk text, raw prompts, or
  `decision.reasoning` (already discarded before reaching `AgentState`
  prior to this milestone; instrumentation only reads the specific fields
  node functions already extracted).
- **Prometheus metrics** (bounded labels only, no query text/tenant/
  document/chunk id/tool-arg ever): `rag_http_requests_total`/
  `rag_http_request_duration_seconds` (`method`/`path`/`status_code`,
  `path` = matched route template not raw URL), `rag_agent_requests_total`/
  `rag_agent_total_latency_seconds` (`route`), `rag_agent_steps`,
  `rag_agent_tool_calls_total`/`rag_agent_tool_latency_seconds`
  (`tool_name`), `rag_agent_node_latency_seconds` (total) and
  `rag_agent_node_llm_latency_seconds` (LLM-inference-only) both by
  `node`, `rag_retrieval_latency_seconds` (`provider`),
  `rag_agent_termination_reason_total`, `rag_agent_evidence_sufficiency_total`,
  `rag_errors_total` (`component`).
- **`GET /metrics`** (`api/routers/metrics.py`) and **`POST
  /agent/query/stream`** (`api/routers/agent_stream.py`, SSE) are both
  always-registered routes that check their own config flag
  (`observability.metrics.enabled`/`observability.live_events.enabled`)
  at request time and 404 when disabled, the same router-level-runtime-
  check pattern `security.dos_limits` already uses. `POST /agent/query`
  itself is untouched. The stream runs `run_agent` in Starlette's worker
  threadpool (`run_in_threadpool`), bridging events to the event loop via
  an `asyncio.Queue` + `call_soon_threadsafe`; `rag/agent/events.py`'s
  `AgentEvent` has no free-text field at all (structurally can't leak
  reasoning/evidence/credentials), and the stream's own internal
  "completed"/"terminated" `AgentEvent` is suppressed in favor of one
  richer, `AgentQueryResponse`-shaped terminal event built from the final
  `AgentRunResult`, never both. SSE was chosen over a WebSocket
  (one-directional progress is all that's needed) and POST over GET
  (matches `/agent/query`'s JSON-body shape; no browser frontend exists
  to need `EventSource`).
- **`GET /`** (`api/main.py`) is new: service name/status plus links to
  `/health`/`/docs`/`/metrics`, no dependency checks (that's `/health`).
  Also returns a `features` block (`FeatureFlags`: `auth_enabled`,
  `insecure_dev_mode`, `authorization_enabled`, `field_redaction_enabled`,
  `rate_limit_enabled`, `agent_enabled`, `vision_provider`,
  `tracing_enabled`), added after
  a real incident: querying an adversarial prompt against whatever config
  happened to be running (auth/authorization/field_redaction all off)
  produced a Base64-encoded credential in the answer, and nothing in the
  UI indicated *why*, the retrieval/redaction code was correct, the
  active config just had every safety control disabled. Booleans and
  provider names only, never a model name/host/connection string/JWT
  setting; `tests/unit/test_root_endpoint.py::test_root_features_never_leak_secrets_or_identifying_config`
  greps the raw response text for a denylist of such values as a
  regression guard, not just checking the known field names. The web
  UI's `FeatureFlagsBar` (`frontend/src/components/FeatureFlagsBar.tsx`)
  renders this as an always-visible pill row (not collapsed, unlike every
  other panel) fetched once on mount plus a manual refresh button, so an
  insecure demo config is never mistaken for a secure one just because
  the chat UI looks identical either way.
- **`GET /info`** (`api/main.py`) is a sibling to `GET /`, not an extension of it: `GET /`'s own
  `test_root_features_never_leak_secrets_or_identifying_config` test
  deliberately asserts no model name ever appears in that response, so a
  second `RuntimeInfo` model exposes the concrete pipeline identity
  `FeatureFlags` deliberately omits, `generation_provider`/
  `generation_model`, `embedding_provider`/`embedding_model`,
  `vectorstore_provider`, `retrieval_mode`, `sparse_retrieval_enabled`,
  `fusion_method` (`"rrf"` or `null`), `reranker_provider`/
  `reranker_enabled`, `agent_enabled`, `mcp_enabled`/`mcp_client_enabled`,
  `vision_provider`, `tracing_enabled`, `rate_limit_enabled`,
  `field_redaction_enabled`, `authorization_enabled`, `auth_enabled`.
  Still bound by the same rule as `FeatureFlags`: never a JWT/signing
  key, DB URL, filesystem path, internal token, raw auth context, prompt
  text, or tenant data, model/provider identifiers are the only "new"
  kind of information exposed relative to `GET /`. The web UI's
  `RuntimeInfoPanel` (`frontend/src/components/RuntimeInfoPanel.tsx`)
  renders this inside a collapsed-by-default "Runtime configuration"
  panel, fetched lazily on first expand (not eagerly on mount like
  `FeatureFlagsBar`, since this is opt-in developer detail, not
  always-visible security posture).
- **`config.observability`** (`ObservabilityConfig`, new top-level
  `AppConfig` field, sibling to `mlflow`): `metrics.enabled` (default
  `True`, in-process only, harmless unless scraped, same precedent as
  `MLflowConfig`), `tracing.enabled` (default `False`, needs a real
  OTLP endpoint to be useful, same no-op-by-default precedent as every
  security toggle), `live_events.enabled` (default `True`, a route
  only, no external dependency). `AppConfig.otlp_endpoint()` resolves
  `OTEL_EXPORTER_OTLP_ENDPOINT`, defaulting to `http://localhost:4318`
  (native dev; `docker-compose.observability.yml` overrides it to
  `http://jaeger:4318` for the container).
- **`docker-compose.observability.yml`** (new, layered on top,
  `docker compose -f docker-compose.yml -f docker-compose.observability.yml
  up -d` / `make observability-up`, never folded into `docker-compose.yml`,
  never brought up by plain `make up`): `jaeger` (`jaegertracing/all-in-one`,
  OTLP-native, no separate OTel Collector, one fewer moving part),
  `prometheus` (scrapes `rag-api:8000/metrics`), `grafana` (provisioned
  datasources + one dashboard,
  `observability/grafana/dashboards/agentic-rag-overview.json`). Bringing
  the stack up doesn't itself enable tracing, `observability.tracing.enabled`
  in `config/default.yaml` still gates the exporter.
- **Known limitation**: reusing Starlette's worker threadpool for the SSE
  endpoint's `run_in_threadpool` calls can occasionally produce a benign
  ERROR-level `opentelemetry.context` "Failed to detach context" log line
  when a pooled thread is reused across requests with a leftover span
  context, swallowed internally by OpenTelemetry (never raises, no test
  failures caused by it), but not eliminated as log noise. See
  `KNOWN_ISSUES.md`.


## Logging

`logging_config.configure_logging()` installs a JSON formatter and a
`request_id` contextvar. `api/middleware.py`'s `RequestIDMiddleware` sets
the request_id for the lifetime of each HTTP request so every log line
across a request can be correlated; `JSONFormatter` also stamps
`trace_id`/`span_id` at format time from whatever OTel span is
ambiently active (see "Observability" above for the tracing/metrics
stack itself, this section covers only the log lines it produces).
Logging a line while a span is still open (before that span's own
`span_cm.__exit__()`) is what makes the trace-id stamping actually find
it; logging after exit silently drops correlation for that one line,
a real bug once fixed in `middleware.py`'s `request_handled` log and
guarded against the same way in `agent/graph.py`'s
`agent_request_completed` log.

Every request gets exactly one `request_handled` line (`method`, `path`,
`route_name`, `status_code`, `duration_ms`). Every classic or agentic
run gets exactly one `agent_request_completed` line (`route`,
`termination_reason`, `step_count`, `tool_call_count`,
`retrieval_attempts`, `evidence_sufficient`, `duration_ms`, `success`),
emitted from `api/routers/query.py` for `classic_rag` and from
`agent/graph.py:run_agent`'s `_finish()` for the agent route, same event
name and field shape either way so a log consumer can filter one event
type regardless of which endpoint served the request. Every tool
dispatch, the in-process agent's local tools (`agent/graph.py:
_execute_tool`) and the MCP server's tools (`mcp/server.py`'s
`_run_tool`/`_run_business_tool`), gets one `tool_call_completed` line
(`tool_name`, `success`, `result_count`, `duration_ms`, `error_type`:
the exception's class name, never `str(exc)` or any argument/result
content). None of these lines ever include query text, answer text,
retrieved chunk content, prompts, or credentials, only shape/timing/
outcome fields, the same discipline `observability/metrics.py`'s
bounded labels already follow.

Built images run uvicorn with `--no-access-log` (`Dockerfile`,
`docker-compose.prod.yml`): its plain-text access line duplicates
`request_handled` field-for-field, just without `status_code`. The dev
compose command keeps it on for quick human-readable terminal feedback
while iterating locally; server lifecycle/error logs (`uvicorn`/
`uvicorn.error`) are unaffected either way.
