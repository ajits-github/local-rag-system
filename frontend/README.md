# Local RAG Chat (frontend)

A small React + Vite + TypeScript chat UI over the existing FastAPI backend
(`/query`, `/agent/query`, `/agent/query/stream`). This is a frontend over
the proven backend, not a redesign of it: no RAG logic lives here, and no
backend code changes were needed to build it.

## Requirements

- Node.js 18.18+ or 20+ for `dev`/`build`/`lint`/`typecheck`/`test`
  (confirmed working on Node 18.12 in practice, despite `npm install`'s
  `EBADENGINE` warnings for `@typescript-eslint`). **Node 20+ is required**
  specifically for `npm run test:e2e`; Playwright refuses to launch at
  all below it (`npx playwright --version` errors outright, not just a
  warning).
- The backend running locally (`make up`, see the repo root README) if you
  want real answers rather than mocked ones.

## Running locally (no Docker)

```
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`. The Vite dev server proxies `/query`,
`/agent/*`, `/health`, `/metrics`, `/docs`, `/openapi.json` to
`http://localhost:8000` (see `vite.config.ts`). This keeps the browser
talking to a single origin, so the FastAPI backend never needs CORS
middleware added to it. Point the proxy at a different backend port with
`VITE_DEV_PROXY_TARGET` (see `.env.example`).

## Running via Docker

```
# from the repo root
make frontend-up      # backend + frontend
make observability-up # add Prometheus/Grafana/Jaeger too, independently
```

or directly:

```
docker compose -f docker-compose.yml -f docker-compose.frontend.yml up -d --build
```

Opens on `http://localhost:3001`. The frontend container's own nginx
serves the built SPA and reverse-proxies backend paths to `rag-api:8000`
(`nginx.conf`). This uses the same-origin design as the dev proxy, just
serving a static build instead. `RAG_API_BASE_URL` (unset by default) can
point the built frontend at a different backend origin instead of using
the built-in proxy. That mode requires CORS to be enabled on that backend,
which `config/default.yaml` does not do by default.

`make up` alone (backend only) is unaffected. The frontend is an
entirely separate, opt-in overlay, following the same pattern as
`docker-compose.observability.yml`.

## Backend configuration

The frontend has no say in which backend config file is loaded.
`RAG_CONFIG_PATH` (see the repo root README/CLAUDE.md) is a `rag-api`
concern. Point `rag-api` at an experiment config as usual and the frontend
will simply reflect whatever that config enables.

## Active feature flags

An always-visible pill row under the header (`FeatureFlagsBar`, never
collapsed, unlike every other panel) shows the connected backend's actual
active security/agent posture: Auth, Authorization, Redaction, Rate limit,
Agent, Vision provider, and Tracing. It is fetched from `GET /`'s
`features` block (`rag.api.main.FeatureFlags`) once on load, plus a manual
refresh button.
Booleans and provider names only, never a model name/host/secret. If
`security.auth.enabled` and `security.auth.insecure_dev_mode` are both on,
an additional "Dev mode" pill calls that out explicitly.

This exists because of a real incident: a Base64-obfuscated credential-
extraction prompt succeeded against whatever config happened to be
running, and nothing in the UI indicated that authorization/field
redaction were simply both off. The retrieval/redaction code itself was
correct, the active config just had every safety control disabled. The
bar makes that visible at a glance instead of by accident. See
`CLAUDE.md`'s "Web UI" section for the full writeup.

## Classic vs. Agentic RAG

- **Classic RAG** calls `POST /query`.
- **Agentic RAG** calls `POST /agent/query/stream` for live progress,
  rendering a collapsible "Agent activity" panel (step, tool name, elapsed
  time, retrieved chunk count, evidence-sufficiency, termination reason:
  the same safe fields `AgentEvent` exposes, nothing else. `AgentEvent` has
  no free-text field at all, so there is no reasoning/prompt/evidence text
  for the UI to accidentally render). If the backend has
  `observability.live_events.enabled: false`, the stream endpoint 404s and
  the UI falls back to the non-streaming `POST /agent/query` automatically,
  with a small notice that live progress isn't available.

Sources are rendered in a collapsible panel with content-type badges
(table/code/configuration/image/chart/prose), section path, page number,
and score, never a filesystem path (the API never returns one).

## Authentication and local development

The **Developer settings** panel (collapsed by default) lets you paste a
bearer token for testing an authenticated backend
(`security.auth.enabled: true`), or set `tenant_id`/`roles`/`as_of`/
`require_trust_level` for a backend running with JWT auth disabled. The
moment a token is entered, the tenant/roles fields grey out client-side.
The backend already ignores those two fields whenever a verified JWT
identity is present (`rag.api.request_auth.build_authorization_context`),
and the UI makes that visible rather than implying they still matter. This
panel is local-development tooling; there is no client-side authorization
of any kind. Every access decision is still made entirely by the
backend.

The token is kept in `sessionStorage` only (cleared when the tab closes),
never `localStorage`.

## Errors and safety states

Authentication failure, rate limiting, oversized-request rejection,
backend/Ollama unavailability, malformed responses, and a disconnected/
malformed event stream each render a distinct, honest error banner (see
`src/components/ErrorBanner.tsx`), never folded into a generic
successful-looking answer bubble. `insufficient_evidence` and max-step/
max-retry/max-tool-call terminations get their own inline notice above the
answer rather than being presented as an ordinary response.

## Known limitations

- No persistent multi-user chat history. Conversations live in
  in-browser state only and are lost on refresh (matches the milestone's
  explicit scope; the backend has no chat-history storage to persist to).
- The `RAG_API_BASE_URL` direct-origin mode is untested against a real
  CORS-enabled backend, since no such configuration exists in this repo
  yet. The documented, tested path is the same-origin proxy.
- Playwright e2e tests (`npm run test:e2e`) require Node 20+ to even
  launch; they were authored and type-checked but could not be executed
  in a Node 18 environment during development. `npm test` (Vitest) has no
  such constraint and was run successfully.

## Testing

```
npm run typecheck   # tsc -b --noEmit
npm run lint         # eslint
npm test              # vitest: unit + integration (mocked backend, jsdom)
npm run test:e2e       # playwright: a handful of critical flows (Node 20+)
```

Tests live under `frontend/tests/` (`unit/`, `integration/`,
`e2e/`), kept out of the repo's Python `tests/` tree.
