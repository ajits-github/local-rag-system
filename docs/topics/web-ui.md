# Web UI

--8<-- "README.md:docs-web-ui"

![Classic RAG answer with the feature-flags bar and expanded source citations](../assets/web-ui-screenshot.png)

*A real `POST /query` round trip against the sample corpus: the feature-flags
bar confirms which security controls are active, and the sources panel shows
the actual retrieved chunks with scores and section paths.*

--8<-- "README.md:docs-web-ui-setup"

Architectural notes for the same-origin proxy design, POST-based event
stream parser, live-events-disabled fallback, and error taxonomy are in
`CLAUDE.md`'s "Web UI" section. Full setup, authentication behavior, and
known limitations are in `frontend/README.md` in the repository. The
frontend has its own `package.json` and tests; it is not part of the
Python package this reference section documents.

## Web UI (web-ui milestone)

A React/Vite/TypeScript chat interface (`frontend/`) over the existing
API: choose Classic (`POST /query`) or Agentic (`POST /agent/query/stream`,
falling back to `POST /agent/query`) RAG, watch live agent progress,
inspect sources/citations, and optionally set local-development identity.
A frontend over the proven backend, not a redesign of it, no RAG logic
lives in `frontend/`, and no backend Python source changed to build it.
See `frontend/README.md` for the user-facing setup/behavior writeup; this
section is the architectural record.

- **Same-origin by design, not CORS.** No CORS middleware exists anywhere
  in the backend (confirmed by inspection before design, not assumed), and
  none was added. The frontend always talks to the API same-origin
  instead: `vite.config.ts`'s dev-server `server.proxy` locally, and
  `frontend/nginx.conf` (the built image's own nginx) in Docker, both
  proxy `/query`, `/agent/*`, `/feedback`, `/info`, `/health`, `/metrics`,
  `/docs`, `/openapi.json` to the real backend (`/info`/`/feedback` were
  a real gap, fixed later. See the "Reverse-proxy gap"
  note under "Developer/demo UI polish" below; both are proxied in both
  places now). `window.__RAG_API_BASE__`
  (`frontend/index.html`, rewritten at container start by
  `frontend/docker-entrypoint.sh` from a `RAG_API_BASE_URL` env var) is an
  escape hatch for pointing the built frontend at a different origin
  directly; that path is documented as needing CORS enabled on that
  backend (not implemented) and is untested.
- **`POST /agent/query/stream` is POST, not GET** (see that router's own
  docstring for why), so the browser's native `EventSource` can't consume
  it. `frontend/src/utils/sse.ts`'s `readSseFrames` hand-parses
  `event:`/`data:` frames off the `fetch` response's `ReadableStream`
  instead. The stream's terminal `completed`/`terminated` frame carries the
  same shape as `POST /agent/query`'s JSON response (by the backend's own
  design, see `agent_stream.py:_final_response`), so Agentic mode issues
  exactly one request, never a stream-for-progress-then-fetch-for-the-
  answer pair.
- **Live-events-disabled fallback.** When
  `config.observability.live_events.enabled` is `False`, the stream
  endpoint 404s. `frontend/src/api/agentQuery.ts:streamAgentQuery` surfaces
  that as a distinct `RagApiError` kind (`not_found`); `useSendMessage`
  catches specifically that kind and retries via the non-streaming
  `POST /agent/query`, so Agentic mode still functions (just without the
  live panel) rather than erroring out.
- **`AgentEvent`'s no-free-text-field guarantee is inherited, not
  re-implemented.** `AgentActivityPanel` renders only the typed fields
  `AgentEvent` (`rag/agent/events.py`) already exposes (event type, step,
  tool name, elapsed ms, retrieved chunk count, evidence-sufficient,
  termination reason, route), there is no code path in the component
  that could surface a field the backend doesn't already consider safe.
  `tests/unit/agentActivityPanel.test.tsx` proves this directly: an event
  object carrying extra `reasoning`/`raw_prompt` keys (simulating a
  buggy/compromised backend) renders with neither string appearing in the
  DOM.
- **Response validation via zod, not `any`.** Every API response
  (`frontend/src/api/types.ts`'s `QueryResponseSchema`/
  `AgentQueryResponseSchema`/`AgentEventSchema`, mirroring the Pydantic
  models in `api/routers/query.py`/`agent_query.py`/`agent/events.py`) is
  parsed through a zod schema; a shape mismatch becomes a distinct
  `malformed_response` error state (`ErrorBanner`) instead of a runtime
  crash or a silently-wrong render.
- **Error taxonomy matches what the backend actually returns.**
  `frontend/src/api/client.ts` classifies by HTTP status/body shape:
  401 -> `authentication_failed`, 429 -> `rate_limited` (both DoS-limit
  422s and FastAPI's list-shaped validation-error 422s) -> `validation_failed`,
  404 -> `not_found` (the stream-disabled case above), a `fetch` rejection
  -> `backend_unavailable`, a zod parse failure -> `malformed_response`.
  There is deliberately no distinct "authorization failed" HTTP error
  handled, because none exists: document-level ACL is a silent SQL filter
  (`retrieval/authorization.py`), so an unauthorized/insufficiently-
  evidenced query surfaces as empty/thin `sources` or
  `termination_reason: insufficient_evidence`, not an HTTP error,
  `MessageBubble` renders that as its own "insufficient evidence" notice,
  never a generic successful-looking answer.
- **Dev identity precedence is mirrored client-side, not re-decided.**
  `DevIdentityPanel` disables the tenant/roles inputs the moment a bearer
  token is entered, matching `request_auth.build_authorization_context`'s
  actual precedence (a verified JWT identity always wins over
  body-supplied `tenant_id`/`roles`), the frontend has no independent
  opinion about authorization, it just avoids implying a field still has
  effect once the backend would ignore it. The token lives in
  `sessionStorage` only.
- **`DevIdentity` also carries a `datasetId`, independent of
  authentication.** Unlike `bearerToken`/`tenantId`/`roles`, this field
  is a retrieval filter, not an identity claim, `buildQueryRequestBody`/
  `buildAgentQueryRequestBody` (`api/query.ts`/`agentQuery.ts`) merge it
  into `filters.dataset_id` whenever it's set, regardless of whether a
  bearer token is present. Added because the manual/dev query path had
  no way to scope retrieval to one ingestion namespace: a shared dev
  Postgres instance routinely holds several unrelated ingestions (the
  real corpus, eval/experiment copies) at once, and leaving this blank
  searches all of them together (see CLAUDE.md's "Document identity"
  section for `dataset_id` itself), `DevIdentityPanel` shows an explicit hint
  rather than leaving that scope silently ambiguous.
- **`docker-compose.frontend.yml`** is an additive overlay (same pattern as
  `docker-compose.observability.yml`), never folded into `docker-compose.yml`
  and never brought up by plain `make up`. `frontend/Dockerfile` is
  multi-stage: `node:20-alpine` builds the static bundle, `nginx:1.27-alpine`
  serves it. `make frontend-up`/`make frontend-down` are the Makefile
  companions.
- **Tests**: `frontend/tests/unit` (component/reducer/client logic,
  mocked `fetch`), `frontend/tests/integration` (full `ChatWindow` flows:
  classic query, agent streaming, error states, auth header handling,
  mode switching, malformed/disconnected stream), `frontend/tests/e2e`
  (Playwright, three flows against a real browser with the backend mocked
  at the network layer via `page.route`), kept under `frontend/tests/`,
  not mixed into the Python `tests/` tree. Playwright requires Node 20+ to
  launch at all (a hard requirement, not a soft warning); it was authored
  and type-checked but not executed end-to-end in this project's Node 18
  dev environment, see `frontend/README.md`'s "Known limitations".

### Developer Settings polish: Apply workflow, JWT countdown, debug cards

This closes three UX gaps in
the developer-facing tooling above, none of which touch retrieval, agent
decision logic, or security policy.

- **Apply-settings workflow.** `DevIdentityPanel` no longer commits a
  field on every keystroke. It edits a local draft and only calls its
  `onChange` (which is what actually persists to `sessionStorage` and
  reaches the next request) from a `handleApply` submit handler, after
  `validateDraft` passes, a structurally-invalid bearer token (see
  below) or an unparseable `as_of` date is rejected inline, tied to its
  field via `aria-describedby`/`role="alert"`, and never applied. A
  successful apply collapses the panel into a `DevSessionSummary`
  (decoded tenant/role for a token, as-of, dataset, trust level, and a
  countdown) with an **Edit settings** action; `DevIdentityPanel` is now
  `forwardRef`-wrapped (`DevIdentityPanelHandle.openForEdit`) so a
  sibling component (the header badge's "Replace token" button) can
  reopen it without lifting its `expanded` state to the parent.
- **Client-side-only JWT countdown, never authentication.**
  `frontend/src/utils/jwt.ts` decodes a token's header/payload
  (structural check only, no signature verification) and extracts a
  fixed allowlist of display claims (`exp`/`sub`/`tenant_id`/`roles`),
  deliberately never the raw payload, so no other claim a token happens
  to carry can leak into the UI. `hooks/useTokenStatus.ts` derives a
  `none`/`invalid`/`no_exp`/`healthy`/`expiring_soon` (<=10min)/
  `critical` (<=2min)/`expired` state machine from it, ticking once per
  second only while a real `exp` claim is present (an interval that
  starts/stops with `exp`, resyncing `now` immediately on that
  transition rather than waiting for the first tick, since `now`'s
  `useState` initializer only runs once at mount and would otherwise be
  stale relative to a token applied later). `components/TokenStatusBadge.tsx`
  renders this in the header (`Authenticated · 47:32`, icon+text per
  state so nothing depends on color alone) and a visually-hidden
  `aria-live="polite"` region that announces only on a state
  *transition*, not every countdown tick. Nothing here ever mints or
  refreshes a token; backend JWT verification (`api/auth.py:verify_jwt`)
  remains the only thing that actually accepts a request.
- **Richer, sectioned per-request debug UI.** `DebugPanel` is now four
  labeled `<section aria-label=...>` blocks (Request/Pipeline/Timings/
  Tools) instead of one flat `<dl>`, and an agentic run's Tools section
  renders a `ToolCallCard` per dispatch (name, `Local`/`MCP remote`,
  `Success`/`Failed`, duration), never args or raw error text, matching
  the same restraint `tool_call_completed`'s audit log already applies.
  This required two small, additive backend fields: `SourceItem.origin`
  (`api/routers/query.py`; mirrors `SearchResult.origin`, which
  `source_dict()` already computed but the API model silently dropped
  since it wasn't declared, now flows straight through for both
  `/query` and `/agent/query`'s citations via `Citation.origin`,
  `agent/state.py`) and `AgentQueryResponse.tool_call_details`
  (`api/routers/agent_query.py`'s `ToolCallDetail` model +
  `_tool_call_details()`, mapping `ToolCallRecord` -> name/execution/
  success/result_count/latency_ms, with `execution` derived from the
  existing `REMOTE_MCP_TOOL_NAMES` lookup rather than a second,
  independently-maintained list; `tool_calls`, the bare name list, is
  unchanged for compatibility). `agent_stream.py`'s SSE terminal frame
  reuses the same `_tool_call_details()` helper, so it matches
  `/agent/query`'s JSON shape exactly. `SourcesPanel` uses the same
  `origin` field to badge each source `RAG`/`RAG (related)`/
  `RAG (tool)`/`MCP remote`, so an MCP business-tool result is never
  visually indistinguishable from a local knowledge-base one.
- **`RuntimeInfoPanel` now fetches on mount**, not lazily on first
  expand as originally designed, a deliberate reversal, since the
  panel's collapsed toggle now doubles as a compact
  `qwen2.5:3b · Hybrid · RRF · MCP` badge (`compactBadge()` in
  `RuntimeInfoPanel.tsx`), which needs the data to render at all. Same
  "always know what the connected backend is actually running"
  rationale `FeatureFlagsBar` already uses for its own eager fetch. The
  detailed field-by-field `<dl>` still stays behind the toggle.
- **Testing gotcha worth remembering**: `@testing-library/user-event`
  interactions hang indefinitely under `vi.useFakeTimers()` unless the
  timers are created with `{ shouldAdvanceTime: true }` (bridges real
  wall-clock time into the fake clock so user-event's internal delays
  resolve). Once real time is allowed to advance under fake timers, a
  `Math.floor(remainingMs / ...)`-based countdown can legitimately roll
  over by one unit between minting a test token and asserting on it,
  `devIdentityPanel.test.tsx`'s summary test asserts the format
  (`/^\d+m$/`) rather than an exact value for that reason, while every
  fully-frozen-time test (`useTokenStatus.test.ts`,
  `tokenStatusBadge.test.tsx`) still asserts exact values.

### Production vs. developer UI mode

One frontend codebase, two UI modes, selected by a single build-time env
var: `src/config/uiMode.ts`'s `getUiModeConfig()` resolves `VITE_UI_MODE`
("developer" | anything else) into four booleans
(`showDeveloperSettings`/`showDebugDetails`/`showRuntimeDetails`/
`showDeveloperModeIndicator`). **The safe default is `production`**,
unset, empty, or a typo'd value all resolve to it, never to
`"developer"`; only the exact literal string `"developer"` opts in. This
is a display flag only, never a security boundary: every control it
gates is independently enforced by the backend regardless (JWT
verification, tenant/role authorization, field redaction), so hiding a
control here changes what the browser *shows*, never what the backend
*allows*.

Gating decisions live in the few composing container components, not
scattered through leaf components: `ChatWindow` decides whether to mount
`DevIdentityPanel`/`FeatureFlagsBar`/`RuntimeInfoPanel`
(`showDeveloperSettings`/`showRuntimeDetails`) and renders the
"Developer Mode" header badge (`showDeveloperModeIndicator`);
`MessageBubble` decides whether to mount the per-answer `DebugPanel` and
passes `showScore={showDebugDetails}` into `SourcesPanel` (the one leaf
component with a partial gate, since the citation list itself stays
visible in both modes, only the raw relevance score is
developer-only). `ErrorBanner`'s `authentication_failed` copy also reads
this flag, so its "Check the bearer token in Developer settings" detail
sentence only appears when that panel actually exists.
`AgentActivityPanel` (the live per-step progress panel during agent
streaming) is deliberately **not** gated, it's the visible form of the
"useful loading state" for Agentic RAG, a keep-item in either mode, not
a post-hoc debug artifact like `DebugPanel`'s tool-call cards.

`frontend/Dockerfile`'s `ARG VITE_UI_MODE=production` means a plain
`docker build` with no `--build-arg` always ships production mode;
`docker-compose.frontend.yml` forwards a `VITE_UI_MODE` shell variable as
that build arg (same `production` default when unset). Local `npm run
dev`/`npm run build` read it from a gitignored `frontend/.env` the same
way every other `VITE_*` value in `.env.example` already does. See
`frontend/README.md`'s "UI modes" section for the full reference
(configuration, exact developer/production behavior, build commands).

### Developer/demo UI polish: neutral unavailable states, compact session summary, empty-state examples

A follow-up pass (`24.0-UI-improvement`) closed a handful of UX rough
edges the previous milestone's screenshot exposed, none touching
retrieval/agent/MCP behavior or backend auth.

- **`RuntimeInfoPanel`'s unreachable state stopped looking like a broken
  app.** The collapsed toggle used to read "Runtime configuration
  (unavailable)" and the expanded body "Runtime configuration
  unavailable" in `--error-text` red (`.runtime-info__status--error`/
  `.runtime-info__badge-error`, both now removed), alarming for a
  panel that describes the pipeline, not the chat itself. Both now read
  the neutral "Runtime info unavailable" (toggle) / "Runtime info
  unavailable. Chat still works normally." (expanded body, `role=
  "status"`, default muted text color), and the panel stays collapsible
  with its existing Refresh button as the retry action, no hook
  changes, presentation only.
- **Feature-flag pills stopped relying on color alone.**
  `FeatureFlagsBar`'s boolean pills previously differed only by border
  color and a dot's color/opacity (`feature-pill--on`/`--off`), no
  textual difference at all, failing "status text in addition to color"
  outright. Each pill now adds a visible "✓" (`.feature-pill__check`,
  `aria-hidden` since it's decorative next to the text) when enabled,
  plus an `sr-only` "enabled"/"disabled" word regardless of state (the
  existing `title` tooltip alone wasn't enough, not always readable
  without hovering). `Vision: none` (the backend's literal value when no
  vision provider is configured) renders as the friendlier `Vision:
  off`; any real provider name is unchanged. Still fully config-derived,
  never a hardcoded state.
- **The collapsed Developer session summary went from up to six
  `<dl>` rows to one line.** `DevIdentityPanel`'s `DevSessionSummary`
  used to render tenant/role/as-of/dataset/trust-level/expiry as
  `debug-grid` dt/dd pairs (one full row each). It's now a single
  `.dev-session-summary__line` of " · "-joined segments (e.g.
  `tenant_beta · tenant_beta_operator · Current · expires in 3h 58m`),
  each fact its own `<span>` (so a direct-text query like
  `getByText("techfusion")` still finds exactly that segment) with a
  plain `·` text separator between them, not a dt/dd pair per field. A
  new `statusSegment()` helper replaces the old `sessionStatusText()`:
  it deliberately does not repeat a bare "Authenticated" word when the
  header's `TokenStatusBadge` already conveys that via its own live
  countdown, the `expires in Xm` segment itself implies authenticated;
  `statusSegment()` only contributes "Authenticated" for the no-`exp`-
  claim edge case (nothing else would convey that), or the problem
  words `Token expired`/`Malformed token` in place of the expiry phrase.
  Detailed per-field editing is unchanged, one click away via **Edit
  settings**.
- **The empty chat state became product-facing instead of generic.**
  `MessageList`'s pre-first-message view changed from a single
  instructional sentence to a heading ("Ask about the TechFusion
  knowledge base"), a short subtitle, and 2-3 example-query chips
  (`EXAMPLE_QUERIES` in `MessageList.tsx`). Each chip's text is a real
  question grounded in actual, tenant-unrestricted `data/knowledge_base`
  content verified by direct inspection (a webhook
  delivery timeout, a production-write-access duration, a case's
  manual-review wait threshold), deliberately not a fabricated case ID
  or document name, and not drawn from `security_evaluation/`'s
  tenant-gated content, which would read as broken for a caller with no
  dev identity configured. Clicking a chip calls a new
  `MessageInputHandle.setValue` imperative method (`MessageInput` is now
  `forwardRef`-wrapped, mirroring `DevIdentityPanelHandle`'s existing
  pattern) to populate and focus the composer, it does not submit,
  matching the composer's existing behavior for any other way text gets
  into it.
- **The composer got a genuine auto-grow, not just a manual resize
  handle.** `MessageInput`'s textarea previously only had CSS
  `resize: vertical` (a user-draggable handle, not automatic) and a
  fixed `rows={2}`. It now starts at `rows={1}` (more compact when
  empty) and a `useEffect` on `value` measures `scrollHeight` and
  applies it as an inline `style.height`, capped at
  `MAX_TEXTAREA_HEIGHT_PX` (160px, matching the CSS `max-height`) beyond
  which it scrolls internally; `resize: none` now, since auto-grow
  supersedes manual dragging. Enter-sends/Shift+Enter-newline/disabled-
  state behavior is unchanged.
- **Tests**: `tests/unit/messageInput.test.tsx` (new, Enter/Shift+Enter,
  disabled state, the imperative handle, and a `scrollHeight`-stubbed
  proof of the height cap, since jsdom has no real layout engine so a
  genuine multi-line measurement can't be exercised directly),
  `tests/unit/messageList.test.tsx` (new, empty-state heading/chips,
  chip-click populates without submitting, chips omitted when no handler
  is given, real messages replace the empty state), `tests/integration/
  exampleQueries.test.tsx` (new, the same chip-click-then-Send flow
  through the real `ChatWindow`, proving no request fires from the click
  itself), extended `tests/unit/featureFlagsBar.test.tsx` (checkmark +
  sr-only text + `Vision: off` mapping) and `tests/unit/
  runtimeInfoPanel.test.tsx` (neutral wording, no error-styling class,
  Refresh still enabled in the unavailable state), and rewrote the one
  `tests/unit/devIdentityPanel.test.tsx` case that asserted the old
  dt/dd "Expires in" structure to assert the new one-line summary
  instead. Every other pre-existing frontend test (150 total before this
  pass) needed no change.
- **Reverse-proxy gap: `/info` and `/feedback` were never proxied,
  discovered by manually testing the built Docker image.** Every unit/
  integration test mocks `fetch` directly, so this was invisible to the
  whole test suite; it only surfaced once `RuntimeInfoPanel` was checked
  against the real `docker-compose.frontend.yml` stack and read "Runtime
  info unavailable" against a healthy backend. Root cause: `nginx.conf`
  (and `vite.config.ts`'s dev-server proxy, same gap) never had a
  location/proxy entry for `/info` or `/feedback`, both fell through to
  the SPA's own catch-all (`try_files $uri /index.html`), so `GET /info`
  silently got back `index.html` (200, `text/html`, fails
  `RuntimeInfoSchema.safeParse`) and `POST /feedback` got nginx's static
  handler's 405, never reaching `rag-api` at all. Neither endpoint is
  new, `/info` and `/feedback` both predate this frontend's own nginx
  config, so this was a real, live gap, not a regression introduced
  here. Fixed by adding `location = /info` and
  `location = /feedback` (nginx) and `"/info"`/`"/feedback"` entries
  (vite.config.ts), mirrored in both places like every other proxied
  path. Verified against the real container: `GET /info` now returns
  `Content-Type: application/json` (was `text/html`), `POST /feedback`
  now returns the backend's own `401` (auth required) instead of
  nginx's `405`.
- **`/query`'s reverse-proxy timeout gap, flagged as open in an earlier
  KNOWN_ISSUES.md entry, recurred and was closed.** `nginx.conf`'s `/query`
  location had no `proxy_read_timeout` override (nginx's stock default
  is 60s), unlike `/agent/`'s existing `300s`, a classic-RAG generation
  call that legitimately takes longer than 60s (CPU-only inference under
  host memory pressure) got a `504` at the nginx layer even though
  `rag-api` itself returned `200` (confirmed directly in its own logs:
  `79105.58ms`, `success: true`). Added the same `proxy_read_timeout
  300s;` to `/query`'s location block. See KNOWN_ISSUES.md's "A 504 from the
  frontend..." entry's follow-up section for the full before/after.
