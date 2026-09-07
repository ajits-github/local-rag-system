export type UiMode = "developer" | "production";

/**
 * The safe fallback when `VITE_UI_MODE` is unset or unrecognized.
 *
 * This project's Developer Settings panel can inject a bearer token,
 * impersonate a tenant/roles, and surface internal pipeline/debug
 * information -- all of it local-development tooling, none of it meant
 * for a real end user. A build that forgets to set `VITE_UI_MODE` (or
 * sets a typo'd value) must not accidentally ship that tooling, so the
 * fallback is "production" (developer tooling hidden), never
 * "developer".
 */
export const SAFE_DEFAULT_UI_MODE: UiMode = "production";

export interface UiModeConfig {
  uiMode: UiMode;
  /** Bearer-token entry and the tenant/role/as-of developer identity controls. */
  showDeveloperSettings: boolean;
  /** Per-answer Debug panel (route, timings, tool-call cards) and raw retrieval scores. */
  showDebugDetails: boolean;
  /** The backend feature-flags bar and the runtime/pipeline info panel. */
  showRuntimeDetails: boolean;
  /** The small "Developer Mode" header badge. */
  showDeveloperModeIndicator: boolean;
}

/**
 * Resolves a raw `VITE_UI_MODE` value to a `UiMode`. Only the exact
 * string "developer" opts into developer mode; anything else (unset,
 * empty, "production", a typo, a different case) resolves to
 * `SAFE_DEFAULT_UI_MODE`.
 */
export function resolveUiMode(raw: string | undefined): UiMode {
  return raw === "developer" ? "developer" : SAFE_DEFAULT_UI_MODE;
}

/**
 * This is a client-side display flag only, resolved from a build-time
 * Vite env var (see frontend/README.md's "UI modes" section). It decides
 * what the UI *shows*, never what the backend *allows* -- every control
 * gated by it (auth token entry, tenant/role impersonation, internal
 * debug/runtime detail) is independently enforced by the backend
 * (JWT verification, authorization, field redaction) regardless of what
 * this resolves to. Read fresh on every call rather than cached at
 * module scope, so tests can override it per-case via
 * `vi.stubEnv("VITE_UI_MODE", ...)` without needing `vi.resetModules()`.
 */
export function getUiModeConfig(): UiModeConfig {
  const uiMode = resolveUiMode(import.meta.env.VITE_UI_MODE);
  const isDeveloper = uiMode === "developer";
  return {
    uiMode,
    showDeveloperSettings: isDeveloper,
    showDebugDetails: isDeveloper,
    showRuntimeDetails: isDeveloper,
    showDeveloperModeIndicator: isDeveloper,
  };
}
