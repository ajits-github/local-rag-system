import type { RagApiErrorKind } from "../api/client";
import { getUiModeConfig } from "../config/uiMode";

const ERROR_COPY: Record<RagApiErrorKind, string> = {
  authentication_failed: "Authentication failed.",
  rate_limited: "Rate limit exceeded. Wait a moment and try again.",
  validation_failed: "The request was rejected as invalid.",
  not_found: "The requested endpoint is not available on this backend.",
  backend_unavailable: "Could not reach the backend. Is the API running (and is Ollama running for generation)?",
  malformed_response: "The server returned a response the UI could not understand.",
  server_error: "The server reported an error.",
};

// Only meaningful in developer mode, where the panel it points at
// actually exists; a production build has no Developer settings panel
// to send someone to.
const DEV_AUTH_FAILED_DETAIL = "Check the bearer token in Developer settings.";

export function ErrorBanner({ kind, message }: { kind: RagApiErrorKind; message?: string }) {
  const { showDeveloperSettings } = getUiModeConfig();
  const copy =
    kind === "authentication_failed" && showDeveloperSettings
      ? `${ERROR_COPY[kind]} ${DEV_AUTH_FAILED_DETAIL}`
      : ERROR_COPY[kind];
  return (
    <div className="error-banner" role="alert">
      <strong>{copy}</strong>
      {message && message !== copy && <p className="error-banner__detail">{message}</p>}
    </div>
  );
}
