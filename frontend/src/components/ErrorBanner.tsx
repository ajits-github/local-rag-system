import type { RagApiErrorKind } from "../api/client";

const ERROR_COPY: Record<RagApiErrorKind, string> = {
  authentication_failed: "Authentication failed. Check the bearer token in Developer settings.",
  rate_limited: "Rate limit exceeded. Wait a moment and try again.",
  validation_failed: "The request was rejected as invalid.",
  not_found: "The requested endpoint is not available on this backend.",
  backend_unavailable: "Could not reach the backend. Is the API running (and is Ollama running for generation)?",
  malformed_response: "The server returned a response the UI could not understand.",
  server_error: "The server reported an error.",
};

export function ErrorBanner({ kind, message }: { kind: RagApiErrorKind; message?: string }) {
  return (
    <div className="error-banner" role="alert">
      <strong>{ERROR_COPY[kind]}</strong>
      {message && message !== ERROR_COPY[kind] && <p className="error-banner__detail">{message}</p>}
    </div>
  );
}
