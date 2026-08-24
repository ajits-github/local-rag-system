/**
 * Shared fetch wrapper: base URL resolution, auth header attachment, and
 * error classification. Every API call in this app goes through here so
 * the error taxonomy (see RagApiError) stays consistent across classic
 * query, agent query, and the agent event stream.
 */
import type { DevIdentity } from "./types";

export type RagApiErrorKind =
  | "authentication_failed"
  | "rate_limited"
  | "validation_failed"
  | "not_found"
  | "backend_unavailable"
  | "malformed_response"
  | "server_error";

export class RagApiError extends Error {
  readonly kind: RagApiErrorKind;
  readonly status?: number;
  readonly detail?: string;

  constructor(kind: RagApiErrorKind, message: string, status?: number, detail?: string) {
    super(message);
    this.name = "RagApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

declare global {
  interface Window {
    __RAG_API_BASE__?: string;
  }
}

/**
 * Resolve the configured API base URL.
 *
 * Empty string (the default) means "same origin": the Vite dev-server
 * proxy or the Docker image's nginx proxy both make relative paths correct
 * without any CORS configuration on the backend. See frontend/README.md.
 */
export function getApiBase(): string {
  return (typeof window !== "undefined" && window.__RAG_API_BASE__) || "";
}

/** Extract a human-readable message from a FastAPI-shaped error body. */
function extractDetail(body: unknown): string | undefined {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : String(item)
        )
        .join("; ");
    }
  }
  return undefined;
}

async function classifyErrorResponse(response: Response): Promise<RagApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  const detail = extractDetail(body);

  if (response.status === 401) {
    return new RagApiError("authentication_failed", "Authentication failed.", 401, detail);
  }
  if (response.status === 429) {
    return new RagApiError("rate_limited", "Rate limit exceeded.", 429, detail);
  }
  if (response.status === 422) {
    return new RagApiError("validation_failed", detail ?? "Request was rejected as invalid.", 422, detail);
  }
  if (response.status === 404) {
    return new RagApiError("not_found", detail ?? "Endpoint not found.", 404, detail);
  }
  return new RagApiError(
    "server_error",
    detail ?? `Request failed with status ${response.status}.`,
    response.status,
    detail
  );
}

export interface RequestOptions {
  identity?: DevIdentity;
  signal?: AbortSignal;
}

/** Build the JSON headers for a request, attaching a bearer token when present. */
export function buildHeaders(identity?: DevIdentity): HeadersInit {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (identity?.bearerToken) {
    headers.Authorization = `Bearer ${identity.bearerToken}`;
  }
  return headers;
}

/**
 * POST JSON to `path` and return the raw Response, or throw a
 * classified RagApiError. Network failures (backend unreachable) are
 * distinguished from HTTP-level failures here.
 */
export async function postJson(
  path: string,
  body: unknown,
  options: RequestOptions = {}
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${getApiBase()}${path}`, {
      method: "POST",
      headers: buildHeaders(options.identity),
      body: JSON.stringify(body),
      signal: options.signal,
    });
  } catch (cause) {
    throw new RagApiError(
      "backend_unavailable",
      "Could not reach the backend. Is the API running?",
      undefined,
      cause instanceof Error ? cause.message : undefined
    );
  }

  if (!response.ok) {
    throw await classifyErrorResponse(response);
  }
  return response;
}
