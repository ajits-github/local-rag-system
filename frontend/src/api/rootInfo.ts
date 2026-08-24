import { getApiBase, RagApiError } from "./client";
import { RootInfoSchema, type RootInfo } from "./types";

/**
 * Fetch GET /: service info plus a safe, non-secret summary of which
 * optional security/agent/vision features the backend currently has
 * active (see src/rag/api/main.py's FeatureFlags docstring). Used to
 * render a visible "what's actually enforced right now" strip so an
 * insecure demo config isn't mistaken for a secure one.
 */
export async function fetchRootInfo(signal?: AbortSignal): Promise<RootInfo> {
  let response: Response;
  try {
    response = await fetch(`${getApiBase()}/`, { signal });
  } catch (cause) {
    throw new RagApiError(
      "backend_unavailable",
      "Could not reach the backend.",
      undefined,
      cause instanceof Error ? cause.message : undefined
    );
  }

  if (!response.ok) {
    throw new RagApiError("server_error", `Request failed with status ${response.status}.`, response.status);
  }

  const json = await response.json();
  const parsed = RootInfoSchema.safeParse(json);
  if (!parsed.success) {
    throw new RagApiError("malformed_response", "The server returned an unexpected response shape.");
  }
  return parsed.data;
}
