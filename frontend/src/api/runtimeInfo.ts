import { getApiBase, RagApiError } from "./client";
import { RuntimeInfoSchema, type RuntimeInfo } from "./types";

/**
 * Fetch GET /info: the effective runtime pipeline configuration (models,
 * retrieval mode, fusion method, reranker, MCP/agent/vision status, etc.)
 * -- see src/rag/api/main.py's RuntimeInfo docstring for the full field
 * list and why it's a separate endpoint from GET /'s FeatureFlags.
 */
export async function fetchRuntimeInfo(signal?: AbortSignal): Promise<RuntimeInfo> {
  let response: Response;
  try {
    response = await fetch(`${getApiBase()}/info`, { signal });
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
  const parsed = RuntimeInfoSchema.safeParse(json);
  if (!parsed.success) {
    throw new RagApiError("malformed_response", "The server returned an unexpected response shape.");
  }
  return parsed.data;
}
