import { buildHeaders, getApiBase, RagApiError, type RequestOptions } from "./client";
import { readSseFrames } from "../utils/sse";
import {
  AgentEventSchema,
  AgentQueryResponseSchema,
  type AgentEvent,
  type AgentQueryResponse,
  type DevIdentity,
} from "./types";

export interface AgentQueryRequestBody {
  query: string;
  filters?: Record<string, unknown>;
  tenant_id?: string;
  roles?: string[];
  as_of?: string;
  require_trust_level?: string;
}

/** Build the request body for POST /agent/query(/stream), using the same precedence rules as buildQueryRequestBody. */
export function buildAgentQueryRequestBody(query: string, identity?: DevIdentity): AgentQueryRequestBody {
  const body: AgentQueryRequestBody = { query };
  if (!identity?.bearerToken) {
    if (identity?.tenantId) body.tenant_id = identity.tenantId;
    if (identity?.roles) {
      const roles = identity.roles.split(",").map((r) => r.trim()).filter(Boolean);
      if (roles.length) body.roles = roles;
    }
  }
  if (identity?.asOf) body.as_of = identity.asOf;
  if (identity?.requireTrustLevel) body.require_trust_level = identity.requireTrustLevel;
  return body;
}

/** Non-streaming agent query, used as the initial call and as the fallback when the stream endpoint is disabled. */
export async function postAgentQuery(
  query: string,
  identity?: DevIdentity,
  options: RequestOptions = {}
): Promise<AgentQueryResponse> {
  const response = await fetch(`${getApiBase()}/agent/query`, {
    method: "POST",
    headers: buildHeaders(identity),
    body: JSON.stringify(buildAgentQueryRequestBody(query, identity)),
    signal: options.signal,
  }).catch((cause) => {
    throw new RagApiError(
      "backend_unavailable",
      "Could not reach the backend. Is the API running?",
      undefined,
      cause instanceof Error ? cause.message : undefined
    );
  });

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = await response.json();
      detail = typeof body?.detail === "string" ? body.detail : undefined;
    } catch {
      detail = undefined;
    }
    if (response.status === 401) throw new RagApiError("authentication_failed", "Authentication failed.", 401, detail);
    if (response.status === 429) throw new RagApiError("rate_limited", "Rate limit exceeded.", 429, detail);
    if (response.status === 422) throw new RagApiError("validation_failed", detail ?? "Request was rejected as invalid.", 422, detail);
    throw new RagApiError("server_error", detail ?? `Request failed with status ${response.status}.`, response.status, detail);
  }

  const json = await response.json();
  const parsed = AgentQueryResponseSchema.safeParse(json);
  if (!parsed.success) {
    throw new RagApiError("malformed_response", "The server returned an unexpected response shape.");
  }
  return parsed.data;
}

export type AgentStreamItem =
  | { type: "event"; event: AgentEvent }
  | { type: "final"; response: AgentQueryResponse };

/**
 * Stream POST /agent/query/stream, yielding parsed AgentEvents followed by
 * one final item carrying the AgentQueryResponse-shaped terminal frame.
 *
 * Throws RagApiError("not_found") when live events are disabled server-side
 * (config.observability.live_events.enabled=False) so callers can fall back
 * to postAgentQuery without losing the request.
 */
export async function* streamAgentQuery(
  query: string,
  identity?: DevIdentity,
  options: RequestOptions = {}
): AsyncGenerator<AgentStreamItem> {
  let response: Response;
  try {
    response = await fetch(`${getApiBase()}/agent/query/stream`, {
      method: "POST",
      headers: buildHeaders(identity),
      body: JSON.stringify(buildAgentQueryRequestBody(query, identity)),
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
    if (response.status === 404) {
      throw new RagApiError("not_found", "Live agent events are disabled on this backend.", 404);
    }
    if (response.status === 401) throw new RagApiError("authentication_failed", "Authentication failed.", 401);
    if (response.status === 429) throw new RagApiError("rate_limited", "Rate limit exceeded.", 429);
    if (response.status === 422) throw new RagApiError("validation_failed", "Request was rejected as invalid.", 422);
    throw new RagApiError("server_error", `Request failed with status ${response.status}.`, response.status);
  }

  for await (const frame of readSseFrames(response, options.signal)) {
    let payload: unknown;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      throw new RagApiError("malformed_response", "The server sent an unparsable stream frame.");
    }

    if (frame.event === "completed" || frame.event === "terminated") {
      const parsed = AgentQueryResponseSchema.safeParse(payload);
      if (!parsed.success) {
        throw new RagApiError("malformed_response", "The server sent an unexpected final response shape.");
      }
      yield { type: "final", response: parsed.data };
      return;
    }

    const parsedEvent = AgentEventSchema.safeParse(payload);
    if (!parsedEvent.success) {
      // Skip unrecognized event shapes rather than aborting an otherwise-healthy stream.
      continue;
    }
    yield { type: "event", event: parsedEvent.data };
  }
}
