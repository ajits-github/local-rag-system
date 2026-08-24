import { postJson, RagApiError, type RequestOptions } from "./client";
import { QueryResponseSchema, type DevIdentity, type QueryResponse } from "./types";

export interface QueryRequestBody {
  query: string;
  top_k?: number;
  filters?: Record<string, unknown>;
  tenant_id?: string;
  roles?: string[];
  as_of?: string;
  require_trust_level?: string;
}

/** Build the request body for POST /query from the raw query text and optional dev identity overrides. */
export function buildQueryRequestBody(query: string, identity?: DevIdentity): QueryRequestBody {
  const body: QueryRequestBody = { query };
  // Only sent when no bearer token is present. The backend ignores these
  // fields whenever a verified JWT identity exists (see request_auth.py),
  // and the Dev Identity panel disables them client-side in that case too.
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

export async function postQuery(query: string, identity?: DevIdentity, options: RequestOptions = {}): Promise<QueryResponse> {
  const response = await postJson("/query", buildQueryRequestBody(query, identity), { ...options, identity });
  const json = await response.json();
  const parsed = QueryResponseSchema.safeParse(json);
  if (!parsed.success) {
    throw new RagApiError("malformed_response", "The server returned an unexpected response shape.");
  }
  return parsed.data;
}
