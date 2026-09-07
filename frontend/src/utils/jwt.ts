/**
 * Client-side-only JWT inspection for display purposes (see DevIdentityPanel /
 * TokenStatusBadge). Decoding here never constitutes authentication or
 * authorization -- the backend (rag.api.auth.verify_jwt) is the only place a
 * signature is actually checked. This module reads the base64url-decoded
 * header/payload JSON only; it makes no claim about whether the token is
 * genuine, current, or accepted by the server.
 */

export interface DecodedJwt {
  header: Record<string, unknown>;
  payload: Record<string, unknown>;
}

function base64UrlDecode(segment: string): string {
  const normalized = segment.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
  return new TextDecoder("utf-8").decode(bytes);
}

function decodeSegment(segment: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(base64UrlDecode(segment));
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Decode a JWT's header and payload without verifying its signature.
 *
 * Returns `null` for anything that isn't shaped like a JWT (wrong segment
 * count, non-base64url segments, or a segment that isn't a JSON object) --
 * malformed input is a display state (see useTokenStatus), never a thrown
 * error.
 */
export function decodeJwt(token: string): DecodedJwt | null {
  const parts = token.trim().split(".");
  if (parts.length !== 3 || parts.some((p) => p.length === 0)) return null;
  const header = decodeSegment(parts[0]);
  const payload = decodeSegment(parts[1]);
  if (!header || !payload) return null;
  return { header, payload };
}

/** Structural check only (decodes header+payload); never validates a signature. */
export function looksLikeJwt(token: string): boolean {
  return decodeJwt(token) !== null;
}

export interface JwtDisplayClaims {
  exp?: number;
  sub?: string;
  tenantId?: string;
  roles?: string[];
}

/**
 * Extract only the specific claims the UI ever displays, from an
 * already-decoded payload.
 *
 * Deliberately returns a small, fixed shape rather than the raw payload
 * object, so no call site can accidentally render an arbitrary claim the
 * token happens to carry.
 */
export function extractDisplayClaims(payload: Record<string, unknown>): JwtDisplayClaims {
  const claims: JwtDisplayClaims = {};
  if (typeof payload.exp === "number") claims.exp = payload.exp;
  if (typeof payload.sub === "string") claims.sub = payload.sub;
  if (typeof payload.tenant_id === "string") claims.tenantId = payload.tenant_id;
  if (Array.isArray(payload.roles) && payload.roles.every((r) => typeof r === "string")) {
    claims.roles = payload.roles as string[];
  }
  return claims;
}
