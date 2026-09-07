/** Shared test-only helpers for building fake JWTs (never real signatures -- these are never verified). */

function base64UrlEncode(obj: Record<string, unknown>): string {
  return btoa(JSON.stringify(obj)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Build a structurally-valid (but unsigned/unverified) JWT string for tests: `header.payload.signature`. */
export function makeJwt(
  payload: Record<string, unknown>,
  header: Record<string, unknown> = { alg: "HS256", typ: "JWT" }
): string {
  return `${base64UrlEncode(header)}.${base64UrlEncode(payload)}.test-signature`;
}

/** `exp` claim value (seconds since epoch) `minutesFromNow` minutes in the future. */
export function expInMinutes(minutesFromNow: number, fromMs: number = Date.now()): number {
  return Math.floor((fromMs + minutesFromNow * 60_000) / 1000);
}
