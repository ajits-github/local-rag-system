import { useEffect, useMemo, useState } from "react";
import { decodeJwt, extractDisplayClaims, type JwtDisplayClaims } from "../utils/jwt";

export type TokenStatusState = "none" | "invalid" | "no_exp" | "healthy" | "expiring_soon" | "critical" | "expired";

export interface TokenStatus {
  hasToken: boolean;
  state: TokenStatusState;
  claims: JwtDisplayClaims | null;
  remainingMs: number | null;
  /** Short countdown for the header badge, e.g. "47:32" or "3h 12m". Null unless `exp` is known and not yet passed. */
  countdownLabel: string | null;
  /** Absolute local expiry time, for the compact summary. Null unless `exp` is known. */
  expiresAtLabel: string | null;
}

const TICK_MS = 1000;
const EXPIRING_SOON_MS = 10 * 60 * 1000;
const CRITICAL_MS = 2 * 60 * 1000;
const ONE_HOUR_MS = 60 * 60 * 1000;

/** Coarser "47m" / "1h 5m" / "32s" form for the compact session summary (vs. the header badge's precise mm:ss). */
export function formatApproxDuration(remainingMs: number): string {
  if (remainingMs <= 0) return "Expired";
  const totalMinutes = Math.floor(remainingMs / 60000);
  if (totalMinutes < 1) {
    const seconds = Math.max(1, Math.floor(remainingMs / 1000));
    return `${seconds}s`;
  }
  if (totalMinutes < 60) return `${totalMinutes}m`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
}

function formatCountdown(remainingMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(remainingMs / 1000));
  if (remainingMs >= ONE_HOUR_MS) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

const NO_TOKEN: TokenStatus = {
  hasToken: false,
  state: "none",
  claims: null,
  remainingMs: null,
  countdownLabel: null,
  expiresAtLabel: null,
};

/**
 * Derive a display-only token status (healthy/expiring/critical/expired)
 * from a bearer token string, decoded entirely client-side.
 *
 * This is never authentication or authorization -- see utils/jwt.ts. When
 * `exp` is present, the countdown re-renders once per second so the header
 * badge / session summary stay live without a manual refresh; the interval
 * is only running while a token with a real `exp` claim is set, so an empty
 * or exp-less token costs nothing.
 */
export function useTokenStatus(token: string): TokenStatus {
  const trimmed = token.trim();
  const decoded = useMemo(() => (trimmed ? decodeJwt(trimmed) : null), [trimmed]);
  const claims = useMemo(() => (decoded ? extractDisplayClaims(decoded.payload) : null), [decoded]);
  const exp = claims?.exp;

  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (exp == null) return undefined;
    // Resync immediately (not just on the first 1s tick): `now` may be stale
    // relative to a just-applied token -- e.g. it was last set when a
    // previous exp-less token stopped the interval, or is still the
    // mount-time value from before the user spent time filling out the form.
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), TICK_MS);
    return () => clearInterval(id);
  }, [exp]);

  return useMemo(() => {
    if (!trimmed) return NO_TOKEN;
    if (!decoded) {
      return { hasToken: true, state: "invalid", claims: null, remainingMs: null, countdownLabel: null, expiresAtLabel: null };
    }
    if (exp == null) {
      return { hasToken: true, state: "no_exp", claims, remainingMs: null, countdownLabel: null, expiresAtLabel: null };
    }
    const expiresAtMs = exp * 1000;
    const remainingMs = expiresAtMs - now;
    const expiresAtLabel = new Date(expiresAtMs).toLocaleString();
    if (remainingMs <= 0) {
      return { hasToken: true, state: "expired", claims, remainingMs, countdownLabel: null, expiresAtLabel };
    }
    const countdownLabel = formatCountdown(remainingMs);
    const state: TokenStatusState =
      remainingMs <= CRITICAL_MS ? "critical" : remainingMs <= EXPIRING_SOON_MS ? "expiring_soon" : "healthy";
    return { hasToken: true, state, claims, remainingMs, countdownLabel, expiresAtLabel };
  }, [trimmed, decoded, claims, exp, now]);
}
