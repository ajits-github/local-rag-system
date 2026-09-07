import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTokenStatus } from "../../src/hooks/useTokenStatus";
import { expInMinutes, makeJwt } from "../testHelpers";

describe("useTokenStatus", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reports 'none' for an empty token", () => {
    const { result } = renderHook(() => useTokenStatus(""));
    expect(result.current).toMatchObject({ hasToken: false, state: "none", claims: null, countdownLabel: null });
  });

  it("reports 'invalid' for a malformed token", () => {
    const { result } = renderHook(() => useTokenStatus("not-a-jwt"));
    expect(result.current).toMatchObject({ hasToken: true, state: "invalid", claims: null });
  });

  it("reports 'no_exp' for a well-formed token with no exp claim", () => {
    const token = makeJwt({ sub: "user-1" });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current).toMatchObject({ hasToken: true, state: "no_exp", countdownLabel: null });
    expect(result.current.claims).toMatchObject({ sub: "user-1" });
  });

  it("reports 'healthy' well before expiry and formats a mm:ss countdown", () => {
    const token = makeJwt({ exp: expInMinutes(47) });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current.state).toBe("healthy");
    expect(result.current.countdownLabel).toBe("47:00");
  });

  it("reports 'expiring_soon' at 10 minutes or less remaining", () => {
    const token = makeJwt({ exp: expInMinutes(9) });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current.state).toBe("expiring_soon");
  });

  it("reports 'critical' at 2 minutes or less remaining", () => {
    const token = makeJwt({ exp: expInMinutes(1) });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current.state).toBe("critical");
  });

  it("reports 'expired' once exp has passed, with no countdown label", () => {
    const token = makeJwt({ exp: expInMinutes(-1) });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current.state).toBe("expired");
    expect(result.current.countdownLabel).toBeNull();
  });

  it("updates the countdown automatically, once per second, without waiting in real time", () => {
    const token = makeJwt({ exp: expInMinutes(1) });
    const { result } = renderHook(() => useTokenStatus(token));
    const initialRemaining = result.current.remainingMs;

    act(() => {
      vi.advanceTimersByTime(10_000);
    });

    expect(result.current.remainingMs).toBeLessThan(initialRemaining!);
    expect(initialRemaining! - result.current.remainingMs!).toBeCloseTo(10_000, -2);
  });

  it("transitions from healthy to expiring_soon to critical to expired as time passes", () => {
    const token = makeJwt({ exp: expInMinutes(11) });
    const { result } = renderHook(() => useTokenStatus(token));
    expect(result.current.state).toBe("healthy");

    act(() => {
      vi.advanceTimersByTime(2 * 60_000); // 9 minutes left
    });
    expect(result.current.state).toBe("expiring_soon");

    act(() => {
      vi.advanceTimersByTime(7.5 * 60_000); // 1.5 minutes left
    });
    expect(result.current.state).toBe("critical");

    act(() => {
      vi.advanceTimersByTime(2 * 60_000); // past expiry
    });
    expect(result.current.state).toBe("expired");
  });

  it("does not run a countdown interval when there is no exp claim to track", () => {
    const token = makeJwt({ sub: "user-1" });
    renderHook(() => useTokenStatus(token));
    // No interval should be scheduled; asserting via the fake-timer count keeps
    // this a behavioral check rather than an implementation-detail spy.
    expect(vi.getTimerCount()).toBe(0);
  });
});
