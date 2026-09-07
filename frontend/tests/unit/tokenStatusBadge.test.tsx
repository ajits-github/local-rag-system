import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TokenStatusBadge } from "../../src/components/TokenStatusBadge";
import { expInMinutes, makeJwt } from "../testHelpers";

describe("TokenStatusBadge", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders nothing when no token is set", () => {
    const { container } = render(<TokenStatusBadge token="" onReplaceToken={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows an authenticated countdown for a healthy token", () => {
    const token = makeJwt({ exp: expInMinutes(47) });
    render(<TokenStatusBadge token={token} onReplaceToken={() => {}} />);
    expect(screen.getByText(/Authenticated · 47:00/)).toBeInTheDocument();
  });

  it("shows 'Token expired' once the token has expired, with a Replace token action", async () => {
    const token = makeJwt({ exp: expInMinutes(-1) });
    const onReplaceToken = vi.fn();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<TokenStatusBadge token={token} onReplaceToken={onReplaceToken} />);

    expect(screen.getByText("Token expired")).toBeInTheDocument();
    expect(screen.queryByText(/Authenticated/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Replace token" }));
    expect(onReplaceToken).toHaveBeenCalledOnce();
  });

  it("shows a Replace token action for a malformed token", () => {
    render(<TokenStatusBadge token="not-a-jwt" onReplaceToken={() => {}} />);
    expect(screen.getByText("Invalid token")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Replace token" })).toBeInTheDocument();
  });

  it("never renders the raw token text anywhere in the badge", () => {
    const rawToken = makeJwt({ exp: expInMinutes(47), sub: "super-secret-subject-value" });
    render(<TokenStatusBadge token={rawToken} onReplaceToken={() => {}} />);
    expect(screen.queryByText(rawToken)).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain(rawToken);
  });

  it("announces a state change via an ARIA live region, not on every countdown tick", () => {
    const token = makeJwt({ exp: expInMinutes(11) });
    render(<TokenStatusBadge token={token} onReplaceToken={() => {}} />);
    const live = screen.getByRole("status", { hidden: true });
    expect(live).toHaveTextContent("");

    act(() => {
      vi.advanceTimersByTime(2 * 60_000); // crosses into expiring_soon (<=10min)
    });
    expect(live).toHaveTextContent(/expiring soon/i);
  });
});
