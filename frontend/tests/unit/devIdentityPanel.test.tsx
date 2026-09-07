import { useRef, useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DevIdentityPanel, type DevIdentityPanelHandle } from "../../src/components/DevIdentityPanel";
import { ChatProvider, useChat } from "../../src/state/chatContext";
import type { DevIdentity } from "../../src/api/types";
import { expInMinutes, makeJwt } from "../testHelpers";

const emptyIdentity: DevIdentity = {
  bearerToken: "",
  tenantId: "",
  roles: "",
  asOf: "",
  requireTrustLevel: "",
  datasetId: "",
};

/** Standalone (non-chatContext) harness: local React state stands in for the caller. */
function StandaloneHarness({ initial = emptyIdentity }: { initial?: DevIdentity }) {
  const [identity, setIdentity] = useState(initial);
  const ref = useRef<DevIdentityPanelHandle>(null);
  return (
    <>
      <button type="button" onClick={() => ref.current?.openForEdit()}>
        external open
      </button>
      <DevIdentityPanel ref={ref} identity={identity} onChange={setIdentity} />
    </>
  );
}

/** Wired to the real ChatProvider/reducer, so Apply's sessionStorage persistence is exercised end to end. */
function ChatContextHarness() {
  const { state, dispatch } = useChat();
  return (
    <DevIdentityPanel
      identity={state.devIdentity}
      onChange={(identity) => dispatch({ type: "SET_DEV_IDENTITY", identity })}
    />
  );
}

function renderStandalone(initial?: DevIdentity) {
  return render(<StandaloneHarness initial={initial} />);
}

function renderWithChatContext() {
  return render(
    <ChatProvider>
      <ChatContextHarness />
    </ChatProvider>
  );
}

function setupUser() {
  return userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
}

describe("DevIdentityPanel", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("starts collapsed with a plain toggle when nothing is configured", () => {
    renderStandalone();
    expect(screen.getByRole("button", { name: "▸ Developer settings" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Bearer token")).not.toBeInTheDocument();
  });

  it("expands the form and focuses the bearer token field when opened", async () => {
    renderStandalone();
    const user = setupUser();
    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));

    expect(screen.getByLabelText("Bearer token")).toHaveFocus();
  });

  it("does not commit field edits until Apply settings is clicked", async () => {
    const onChange = vi.fn();
    render(<DevIdentityPanel identity={emptyIdentity} onChange={onChange} />);
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Dataset ID"), "techfusion");

    expect(onChange).not.toHaveBeenCalled();
  });

  it("applies valid settings, collapses the panel, and shows a compact summary", async () => {
    renderStandalone();
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Dataset ID"), "techfusion");
    await user.type(screen.getByLabelText("Require trust level"), "authoritative");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    expect(screen.queryByLabelText("Bearer token")).not.toBeInTheDocument();
    const summary = screen.getByText("Developer session").closest<HTMLElement>(".dev-session-summary")!;
    expect(within(summary).getByText("techfusion")).toBeInTheDocument();
    expect(within(summary).getByText("authoritative")).toBeInTheDocument();
    expect(within(summary).getByText("Current")).toBeInTheDocument(); // As of defaults to "Current"
  });

  it("moves focus to Edit settings after a successful Apply", async () => {
    renderStandalone();
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Dataset ID"), "techfusion");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    expect(screen.getByRole("button", { name: "Edit settings" })).toHaveFocus();
  });

  it("reopens Edit settings pre-filled with the last applied values", async () => {
    renderStandalone();
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Dataset ID"), "techfusion");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    await user.click(screen.getByRole("button", { name: "Edit settings" }));
    expect(screen.getByLabelText("Dataset ID")).toHaveValue("techfusion");
  });

  it("can be opened externally (e.g. by the header's Replace token action) via the imperative handle", async () => {
    renderStandalone();
    const user = setupUser();
    await user.click(screen.getByRole("button", { name: "external open" }));
    expect(screen.getByLabelText("Bearer token")).toBeInTheDocument();
  });

  it("rejects a malformed bearer token, ties the error to the field, and never applies it", async () => {
    const onChange = vi.fn();
    render(<DevIdentityPanel identity={emptyIdentity} onChange={onChange} />);
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Bearer token"), "signed.jwt.token");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/doesn't look like a valid JWT/);
    const field = screen.getByLabelText("Bearer token");
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(field).toHaveAttribute("aria-describedby", alert.id);
    expect(field).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("accepts a valid as-of date and applies it", async () => {
    renderStandalone();
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    const asOfInput = screen.getByLabelText("As of (date)");
    await user.type(asOfInput, "2026-01-15");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    const summary = screen.getByText("Developer session").closest<HTMLElement>(".dev-session-summary")!;
    expect(within(summary).getByText("2026-01-15")).toBeInTheDocument();
  });

  it("never renders the raw bearer token in the collapsed summary", async () => {
    renderStandalone();
    const user = setupUser();
    const token = makeJwt({ sub: "user-1", tenant_id: "tenant-alpha", exp: expInMinutes(47) });

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Bearer token"), token);
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    expect(document.body.textContent).not.toContain(token);
    // The decoded tenant claim is safe/expected to show; the raw token substring must not appear anywhere.
    expect(screen.getByText("tenant-alpha")).toBeInTheDocument();
  });

  it("shows decoded tenant/roles and an expiry countdown as one compact horizontal summary line", async () => {
    renderStandalone();
    const user = setupUser();
    const token = makeJwt({ sub: "user-1", tenant_id: "tenant-beta", roles: ["tenant_beta_operator"], exp: expInMinutes(47) });

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Bearer token"), token);
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    const summary = screen.getByText("Developer session").closest<HTMLElement>(".dev-session-summary")!;
    // A single compact line ("tenant · role · Current · expires in Xm"),
    // not a multi-row field list -- exactly one line element for the facts.
    expect(summary.querySelectorAll(".dev-session-summary__line")).toHaveLength(1);
    expect(within(summary).getByText("tenant-beta")).toBeInTheDocument();
    expect(within(summary).getByText("tenant_beta_operator")).toBeInTheDocument();
    expect(within(summary).getByText("Current")).toBeInTheDocument();
    // Not pinned to an exact minute count: real time elapses between minting
    // the token and this assertion (user-event interactions take real wall-clock
    // ms even under fake timers), so only the format is checked here -- the
    // exact countdown arithmetic is covered deterministically, under fully
    // frozen time, by useTokenStatus.test.ts and tokenStatusBadge.test.tsx.
    // The bare "Authenticated" text is deliberately not repeated here --
    // the header's TokenStatusBadge already owns that -- so this asserts
    // the expiry phrasing directly instead.
    const expirySegment = within(summary).getByText(/^expires in \d+m$/);
    expect(expirySegment).toBeInTheDocument();
  });

  it("disables tenant/roles inputs while a bearer token is being entered, before Apply", async () => {
    renderStandalone();
    const user = setupUser();
    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Bearer token"), "x.y.z");
    expect(screen.getByLabelText("Tenant ID")).toBeDisabled();
    expect(screen.getByLabelText("Roles (comma-separated)")).toBeDisabled();
  });

  it("persists applied settings to sessionStorage via the chat context", async () => {
    renderWithChatContext();
    const user = setupUser();

    await user.click(screen.getByRole("button", { name: "▸ Developer settings" }));
    await user.type(screen.getByLabelText("Dataset ID"), "techfusion");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    const stored = window.sessionStorage.getItem("rag-ui:dev-identity");
    expect(stored).not.toBeNull();
    expect(JSON.parse(stored!)).toMatchObject({ datasetId: "techfusion" });
  });
});
