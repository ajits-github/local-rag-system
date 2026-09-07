import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeJwt, expInMinutes } from "../testHelpers";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

describe("auth header handling", () => {
  beforeEach(() => {
    // Developer settings are gated behind developer mode; this suite
    // exercises that panel through the real ChatWindow, so opt in
    // explicitly rather than relying on the safe production default.
    vi.stubEnv("VITE_UI_MODE", "developer");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("attaches a Bearer token once applied, and disables manual tenant/roles while editing", async () => {
    const fetchMock = routedFetchMock({
      "/query": () => jsonResponse(200, { answer: "ok", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();
    const token = makeJwt({ sub: "dev-user", tenant_id: "tenant-alpha", exp: expInMinutes(60) });

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Bearer token"), token);

    expect(screen.getByLabelText("Tenant ID")).toBeDisabled();
    expect(screen.getByLabelText("Roles (comma-separated)")).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    // Applying collapses the panel into a compact session summary.
    await waitFor(() => expect(screen.getByText("Developer session")).toBeInTheDocument());
    expect(screen.queryByLabelText("Bearer token")).not.toBeInTheDocument();

    await user.type(screen.getByLabelText("Message"), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    const headers = init!.headers as Record<string, string>;
    expect(headers.Authorization).toBe(`Bearer ${token}`);
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("tenant_id");
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("roles");
  });

  it("sends caller-supplied tenant_id/roles only when no token is applied", async () => {
    const fetchMock = routedFetchMock({
      "/query": () => jsonResponse(200, { answer: "ok", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Tenant ID"), "tenant-alpha");
    await user.type(screen.getByLabelText("Roles (comma-separated)"), "viewer, support");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    await waitFor(() => expect(screen.getByText("Developer session")).toBeInTheDocument());

    await user.type(screen.getByLabelText("Message"), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    const headers = init!.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
    expect(JSON.parse(init!.body as string)).toMatchObject({ tenant_id: "tenant-alpha", roles: ["viewer", "support"] });
  });

  it("rejects a malformed bearer token on Apply instead of silently sending it", async () => {
    const fetchMock = routedFetchMock({});
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Bearer token"), "not-a-jwt");
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    expect(screen.getByRole("alert")).toHaveTextContent(/doesn't look like a valid JWT/);
    // The panel stays open (not applied), and the invalid token was never persisted/sent.
    expect(screen.getByLabelText("Bearer token")).toBeInTheDocument();
    expect(screen.queryByText("Developer session")).not.toBeInTheDocument();
  });
});
