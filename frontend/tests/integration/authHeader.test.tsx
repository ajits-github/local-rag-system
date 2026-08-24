import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

describe("auth header handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("attaches a Bearer token when one is set in Developer settings, and disables manual tenant/roles", async () => {
    const fetchMock = routedFetchMock({
      "/query": () => jsonResponse(200, { answer: "ok", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Bearer token"), "signed.jwt.token");

    expect(screen.getByLabelText("Tenant ID")).toBeDisabled();
    expect(screen.getByLabelText("Roles (comma-separated)")).toBeDisabled();

    await user.type(screen.getByLabelText("Message"), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    const headers = init!.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer signed.jwt.token");
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("tenant_id");
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("roles");
  });

  it("sends caller-supplied tenant_id/roles only when no token is set", async () => {
    const fetchMock = routedFetchMock({
      "/query": () => jsonResponse(200, { answer: "ok", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Tenant ID"), "tenant-alpha");
    await user.type(screen.getByLabelText("Roles (comma-separated)"), "viewer, support");

    await user.type(screen.getByLabelText("Message"), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    const headers = init!.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
    expect(JSON.parse(init!.body as string)).toMatchObject({ tenant_id: "tenant-alpha", roles: ["viewer", "support"] });
  });
});
