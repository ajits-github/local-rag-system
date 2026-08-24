import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FeatureFlagsBar } from "../../src/components/FeatureFlagsBar";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function rootInfoBody(featureOverrides: Record<string, unknown> = {}) {
  return {
    service: "local-rag-system",
    status: "ok",
    docs: "/docs",
    health: "/health",
    metrics: "/metrics",
    features: {
      auth_enabled: false,
      insecure_dev_mode: false,
      authorization_enabled: false,
      field_redaction_enabled: false,
      rate_limit_enabled: false,
      agent_enabled: true,
      vision_provider: "none",
      tracing_enabled: false,
      ...featureOverrides,
    },
  };
}

describe("FeatureFlagsBar", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows enabled/disabled pills reflecting the backend's active features", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          200,
          rootInfoBody({ authorization_enabled: true, field_redaction_enabled: true, vision_provider: "ollama" })
        )
      )
    );

    render(<FeatureFlagsBar />);

    await waitFor(() => expect(screen.getByTitle("Authorization: enabled")).toBeInTheDocument());
    expect(screen.getByTitle("Auth: disabled")).toBeInTheDocument();
    expect(screen.getByTitle("Redaction: enabled")).toBeInTheDocument();
    expect(screen.getByTitle("Vision provider: ollama")).toHaveTextContent("Vision: ollama");
  });

  it("flags insecure_dev_mode explicitly when auth is enabled with dev mode on", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(200, rootInfoBody({ auth_enabled: true, insecure_dev_mode: true })))
    );

    render(<FeatureFlagsBar />);

    await waitFor(() => expect(screen.getByText("Dev mode")).toBeInTheDocument());
  });

  it("does not show a dev-mode warning when auth is disabled", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(200, rootInfoBody({ auth_enabled: false, insecure_dev_mode: true })))
    );

    render(<FeatureFlagsBar />);

    await waitFor(() => expect(screen.getByTitle("Auth: disabled")).toBeInTheDocument());
    expect(screen.queryByText("Dev mode")).not.toBeInTheDocument();
  });

  it("shows a status message while loading and an error state when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    render(<FeatureFlagsBar />);

    expect(screen.getByText(/Checking backend configuration/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Backend status unavailable")).toBeInTheDocument());
  });

  it("re-fetches when the refresh button is clicked", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, rootInfoBody({ authorization_enabled: false })))
      .mockResolvedValueOnce(jsonResponse(200, rootInfoBody({ authorization_enabled: true })));
    vi.stubGlobal("fetch", fetchMock);

    render(<FeatureFlagsBar />);
    await waitFor(() => expect(screen.getByTitle("Authorization: disabled")).toBeInTheDocument());

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Refresh backend status/ }));

    await waitFor(() => expect(screen.getByTitle("Authorization: enabled")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
