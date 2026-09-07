import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DebugPanel } from "../../src/components/DebugPanel";
import type { DebugInfo } from "../../src/state/types";

describe("DebugPanel", () => {
  it("renders nothing when there is no debug info", () => {
    const { container } = render(<DebugPanel debug={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("stays collapsed by default", () => {
    render(<DebugPanel debug={{ route: "classic_rag", requestId: "req-1" }} />);
    expect(screen.getByRole("button", { name: /Debug/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Request")).not.toBeInTheDocument();
  });

  it("separates request, pipeline, and timings into distinct labeled sections", async () => {
    const debug: DebugInfo = {
      route: "agent",
      terminationReason: "synthesized",
      requestId: "req-42",
      retrievalMs: 120.4,
      generationMs: 900.1,
      totalMs: 1020.5,
    };
    render(<DebugPanel debug={debug} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Debug/ }));

    expect(screen.getByRole("region", { name: "Request" })).toHaveTextContent("req-42");
    expect(screen.getByRole("region", { name: "Pipeline" })).toHaveTextContent("agent");
    expect(screen.getByRole("region", { name: "Pipeline" })).toHaveTextContent("synthesized");
    expect(screen.getByRole("region", { name: "Timings" })).toHaveTextContent("120 ms");
    expect(screen.getByRole("region", { name: "Timings" })).toHaveTextContent("900 ms");
    expect(screen.getByRole("region", { name: "Timings" })).toHaveTextContent("1021 ms");
  });

  it("renders a safe card per tool call: name, execution, status, duration", async () => {
    const debug: DebugInfo = {
      route: "agent",
      steps: 2,
      toolCallDetails: [
        { tool_name: "search_knowledge_base", execution: "local", success: true, result_count: 3, latency_ms: 45 },
        { tool_name: "get_case_status", execution: "mcp_remote", success: true, result_count: 1, latency_ms: 83 },
        { tool_name: "update_case_status", execution: "mcp_remote", success: false, result_count: 0, latency_ms: 12 },
      ],
    };
    render(<DebugPanel debug={debug} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Debug/ }));

    const tools = screen.getByRole("region", { name: "Agent and tool execution" });
    expect(tools).toHaveTextContent("search_knowledge_base");
    expect(tools).toHaveTextContent("Local");
    expect(tools).toHaveTextContent("get_case_status");
    expect(tools).toHaveTextContent("MCP remote");
    expect(tools).toHaveTextContent("83 ms");
    expect(tools).toHaveTextContent("Success");
    expect(tools).toHaveTextContent("update_case_status");
    expect(tools).toHaveTextContent("Failed");
  });

  it("falls back to a plain tool-name list when no per-call detail is available", async () => {
    render(<DebugPanel debug={{ route: "agent", toolCalls: ["search_knowledge_base"] }} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Debug/ }));

    expect(screen.getByText("search_knowledge_base")).toBeInTheDocument();
  });

  it("never renders a tool call's raw arguments or error text (not present on DebugInfo at all)", async () => {
    const debug: DebugInfo = {
      route: "agent",
      toolCallDetails: [
        { tool_name: "search_knowledge_base", execution: "local", success: false, result_count: 0, latency_ms: 5 },
      ],
    };
    render(<DebugPanel debug={debug} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Debug/ }));

    // Only the fields ToolCallDetail actually carries should ever render.
    expect(screen.queryByText(/reasoning/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/traceback/i)).not.toBeInTheDocument();
  });
});
