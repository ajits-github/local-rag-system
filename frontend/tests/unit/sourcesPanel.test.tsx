import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { SourcesPanel } from "../../src/components/SourcesPanel";
import type { SourceItem } from "../../src/api/types";

function source(overrides: Partial<SourceItem> = {}): SourceItem {
  return {
    chunk_id: "doc_0",
    document_id: "doc",
    source: "a.md",
    score: 0.9,
    vision_generated: false,
    origin: "retrieved",
    ...overrides,
  };
}

describe("SourcesPanel", () => {
  it("renders nothing for an empty source list", () => {
    const { container } = render(<SourcesPanel sources={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("labels a normal retrieved chunk as a RAG source", async () => {
    render(<SourcesPanel sources={[source({ origin: "retrieved" })]} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Sources/ }));

    expect(screen.getByText("RAG")).toBeInTheDocument();
    expect(screen.queryByText("MCP remote")).not.toBeInTheDocument();
  });

  it("clearly distinguishes an MCP remote/business source from a local RAG source", async () => {
    render(
      <SourcesPanel
        sources={[
          source({ chunk_id: "doc_0", origin: "retrieved" }),
          source({ chunk_id: "mcp:get_case_status:case-1", source: "mcp://business/case-1", origin: "mcp_remote" }),
        ]}
      />
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Sources/ }));

    expect(screen.getByText("RAG")).toBeInTheDocument();
    const mcpBadge = screen.getByText("MCP remote");
    expect(mcpBadge).toHaveClass("origin-badge--mcp");
  });
});
