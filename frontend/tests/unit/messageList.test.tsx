import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MessageList } from "../../src/components/chat/MessageList";
import type { ChatMessage } from "../../src/state/types";

function userMessage(text: string): ChatMessage {
  return {
    id: "m1",
    role: "user",
    mode: "classic",
    status: "done",
    text,
    sources: [],
    agentEvents: [],
    createdAt: Date.now(),
  };
}

describe("MessageList empty state", () => {
  it("shows a product-like heading, subtitle, and example-query chips when there are no messages", () => {
    render(<MessageList messages={[]} onExampleQuery={vi.fn()} />);

    expect(screen.getByText("Ask about the TechFusion knowledge base")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Example questions" })).toBeInTheDocument();
    expect(screen.getAllByRole("button").length).toBeGreaterThanOrEqual(2);
  });

  it("populates the composer via onExampleQuery when a chip is clicked, without sending anything itself", async () => {
    const onExampleQuery = vi.fn();
    render(<MessageList messages={[]} onExampleQuery={onExampleQuery} />);
    const user = userEvent.setup();

    const chip = screen.getAllByRole("button")[0];
    const chipText = chip.textContent;
    await user.click(chip);

    expect(onExampleQuery).toHaveBeenCalledTimes(1);
    expect(onExampleQuery).toHaveBeenCalledWith(chipText);
  });

  it("renders no example chips when no onExampleQuery handler is given", () => {
    render(<MessageList messages={[]} />);
    expect(screen.getByText("Ask about the TechFusion knowledge base")).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Example questions" })).not.toBeInTheDocument();
  });

  it("shows real messages instead of the empty state once there are any", () => {
    render(<MessageList messages={[userMessage("hi")]} onExampleQuery={vi.fn()} />);
    expect(screen.queryByText("Ask about the TechFusion knowledge base")).not.toBeInTheDocument();
    expect(screen.getByText("hi")).toBeInTheDocument();
  });
});
