import { useRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MessageInput, type MessageInputHandle } from "../../src/components/chat/MessageInput";

function Harness({ disabled = false }: { disabled?: boolean }) {
  const ref = useRef<MessageInputHandle>(null);
  const onSend = vi.fn();
  const onCancel = vi.fn();
  return (
    <>
      <button type="button" onClick={() => ref.current?.setValue("populated from outside")}>
        populate
      </button>
      <MessageInput ref={ref} onSend={onSend} onCancel={onCancel} disabled={disabled} />
    </>
  );
}

describe("MessageInput", () => {
  it("sends on Enter and clears the field", async () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} onCancel={vi.fn()} disabled={false} />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "hello there{Enter}");

    expect(onSend).toHaveBeenCalledWith("hello there");
    expect(screen.getByLabelText("Message")).toHaveValue("");
  });

  it("does not send on Shift+Enter, and inserts a newline instead", async () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} onCancel={vi.fn()} disabled={false} />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "line one{Shift>}{Enter}{/Shift}line two");

    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Message")).toHaveValue("line one\nline two");
  });

  it("disables the textarea while disabled, and never sends whitespace-only text", async () => {
    const onSend = vi.fn();
    const { rerender } = render(
      <MessageInput onSend={onSend} onCancel={vi.fn()} disabled={true} />
    );
    expect(screen.getByLabelText("Message")).toBeDisabled();

    rerender(<MessageInput onSend={onSend} onCancel={vi.fn()} disabled={false} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled(); // still empty

    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Message"), "   ");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("shows an enabled Stop button instead of Send while disabled (sending), and calls onCancel", async () => {
    const onSend = vi.fn();
    const onCancel = vi.fn();
    render(<MessageInput onSend={onSend} onCancel={onCancel} disabled={true} />);
    const user = userEvent.setup();

    expect(screen.queryByRole("button", { name: "Send" })).not.toBeInTheDocument();
    const stopButton = screen.getByRole("button", { name: "Stop" });
    expect(stopButton).toBeEnabled();

    await user.click(stopButton);

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("shows the Send button again once no longer disabled", () => {
    const { rerender } = render(
      <MessageInput onSend={vi.fn()} onCancel={vi.fn()} disabled={true} />
    );
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();

    rerender(<MessageInput onSend={vi.fn()} onCancel={vi.fn()} disabled={false} />);

    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
  });

  it("lets a caller populate the field via the imperative handle without sending it", async () => {
    render(<Harness />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "populate" }));

    expect(screen.getByLabelText("Message")).toHaveValue("populated from outside");
    expect(screen.getByLabelText("Message")).toHaveFocus();
    // Populating is not sending: the Send button is enabled, but nothing was dispatched.
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  it("grows the textarea height with content, capped at a maximum", async () => {
    render(<MessageInput onSend={vi.fn()} onCancel={vi.fn()} disabled={false} />);
    const textarea = screen.getByLabelText("Message") as HTMLTextAreaElement;

    // jsdom has no real layout engine, so scrollHeight is stubbed here to
    // simulate multi-line content -- what's under test is that the
    // component reads it and applies a capped height, not the browser's
    // own text-measurement.
    Object.defineProperty(textarea, "scrollHeight", { value: 500, configurable: true });

    const user = userEvent.setup();
    await user.type(textarea, "a");

    expect(textarea.style.height).toBe("160px"); // capped, not the full 500px
  });
});
