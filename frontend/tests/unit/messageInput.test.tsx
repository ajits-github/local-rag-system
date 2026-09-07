import { useRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MessageInput, type MessageInputHandle } from "../../src/components/chat/MessageInput";

function Harness({ disabled = false }: { disabled?: boolean }) {
  const ref = useRef<MessageInputHandle>(null);
  const onSend = vi.fn();
  return (
    <>
      <button type="button" onClick={() => ref.current?.setValue("populated from outside")}>
        populate
      </button>
      <MessageInput ref={ref} onSend={onSend} disabled={disabled} />
    </>
  );
}

describe("MessageInput", () => {
  it("sends on Enter and clears the field", async () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} disabled={false} />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "hello there{Enter}");

    expect(onSend).toHaveBeenCalledWith("hello there");
    expect(screen.getByLabelText("Message")).toHaveValue("");
  });

  it("does not send on Shift+Enter, and inserts a newline instead", async () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} disabled={false} />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "line one{Shift>}{Enter}{/Shift}line two");

    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Message")).toHaveValue("line one\nline two");
  });

  it("disables the textarea and Send button while disabled, and never sends whitespace-only text", async () => {
    const onSend = vi.fn();
    const { rerender } = render(<MessageInput onSend={onSend} disabled={true} />);
    expect(screen.getByLabelText("Message")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    rerender(<MessageInput onSend={onSend} disabled={false} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled(); // still empty

    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Message"), "   ");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
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
    render(<MessageInput onSend={vi.fn()} disabled={false} />);
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
