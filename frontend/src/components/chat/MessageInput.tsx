import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type KeyboardEvent } from "react";

/** Lets a sibling (an empty-state example-query chip) populate the composer without submitting it. */
export interface MessageInputHandle {
  setValue: (text: string) => void;
}

// Keeps the composer compact when empty/short, but lets it grow with
// multi-line content up to a cap (matches the CSS max-height below);
// beyond the cap the textarea scrolls internally instead of growing
// further.
const MAX_TEXTAREA_HEIGHT_PX = 160;

export const MessageInput = forwardRef<
  MessageInputHandle,
  { onSend: (text: string) => void; disabled: boolean }
>(function MessageInput({ onSend, disabled }, ref) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(ref, () => ({
    setValue: (text: string) => {
      setValue(text);
      textareaRef.current?.focus();
    },
  }));

  // Auto-grow: measure natural content height on every change and apply
  // it directly, capped so a very long paste scrolls instead of taking
  // over the page.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || disabled) return;
    onSend(value);
    setValue("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
    // Shift+Enter falls through to the textarea's default newline behavior.
  };

  return (
    <div className="message-input">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question... (Enter to send, Shift+Enter for a new line)"
        rows={1}
        disabled={disabled}
        aria-label="Message"
      />
      <button type="button" onClick={submit} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  );
});
