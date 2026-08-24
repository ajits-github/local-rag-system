import { useState, type KeyboardEvent } from "react";

export function MessageInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void;
  disabled: boolean;
}) {
  const [value, setValue] = useState("");

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
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question... (Enter to send, Shift+Enter for a new line)"
        rows={2}
        disabled={disabled}
        aria-label="Message"
      />
      <button type="button" onClick={submit} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  );
}
