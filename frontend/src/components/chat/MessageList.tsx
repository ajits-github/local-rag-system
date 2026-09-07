import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../state/types";
import { MessageBubble } from "./MessageBubble";

// Grounded in real, tenant-unrestricted TechFusion knowledge-base
// content (see PROJECT_JOURNAL.md's "Empty chat state" entry for where
// each one was verified) rather than a fabricated case ID or document
// name, per this milestone's "don't hard-code misleading examples" rule.
const EXAMPLE_QUERIES = [
  "What is the webhook delivery timeout?",
  "How long can production write access last?",
  "How long can a case wait before it needs manual review?",
];

export function MessageList({
  messages,
  onExampleQuery,
}: {
  messages: ChatMessage[];
  onExampleQuery?: (text: string) => void;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="message-list message-list--empty">
        <p className="message-list__empty-title">Ask about the TechFusion knowledge base</p>
        <p className="message-list__empty-subtitle">
          Answers are grounded in the ingested documents, with citations you can inspect.
        </p>
        {onExampleQuery && (
          <div className="message-list__examples" role="group" aria-label="Example questions">
            {EXAMPLE_QUERIES.map((query) => (
              <button
                key={query}
                type="button"
                className="example-query-chip"
                onClick={() => onExampleQuery(query)}
              >
                {query}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="message-list" aria-live="polite">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
