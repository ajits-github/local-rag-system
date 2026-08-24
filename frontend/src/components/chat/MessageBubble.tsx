import { AgentActivityPanel } from "../AgentActivityPanel";
import { DebugPanel } from "../DebugPanel";
import { ErrorBanner } from "../ErrorBanner";
import { SourcesPanel } from "../SourcesPanel";
import type { ChatMessage } from "../../state/types";
import { Markdown } from "./Markdown";

const TERMINATION_NOTICE: Record<string, string> = {
  max_steps: "The agent stopped after reaching its step limit before fully resolving this question.",
  max_retrieval_attempts: "The agent stopped after its retry limit without finding sufficient evidence.",
  max_tool_calls: "The agent stopped after reaching its tool-call limit.",
  insufficient_evidence: "The agent could not find sufficient evidence to answer confidently.",
};

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const terminationReason = message.debug?.terminationReason;
  const terminationNotice = terminationReason ? TERMINATION_NOTICE[terminationReason] : undefined;

  return (
    <div className={`message-bubble message-bubble--${message.role}`} data-status={message.status}>
      <div className="message-bubble__avatar">{isUser ? "You" : "AI"}</div>
      <div className="message-bubble__content">
        {isUser && <p className="message-bubble__text">{message.text}</p>}

        {!isUser && message.status === "pending" && (
          <p className="message-bubble__loading" aria-live="polite">
            <span className="loading-dots" /> Thinking&hellip;
          </p>
        )}

        {!isUser &&
          (message.status === "streaming" || message.agentEvents.length > 0 || message.streamFellBack) && (
          <AgentActivityPanel
            events={message.agentEvents}
            isLive={message.status === "streaming"}
            streamFellBack={message.streamFellBack}
          />
        )}

        {!isUser && message.status === "error" && message.errorKind && (
          <ErrorBanner kind={message.errorKind} message={message.errorMessage} />
        )}

        {!isUser && message.status === "done" && (
          <>
            {message.insufficientEvidence && (
              <div className="notice notice--insufficient-evidence" role="status">
                Insufficient evidence was retrieved to fully answer this question.
              </div>
            )}
            {terminationNotice && !message.insufficientEvidence && (
              <div className="notice notice--termination" role="status">
                {terminationNotice}
              </div>
            )}
            <Markdown text={message.text} />
            <SourcesPanel sources={message.sources} />
            <DebugPanel debug={message.debug} />
          </>
        )}
      </div>
    </div>
  );
}
