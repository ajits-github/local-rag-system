import { useRef } from "react";
import { DevIdentityPanel, type DevIdentityPanelHandle } from "../DevIdentityPanel";
import { FeatureFlagsBar } from "../FeatureFlagsBar";
import { RuntimeInfoPanel } from "../RuntimeInfoPanel";
import { ModeToggle } from "../ModeToggle";
import { TokenStatusBadge } from "../TokenStatusBadge";
import { getUiModeConfig } from "../../config/uiMode";
import { useChat } from "../../state/chatContext";
import { useSendMessage } from "../../hooks/useSendMessage";
import { MessageInput, type MessageInputHandle } from "./MessageInput";
import { MessageList } from "./MessageList";

/**
 * Developer- vs. production-mode composition lives here and in
 * MessageBubble, not scattered per-leaf-component: ChatWindow decides
 * *whether* to mount the developer-only panels, the panels themselves
 * stay unaware of the mode. See src/config/uiMode.ts and
 * frontend/README.md's "UI modes" section.
 */
export function ChatWindow() {
  const { state, dispatch } = useChat();
  const { sendMessage, isSending } = useSendMessage();
  const devIdentityRef = useRef<DevIdentityPanelHandle>(null);
  const messageInputRef = useRef<MessageInputHandle>(null);
  const { showDeveloperSettings, showRuntimeDetails, showDeveloperModeIndicator } = getUiModeConfig();

  return (
    <div className="chat-window">
      <header className="chat-window__header">
        <h1>Local RAG Chat</h1>
        <div className="chat-window__controls">
          {showDeveloperModeIndicator && (
            <span
              className="developer-mode-badge"
              title="Developer/demo build -- see frontend/README.md's UI modes section"
            >
              Developer Mode
            </span>
          )}
          <TokenStatusBadge
            token={state.devIdentity.bearerToken}
            onReplaceToken={() => devIdentityRef.current?.openForEdit()}
          />
          <ModeToggle mode={state.mode} onChange={(mode) => dispatch({ type: "SET_MODE", mode })} />
          <button
            type="button"
            className="new-chat-button"
            onClick={() => dispatch({ type: "NEW_CHAT" })}
            disabled={state.messages.length === 0}
          >
            New chat
          </button>
        </div>
      </header>

      {showRuntimeDetails && (
        <>
          <FeatureFlagsBar />
          <RuntimeInfoPanel />
        </>
      )}

      {showDeveloperSettings && (
        <DevIdentityPanel
          ref={devIdentityRef}
          identity={state.devIdentity}
          onChange={(identity) => dispatch({ type: "SET_DEV_IDENTITY", identity })}
        />
      )}

      <MessageList
        messages={state.messages}
        onExampleQuery={(text) => messageInputRef.current?.setValue(text)}
      />

      <MessageInput ref={messageInputRef} onSend={sendMessage} disabled={isSending} />
    </div>
  );
}
