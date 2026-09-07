import { useRef } from "react";
import { DevIdentityPanel, type DevIdentityPanelHandle } from "../DevIdentityPanel";
import { FeatureFlagsBar } from "../FeatureFlagsBar";
import { RuntimeInfoPanel } from "../RuntimeInfoPanel";
import { ModeToggle } from "../ModeToggle";
import { TokenStatusBadge } from "../TokenStatusBadge";
import { useChat } from "../../state/chatContext";
import { useSendMessage } from "../../hooks/useSendMessage";
import { MessageInput } from "./MessageInput";
import { MessageList } from "./MessageList";

export function ChatWindow() {
  const { state, dispatch } = useChat();
  const { sendMessage, isSending } = useSendMessage();
  const devIdentityRef = useRef<DevIdentityPanelHandle>(null);

  return (
    <div className="chat-window">
      <header className="chat-window__header">
        <h1>Local RAG Chat</h1>
        <div className="chat-window__controls">
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

      <FeatureFlagsBar />

      <RuntimeInfoPanel />

      <DevIdentityPanel
        ref={devIdentityRef}
        identity={state.devIdentity}
        onChange={(identity) => dispatch({ type: "SET_DEV_IDENTITY", identity })}
      />

      <MessageList messages={state.messages} />

      <MessageInput onSend={sendMessage} disabled={isSending} />
    </div>
  );
}
