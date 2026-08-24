import { ChatWindow } from "./components/chat/ChatWindow";
import { ChatProvider } from "./state/chatContext";

export function App() {
  return (
    <ChatProvider>
      <ChatWindow />
    </ChatProvider>
  );
}
