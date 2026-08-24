import { createContext, useContext, useMemo, useReducer, type ReactNode } from "react";
import type { DevIdentity, RagMode, SourceItem } from "../api/types";
import type { RagApiErrorKind } from "../api/client";
import type { ChatMessage, DebugInfo } from "./types";

const DEV_IDENTITY_STORAGE_KEY = "rag-ui:dev-identity";

const emptyIdentity: DevIdentity = {
  bearerToken: "",
  tenantId: "",
  roles: "",
  asOf: "",
  requireTrustLevel: "",
};

function loadDevIdentity(): DevIdentity {
  if (typeof window === "undefined") return emptyIdentity;
  try {
    const raw = window.sessionStorage.getItem(DEV_IDENTITY_STORAGE_KEY);
    if (!raw) return emptyIdentity;
    return { ...emptyIdentity, ...(JSON.parse(raw) as Partial<DevIdentity>) };
  } catch {
    return emptyIdentity;
  }
}

function saveDevIdentity(identity: DevIdentity): void {
  if (typeof window === "undefined") return;
  // Dev-only convenience token, kept out of localStorage deliberately so it
  // doesn't outlive the browser tab/session.
  window.sessionStorage.setItem(DEV_IDENTITY_STORAGE_KEY, JSON.stringify(identity));
}

interface ChatState {
  mode: RagMode;
  devIdentity: DevIdentity;
  messages: ChatMessage[];
}

type ChatAction =
  | { type: "SET_MODE"; mode: RagMode }
  | { type: "SET_DEV_IDENTITY"; identity: DevIdentity }
  | { type: "ADD_USER_MESSAGE"; id: string; text: string; mode: RagMode }
  | { type: "ADD_ASSISTANT_PLACEHOLDER"; id: string; mode: RagMode }
  | { type: "APPEND_AGENT_EVENT"; id: string; event: ChatMessage["agentEvents"][number] }
  | { type: "SET_STREAM_FELL_BACK"; id: string }
  | {
      type: "COMPLETE_ASSISTANT_ANSWER";
      id: string;
      text: string;
      sources: SourceItem[];
      debug: DebugInfo;
      insufficientEvidence: boolean;
    }
  | { type: "FAIL_ASSISTANT_MESSAGE"; id: string; kind: RagApiErrorKind; message: string }
  | { type: "NEW_CHAT" };

function initialState(): ChatState {
  return { mode: "classic", devIdentity: loadDevIdentity(), messages: [] };
}

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "SET_MODE":
      return { ...state, mode: action.mode };
    case "SET_DEV_IDENTITY":
      saveDevIdentity(action.identity);
      return { ...state, devIdentity: action.identity };
    case "ADD_USER_MESSAGE":
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: action.id,
            role: "user",
            mode: action.mode,
            status: "done",
            text: action.text,
            sources: [],
            agentEvents: [],
            createdAt: Date.now(),
          },
        ],
      };
    case "ADD_ASSISTANT_PLACEHOLDER":
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: action.id,
            role: "assistant",
            mode: action.mode,
            status: "pending",
            text: "",
            sources: [],
            agentEvents: [],
            createdAt: Date.now(),
          },
        ],
      };
    case "APPEND_AGENT_EVENT":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id
            ? { ...m, status: "streaming", agentEvents: [...m.agentEvents, action.event] }
            : m
        ),
      };
    case "SET_STREAM_FELL_BACK":
      return {
        ...state,
        messages: state.messages.map((m) => (m.id === action.id ? { ...m, streamFellBack: true } : m)),
      };
    case "COMPLETE_ASSISTANT_ANSWER":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id
            ? {
                ...m,
                status: "done",
                text: action.text,
                sources: action.sources,
                debug: action.debug,
                insufficientEvidence: action.insufficientEvidence,
              }
            : m
        ),
      };
    case "FAIL_ASSISTANT_MESSAGE":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id
            ? { ...m, status: "error", errorKind: action.kind, errorMessage: action.message }
            : m
        ),
      };
    case "NEW_CHAT":
      return { ...state, messages: [] };
    default:
      return state;
  }
}

interface ChatContextValue {
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, undefined, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within a ChatProvider");
  return ctx;
}

export { chatReducer, initialState as initialChatState };
export type { ChatState, ChatAction };
