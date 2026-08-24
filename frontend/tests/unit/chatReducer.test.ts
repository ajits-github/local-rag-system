import { describe, expect, it } from "vitest";
import { chatReducer, initialChatState } from "../../src/state/chatContext";

describe("chatReducer", () => {
  it("adds a user message and an assistant placeholder", () => {
    let state = initialChatState();
    state = chatReducer(state, { type: "ADD_USER_MESSAGE", id: "u1", text: "hello", mode: "classic" });
    state = chatReducer(state, { type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });

    expect(state.messages).toHaveLength(2);
    expect(state.messages[0]).toMatchObject({ role: "user", text: "hello", status: "done" });
    expect(state.messages[1]).toMatchObject({ role: "assistant", status: "pending" });
  });

  it("appends agent events and marks the message as streaming", () => {
    let state = initialChatState();
    state = chatReducer(state, { type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "agent" });
    state = chatReducer(state, {
      type: "APPEND_AGENT_EVENT",
      id: "a1",
      event: { event_type: "tool_started", tool_name: "search_knowledge_base" },
    });

    expect(state.messages[0].status).toBe("streaming");
    expect(state.messages[0].agentEvents).toHaveLength(1);
  });

  it("completes an assistant answer with sources and debug info", () => {
    let state = initialChatState();
    state = chatReducer(state, { type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });
    state = chatReducer(state, {
      type: "COMPLETE_ASSISTANT_ANSWER",
      id: "a1",
      text: "the answer",
      sources: [],
      debug: { route: "classic_rag" },
      insufficientEvidence: false,
    });

    expect(state.messages[0]).toMatchObject({ status: "done", text: "the answer", insufficientEvidence: false });
  });

  it("marks a message as errored without discarding it", () => {
    let state = initialChatState();
    state = chatReducer(state, { type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });
    state = chatReducer(state, {
      type: "FAIL_ASSISTANT_MESSAGE",
      id: "a1",
      kind: "backend_unavailable",
      message: "Could not reach the backend.",
    });

    expect(state.messages[0]).toMatchObject({ status: "error", errorKind: "backend_unavailable" });
  });

  it("clears messages on NEW_CHAT but keeps mode and dev identity", () => {
    let state = initialChatState();
    state = chatReducer(state, { type: "SET_MODE", mode: "agent" });
    state = chatReducer(state, { type: "ADD_USER_MESSAGE", id: "u1", text: "hi", mode: "agent" });
    state = chatReducer(state, { type: "NEW_CHAT" });

    expect(state.messages).toHaveLength(0);
    expect(state.mode).toBe("agent");
  });
});
