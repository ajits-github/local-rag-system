import { useCallback, useRef, useState } from "react";
import { postAgentQuery, streamAgentQuery } from "../api/agentQuery";
import { RagApiError } from "../api/client";
import { postQuery } from "../api/query";
import type { AgentQueryResponse, QueryResponse } from "../api/types";
import { useChat } from "../state/chatContext";
import { debugFromAgentResponse, debugFromQueryResponse } from "../state/types";

let nextId = 0;
function newMessageId(): string {
  nextId += 1;
  return `msg-${Date.now()}-${nextId}`;
}

function isInsufficientEvidence(response: QueryResponse | AgentQueryResponse): boolean {
  if ("termination_reason" in response && response.termination_reason === "insufficient_evidence") {
    return true;
  }
  return response.sources.length === 0;
}

export function useSendMessage() {
  const { state, dispatch } = useChat();
  const [isSending, setIsSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (rawText: string) => {
      const text = rawText.trim();
      if (!text || isSending) return;

      const userId = newMessageId();
      const assistantId = newMessageId();
      const mode = state.mode;
      const identity = state.devIdentity;

      dispatch({ type: "ADD_USER_MESSAGE", id: userId, text, mode });
      dispatch({ type: "ADD_ASSISTANT_PLACEHOLDER", id: assistantId, mode });
      setIsSending(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        if (mode === "classic") {
          const response = await postQuery(text, identity, { signal: controller.signal });
          dispatch({
            type: "COMPLETE_ASSISTANT_ANSWER",
            id: assistantId,
            text: response.answer,
            sources: response.sources,
            debug: debugFromQueryResponse(response),
            insufficientEvidence: isInsufficientEvidence(response),
          });
          return;
        }

        // Agent mode: stream for live progress, falling back to the
        // non-streaming endpoint if live events are disabled server-side.
        try {
          for await (const item of streamAgentQuery(text, identity, { signal: controller.signal })) {
            if (item.type === "event") {
              dispatch({ type: "APPEND_AGENT_EVENT", id: assistantId, event: item.event });
            } else {
              dispatch({
                type: "COMPLETE_ASSISTANT_ANSWER",
                id: assistantId,
                text: item.response.answer,
                sources: item.response.sources,
                debug: debugFromAgentResponse(item.response),
                insufficientEvidence: isInsufficientEvidence(item.response),
              });
            }
          }
        } catch (err) {
          if (err instanceof RagApiError && err.kind === "not_found") {
            dispatch({ type: "SET_STREAM_FELL_BACK", id: assistantId });
            const response = await postAgentQuery(text, identity, { signal: controller.signal });
            dispatch({
              type: "COMPLETE_ASSISTANT_ANSWER",
              id: assistantId,
              text: response.answer,
              sources: response.sources,
              debug: debugFromAgentResponse(response),
              insufficientEvidence: isInsufficientEvidence(response),
            });
            return;
          }
          throw err;
        }
      } catch (err) {
        const kind = err instanceof RagApiError ? err.kind : "server_error";
        const message = err instanceof Error ? err.message : "Something went wrong.";
        dispatch({ type: "FAIL_ASSISTANT_MESSAGE", id: assistantId, kind, message });
      } finally {
        setIsSending(false);
        abortRef.current = null;
      }
    },
    [dispatch, isSending, state.devIdentity, state.mode]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { sendMessage, cancel, isSending };
}
