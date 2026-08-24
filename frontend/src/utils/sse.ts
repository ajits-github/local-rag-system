/**
 * Minimal parser for the `text/event-stream` frames emitted by
 * POST /agent/query/stream (src/rag/api/routers/agent_stream.py's
 * `_sse_message`: `event: <type>\ndata: <json>\n\n`).
 *
 * The browser's native EventSource can't be used here; it only supports
 * GET requests, and this endpoint is POST by design (see that router's
 * module docstring), so frames are parsed by hand off the response body's
 * ReadableStream.
 */

export interface SseFrame {
  event: string;
  data: string;
}

/** Split raw stream text into complete `event:`/`data:` frames, holding back any trailing partial frame. */
export function splitSseFrames(buffer: string): { frames: SseFrame[]; remainder: string } {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() ?? "";
  const frames: SseFrame[] = [];
  for (const part of parts) {
    if (!part.trim()) continue;
    let event = "message";
    let data = "";
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        data = line.slice("data:".length).trim();
      }
    }
    frames.push({ event, data });
  }
  return { frames, remainder };
}

/** Read a fetch Response's body as a sequence of SSE frames. */
export async function* readSseFrames(response: Response, signal?: AbortSignal): AsyncGenerator<SseFrame> {
  if (!response.body) {
    throw new Error("Response has no readable body for streaming.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { frames, remainder } = splitSseFrames(buffer);
      buffer = remainder;
      for (const frame of frames) {
        yield frame;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
