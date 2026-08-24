import { describe, expect, it } from "vitest";
import { readSseFrames, splitSseFrames } from "../../src/utils/sse";

describe("splitSseFrames", () => {
  it("parses complete event/data frames and holds back a trailing partial frame", () => {
    const buffer = 'event: tool_started\ndata: {"a":1}\n\nevent: completed\ndata: {"b":2}\n\nevent: par';
    const { frames, remainder } = splitSseFrames(buffer);
    expect(frames).toEqual([
      { event: "tool_started", data: '{"a":1}' },
      { event: "completed", data: '{"b":2}' },
    ]);
    expect(remainder).toBe("event: par");
  });

  it("defaults to event type 'message' when no event: line is present", () => {
    const { frames } = splitSseFrames('data: {"x":1}\n\n');
    expect(frames[0].event).toBe("message");
  });
});

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i]));
        i += 1;
      } else {
        controller.close();
      }
    },
  });
}

describe("readSseFrames", () => {
  it("yields frames as they arrive, even when a frame is split across chunks", async () => {
    const body = streamFrom(["event: tool_st", 'arted\ndata: {"n":1}\n\n', "event: completed\ndata: {}\n\n"]);
    const response = { body } as unknown as Response;

    const frames = [];
    for await (const frame of readSseFrames(response)) {
      frames.push(frame);
    }

    expect(frames).toEqual([
      { event: "tool_started", data: '{"n":1}' },
      { event: "completed", data: "{}" },
    ]);
  });

  it("throws when the response has no body", async () => {
    const response = { body: null } as unknown as Response;
    await expect(async () => {
      const iterator = readSseFrames(response)[Symbol.asyncIterator]();
      await iterator.next();
    }).rejects.toThrow();
  });
});
