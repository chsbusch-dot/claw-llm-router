import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import {
  BYPASS_SENTINEL,
  StreamingSignatureExtractor,
  _cacheSize,
  _clearCache,
  extractSignaturesFromResponse,
  injectThoughtSignatures,
  lookupSignature,
  rememberSignature,
} from "../../providers/gemini-thought-signatures.js";

// ── Cache ───────────────────────────────────────────────────────────────────

describe("gemini-thought-signatures cache", () => {
  beforeEach(() => _clearCache());

  it("stores and retrieves signatures by tool_call.id", () => {
    rememberSignature("call_1", "sigA");
    rememberSignature("call_2", "sigB");
    assert.equal(lookupSignature("call_1"), "sigA");
    assert.equal(lookupSignature("call_2"), "sigB");
    assert.equal(_cacheSize(), 2);
  });

  it("ignores empty id or empty signature", () => {
    rememberSignature("", "sig");
    rememberSignature("id", "");
    assert.equal(_cacheSize(), 0);
  });

  it("returns undefined for unknown id or empty id", () => {
    rememberSignature("call_1", "sigA");
    assert.equal(lookupSignature("call_2"), undefined);
    assert.equal(lookupSignature(""), undefined);
  });

  it("updates value when the same id is set again", () => {
    rememberSignature("call_1", "sigA");
    rememberSignature("call_1", "sigB");
    assert.equal(lookupSignature("call_1"), "sigB");
    assert.equal(_cacheSize(), 1);
  });
});

// ── Injection ───────────────────────────────────────────────────────────────

describe("injectThoughtSignatures", () => {
  beforeEach(() => _clearCache());

  it("injects bypass sentinel when no cached signature exists", () => {
    const payload = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "f", arguments: "{}" } },
          ],
        },
      ],
    } as Record<string, unknown>;

    const stats = injectThoughtSignatures(payload);
    assert.equal(stats.bypassed, 1);
    assert.equal(stats.matched, 0);
    assert.equal(stats.preserved, 0);

    const tc = (payload.messages as Array<{ tool_calls: Array<Record<string, unknown>> }>)[0]
      .tool_calls[0];
    const sig = (tc.extra_content as { google: { thought_signature: string } }).google
      .thought_signature;
    assert.equal(sig, BYPASS_SENTINEL);
  });

  it("injects cached signature when available", () => {
    rememberSignature("call_1", "real_signature_xyz");
    const payload = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "f", arguments: "{}" } },
          ],
        },
      ],
    } as Record<string, unknown>;

    const stats = injectThoughtSignatures(payload);
    assert.equal(stats.matched, 1);
    assert.equal(stats.bypassed, 0);

    const tc = (payload.messages as Array<{ tool_calls: Array<Record<string, unknown>> }>)[0]
      .tool_calls[0];
    const sig = (tc.extra_content as { google: { thought_signature: string } }).google
      .thought_signature;
    assert.equal(sig, "real_signature_xyz");
  });

  it("preserves existing signature without overwriting", () => {
    rememberSignature("call_1", "would_overwrite");
    const payload = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "f", arguments: "{}" },
              extra_content: { google: { thought_signature: "already_present" } },
            },
          ],
        },
      ],
    } as Record<string, unknown>;

    const stats = injectThoughtSignatures(payload);
    assert.equal(stats.preserved, 1);
    assert.equal(stats.matched, 0);
    assert.equal(stats.bypassed, 0);

    const tc = (payload.messages as Array<{ tool_calls: Array<Record<string, unknown>> }>)[0]
      .tool_calls[0];
    const sig = (tc.extra_content as { google: { thought_signature: string } }).google
      .thought_signature;
    assert.equal(sig, "already_present");
  });

  it("ignores non-assistant messages and messages without tool_calls", () => {
    const payload = {
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
        { role: "tool", tool_call_id: "call_1", content: "result" },
      ],
    } as Record<string, unknown>;

    const stats = injectThoughtSignatures(payload);
    assert.equal(stats.matched + stats.bypassed + stats.preserved, 0);
  });

  it("merges into existing extra_content/google without clobbering other fields", () => {
    const payload = {
      messages: [
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "f", arguments: "{}" },
              extra_content: { google: { other_field: "keep_me" }, vendor_x: 1 },
            },
          ],
        },
      ],
    } as Record<string, unknown>;

    injectThoughtSignatures(payload);
    const tc = (payload.messages as Array<{ tool_calls: Array<Record<string, unknown>> }>)[0]
      .tool_calls[0];
    const extra = tc.extra_content as { google: Record<string, unknown>; vendor_x: unknown };
    assert.equal(extra.vendor_x, 1);
    assert.equal(extra.google.other_field, "keep_me");
    assert.equal(extra.google.thought_signature, BYPASS_SENTINEL);
  });
});

// ── Streaming extraction ────────────────────────────────────────────────────

describe("StreamingSignatureExtractor", () => {
  it("captures id and signature when arriving in the same chunk", () => {
    const ex = new StreamingSignatureExtractor();
    const evt =
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""},"extra_content":{"google":{"thought_signature":"sig_in_one_chunk"}}}]}}]}\n\n';
    ex.feed(evt);
    ex.finish();
    assert.deepEqual(ex.flush(), [{ id: "call_1", signature: "sig_in_one_chunk" }]);
  });

  it("merges id and signature arriving in separate deltas", () => {
    const ex = new StreamingSignatureExtractor();
    ex.feed(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_split","type":"function","function":{"name":"f"}}]}}]}\n\n',
    );
    ex.feed(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"extra_content":{"google":{"thought_signature":"sig_later"}}}]}}]}\n\n',
    );
    ex.finish();
    assert.deepEqual(ex.flush(), [{ id: "call_split", signature: "sig_later" }]);
  });

  it("handles SSE event split across multiple feed() calls", () => {
    const ex = new StreamingSignatureExtractor();
    const evt =
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_chunked","type":"function","extra_content":{"google":{"thought_signature":"sig_chunked"}}}]}}]}\n\n';
    // Feed the event one byte at a time.
    for (const ch of evt) ex.feed(ch);
    ex.finish();
    assert.deepEqual(ex.flush(), [{ id: "call_chunked", signature: "sig_chunked" }]);
  });

  it("ignores [DONE] and malformed data lines", () => {
    const ex = new StreamingSignatureExtractor();
    ex.feed("data: [DONE]\n\n");
    ex.feed("data: not-json\n\n");
    ex.feed("data: {}\n\n");
    ex.finish();
    assert.deepEqual(ex.flush(), []);
  });

  it("captures multiple distinct tool_calls in a single response", () => {
    const ex = new StreamingSignatureExtractor();
    ex.feed(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_A","extra_content":{"google":{"thought_signature":"sigA"}}}]}}]}\n\n',
    );
    ex.feed(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_B","extra_content":{"google":{"thought_signature":"sigB"}}}]}}]}\n\n',
    );
    ex.finish();
    const out = ex.flush().sort((a, b) => a.id.localeCompare(b.id));
    assert.deepEqual(out, [
      { id: "call_A", signature: "sigA" },
      { id: "call_B", signature: "sigB" },
    ]);
  });

  it("emits nothing when id is present but signature never arrived", () => {
    const ex = new StreamingSignatureExtractor();
    ex.feed(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_no_sig"}]}}]}\n\n',
    );
    ex.finish();
    assert.deepEqual(ex.flush(), []);
  });
});

// ── Non-streaming extraction ────────────────────────────────────────────────

describe("extractSignaturesFromResponse", () => {
  it("pulls signatures out of choices[N].message.tool_calls", () => {
    const body = {
      choices: [
        {
          message: {
            role: "assistant",
            tool_calls: [
              {
                id: "call_X",
                type: "function",
                function: { name: "f", arguments: "{}" },
                extra_content: { google: { thought_signature: "sigX" } },
              },
            ],
          },
        },
      ],
    } as Record<string, unknown>;
    const out = extractSignaturesFromResponse(body);
    assert.deepEqual(out, [{ id: "call_X", signature: "sigX" }]);
  });

  it("returns empty when no tool_calls", () => {
    const body = {
      choices: [{ message: { role: "assistant", content: "plain text" } }],
    } as Record<string, unknown>;
    assert.deepEqual(extractSignaturesFromResponse(body), []);
  });

  it("returns empty when choices missing", () => {
    assert.deepEqual(extractSignaturesFromResponse({}), []);
  });
});
