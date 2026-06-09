/**
 * Claw LLM Router — Gemini thought_signature cache + replay.
 *
 * Gemini 3+ models attach a cryptographically-signed `thought_signature` to
 * every tool_call they emit via the OpenAI-compatible endpoint:
 *
 *   choices[N].delta.tool_calls[M].extra_content.google.thought_signature
 *
 * On subsequent requests, the assistant message that carries the same
 * tool_call MUST include the same signature, or Gemini returns:
 *
 *   400 — "Function call is missing a thought_signature in functionCall parts."
 *
 * OpenClaw's agent layer doesn't know about this Google-specific field and
 * drops it when reconstructing history, so we cache signatures here keyed by
 * tool_call.id and replay them on the next request. For tool_calls whose
 * signature we never saw (e.g. session history from before this cache
 * existed, or a different runtime), we fall back to the documented
 * `skip_thought_signature_validator` sentinel string — Google's own escape
 * hatch for "transferring history from different models."
 *
 * Cache is in-process, capped, FIFO-evicted, and lost on plugin reload.
 * A reload is fine: every subsequent assistant turn will repopulate the
 * cache from Gemini's response before the next-turn replay needs it.
 */

const MAX_CACHE_ENTRIES = 5000;

/**
 * Documented Google bypass sentinel. Set as the `thought_signature` value
 * when no real signature is available. May lead to degraded model
 * performance — preferred only as a fallback.
 *
 * See: https://ai.google.dev/gemini-api/docs/thought-signatures
 */
export const BYPASS_SENTINEL = "skip_thought_signature_validator";

const cache = new Map<string, string>();

export function rememberSignature(toolCallId: string, signature: string): void {
  if (!toolCallId || !signature) return;
  // Move to insertion-newest position (Map iteration order = insertion order).
  if (cache.has(toolCallId)) cache.delete(toolCallId);
  cache.set(toolCallId, signature);
  // Evict oldest entries when over cap.
  while (cache.size > MAX_CACHE_ENTRIES) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

export function lookupSignature(toolCallId: string): string | undefined {
  if (!toolCallId) return undefined;
  return cache.get(toolCallId);
}

// Exposed for tests.
export function _clearCache(): void {
  cache.clear();
}
export function _cacheSize(): number {
  return cache.size;
}

// ── Request-side: inject signatures into outgoing tool_calls ────────────────

type ToolCall = {
  id?: unknown;
  extra_content?: unknown;
  [k: string]: unknown;
};
type Message = {
  role?: unknown;
  tool_calls?: unknown;
  [k: string]: unknown;
};

export type InjectionStats = { matched: number; bypassed: number; preserved: number };

/**
 * Mutates `payload.messages` so every assistant tool_call has a
 * `extra_content.google.thought_signature`:
 *   - existing signature → preserved
 *   - cached signature for this tool_call.id → injected
 *   - otherwise → bypass sentinel injected
 *
 * Returns counts for logging.
 */
export function injectThoughtSignatures(payload: Record<string, unknown>): InjectionStats {
  const stats: InjectionStats = { matched: 0, bypassed: 0, preserved: 0 };
  const messages = Array.isArray(payload.messages) ? (payload.messages as Message[]) : [];

  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const toolCalls = message.tool_calls;
    if (!Array.isArray(toolCalls)) continue;

    for (const rawTc of toolCalls) {
      if (!rawTc || typeof rawTc !== "object") continue;
      const tc = rawTc as ToolCall;

      const extra = (tc.extra_content as Record<string, unknown> | undefined) ?? {};
      const google = (extra.google as Record<string, unknown> | undefined) ?? {};

      if (typeof google.thought_signature === "string" && google.thought_signature.length > 0) {
        stats.preserved += 1;
        continue;
      }

      const id = typeof tc.id === "string" ? tc.id : "";
      const cached = id ? lookupSignature(id) : undefined;
      const signature = cached ?? BYPASS_SENTINEL;
      if (cached) stats.matched += 1;
      else stats.bypassed += 1;

      tc.extra_content = {
        ...extra,
        google: { ...google, thought_signature: signature },
      };
    }
  }

  return stats;
}

// ── Response-side: extract signatures from streamed/non-streamed responses ──

/**
 * Streaming SSE parser. Feed chunks of decoded SSE text; on stream end,
 * call `flush()` to receive every (id, signature) pair observed.
 *
 * Handles tool_calls split across deltas — Gemini may send `id` and
 * `extra_content.google.thought_signature` in different chunks. We
 * accumulate per (choice.index, tool_call.index) and emit only the pairs
 * where both arrived.
 */
export class StreamingSignatureExtractor {
  private buffer = "";
  // choice.index → tool_call.index → { id, sig }
  private byChoice = new Map<number, Map<number, { id?: string; sig?: string }>>();

  feed(text: string): void {
    this.buffer += text;
    // SSE event boundary = blank line. Keep the trailing partial event.
    const events = this.buffer.split("\n\n");
    this.buffer = events.pop() ?? "";
    for (const evt of events) this.parseEvent(evt);
  }

  finish(): void {
    // Drain any complete trailing event.
    if (this.buffer.length > 0) {
      this.parseEvent(this.buffer);
      this.buffer = "";
    }
  }

  flush(): Array<{ id: string; signature: string }> {
    const out: Array<{ id: string; signature: string }> = [];
    for (const perChoice of this.byChoice.values()) {
      for (const acc of perChoice.values()) {
        if (acc.id && acc.sig) out.push({ id: acc.id, signature: acc.sig });
      }
    }
    return out;
  }

  private parseEvent(evt: string): void {
    for (const rawLine of evt.split("\n")) {
      const line = rawLine.trimStart();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;

      let json: unknown;
      try {
        json = JSON.parse(payload);
      } catch {
        continue;
      }
      const choices = (json as { choices?: unknown }).choices;
      if (!Array.isArray(choices)) continue;

      for (const c of choices) {
        if (!c || typeof c !== "object") continue;
        const choice = c as { index?: unknown; delta?: unknown };
        const idx = typeof choice.index === "number" ? choice.index : 0;
        const delta = choice.delta as { tool_calls?: unknown } | undefined;
        if (!delta) continue;
        const tcs = delta.tool_calls;
        if (!Array.isArray(tcs)) continue;

        let perTc = this.byChoice.get(idx);
        if (!perTc) {
          perTc = new Map();
          this.byChoice.set(idx, perTc);
        }

        for (const t of tcs) {
          if (!t || typeof t !== "object") continue;
          const tc = t as { index?: unknown; id?: unknown; extra_content?: unknown };
          const tcIdx = typeof tc.index === "number" ? tc.index : 0;
          let acc = perTc.get(tcIdx);
          if (!acc) {
            acc = {};
            perTc.set(tcIdx, acc);
          }
          if (typeof tc.id === "string" && tc.id) acc.id = tc.id;
          const extra = tc.extra_content as { google?: unknown } | undefined;
          const google = extra?.google as { thought_signature?: unknown } | undefined;
          if (typeof google?.thought_signature === "string" && google.thought_signature) {
            acc.sig = google.thought_signature;
          }
        }
      }
    }
  }
}

/**
 * Extract (id, signature) pairs from a non-streamed JSON response body.
 */
export function extractSignaturesFromResponse(
  body: Record<string, unknown>,
): Array<{ id: string; signature: string }> {
  const out: Array<{ id: string; signature: string }> = [];
  const choices = body.choices;
  if (!Array.isArray(choices)) return out;
  for (const c of choices) {
    const msg = (c as { message?: { tool_calls?: unknown } }).message;
    const tcs = msg?.tool_calls;
    if (!Array.isArray(tcs)) continue;
    for (const t of tcs) {
      const tc = t as {
        id?: unknown;
        extra_content?: { google?: { thought_signature?: unknown } };
      };
      const id = typeof tc.id === "string" ? tc.id : "";
      const sig = tc.extra_content?.google?.thought_signature;
      if (id && typeof sig === "string" && sig) out.push({ id, signature: sig });
    }
  }
  return out;
}
