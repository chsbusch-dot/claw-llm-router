/**
 * Claw LLM Router — OpenAI-Compatible Provider
 *
 * Handles: Google Gemini, OpenAI, Groq, Mistral, DeepSeek, Together,
 * Fireworks, Perplexity, xAI, and any other OpenAI-compatible API.
 *
 * POST to {baseUrl}/chat/completions with Bearer auth.
 * Forwards request body with only standard OpenAI chat completion params
 * (non-standard fields like `store` cause 400 errors on Google/Groq/etc).
 */

import type { ServerResponse } from "node:http";
import { REQUEST_TIMEOUT_MS, type LLMProvider, type PluginLogger } from "./types.js";
import { RouterLogger } from "../router-logger.js";
import {
  StreamingSignatureExtractor,
  extractSignaturesFromResponse,
  injectThoughtSignatures,
  rememberSignature,
} from "./gemini-thought-signatures.js";

// Standard OpenAI chat completion parameters that providers generally accept.
// Non-standard or provider-specific fields (e.g. `store`, `metadata`) are stripped
// to avoid 400 errors from providers like Google Gemini.
const ALLOWED_PARAMS = new Set([
  "messages",
  "model",
  "stream",
  "max_tokens",
  "max_completion_tokens",
  "temperature",
  "top_p",
  "n",
  "stop",
  "presence_penalty",
  "frequency_penalty",
  "logit_bias",
  "logprobs",
  "top_logprobs",
  "response_format",
  "seed",
  "tools",
  "tool_choice",
  "parallel_tool_calls",
  "user",
  "stream_options",
  "service_tier",
]);

function sanitizeBody(body: Record<string, unknown>): Record<string, unknown> {
  const clean: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(body)) {
    if (ALLOWED_PARAMS.has(key)) {
      clean[key] = value;
    }
  }
  return clean;
}

function isGeminiOpenAICompatible(spec: { modelId: string; baseUrl: string }): boolean {
  return (
    spec.modelId.startsWith("gemini-") || spec.baseUrl.includes("generativelanguage.googleapis.com")
  );
}

export class OpenAICompatibleProvider implements LLMProvider {
  readonly name = "openai-compatible";

  async chatCompletion(
    body: Record<string, unknown>,
    spec: { modelId: string; apiKey: string; baseUrl: string },
    stream: boolean,
    res: ServerResponse,
    log: PluginLogger,
  ): Promise<void> {
    const url = `${spec.baseUrl}/chat/completions`;
    const payload: Record<string, unknown> = {
      ...sanitizeBody(body),
      model: spec.modelId,
      stream,
    };

    // Gemini 3+ requires that every assistant tool_call carry a signed
    // `thought_signature` in extra_content.google. Inject from cache or
    // fall back to Google's documented bypass sentinel. See:
    //   providers/gemini-thought-signatures.ts
    const isGemini = isGeminiOpenAICompatible(spec);
    if (isGemini && Array.isArray(payload.messages)) {
      const stats = injectThoughtSignatures(payload);
      if (stats.matched + stats.bypassed > 0) {
        log.info(
          `[claw-llm-router] gemini thought_signature: ${stats.matched} cached, ${stats.bypassed} bypassed, ${stats.preserved} preserved`,
        );
      }
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${spec.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`${spec.modelId} ${resp.status}: ${errText.slice(0, 300)}`);
      }

      const rlog = new RouterLogger(log);

      if (stream) {
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        });
        const reader = resp.body?.getReader();
        if (!reader) throw new Error(`No response body from ${spec.modelId}`);
        const decoder = new TextDecoder();
        // For Gemini: tee the stream into a signature extractor so we can
        // cache each tool_call.id → thought_signature pair for replay on
        // the next turn. Bytes still forward to the client untouched.
        const extractor = isGemini ? new StreamingSignatureExtractor() : null;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          if (extractor) extractor.feed(text);
          if (!res.writableEnded) res.write(text);
        }
        if (extractor) {
          extractor.finish();
          const pairs = extractor.flush();
          for (const { id, signature } of pairs) rememberSignature(id, signature);
          if (pairs.length > 0) {
            log.info(
              `[claw-llm-router] gemini cached ${pairs.length} thought_signature(s) from stream`,
            );
          }
        }
        if (!res.writableEnded) res.end();
        rlog.done({ model: spec.modelId, via: "direct", streamed: true });
      } else {
        const data = (await resp.json()) as Record<string, unknown>;
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(data));
        if (isGemini) {
          const pairs = extractSignaturesFromResponse(data);
          for (const { id, signature } of pairs) rememberSignature(id, signature);
          if (pairs.length > 0) {
            log.info(
              `[claw-llm-router] gemini cached ${pairs.length} thought_signature(s) from response`,
            );
          }
        }
        const usage = (data.usage ?? {}) as Record<string, number>;
        rlog.done({
          model: spec.modelId,
          via: "direct",
          streamed: false,
          tokensIn: usage.prompt_tokens ?? "?",
          tokensOut: usage.completion_tokens ?? "?",
        });
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new Error(`${spec.modelId} request timed out after ${REQUEST_TIMEOUT_MS}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
