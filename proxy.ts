/**
 * Claw LLM Router — In-Process HTTP Proxy
 *
 * Runs inside the OpenClaw gateway process (no subprocess).
 * Classifies prompts locally, then routes to the right model via
 * direct provider calls (OpenAI-compatible, Anthropic Messages API,
 * or gateway fallback for OAuth tokens).
 *
 * Auth is NEVER stored in the plugin — keys are read from OpenClaw's auth stores.
 */

import { createServer, type Server, type IncomingMessage, type ServerResponse } from "node:http";
import { classify, tierFromModelId, FALLBACK_CHAIN, type Tier } from "./classifier.js";
import { PROXY_PORT } from "./models.js";
import { loadTierConfig } from "./tier-config.js";
import { callProvider, MissingApiKeyError } from "./providers/index.js";
import type { PluginLogger, ChatMessage } from "./providers/types.js";
import { RouterLogger } from "./router-logger.js";

// ── Message extraction ───────────────────────────────────────────────────────

function extractUserPrompt(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "user") {
      if (typeof m.content === "string") return m.content;
      if (Array.isArray(m.content)) {
        return (m.content as Array<{ type: string; text?: string }>)
          .filter((c) => c.type === "text")
          .map((c) => c.text ?? "")
          .join(" ");
      }
    }
  }
  return "";
}

function extractSystemPrompt(messages: ChatMessage[]): string {
  return messages
    .filter((m) => m.role === "system")
    .map((m) => (typeof m.content === "string" ? m.content : ""))
    .join(" ");
}

function estimateMessageTokens(messages: ChatMessage[]): number {
  let chars = 0;
  for (const message of messages) {
    if (typeof message.content === "string") {
      chars += message.content.length;
      continue;
    }
    if (Array.isArray(message.content)) {
      chars += message.content
        .filter((item) => item.type === "text")
        .map((item) => item.text ?? "")
        .join(" ").length;
    }
  }
  return Math.floor(chars / 4);
}

function isLocalLlamaCppProvider(spec: { provider: string; baseUrl: string }): boolean {
  return (
    spec.provider === "llamacpp" ||
    spec.baseUrl.includes("127.0.0.1:8080") ||
    spec.baseUrl.includes("localhost:8080")
  );
}

// ── Request router ────────────────────────────────────────────────────────────

async function handleChatCompletion(
  _req: IncomingMessage,
  res: ServerResponse,
  body: Record<string, unknown>,
  log: PluginLogger,
): Promise<void> {
  const messages = (body.messages ?? []) as ChatMessage[];
  const stream = (body.stream as boolean) ?? false;
  const modelId = ((body.model as string) ?? "auto").replace("claw-llm-router/", "");

  const rlog = new RouterLogger(log);
  const userPrompt = extractUserPrompt(messages);
  const systemPrompt = extractSystemPrompt(messages);
  const estimatedRequestTokens = estimateMessageTokens(messages);

  // ── Extract classifiable prompt ──────────────────────────────────────────
  // The user message may contain more than just the user's input:
  //  1. Packed context (group chats / subagents): history + current message
  //  2. Embedded system prompt: system instructions prepended to user text
  // We need to isolate the actual user text for accurate classification.

  const isPackedContext =
    userPrompt.startsWith("[Chat messages since") || userPrompt.startsWith("[chat messages since");

  let classifiablePrompt = userPrompt;

  // Case 0: Strip "Conversation info (untrusted metadata)" wrapper
  // OpenClaw prepends message metadata as a fenced JSON block:
  //   Conversation info (untrusted metadata): ```json { ... }```\n\nActual prompt
  // Strip it before classification so ```/json don't pollute scoring.
  const metadataPrefix = "Conversation info (untrusted metadata):";
  if (classifiablePrompt.startsWith(metadataPrefix)) {
    const closingFence = classifiablePrompt.indexOf("```", metadataPrefix.length + 4); // skip opening ```
    if (closingFence !== -1) {
      classifiablePrompt = classifiablePrompt.slice(closingFence + 3).trim();
    }
  }

  if (isPackedContext) {
    // Case 1: Packed context — extract text after the current-message marker
    const marker = "[Current message - respond to this]";
    const markerIdx = userPrompt.indexOf(marker);
    if (markerIdx !== -1) {
      classifiablePrompt = userPrompt.slice(markerIdx + marker.length).trim();
    }
  } else if (systemPrompt && userPrompt.length > systemPrompt.length) {
    // Case 2: System prompt embedded in user message — strip it
    // Some paths (e.g. webchat) prepend the system prompt to the user message
    // instead of sending it as a separate system-role message.
    const sysIdx = userPrompt.indexOf(systemPrompt);
    if (sysIdx !== -1) {
      const stripped = (
        userPrompt.slice(0, sysIdx) + userPrompt.slice(sysIdx + systemPrompt.length)
      ).trim();
      if (stripped) classifiablePrompt = stripped;
    }
  } else if (!systemPrompt && userPrompt.length > 500) {
    // Case 3: No separate system message and user message is suspiciously long.
    // The system prompt is likely embedded. The actual user text is at the end,
    // after the last paragraph break.
    const lastBreak = userPrompt.lastIndexOf("\n\n");
    if (lastBreak !== -1) {
      const tail = userPrompt.slice(lastBreak).trim();
      if (tail && tail.length < 500) {
        classifiablePrompt = tail;
      }
    }
  }

  // ── Classify ────────────────────────────────────────────────────────────
  const extracted = classifiablePrompt !== userPrompt;
  rlog.request({
    model: modelId,
    stream,
    prompt: classifiablePrompt,
    extraction: extracted ? { from: userPrompt.length, to: classifiablePrompt.length } : undefined,
  });

  let tier: Tier;
  let classificationMethod: string;

  const tierOverride = tierFromModelId(modelId);
  if (tierOverride) {
    tier = tierOverride;
    classificationMethod = "forced";
    rlog.classify({ tier, method: "forced", detail: `(model=${modelId})` });
  } else if (isPackedContext && !classifiablePrompt) {
    tier = "MEDIUM";
    classificationMethod = "packed-default";
    rlog.classify({
      tier: "MEDIUM",
      method: "packed-default",
      detail: `(no current-message marker in ${userPrompt.length}-char packed context)`,
    });
  } else {
    const result = classify(classifiablePrompt);
    tier = result.tier;
    classificationMethod = "rule-based";
    rlog.classify({
      tier,
      method: "rule-based",
      score: result.score,
      confidence: result.confidence,
      signals: result.signals,
    });
  }

  // ── Route ──────────────────────────────────────────────────────────────
  const tierConfig = loadTierConfig();
  const chain = FALLBACK_CHAIN[tier];
  const targetSpec = tierConfig[tier];
  const LOCAL_CONTEXT_SOFT_LIMIT = 59000;
  const shouldSkipLocalForContext = estimatedRequestTokens >= LOCAL_CONTEXT_SOFT_LIMIT;
  // Tool-use requests (body.tools is a non-empty array) skip local llamacpp
  // tiers entirely. Small local models like Qwen3-30B accept tool definitions
  // but in practice tend to ignore them and hallucinate file contents instead.
  // Empirically a much bigger correctness win than the cost saving is worth.
  const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
  const shouldSkipLocalForTools = hasTools;
  const shouldSkipLocal = shouldSkipLocalForContext || shouldSkipLocalForTools;
  const attemptChain = shouldSkipLocal
    ? chain.filter((attemptTier) => !isLocalLlamaCppProvider(tierConfig[attemptTier]))
    : chain;
  rlog.route({
    tier,
    provider: targetSpec.provider,
    model: targetSpec.modelId,
    method: classificationMethod,
    chain: attemptChain.length > 0 ? attemptChain : chain,
  });

  if (shouldSkipLocalForContext && attemptChain.length > 0) {
    log.warn(
      `[claw-llm-router] skipping local tiers for large request (~${estimatedRequestTokens} tokens > ${LOCAL_CONTEXT_SOFT_LIMIT})`,
    );
  }
  if (shouldSkipLocalForTools && !shouldSkipLocalForContext && attemptChain.length > 0) {
    log.warn(
      `[claw-llm-router] skipping local tiers for tool-use request (${(body.tools as unknown[]).length} tools)`,
    );
  }

  let lastError: Error | undefined;
  let allMissingKeys = true;

  for (const attemptTier of attemptChain.length > 0 ? attemptChain : chain) {
    const spec = tierConfig[attemptTier];
    try {
      await callProvider(spec, body, stream, res, log);
      return; // success
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (!(err instanceof MissingApiKeyError)) allMissingKeys = false;
      rlog.fallback({
        tier: attemptTier,
        provider: spec.provider,
        model: spec.modelId,
        error: lastError.message,
      });
    }
  }

  rlog.failed({ chain, error: lastError?.message ?? "unknown" });
  if (!res.headersSent) {
    res.writeHead(502, { "Content-Type": "application/json" });
  }
  if (!res.writableEnded) {
    const message = allMissingKeys
      ? `No API keys configured. Run /router doctor to see what's needed, or set API keys for your providers (e.g. GEMINI_API_KEY, ANTHROPIC_API_KEY). See: https://github.com/donnfelker/claw-llm-router#troubleshooting`
      : `All providers failed: ${lastError?.message}`;
    res.end(
      JSON.stringify({
        error: { message, type: "router_error" },
      }),
    );
  }
}

// ── Server ────────────────────────────────────────────────────────────────────

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

const CREATED_AT = Math.floor(Date.now() / 1000);

export async function startProxy(log: PluginLogger): Promise<Server> {
  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    try {
      // Health check
      if (req.url === "/health" || req.url?.startsWith("/health?")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ok", version: "1.0.0" }));
        return;
      }

      // Models list
      if (req.url === "/v1/models" && req.method === "GET") {
        const { ROUTER_MODELS, PROVIDER_ID } = await import("./models.js");
        const models = ROUTER_MODELS.map((m) => ({
          id: m.id,
          object: "model",
          created: CREATED_AT,
          owned_by: PROVIDER_ID,
        }));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: models }));
        return;
      }

      // Chat completions
      if (req.url === "/v1/chat/completions" && req.method === "POST") {
        const rawBody = await readBody(req);
        let body: Record<string, unknown>;
        try {
          body = JSON.parse(rawBody.toString()) as Record<string, unknown>;
        } catch {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: "Invalid JSON", type: "invalid_request" } }));
          return;
        }
        await handleChatCompletion(req, res, body, log);
        return;
      }

      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Not found", type: "not_found" } }));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error(`Proxy error: ${msg}`);
      if (!res.headersSent) {
        res.writeHead(502, { "Content-Type": "application/json" });
      }
      if (!res.writableEnded) {
        res.end(JSON.stringify({ error: { message: msg, type: "proxy_error" } }));
      }
    }
  });

  return new Promise((resolve, reject) => {
    server.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "EADDRINUSE") {
        log.warn(`Port ${PROXY_PORT} already in use — proxy may already be running`);
        reject(err);
      } else {
        reject(err);
      }
    });

    server.listen(PROXY_PORT, "127.0.0.1", () => {
      log.info(`Proxy started on http://127.0.0.1:${PROXY_PORT}`);
      resolve(server);
    });
  });
}
