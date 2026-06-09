/**
 * Claw LLM Router — Tier Configuration
 *
 * Reads/writes tier-to-model mappings from a local config file (router-config.json).
 * Resolves provider baseUrl and apiKey for any OpenClaw provider.
 * Auth is NEVER stored in the plugin — always read from OpenClaw's auth stores.
 *
 * Config is stored in the plugin directory (not in openclaw.json) because
 * OpenClaw validates plugins.entries schema and rejects unknown keys.
 */

import { readFileSync, renameSync, writeFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Tier } from "./classifier.js";

const HOME = process.env.HOME;
if (!HOME) throw new Error("[claw-llm-router] HOME environment variable not set");

const OPENCLAW_CONFIG_PATH = `${HOME}/.openclaw/openclaw.json`;
const AUTH_PROFILES_PATH = `${HOME}/.openclaw/agents/main/agent/auth-profiles.json`;
const AUTH_JSON_PATH = `${HOME}/.openclaw/agents/main/agent/auth.json`;

// Router config stored in plugin directory to avoid openclaw.json schema conflicts
const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url));
const ROUTER_CONFIG_PATH = `${PLUGIN_DIR}/router-config.json`;

// ── Types ────────────────────────────────────────────────────────────────────

export type TierModelSpec = {
  provider: string;
  modelId: string;
  baseUrl: string;
  apiKey: string;
  isAnthropic: boolean;
  isOAuth: boolean;
};

export type TierConfig = Record<Tier, TierModelSpec>;

type RouterConfig = {
  tiers: Record<Tier, string>;
  // Pin a specific model to a specific auth profile. Key format: "provider/model".
  // Value format: full profile name as it appears in auth-profiles.json
  // (e.g. "openai:christian.busch@hotmail.com").
  //
  // When a model is pinned: only the named profile is tried — env vars,
  // ${provider}:default, and any other profiles are ignored.
  //
  // The pinned profile is also REMOVED from the generic fallback list for
  // unpinned models in the same provider — so a profile that's reserved for
  // one model cannot accidentally be picked up by another.
  profilePins?: Record<string, string>;
};

// ── Defaults ─────────────────────────────────────────────────────────────────

export const DEFAULT_TIERS: Record<Tier, string> = {
  SIMPLE: "google/gemini-2.5-flash",
  MEDIUM: "anthropic/claude-haiku-4-5-20251001",
  COMPLEX: "anthropic/claude-sonnet-4-6",
  REASONING: "anthropic/claude-opus-4-6",
};

// ── Well-known provider base URLs ────────────────────────────────────────────

const WELL_KNOWN_BASE_URLS: Record<string, string> = {
  anthropic: "https://api.anthropic.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta/openai",
  openai: "https://api.openai.com/v1",
  groq: "https://api.groq.com/openai/v1",
  mistral: "https://api.mistral.ai/v1",
  deepseek: "https://api.deepseek.com/v1",
  together: "https://api.together.xyz/v1",
  fireworks: "https://api.fireworks.ai/inference/v1",
  perplexity: "https://api.perplexity.ai",
  xai: "https://api.x.ai/v1",
  minimax: "https://api.minimax.io/v1",
  moonshot: "https://api.moonshot.ai/v1",
};

// Provider name → env var name mapping (when not just PROVIDER_API_KEY)
const ENV_VAR_OVERRIDES: Record<string, string> = {
  google: "GEMINI_API_KEY",
};

// Provider name → auth-profiles.json provider ID mapping
// Some providers use a different ID in OpenClaw's auth store
const AUTH_PROFILE_ALIASES: Record<string, string> = {
  minimax: "minimax-portal",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function readOpenClawConfig(): Record<string, unknown> {
  const raw = readFileSync(OPENCLAW_CONFIG_PATH, "utf8");
  return JSON.parse(raw) as Record<string, unknown>;
}

function readRouterConfig(): RouterConfig | undefined {
  try {
    if (!existsSync(ROUTER_CONFIG_PATH)) return undefined;
    const raw = readFileSync(ROUTER_CONFIG_PATH, "utf8");
    return JSON.parse(raw) as RouterConfig;
  } catch {
    return undefined;
  }
}

function writeRouterConfigFile(config: RouterConfig): void {
  const tmp = `${ROUTER_CONFIG_PATH}.tmp.${process.pid}`;
  writeFileSync(tmp, JSON.stringify(config, null, 2), "utf8");
  JSON.parse(readFileSync(tmp, "utf8")); // Validate
  renameSync(tmp, ROUTER_CONFIG_PATH);
}

export function envVarName(provider: string): string {
  return ENV_VAR_OVERRIDES[provider] ?? `${provider.toUpperCase()}_API_KEY`;
}

// ── API Key Loading ──────────────────────────────────────────────────────────
// Priority: env var → auth-profiles.json → auth.json → openclaw.json env.vars
// NEVER stores keys — always reads from existing OpenClaw auth stores.
// NEVER logs keys or any portion of them.

type LogFn = ((msg: string) => void) | undefined;

export type ApiKeyResult = { key: string; isOAuth: boolean };

// Exported for testing — parses a single auth profile entry into key + OAuth flag.
export type AuthProfile = { token?: string; key?: string; access?: string; type?: string };

export function parseProfileCredential(profile: AuthProfile): ApiKeyResult | null {
  const profileIsOAuth = profile.type === "oauth";
  // OAuth credentials (e.g., MiniMax) store the token in `access`.
  // Only read `access` for type:"oauth" to avoid stale data on non-OAuth profiles.
  const key = profileIsOAuth
    ? (profile.access ?? profile.token ?? profile.key)
    : (profile.token ?? profile.key);
  if (!key || key === "proxy-handles-auth") return null;
  // Detect OAuth by profile type OR key prefix (OpenClaw stores
  // Anthropic OAuth tokens with type:"token", not type:"oauth")
  const isOAuth = profileIsOAuth || key.startsWith("sk-ant-oat01-");
  return { key, isOAuth };
}

// Profile pinning helpers (read profilePins from router-config.json).

// Resolve the profile pinned to a specific provider/model, if any.
export function getPinnedProfile(
  provider: string,
  modelId: string | undefined,
): string | undefined {
  if (!modelId) return undefined;
  const config = readRouterConfig();
  return config?.profilePins?.[`${provider}/${modelId}`];
}

// All profiles that are pinned somewhere — these are skipped by the generic
// fallback resolution so they cannot accidentally serve a different model.
export function getRestrictedProfiles(): Set<string> {
  const config = readRouterConfig();
  return new Set(Object.values(config?.profilePins ?? {}));
}

// Pure helper exposed for tests: filter a candidate profile list to drop
// restricted profiles (those pinned to other models).
export function filterRestrictedProfiles(
  profileNames: string[],
  restricted: Set<string>,
): string[] {
  return profileNames.filter((name) => !restricted.has(name));
}

export function loadApiKey(
  provider: string,
  options?: { log?: LogFn; modelId?: string },
): ApiKeyResult {
  const log = options?.log;
  const modelId = options?.modelId;
  const envKey = envVarName(provider);
  const pinned = getPinnedProfile(provider, modelId);

  // Pinned model: ONLY the named profile is acceptable. No env-var fallback,
  // no other profiles. Fails closed if the pinned profile has no usable key.
  if (pinned) {
    try {
      const raw = readFileSync(AUTH_PROFILES_PATH, "utf8");
      const store = JSON.parse(raw) as { profiles?: Record<string, AuthProfile> };
      const profile = store.profiles?.[pinned];
      if (profile) {
        const result = parseProfileCredential(profile);
        if (result) {
          log?.(
            `[auth] ${provider}/${modelId}: using pinned profile ${pinned}${result.isOAuth ? " (OAuth)" : ""}`,
          );
          return result;
        }
      }
    } catch {
      /* fall through to empty result */
    }
    log?.(
      `[auth] ${provider}/${modelId}: pinned profile ${pinned} not found or has no usable credential`,
    );
    return { key: "", isOAuth: false };
  }

  const restricted = getRestrictedProfiles();

  // 1. Environment variable
  if (process.env[envKey]) {
    const key = process.env[envKey]!;
    log?.(`[auth] ${provider}: using key from env var ${envKey}`);
    // Detect OAuth tokens passed via env var
    const isOAuth = key.startsWith("sk-ant-oat01-");
    return { key, isOAuth };
  }

  // 2. auth-profiles.json (canonical credential store)
  // Honor OpenClaw's configured profile order, then try the provider default
  // and any alias (e.g., minimax → minimax-portal).
  const alias = AUTH_PROFILE_ALIASES[provider];
  let profileNames: string[] = [];
  try {
    const config = readOpenClawConfig();
    const auth = config.auth as { order?: Record<string, string[]> } | undefined;
    profileNames = auth?.order?.[provider] ?? [];
  } catch {
    /* fall through */
  }
  profileNames.push(`${provider}:default`);
  if (alias) profileNames.push(`${alias}:default`);
  profileNames = filterRestrictedProfiles([...new Set(profileNames)], restricted);

  try {
    const raw = readFileSync(AUTH_PROFILES_PATH, "utf8");
    const store = JSON.parse(raw) as {
      profiles?: Record<string, AuthProfile>;
    };

    for (const profileName of profileNames) {
      const profile = store.profiles?.[profileName];
      if (!profile) continue;

      const result = parseProfileCredential(profile);
      if (result) {
        log?.(
          `[auth] ${provider}: using key from auth-profiles.json (${profileName}${result.isOAuth ? ", OAuth" : ""})`,
        );
        return result;
      }
    }
  } catch {
    /* fall through */
  }

  // 3. auth.json (runtime cache)
  try {
    const raw = readFileSync(AUTH_JSON_PATH, "utf8");
    const store = JSON.parse(raw) as Record<string, { key?: string; token?: string }>;
    const entry = store[provider];
    const key = entry?.key ?? entry?.token;
    if (key && key !== "proxy-handles-auth") {
      log?.(`[auth] ${provider}: using key from auth.json`);
      return { key, isOAuth: false };
    }
  } catch {
    /* fall through */
  }

  // 4. openclaw.json env.vars
  try {
    const config = readOpenClawConfig();
    const env = config.env as { vars?: Record<string, string> } | undefined;
    const val = env?.vars?.[envKey];
    if (val) {
      log?.(`[auth] ${provider}: using key from openclaw.json env.vars`);
      return { key: val, isOAuth: false };
    }
  } catch {
    /* fall through */
  }

  log?.(
    `[auth] ${provider}: NO API KEY FOUND (checked: env ${envKey}, auth-profiles.json, auth.json, openclaw.json env.vars)`,
  );
  return { key: "", isOAuth: false };
}

// ── Provider Resolution ──────────────────────────────────────────────────────

function resolveBaseUrl(provider: string): string {
  // First check openclaw.json models.providers for a configured baseUrl
  try {
    const config = readOpenClawConfig();
    const models = config.models as
      | { providers?: Record<string, { baseUrl?: string }> }
      | undefined;
    const providerConfig = models?.providers?.[provider];
    if (providerConfig?.baseUrl) {
      if (provider === "google") {
        return providerConfig.baseUrl.includes("/openai")
          ? providerConfig.baseUrl
          : `${providerConfig.baseUrl.replace(/\/$/, "")}/openai`;
      }
      return providerConfig.baseUrl;
    }
  } catch {
    /* fall through */
  }

  // Fall back to well-known URLs
  const wellKnown = WELL_KNOWN_BASE_URLS[provider];
  if (wellKnown) return wellKnown;

  throw new Error(
    `[claw-llm-router] Unknown provider "${provider}" — not in openclaw.json and no well-known URL. Configure it in openclaw.json models.providers or use /router set.`,
  );
}

// ── Tier Model Resolution ────────────────────────────────────────────────────

export function resolveTierModel(tierString: string, log?: LogFn): TierModelSpec {
  const slashIdx = tierString.indexOf("/");
  if (slashIdx === -1) {
    throw new Error(
      `[claw-llm-router] Invalid tier model format: "${tierString}" — expected "provider/model-id"`,
    );
  }

  const provider = tierString.slice(0, slashIdx);
  const modelId = tierString.slice(slashIdx + 1);
  const baseUrl = resolveBaseUrl(provider);
  const { key: apiKey, isOAuth } = loadApiKey(provider, { log, modelId });
  const isAnthropic = provider === "anthropic" || baseUrl.includes("anthropic.com");

  return { provider, modelId, baseUrl, apiKey, isAnthropic, isOAuth };
}

// ── Tier Config Read/Write ───────────────────────────────────────────────────

export function isTierConfigured(): boolean {
  const config = readRouterConfig();
  return !!config?.tiers && Object.keys(config.tiers).length === 4;
}

export function loadTierConfig(log?: LogFn): TierConfig {
  const config = readRouterConfig();
  const tiers = config?.tiers ?? DEFAULT_TIERS;

  return {
    SIMPLE: resolveTierModel(tiers.SIMPLE ?? DEFAULT_TIERS.SIMPLE, log),
    MEDIUM: resolveTierModel(tiers.MEDIUM ?? DEFAULT_TIERS.MEDIUM, log),
    COMPLEX: resolveTierModel(tiers.COMPLEX ?? DEFAULT_TIERS.COMPLEX, log),
    REASONING: resolveTierModel(tiers.REASONING ?? DEFAULT_TIERS.REASONING, log),
  };
}

export function getTierStrings(): Record<Tier, string> {
  const config = readRouterConfig();
  const tiers = config?.tiers;
  return {
    SIMPLE: tiers?.SIMPLE ?? DEFAULT_TIERS.SIMPLE,
    MEDIUM: tiers?.MEDIUM ?? DEFAULT_TIERS.MEDIUM,
    COMPLEX: tiers?.COMPLEX ?? DEFAULT_TIERS.COMPLEX,
    REASONING: tiers?.REASONING ?? DEFAULT_TIERS.REASONING,
  };
}

export function writeTierConfig(tiers: Record<Tier, string>): void {
  const existing = readRouterConfig();
  writeRouterConfigFile({
    ...existing,
    tiers,
  });
}
