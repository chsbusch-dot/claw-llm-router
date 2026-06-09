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
import { DatabaseSync } from "node:sqlite";
import type { Tier } from "./classifier.js";

const HOME = process.env.HOME;
if (!HOME) throw new Error("[claw-llm-router] HOME environment variable not set");

const OPENCLAW_CONFIG_PATH = `${HOME}/.openclaw/openclaw.json`;
const AUTH_PROFILES_PATH = `${HOME}/.openclaw/agents/main/agent/auth-profiles.json`;
const AUTH_JSON_PATH = `${HOME}/.openclaw/agents/main/agent/auth.json`;
// OpenClaw 2026.6.5+ migrated auth profiles from auth-profiles.json into a
// sqlite store. The JSON file becomes a placeholder containing only
// `claw-llm-router:default → proxy-handles-auth`. We read the sqlite store
// directly when present, so credentials survive future migrations.
const AUTH_SQLITE_PATH = `${HOME}/.openclaw/agents/main/agent/openclaw-agent.sqlite`;

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

// ── Auth profile store reader ───────────────────────────────────────────────
// OpenClaw 2026.6.5 migrated auth profiles from auth-profiles.json into a
// sqlite store at openclaw-agent.sqlite (table `auth_profile_store`, row
// `store_key='primary'`, value `store_json` blob). The JSON file becomes a
// placeholder containing only `claw-llm-router:default → proxy-handles-auth`.
//
// We prefer the sqlite store when it exists and contains real profiles, then
// fall back to the JSON file for older OpenClaw versions or scenarios where
// the user manually restored the JSON. Uses Node's built-in node:sqlite (no
// new npm dependency).
//
// `parseProfileCredential` already filters the `proxy-handles-auth` placeholder,
// so a placeholder-only JSON store harmlessly returns no usable credential.

type AuthProfileStore = { profiles?: Record<string, AuthProfile> };

function readAuthProfileStoreFromSqlite(): AuthProfileStore | null {
  if (!existsSync(AUTH_SQLITE_PATH)) return null;
  let db: DatabaseSync | null = null;
  try {
    db = new DatabaseSync(AUTH_SQLITE_PATH, { readOnly: true });
    const stmt = db.prepare(
      "SELECT store_json FROM auth_profile_store WHERE store_key = 'primary'",
    );
    const row = stmt.get() as { store_json?: string } | undefined;
    if (!row?.store_json) return null;
    const parsed = JSON.parse(row.store_json) as AuthProfileStore;
    if (parsed.profiles && Object.keys(parsed.profiles).length > 0) return parsed;
    return null;
  } catch {
    return null;
  } finally {
    db?.close();
  }
}

function readAuthProfileStoreFromJson(): AuthProfileStore | null {
  try {
    const raw = readFileSync(AUTH_PROFILES_PATH, "utf8");
    return JSON.parse(raw) as AuthProfileStore;
  } catch {
    return null;
  }
}

// Read the auth profile store from whichever source has data. Returns an
// empty object (not null) when neither source is available, so callers can
// uniformly do `store.profiles?.[name]`.
export function readAuthProfileStore(): AuthProfileStore {
  return readAuthProfileStoreFromSqlite() ?? readAuthProfileStoreFromJson() ?? {};
}

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

export function loadApiKey(provider: string, log?: LogFn): ApiKeyResult {
  const envKey = envVarName(provider);

  // 1. Environment variable
  if (process.env[envKey]) {
    const key = process.env[envKey]!;
    log?.(`[auth] ${provider}: using key from env var ${envKey}`);
    // Detect OAuth tokens passed via env var
    const isOAuth = key.startsWith("sk-ant-oat01-");
    return { key, isOAuth };
  }

  // 2. Auth profile store (sqlite on OpenClaw 2026.6.5+, JSON on older)
  // Try both the provider name and any alias (e.g., minimax → minimax-portal)
  const alias = AUTH_PROFILE_ALIASES[provider];
  const profileNames = [`${provider}:default`];
  if (alias) profileNames.push(`${alias}:default`);

  const store = readAuthProfileStore();
  for (const profileName of profileNames) {
    const profile = store.profiles?.[profileName];
    if (!profile) continue;

    const result = parseProfileCredential(profile);
    if (result) {
      log?.(
        `[auth] ${provider}: using key from auth profile store (${profileName}${result.isOAuth ? ", OAuth" : ""})`,
      );
      return result;
    }
  }

  // 3. auth.json (runtime cache)
  try {
    const raw = readFileSync(AUTH_JSON_PATH, "utf8");
    const cache = JSON.parse(raw) as Record<string, { key?: string; token?: string }>;
    const entry = cache[provider];
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
    if (providerConfig?.baseUrl) return providerConfig.baseUrl;
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
  const { key: apiKey, isOAuth } = loadApiKey(provider, log);
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
