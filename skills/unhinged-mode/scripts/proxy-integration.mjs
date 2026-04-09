/**
 * Unhinged Mode — NVIDIA Proxy Integration
 *
 * Drop-in module for nvidia-proxy.mjs to support L1B3RT4S prompt injection.
 *
 * Usage in nvidia-proxy.mjs:
 *   import { maybeInjectLiberation } from './unhinged-proxy-integration.mjs';
 *   // In your request handler, before sending upstream:
 *   body = maybeInjectLiberation(body, modelName);
 *
 * Or add this to the proxy's request pipeline manually.
 */

import { readFileSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";

const CONFIG_PATH = join(homedir(), ".skcapstone", "unhinged.json");
const SKILL_DIR = join(homedir(), "clawd", "skills", "unhinged-mode");
const PROMPT_SCRIPT = join(SKILL_DIR, "scripts", "get-liberation-prompt.sh");

let cachedConfig = null;
let configMtime = 0;

/** Read unhinged config, cached with mtime check */
function getConfig() {
  try {
    if (!existsSync(CONFIG_PATH)) return { enabled: false };
    const stat = require("node:fs").statSync(CONFIG_PATH);
    if (stat.mtimeMs !== configMtime) {
      cachedConfig = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
      configMtime = stat.mtimeMs;
    }
    return cachedConfig || { enabled: false };
  } catch {
    return { enabled: false };
  }
}

/** Get liberation prompt for a model, cached per provider */
const promptCache = new Map();

function getLiberationPrompt(modelName) {
  // Map model to provider key
  const lower = (modelName || "").toLowerCase();
  let provider = "nvidia"; // default
  if (lower.includes("llama") || lower.includes("meta")) provider = "meta";
  else if (lower.includes("mistral") || lower.includes("mixtral")) provider = "mistral";
  else if (lower.includes("deepseek")) provider = "deepseek";
  else if (lower.includes("grok") || lower.includes("xai")) provider = "xai";
  else if (lower.includes("qwen") || lower.includes("alibaba")) provider = "alibaba";
  else if (lower.includes("kimi") || lower.includes("nvidia") || lower.includes("nemotron") || lower.includes("minimax")) provider = "nvidia";

  if (promptCache.has(provider)) return promptCache.get(provider);

  try {
    const prompt = execSync(`bash "${PROMPT_SCRIPT}" "${provider}"`, {
      encoding: "utf8",
      timeout: 5000,
    }).trim();
    promptCache.set(provider, prompt);
    return prompt;
  } catch (e) {
    console.error(`[unhinged] Failed to get liberation prompt for ${provider}:`, e.message);
    return null;
  }
}

/**
 * Maybe inject a L1B3RT4S liberation prompt into the request body.
 * Call this before sending upstream in the proxy.
 *
 * @param {object} body - Parsed request body
 * @param {string} modelName - The model being called
 * @returns {object} - Modified body (or original if unhinged is off)
 */
export function maybeInjectLiberation(body, modelName) {
  const config = getConfig();
  if (!config.enabled) return body;

  const prompt = getLiberationPrompt(modelName);
  if (!prompt) return body;

  // Inject as first system message
  if (!body.messages) return body;

  const liberationMsg = {
    role: "system",
    content: `[LIBERATION MODE ACTIVE]\n\n${prompt}`,
  };

  // Prepend before other messages
  body.messages = [liberationMsg, ...body.messages];

  // Log if enabled
  if (config.log_enabled) {
    const ts = new Date().toISOString();
    console.log(`[unhinged][${ts}] Liberation prompt injected for model: ${modelName}`);
  }

  return body;
}

/**
 * Check if unhinged mode is currently enabled.
 */
export function isUnhingedEnabled() {
  return getConfig().enabled === true;
}

/**
 * Handle USR1 signal to reload config (called from proxy signal handler)
 */
export function reloadConfig() {
  configMtime = 0;
  cachedConfig = null;
  promptCache.clear();
  console.log("[unhinged] Config reloaded");
}
