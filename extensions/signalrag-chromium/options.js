const DEFAULTS = {
  apiBase: "http://127.0.0.1:8000",
  mode: "pro",
  citationVerifier: "auto",
};

const form = document.querySelector("#options-form");
const apiBase = document.querySelector("#api-base");
const mode = document.querySelector("#mode");
const verifier = document.querySelector("#citation-verifier");
const message = document.querySelector("#message");
const test = document.querySelector("#test");

load();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const next = {
    apiBase: normalizeBase(apiBase.value),
    mode: normalizeMode(mode.value),
    citationVerifier: normalizeVerifier(verifier.value),
  };
  await chrome.storage.sync.set(next);
  message.textContent = "Saved.";
});

test.addEventListener("click", async () => {
  message.textContent = "Testing...";
  const base = normalizeBase(apiBase.value);
  try {
    const res = await fetch(`${base}/api/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const provider = data.deepseek ? "DeepSeek" : data.openai ? "OpenAI" : "Extractive";
    message.textContent = `Connected: ${provider}`;
  } catch (error) {
    message.textContent = `Connection failed: ${error.message}`;
  }
});

async function load() {
  const settings = await chrome.storage.sync.get(DEFAULTS);
  apiBase.value = settings.apiBase;
  mode.value = normalizeMode(settings.mode);
  verifier.value = normalizeVerifier(settings.citationVerifier);
}

function normalizeBase(value) {
  const trimmed = String(value || DEFAULTS.apiBase).trim().replace(/\/+$/, "");
  try {
    const url = new URL(trimmed);
    if (url.protocol === "http:" || url.protocol === "https:") {
      return url.toString().replace(/\/+$/, "");
    }
  } catch {
    return DEFAULTS.apiBase;
  }
  return DEFAULTS.apiBase;
}

function normalizeMode(value) {
  return ["fast", "pro", "deep"].includes(value) ? value : "pro";
}

function normalizeVerifier(value) {
  return ["auto", "deepseek", "lexical"].includes(value) ? value : "auto";
}
