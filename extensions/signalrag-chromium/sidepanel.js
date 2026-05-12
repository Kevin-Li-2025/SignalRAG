const DEFAULTS = {
  apiBase: "http://127.0.0.1:8000",
  mode: "pro",
  citationVerifier: "auto",
};

const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeInput = document.querySelector("#mode");
const verifierInput = document.querySelector("#citation-verifier");
const includeDomains = document.querySelector("#include-domains");
const excludeDomains = document.querySelector("#exclude-domains");
const recency = document.querySelector("#recency");
const country = document.querySelector("#country");
const language = document.querySelector("#language");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const offlineCard = document.querySelector("#offline-card");
const retryHealth = document.querySelector("#retry-health");
const useTab = document.querySelector("#use-tab");
const openFull = document.querySelector("#open-full");
const metrics = document.querySelector("#metrics");
const trace = document.querySelector("#trace");
const answer = document.querySelector("#answer");
const claims = document.querySelector("#claims");
const sources = document.querySelector("#sources");
const chips = [...document.querySelectorAll(".chip")];

let settings = { ...DEFAULTS };
let lastQuery = "";

init();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  await runSearch(query);
});

retryHealth.addEventListener("click", () => {
  loadHealth();
});

useTab.addEventListener("click", async () => {
  const tab = await activeTab();
  if (!tab) return;
  const title = tab.title || "Current page";
  const url = tab.url || "";
  queryInput.value = `${queryInput.value.trim()}\n\nCurrent page: ${title}\n${url}`.trim();
  queryInput.focus();
});

openFull.addEventListener("click", async () => {
  const query = queryInput.value.trim() || lastQuery || "SignalRAG";
  const url = new URL("/engine", baseUrl());
  url.searchParams.set("q", query);
  url.searchParams.set("mode", modeInput.value);
  await chrome.tabs.create({ url: url.toString() });
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.dataset.query || "";
    queryInput.value = queryInput.value.trim() ? `${queryInput.value.trim()}\n${text}` : text;
    queryInput.focus();
  });
});

async function init() {
  settings = await chrome.storage.sync.get(DEFAULTS);
  modeInput.value = normalizeMode(settings.mode);
  verifierInput.value = normalizeVerifier(settings.citationVerifier);
  await loadHealth();
}

async function loadHealth() {
  setStatus("checking", "Checking local engine...");
  try {
    const res = await fetch(`${baseUrl()}/api/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const provider = data.deepseek ? "DeepSeek" : data.openai ? "OpenAI" : "Extractive";
    const search = data.brave ? "Brave" : "DuckDuckGo";
    setStatus("online", `${provider} · ${search}`);
    offlineCard.hidden = true;
  } catch {
    setStatus("offline", "Local engine offline");
    offlineCard.hidden = false;
  }
}

async function runSearch(query) {
  lastQuery = query;
  setLoading(true);
  try {
    const res = await fetch(`${baseUrl()}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode: modeInput.value,
        max_results: modeInput.value === "deep" ? 14 : modeInput.value === "pro" ? 10 : 8,
        include_domains: parseDomains(includeDomains.value),
        exclude_domains: parseDomains(excludeDomains.value),
        recency: recency.value,
        country: country.value.trim(),
        language: language.value.trim(),
        citation_verifier: verifierInput.value,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus("online", `${labelFor(data.answer_mode)} · ${data.meta.crag_status}`);
    offlineCard.hidden = true;
    render(data);
  } catch (error) {
    answer.classList.add("error");
    answer.textContent = `Search failed: ${error.message}`;
    setStatus("offline", "Search failed");
    offlineCard.hidden = false;
  } finally {
    setLoading(false);
  }
}

function render(data) {
  answer.textContent = data.answer;
  renderMetrics(data);
  renderTrace(data);
  renderClaims(data.claim_citations || []);
  renderSources(data.used_citations || data.citations || []);
}

function renderMetrics(data) {
  metrics.hidden = false;
  const meta = data.meta;
  const items = [
    ["time", `${meta.elapsed_ms} ms`],
    ["mode", meta.effective_mode],
    ["queries", meta.queries.length],
    ["sources", meta.used_citations],
    ["verifier", meta.citation_verifier],
  ];
  metrics.innerHTML = items
    .map(([label, value]) => `<span><b>${escapeHtml(label)}</b>${escapeHtml(value)}</span>`)
    .join("");
}

function renderTrace(data) {
  const crag = data.crag;
  const research = data.research_trace || [];
  if (!crag && !research.length) {
    trace.hidden = true;
    trace.innerHTML = "";
    return;
  }
  trace.hidden = false;
  const cragHtml = crag
    ? `
      <article class="trace-card">
        <div class="trace-title">
          <strong>Retrieval check</strong>
          <span>${escapeHtml(crag.after?.status || crag.before.status)}</span>
        </div>
        <p>${Math.round((crag.after?.confidence || crag.before.confidence) * 100)}% confidence${crag.corrected ? " · corrected" : ""}</p>
      </article>
    `
    : "";
  const researchHtml = research
    .map(
      (item) => `
        <article class="trace-card compact">
          <div class="trace-title">
            <strong>${escapeHtml(item.label)}</strong>
            <span>${escapeHtml(item.documents)} docs</span>
          </div>
          <p>${escapeHtml(item.purpose)}</p>
        </article>
      `
    )
    .join("");
  trace.innerHTML = cragHtml + researchHtml;
}

function renderClaims(items) {
  claims.hidden = items.length === 0;
  claims.innerHTML = items
    .map((item) => {
      const refs = item.citation_ids.length ? item.citation_ids.map((id) => `[${id}]`).join(" ") : "no citation";
      const quote = item.supporting_quote ? `<blockquote>${escapeHtml(item.supporting_quote)}</blockquote>` : "";
      return `
        <article class="claim ${escapeHtml(item.status)}">
          <div class="claim-head">
            <strong>${escapeHtml(item.status)}</strong>
            <span>${Math.round(item.support_score * 100)}%</span>
            <span>${escapeHtml(refs)}</span>
          </div>
          <p>${escapeHtml(item.claim)}</p>
          ${quote}
        </article>
      `;
    })
    .join("");
}

function renderSources(items) {
  sources.innerHTML = items
    .map(
      (item) => `
        <article class="source">
          <div class="source-head">
            <span>[${item.id}]</span>
            <span>${escapeHtml(item.provider)}</span>
            <span>${escapeHtml(item.score)}</span>
          </div>
          <a href="${item.url}" target="_blank" rel="noreferrer">${escapeHtml(item.title || item.url)}</a>
          <p>${escapeHtml(item.passage)}</p>
        </article>
      `
    )
    .join("");
}

function setLoading(value) {
  submit.disabled = value;
  form.classList.toggle("busy", value);
  answer.classList.toggle("loading", value);
  if (value) {
    answer.classList.remove("error");
    answer.textContent = "Planning query, checking retrieval, and ranking evidence...";
    metrics.hidden = true;
    trace.hidden = true;
    claims.hidden = true;
    claims.innerHTML = "";
    sources.innerHTML = "";
  }
}

function setStatus(kind, text) {
  status.className = `status ${kind}`;
  status.innerHTML = `<span></span>${escapeHtml(text)}`;
}

async function activeTab() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    return tabs[0] || null;
  } catch {
    return null;
  }
}

function baseUrl() {
  return String(settings.apiBase || DEFAULTS.apiBase).replace(/\/+$/, "");
}

function parseDomains(value) {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalizeMode(value) {
  return ["fast", "pro", "deep"].includes(value) ? value : "pro";
}

function normalizeVerifier(value) {
  return ["auto", "deepseek", "lexical"].includes(value) ? value : "auto";
}

function labelFor(value) {
  return value === "deepseek" ? "DeepSeek" : value === "openai" ? "OpenAI" : "Extractive";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
