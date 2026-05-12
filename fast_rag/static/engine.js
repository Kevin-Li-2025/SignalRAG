const form = document.querySelector("#engine-form");
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const input = document.querySelector("#engine-query");
const submit = document.querySelector("#engine-submit");
const answer = document.querySelector("#engine-answer");
const metrics = document.querySelector("#engine-metrics");
const claimsSection = document.querySelector("#engine-claims-section");
const claims = document.querySelector("#engine-claims");
const sources = document.querySelector("#engine-sources");

const params = new URLSearchParams(window.location.search);
const initialQuery = params.get("q") || params.get("query") || "";
const initialMode = normalizeMode(params.get("mode") || "pro");

if (initialQuery) {
  input.value = initialQuery;
  runSearch(initialQuery, initialMode);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  const next = new URL(window.location.href);
  next.searchParams.set("q", query);
  next.searchParams.set("mode", initialMode);
  window.history.replaceState({}, "", next);
  runSearch(query, initialMode);
});

async function runSearch(query, mode) {
  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode,
        max_results: mode === "deep" ? 14 : mode === "pro" ? 10 : 8,
        citation_verifier: mode === "fast" ? "lexical" : "auto",
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    answer.textContent = data.answer;
    renderMetrics(data.meta, data.answer_mode);
    renderClaims(data.claim_citations || []);
    renderSources(data.used_citations || data.citations || []);
  } catch (error) {
    answer.classList.add("error");
    answer.textContent = `搜索失败：${error.message}`;
  } finally {
    setLoading(false);
  }
}

function setLoading(value) {
  submit.disabled = value;
  answer.classList.toggle("loading", value);
  if (value) {
    answer.classList.remove("error");
    answer.textContent = "正在搜索 SignalRAG...";
    metrics.hidden = true;
    claimsSection.hidden = true;
    claims.innerHTML = "";
    sources.innerHTML = "";
  }
}

function renderMetrics(meta, answerMode) {
  metrics.hidden = false;
  const data = [
    ["time", `${meta.elapsed_ms} ms`],
    ["mode", meta.effective_mode],
    ["queries", meta.queries.length],
    ["sources", meta.used_citations],
    ["verifier", meta.citation_verifier],
    ["answer", answerMode],
  ];
  metrics.innerHTML = data.map(([label, value]) => `<span class="metric">${label}: ${escapeHtml(value)}</span>`).join("");
}

function renderClaims(items) {
  if (!items.length) {
    claimsSection.hidden = true;
    claims.innerHTML = "";
    return;
  }
  claimsSection.hidden = false;
  claims.innerHTML = items
    .map((item) => {
      const refs = item.citation_ids.length ? item.citation_ids.map((id) => `[${id}]`).join(" ") : "no citation";
      return `
        <article class="claim ${escapeHtml(item.status)}">
          <div class="claim-top">
            <span class="badge ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span>
            <span class="claim-score">${Math.round(item.support_score * 100)}%</span>
            <span class="claim-refs">${escapeHtml(refs)}</span>
          </div>
          <p>${escapeHtml(item.claim)}</p>
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
          <div>
            <span class="badge">[${item.id}] score ${item.score}</span>
            <span class="badge">${escapeHtml(item.provider)}</span>
          </div>
          <a href="${item.url}" target="_blank" rel="noreferrer">${escapeHtml(item.title || item.url)}</a>
          <p>${escapeHtml(item.passage)}</p>
        </article>
      `
    )
    .join("");
}

function normalizeMode(value) {
  return ["fast", "pro", "deep"].includes(value) ? value : "pro";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
