const form = document.querySelector("#search-form");
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const queryInput = document.querySelector("#query");
const clearQuery = document.querySelector("#clear-query");
const submit = document.querySelector("#submit");
const includeDomains = document.querySelector("#include-domains");
const excludeDomains = document.querySelector("#exclude-domains");
const scope = document.querySelector("#scope");
const recency = document.querySelector("#recency");
const country = document.querySelector("#country");
const language = document.querySelector("#language");
const citationVerifier = document.querySelector("#citation-verifier");
const resetControls = document.querySelector("#reset-controls");
const answer = document.querySelector("#answer");
const statusPill = document.querySelector("#engine-status");
const footerStatus = document.querySelector("#footer-status");
const footerStatusDot = document.querySelector("#footer-status-dot");
const footerMode = document.querySelector("#footer-mode");
const footerVerifier = document.querySelector("#footer-verifier");
const runSummary = document.querySelector("#run-summary");
const cragStrip = document.querySelector("#crag-strip");
const questions = document.querySelector("#questions");
const sourceCount = document.querySelector("#source-count");
const claimCount = document.querySelector("#claim-count");
const panels = {
  sources: document.querySelector("#panel-sources"),
  claims: document.querySelector("#panel-claims"),
  retrieval: document.querySelector("#panel-retrieval"),
  trace: document.querySelector("#panel-trace"),
};
const modeButtons = [...document.querySelectorAll(".mode-btn")];
const tabButtons = [...document.querySelectorAll(".tab-btn")];
const quickChips = [...document.querySelectorAll(".quick-chip")];
const clearChat = document.querySelector("#clear-chat");

let mode = "pro";
let lastData = null;
let startedAt = 0;

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    mode = button.dataset.mode;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
    footerMode.textContent = labelMode(mode);
  });
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

quickChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.dataset.query || "";
    if (chip.dataset.mode) {
      setMode(chip.dataset.mode);
    }
    queryInput.value = queryInput.value.trim() ? `${queryInput.value.trim()}\n${text}` : text;
    queryInput.focus();
  });
});

clearQuery.addEventListener("click", () => {
  queryInput.value = "";
  queryInput.focus();
});

clearChat.addEventListener("click", () => {
  lastData = null;
  answer.textContent = "Ask a question to start a cited search.";
  runSummary.hidden = true;
  cragStrip.hidden = true;
  questions.innerHTML = "";
  Object.values(panels).forEach((panel) => {
    panel.innerHTML = "";
  });
  sourceCount.textContent = "0";
  claimCount.textContent = "0";
});

resetControls.addEventListener("click", () => {
  includeDomains.value = "";
  excludeDomains.value = "";
  recency.value = "any";
  country.value = "";
  language.value = "";
  citationVerifier.value = "auto";
  footerVerifier.textContent = "DeepSeek Verifier";
});

scope.addEventListener("change", () => {
  if (scope.value === "news" && recency.value === "any") {
    recency.value = "week";
  }
});

citationVerifier.addEventListener("change", () => {
  footerVerifier.textContent = verifierLabel(citationVerifier.value);
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    queryInput.focus();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode,
        lens: scope.value,
        max_results: mode === "deep" ? 14 : mode === "pro" ? 10 : 8,
        include_domains: parseDomains(includeDomains.value),
        exclude_domains: parseDomains(excludeDomains.value),
        recency: recency.value,
        country: country.value.trim(),
        language: language.value.trim(),
        citation_verifier: citationVerifier.value,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    lastData = data;
    render(data);
  } catch (error) {
    answer.classList.add("error");
    answer.textContent = `Search failed: ${error.message}`;
    setRunState("Failed", "error");
  } finally {
    setLoading(false);
  }
});

async function loadHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    const data = await res.json();
    const provider = data.deepseek ? "DeepSeek" : data.openai ? "OpenAI" : "Extractive";
    const search = data.brave ? "Brave + HTML" : "DDG/Bing/Yahoo";
    setEngineStatus("online", `Local Engine Online`);
    footerStatus.textContent = "Connected";
    footerStatusDot.classList.add("online");
    const providerPills = document.querySelectorAll(".provider-pill");
    providerPills[0].textContent = provider;
    providerPills[1].textContent = search;
    providerPills[2].textContent = data.brave ? "Brave Search" : "No Brave key";
  } catch {
    setEngineStatus("offline", "Local Engine Offline");
    footerStatus.textContent = "Offline";
    footerStatusDot.classList.remove("online");
  }
}

function render(data) {
  answer.classList.remove("loading", "error");
  answer.innerHTML = formatAnswer(data.answer);
  renderSummary(data);
  renderCragStrip(data.crag, data.meta);
  renderSources(data.used_citations || data.citations || []);
  renderClaims(data.claim_citations || []);
  renderRetrieval(data);
  renderTrace(data);
  renderQuestions(data.query);
  sourceCount.textContent = String((data.used_citations || data.citations || []).length);
  claimCount.textContent = String((data.claim_citations || []).length);
  footerMode.textContent = labelMode(data.mode);
  footerVerifier.textContent = verifierLabel(data.meta.citation_verifier);
  activateTab("sources");
}

function renderSummary(data) {
  const elapsed = data.meta.elapsed_ms ? `${(data.meta.elapsed_ms / 1000).toFixed(1)}s` : `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
  runSummary.hidden = false;
  runSummary.innerHTML = `
    <span class="completed-dot">✓</span>
    <span>Completed</span>
    <span>·</span>
    <span>${escapeHtml(data.meta.used_citations)} sources</span>
    <span>·</span>
    <span>${escapeHtml(data.meta.verified_claims)} claims verified</span>
    <span>·</span>
    <span>${escapeHtml(elapsed)}</span>
  `;
}

function renderCragStrip(crag, meta) {
  if (!crag) {
    cragStrip.hidden = true;
    cragStrip.innerHTML = "";
    return;
  }
  const assessment = crag.after || crag.before;
  const confidence = Math.round((assessment.confidence || 0) * 100);
  cragStrip.hidden = false;
  cragStrip.innerHTML = `
    <div class="crag-main">
      <strong>CRAG Confidence</strong>
      <span>${qualityLabel(confidence)}</span>
      <div class="confidence-bars">${bars(confidence)}</div>
      <p>Evidence is ${escapeHtml(assessment.status)} across selected sources.</p>
    </div>
    <div class="crag-stat"><span>Retrieval</span><strong>${escapeHtml(assessment.status)}</strong></div>
    <div class="crag-stat"><span>Rerank</span><strong>${escapeHtml(meta.ranked_evidence)} passages</strong></div>
    <div class="crag-stat"><span>Diversity</span><strong>${escapeHtml(assessment.metrics.domains || 0)} domains</strong></div>
    <div class="crag-stat"><span>Freshness</span><strong>${escapeHtml(recency.value === "any" ? "Any" : recency.value)}</strong></div>
    <div class="crag-stat"><span>Countercheck</span><strong>${crag.corrected ? "Corrected" : "Passed"}</strong></div>
  `;
}

function renderSources(items) {
  panels.sources.innerHTML = `
    <div class="source-table">
      ${items
        .map((item, index) => {
          const domain = domainFor(item.url);
          return `
            <a class="source-row" href="${item.url}" target="_blank" rel="noreferrer">
              <span class="source-index">${index + 1}</span>
              <span class="source-favicon">${domain.slice(0, 2).toUpperCase()}</span>
              <span class="source-domain">${escapeHtml(domain)}</span>
              <strong>${escapeHtml(item.title || item.url)}</strong>
              <span class="source-date">${escapeHtml(item.provider)}</span>
            </a>
          `;
        })
        .join("")}
    </div>
    ${items.length ? `<button class="view-link" type="button">View all sources ↗</button>` : emptyPanel("Sources", "Run a search to see the sources used in the answer.")}
  `;
}

function renderClaims(items) {
  const counts = {
    supported: items.filter((item) => item.status === "supported").length,
    weak: items.filter((item) => item.status === "weak").length,
    contradicted: items.filter((item) => item.status === "contradicted").length,
    insufficient: items.filter((item) => item.status === "insufficient" || item.status === "needs_review").length,
  };
  if (!items.length) {
    panels.claims.innerHTML = emptyPanel("Claim Citations", "Verified claims will appear here after the answer is generated.");
    return;
  }
  panels.claims.innerHTML = `
    <article class="side-card">
      <div class="side-card-title">Claim Citations <span>${items.length}</span></div>
      ${claimSummaryRow("supported", "Supported", counts.supported, items)}
      ${claimSummaryRow("weak", "Weak", counts.weak, items)}
      ${claimSummaryRow("contradicted", "Contradicted", counts.contradicted, items)}
      ${claimSummaryRow("insufficient", "Insufficient", counts.insufficient, items)}
      <button class="view-link" type="button">View all claims →</button>
    </article>
  `;
}

function renderRetrieval(data) {
  const assessment = data.crag?.after || data.crag?.before;
  const metrics = assessment?.metrics || {};
  const packing = data.meta?.context_packing || {};
  panels.retrieval.innerHTML = `
    <article class="side-card retrieval-card">
      <div class="side-card-title">Retrieval Check</div>
      <div class="retrieval-grid">
        <div><span>Query Intent</span><strong>${escapeHtml(data.query_plan.intent)}</strong></div>
        <div><span>Lens</span><strong>${escapeHtml(lensLabel(data.meta.filters?.lens || "web"))}</strong></div>
        <div><span>Coverage</span><strong>${metricQuality(metrics.query_token_coverage || 0)}</strong></div>
        <div><span>Evidence Quality</span><strong>${metricQuality((assessment?.confidence || 0))}</strong></div>
        <div><span>Consistency</span><strong>${escapeHtml(assessment?.status || "pending")}</strong></div>
        <div><span>Confidence</span><strong>${qualityLabel(Math.round((assessment?.confidence || 0) * 100))}</strong></div>
        <div><span>Context Pack</span><strong>${escapeHtml(packing.strategy ? `${packing.packed_evidence}/${packing.input_evidence}` : "Pending")}</strong></div>
        <div><span>Compression</span><strong>${escapeHtml(formatRatio(packing.compression_ratio))}</strong></div>
      </div>
    </article>
  `;
}

function renderTrace(data) {
  const steps = [
    ["Query understood", 0.3],
    [`Search generated (${data.meta.queries.length} queries)`, 1.2],
    [`Sources retrieved (${data.meta.raw_results} results)`, 3.1],
    [`Ranked (${data.meta.ranked_evidence} selected)`, 5.4],
    ["Evidence extracted", 6.7],
    [`Retrieval check ${data.meta.crag_status}`, 7.2],
    ["Answer generated & claims verified", data.meta.elapsed_ms ? data.meta.elapsed_ms / 1000 : 8.2],
  ];
  const research = data.research_trace || [];
  panels.trace.innerHTML = `
    <article class="side-card">
      <div class="side-card-title">Research Trace <span>${labelMode(data.mode)}</span></div>
      <div class="timeline">
        ${steps
          .map(
            ([label, time]) => `
              <div class="timeline-row">
                <span>✓</span>
                <p>${escapeHtml(label)}</p>
                <em>${Number(time).toFixed(1)}s</em>
              </div>
            `
          )
          .join("")}
      </div>
      ${
        research.length
          ? `<div class="research-mini">${research
              .map((item) => `<div><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.documents)} docs</span></div>`)
              .join("")}</div>`
          : ""
      }
      <button class="view-link" type="button">View full trace ↗</button>
    </article>
  `;
}

function emptyPanel(title, detail) {
  return `
    <div class="empty-panel">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(detail)}</p>
    </div>
  `;
}

function renderQuestions(query) {
  const base = query.replace(/[?？]\s*$/, "");
  const list = [
    `What are the strongest sources for ${base}?`,
    `What evidence is uncertain or conflicting about ${base}?`,
    `Can recent changes affect the answer to ${base}?`,
  ];
  questions.innerHTML = list.map((item) => `<button type="button">${escapeHtml(item)}<span>⌄</span></button>`).join("");
}

function claimSummaryRow(status, label, count, items) {
  const sample = items.find((item) => item.status === status || (status === "insufficient" && item.status === "needs_review"));
  const refs = sample?.citation_ids?.length ? sample.citation_ids.map((id) => `<span>${id}</span>`).join("") : "";
  return `
    <div class="claim-summary-row ${status}">
      <div><span class="claim-dot"></span>${label}</div>
      <strong>${count}</strong>
      <p>${escapeHtml(sample?.claim || "–")}</p>
      <div class="claim-ref-list">${refs}</div>
    </div>
  `;
}

function setLoading(value) {
  submit.disabled = value;
  answer.classList.toggle("loading", value);
  if (value) {
    startedAt = Date.now();
    answer.classList.remove("error");
    answer.textContent = "Planning query, checking retrieval quality, ranking evidence, and verifying claims...";
    setRunState("Searching", "loading");
    runSummary.hidden = false;
    runSummary.innerHTML = `<span class="spinner"></span><span>Searching</span><span>·</span><span>retrieval check running</span>`;
    cragStrip.hidden = true;
    panels.sources.innerHTML = skeletonRows(5);
    panels.claims.innerHTML = skeletonRows(4);
    panels.retrieval.innerHTML = skeletonRows(3);
    panels.trace.innerHTML = skeletonRows(6);
  }
}

function setRunState(text, state) {
  runSummary.hidden = false;
  runSummary.className = `run-summary ${state}`;
  runSummary.innerHTML = `<span>${escapeHtml(text)}</span>`;
}

function setEngineStatus(state, text) {
  statusPill.className = `status-pill ${state}`;
  statusPill.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(text)}</span>`;
}

function setMode(nextMode) {
  mode = nextMode;
  modeButtons.forEach((item) => item.classList.toggle("active", item.dataset.mode === nextMode));
  footerMode.textContent = labelMode(nextMode);
}

function activateTab(name) {
  tabButtons.forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle("active", key === name));
}

function formatAnswer(text) {
  return escapeHtml(text)
    .replace(/\[(\d+)\]/g, '<sup>$1</sup>')
    .replace(/\n-\s+/g, "\n• ")
    .replace(/\n/g, "<br>");
}

function parseDomains(value) {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function domainFor(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "source";
  }
}

function bars(confidence) {
  const filled = Math.round(confidence / 14.3);
  return Array.from({ length: 7 }, (_, index) => `<span class="${index < filled ? "filled" : ""}"></span>`).join("");
}

function skeletonRows(count) {
  return `
    <div class="skeleton-list">
      ${Array.from({ length: count }, () => `<span></span>`).join("")}
    </div>
  `;
}

function qualityLabel(score) {
  if (score >= 75) return "High";
  if (score >= 45) return "Medium";
  return "Low";
}

function metricQuality(value) {
  if (value >= 0.7) return "High";
  if (value >= 0.35) return "Medium";
  return "Low";
}

function formatRatio(value) {
  if (typeof value !== "number") return "Pending";
  return `${Math.round(value * 100)}%`;
}

function labelMode(value) {
  if (value === "deep") return "Deep Research";
  if (value === "fast") return "Fast";
  return "Pro";
}

function lensLabel(value) {
  const labels = {
    web: "Web",
    official: "Official",
    academic: "Academic",
    forums: "Forums",
    news: "News",
    pdf: "PDFs",
    finance: "Finance",
  };
  return labels[value] || "Web";
}

function verifierLabel(value) {
  if (value === "deepseek") return "DeepSeek Verifier";
  if (value === "lexical") return "Lexical";
  if (value === "deepseek" || value === "auto") return "DeepSeek Verifier";
  return String(value || "Verifier");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadHealth();
