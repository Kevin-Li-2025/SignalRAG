const DEFAULTS = {
  apiBase: "http://127.0.0.1:8000",
  mode: "pro",
  citationVerifier: "auto",
};

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.sync.set({ ...(await loadSettings()) });
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "signalrag-selection-pro",
      title: "Search selection with SignalRAG",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "signalrag-selection-deep",
      title: "Deep Research selection with SignalRAG",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "signalrag-page",
      title: "Open SignalRAG side panel",
      contexts: ["page", "selection", "link"],
    });
  });

  if (chrome.sidePanel?.setPanelBehavior) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
  }
});

chrome.action.onClicked.addListener(async (tab) => {
  await openSidePanel(tab);
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "signalrag-page") {
    await openSidePanel(tab);
    return;
  }

  const query = (info.selectionText || "").trim();
  if (!query) return;
  if (info.menuItemId === "signalrag-selection-deep") {
    await openSearchTab(query, "deep");
    return;
  }
  await openSearchTab(query, "pro");
});

chrome.omnibox.setDefaultSuggestion({
  description: "Search SignalRAG: %s",
});

chrome.omnibox.onInputChanged.addListener((text, suggest) => {
  const query = text.trim();
  const suggestions = [
    {
      content: query,
      description: `Search SignalRAG Pro: ${escapeXml(query || "type a query")}`,
    },
    {
      content: `deep:${query}`,
      description: `Deep Research in SignalRAG: ${escapeXml(query || "type a query")}`,
    },
  ];
  suggest(suggestions);
});

chrome.omnibox.onInputEntered.addListener(async (text) => {
  const raw = text.trim();
  if (!raw) return;
  if (raw.startsWith("deep:")) {
    await openSearchTab(raw.slice(5).trim(), "deep");
    return;
  }
  const settings = await loadSettings();
  await openSearchTab(raw, settings.mode);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "open-search" && message.query) {
    openSearchTab(message.query, message.mode || "pro")
      .then(() => sendResponse({ ok: true }))
      .catch((error) => sendResponse({ ok: false, error: String(error) }));
    return true;
  }
  return false;
});

async function openSidePanel(tab) {
  if (!chrome.sidePanel?.open) return;
  const windowId = tab?.windowId;
  if (typeof windowId === "number") {
    await chrome.sidePanel.open({ windowId });
  }
}

async function openSearchTab(query, mode) {
  const settings = await loadSettings();
  const url = new URL("/engine", normalizeBase(settings.apiBase));
  url.searchParams.set("q", query);
  url.searchParams.set("mode", normalizeMode(mode || settings.mode));
  await chrome.tabs.create({ url: url.toString() });
}

async function loadSettings() {
  return chrome.storage.sync.get(DEFAULTS);
}

function normalizeBase(value) {
  const trimmed = String(value || DEFAULTS.apiBase).replace(/\/+$/, "");
  return trimmed || DEFAULTS.apiBase;
}

function normalizeMode(value) {
  return ["fast", "pro", "deep"].includes(value) ? value : "pro";
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
