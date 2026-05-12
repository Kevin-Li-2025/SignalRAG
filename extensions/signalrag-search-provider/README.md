# SignalRAG Search Provider

Minimal Chrome/Chromium extension that registers SignalRAG as a search provider.

Search URL:

```text
http://127.0.0.1:8000/engine?q={searchTerms}&mode=pro
```

This extension intentionally has no extra permissions. Chrome's search-provider
override policy is stricter when an extension combines search override with
unrelated capabilities, so the main side-panel extension lives separately in
`extensions/signalrag-chromium`.
