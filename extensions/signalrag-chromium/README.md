# SignalRAG for Chromium

Unpacked Chromium/Chrome extension for the local SignalRAG engine.

## Load the Extension

1. Start SignalRAG at `http://127.0.0.1:8000`.
2. Open `chrome://extensions`.
3. Enable Developer mode.
4. Click "Load unpacked".
5. Select this folder:

```text
extensions/signalrag-chromium
```

## Use It

- Type `sr` in the address bar, press Space, then type a query.
- Type `sr`, press Space, then `deep: your query` for Deep Research.
- Select text on any page, right-click, and choose SignalRAG search.
- Click the extension action to open the side panel.

## Add as a Chromium Search Engine

Chromium can use SignalRAG's local search endpoint as a custom search engine:

```text
http://127.0.0.1:8000/engine?q=%s&mode=pro
```

Deep Research variant:

```text
http://127.0.0.1:8000/engine?q=%s&mode=deep
```

Chrome's manifest `chrome_settings_overrides.search_provider` can override the
default search provider, but published extensions need verified domains and
Chrome may flag broad extensions that combine search override with unrelated
permissions. This extension uses the safer omnibox/context menu/side panel path.
