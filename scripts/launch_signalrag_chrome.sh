#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME="/Users/yinxiaogou/Documents/New project 7/.browsers/chrome/148.0.7778.97-mac-arm64/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
PROFILE="$ROOT/.browsers/signalrag-profile"
EXT_MAIN="$ROOT/extensions/signalrag-chromium"
EXT_PROVIDER="$ROOT/extensions/signalrag-search-provider"
START_URL="${1:-http://127.0.0.1:8000/engine?q=SignalRAG&mode=pro}"

if ! curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "SignalRAG server is not reachable at http://127.0.0.1:8000"
  echo "Start it first with: python -m fast_rag.app"
  exit 1
fi

mkdir -p "$PROFILE"
exec "$CHROME" \
  --user-data-dir="$PROFILE" \
  --no-first-run \
  --no-default-browser-check \
  --disable-first-run-ui \
  --load-extension="$EXT_MAIN,$EXT_PROVIDER" \
  "$START_URL"
