#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-hhai-zijun}"
REMOTE_DIR="${REMOTE_DIR:-~/llm-claim-bench}"

cd "$(dirname "$0")/.."

ssh "${HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -az \
  --delete \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "reports" \
  ./ "${HOST}:${REMOTE_DIR}/"
ssh "${HOST}" "cd ${REMOTE_DIR} && bash scripts/bootstrap_and_run.sh"
