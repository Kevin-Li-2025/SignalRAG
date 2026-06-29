#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python -m pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
python -m pip install -e ".[dev,vllm]"

bash scripts/run_qwen25_7b_mixed_vllm.sh
