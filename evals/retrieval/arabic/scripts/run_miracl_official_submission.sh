#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-remote_outputs/miracl-official-ar-bge-m3-hybrid-r100-v43}"
MODEL_PATH="${2:-/home/hhai/hf-models/BAAI-bge-m3}"
RUN_ID="${3:-bge-m3-hybrid-r100-ar}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

RUNFILES="$BASE/runfiles"
mkdir -p "$RUNFILES"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --subset ar
  --query-source miracl-raw
  --candidate-top-k 100
  --rerank-limit 100
  --top-k 100
  --metric-k 10
  --batch-size 32
  --rerank-batch-size 32
  --encode-chunk-size 1000
  --run-id "$RUN_ID"
)

echo "[official-submission] start $(date -Is)"
echo "[official-submission] output=$BASE run_id=$RUN_ID"

"$PYTHON_BIN" scripts/run_miracl_bge_m3_hybrid.py \
  --output-dir "$BASE/dev" \
  --split dev \
  "${COMMON_ARGS[@]}" \
  --force-rebuild
cp "$BASE/dev/ar_dev.txt" "$RUNFILES/ar_dev.txt"

"$PYTHON_BIN" scripts/run_miracl_bge_m3_hybrid.py \
  --output-dir "$BASE/test-a" \
  --split test-a \
  "${COMMON_ARGS[@]}" \
  --force-rebuild
cp "$BASE/test-a/ar_test-a.txt" "$RUNFILES/ar_test-a.txt"

"$PYTHON_BIN" scripts/prepare_miracl_submission.py \
  --run-dir "$RUNFILES" \
  --output-zip "$BASE/miracl_ar_bge_m3_hybrid_r100.zip" \
  --subset ar \
  --splits dev test-a \
  --depth 100 \
  --package-dir-name miracl_submission \
  --validation-json "$BASE/validation.json"

echo "[official-submission] done $(date -Is)"
