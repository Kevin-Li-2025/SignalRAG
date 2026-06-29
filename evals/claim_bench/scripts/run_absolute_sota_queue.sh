#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_DIR="${CACHE_DIR:-/home/hhai/models/modelscope}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" reports

export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

wait_for_sessions() {
  local session
  for session in ${WAIT_FOR_TMUX_SESSIONS:-}; do
    while tmux has-session -t "${session}" 2>/dev/null; do
      log "Waiting for tmux session ${session}"
      sleep 30
    done
  done
}

ensure_model() {
  local model_id="$1"
  log "Ensuring ${model_id}"
  "${PYTHON_BIN}" scripts/download_modelscope.py "${model_id}" --cache-dir "${CACHE_DIR}" | tail -n 1
}

run_step() {
  local name="$1"
  shift
  log "START ${name}"
  set +e
  "$@"
  local status=$?
  set -e
  log "END ${name} status=${status}"
  return 0
}

log "Disk before queue"
df -h "${CACHE_DIR%/modelscope}" || true

qwen3_coder_path="$(ensure_model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)"
wait_for_sessions
run_step qwen3_coder30b_fp8_humaneval \
  env MODEL="${qwen3_coder_path}" PYTHON_BIN="${PYTHON_BIN}" \
    OUT=reports/qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope \
    bash scripts/run_qwen3_coder30b_fp8_humaneval_vllm.sh \
  >"${LOG_DIR}/qwen3_coder30b_fp8_humaneval.log" 2>&1

math72_path="$(ensure_model ShelterW/Qwen2.5-Math-72B-Instruct-AWQ)"
wait_for_sessions
run_step qwen25_math72b_awq_gsm8k_smoke \
  env MODEL="${math72_path}" PYTHON_BIN="${PYTHON_BIN}" LIMIT=8 GSM8K_GATE=0 \
    OUT=reports/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-smoke \
    bash scripts/run_qwen25_math72b_awq_gsm8k_vllm.sh \
  >"${LOG_DIR}/qwen25_math72b_awq_gsm8k_smoke.log" 2>&1

if [[ -f reports/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-smoke/summary.json ]]; then
  wait_for_sessions
  run_step qwen25_math72b_awq_gsm8k_full \
    env MODEL="${math72_path}" PYTHON_BIN="${PYTHON_BIN}" \
      OUT=reports/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k \
      bash scripts/run_qwen25_math72b_awq_gsm8k_vllm.sh \
    >"${LOG_DIR}/qwen25_math72b_awq_gsm8k.log" 2>&1
fi

qwen25_72b_path="$(ensure_model Qwen/Qwen2.5-72B-Instruct-AWQ)"
wait_for_sessions
run_step qwen25_72b_awq_mixed_smoke \
  env MODEL="${qwen25_72b_path}" PYTHON_BIN="${PYTHON_BIN}" LIMIT=4 HUMANEVAL_GATE=0 GSM8K_GATE=0 \
    OUT=reports/qwen25-72b-instruct-awq-marlin-vllm-smoke \
    bash scripts/run_qwen25_72b_awq_vllm.sh \
  >"${LOG_DIR}/qwen25_72b_awq_mixed_smoke.log" 2>&1

if [[ -f reports/qwen25-72b-instruct-awq-marlin-vllm-smoke/summary.json ]]; then
  wait_for_sessions
  run_step qwen25_72b_awq_mixed_full \
    env MODEL="${qwen25_72b_path}" PYTHON_BIN="${PYTHON_BIN}" \
      OUT=reports/qwen25-72b-instruct-awq-marlin-vllm \
      bash scripts/run_qwen25_72b_awq_vllm.sh \
    >"${LOG_DIR}/qwen25_72b_awq_mixed.log" 2>&1
fi

log "Disk after queue"
df -h "${CACHE_DIR%/modelscope}" || true
