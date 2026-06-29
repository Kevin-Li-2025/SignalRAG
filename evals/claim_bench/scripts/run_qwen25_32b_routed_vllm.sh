#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen25-32b-routed-vllm}"
CODE_MODEL="${CODE_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4}"
MATH_MODEL="${MATH_MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
CODE_BATCH_SIZE="${CODE_BATCH_SIZE:-${BATCH_SIZE:-16}}"
MATH_BATCH_SIZE="${MATH_BATCH_SIZE:-4}"
CODE_DTYPE="${CODE_DTYPE:-${DTYPE:-auto}}"
MATH_DTYPE="${MATH_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
CODE_VLLM_MAX_MODEL_LEN="${CODE_VLLM_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-8192}}"
CODE_VLLM_MAX_NUM_SEQS="${CODE_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-16}}"
MATH_VLLM_MAX_MODEL_LEN="${MATH_VLLM_MAX_MODEL_LEN:-4096}"
MATH_VLLM_MAX_NUM_SEQS="${MATH_VLLM_MAX_NUM_SEQS:-32}"
CODE_VLLM_QUANTIZATION="${CODE_VLLM_QUANTIZATION:-gptq_marlin}"
MATH_VLLM_QUANTIZATION="${MATH_VLLM_QUANTIZATION:-}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
HUMANEVAL_TEMPLATE="${HUMANEVAL_TEMPLATE:-evalscope}"
HUMANEVAL_GATE="${HUMANEVAL_GATE:-0.908}"
GSM8K_GATE="${GSM8K_GATE:-0.93}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

run_eval() {
  local model="$1"
  local out="$2"
  local task="$3"
  local quantization="$4"
  local batch_size="$5"
  local dtype="$6"
  local max_model_len="$7"
  local max_num_seqs="$8"
  local quantization_arg=()
  if [[ -n "${quantization}" ]]; then
    quantization_arg=(--vllm-quantization "${quantization}")
  fi

  "${PYTHON_BIN}" -m claimbench.cli eval \
    --model "${model}" \
    --backend vllm \
    --tasks "${task}" \
    --out "${out}" \
    --dtype "${dtype}" \
    --batch-size "${batch_size}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm-max-model-len "${max_model_len}" \
    --vllm-max-num-seqs "${max_num_seqs}" \
    "${quantization_arg[@]}" \
    --vllm-cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}" \
    --gsm8k-n-shot 4 \
    --gsm8k-template evalscope \
    --humaneval-template "${HUMANEVAL_TEMPLATE}" \
    --humaneval-timeout 4.0 \
    --temperature-code 0.0 \
    --temperature-math 0.0 \
    "${LIMIT_ARG[@]}"
}

"${PYTHON_BIN}" -m claimbench.cli inspect-env
run_eval "${CODE_MODEL}" "${OUT}/humaneval" humaneval "${CODE_VLLM_QUANTIZATION}" "${CODE_BATCH_SIZE}" "${CODE_DTYPE}" "${CODE_VLLM_MAX_MODEL_LEN}" "${CODE_VLLM_MAX_NUM_SEQS}"
run_eval "${MATH_MODEL}" "${OUT}/gsm8k" gsm8k "${MATH_VLLM_QUANTIZATION}" "${MATH_BATCH_SIZE}" "${MATH_DTYPE}" "${MATH_VLLM_MAX_MODEL_LEN}" "${MATH_VLLM_MAX_NUM_SEQS}"

"${PYTHON_BIN}" - "${OUT}" "${CODE_MODEL}" "${MATH_MODEL}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
code_model = sys.argv[2]
math_model = sys.argv[3]
human_summary = json.loads((out / "humaneval" / "summary.json").read_text(encoding="utf-8"))
math_summary = json.loads((out / "gsm8k" / "summary.json").read_text(encoding="utf-8"))

summary = {
    "model": f"routed: humaneval={code_model}; gsm8k={math_model}",
    "revision": None,
    "settings": {
        "claim_scope": "routed_system_not_single_model",
        "humaneval_model": code_model,
        "gsm8k_model": math_model,
        "humaneval_summary": str(out / "humaneval" / "summary.json"),
        "gsm8k_summary": str(out / "gsm8k" / "summary.json"),
    },
    "environment": {
        "humaneval": human_summary.get("environment", {}),
        "gsm8k": math_summary.get("environment", {}),
    },
    "results": {
        "humaneval": human_summary["results"]["humaneval"],
        "gsm8k": math_summary["results"]["gsm8k"],
    },
}

out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

human = summary["results"]["humaneval"]
math = summary["results"]["gsm8k"]
(out / "summary.md").write_text(
    "\n".join(
        [
            "# Routed Benchmark Summary",
            "",
            f"- HumanEval model: `{code_model}`",
            f"- GSM8K model: `{math_model}`",
            "- Claim scope: routed system, not single model",
            f"- HumanEval Pass@1: {human['pass_at_1'] * 100:.2f}% ({human['passed']} / {human['total']})",
            f"- GSM8K exact match: {math['exact_match'] * 100:.2f}% ({math['correct']} / {math['total']})",
            "",
        ]
    ),
    encoding="utf-8",
)
print((out / "summary.md").read_text(encoding="utf-8"))
PY

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval "${HUMANEVAL_GATE}" \
  --gsm8k "${GSM8K_GATE}"
