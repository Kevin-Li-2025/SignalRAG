#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-7b-instruct-vllm-mixed-summary.json
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope-summary.json \
  --humaneval 0.92 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-recommended-s1-summary.json \
  --humaneval 0.932 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-t02-s1-summary.json \
  --humaneval 0.914 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-math-7b-instruct-vllm-gsm8k-summary.json \
  --humaneval none \
  --gsm8k 0.93
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-full-u099-l4096-eager-summary.json \
  --humaneval none \
  --gsm8k 0.943
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-summary.json \
  --humaneval none \
  --gsm8k 0.957
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-majority8-summary.json \
  --humaneval none \
  --gsm8k 0.959
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-claimbench-sweep-summary.json \
  --humaneval 0.89 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-sweep-summary.json \
  --humaneval 0.91 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-concise-sweep-summary.json \
  --humaneval 0.90 \
  --gsm8k none
PYTHONPATH=src "${PYTHON_BIN}" -m claimbench.cli gate \
  --summary evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-raw-sweep-summary.json \
  --humaneval 0.53 \
  --gsm8k none
