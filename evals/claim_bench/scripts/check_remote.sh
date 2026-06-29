#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-hhai-zijun}"

ssh "${HOST}" '
set -e
hostname
command -v python3 || true
python3 --version || true
command -v nvidia-smi || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || nvidia-smi
'
