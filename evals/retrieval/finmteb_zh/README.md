# FinMTEB ZH Reranker SOTA

[![FinanceMTEB Reranking_zh](https://img.shields.io/badge/FinanceMTEB_Reranking__zh-0.9978_MAP-2ea44f)](reports/public_reranking_zh_snapshot_comparison.md)
[![Public Snapshot SOTA](https://img.shields.io/badge/Public_snapshot-SOTA-blue)](reports/public_reranking_zh_snapshot_comparison.md)
[![CI](https://img.shields.io/github/checks-status/Kevin-Li-2025/finmteb-zh-reranker-sota/main?label=CI)](https://github.com/Kevin-Li-2025/finmteb-zh-reranker-sota/commits/main)
[![CD](https://img.shields.io/badge/CD-release_workflow_configured-blue)](.github/workflows/release.yml)
[![Model](https://img.shields.io/badge/Model-Qwen3--Reranker--8B-black)](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

Public snapshot SOTA finance-domain Chinese reranking system for
FinanceMTEB `Reranking_zh`.

## Result

This project targets the FinanceMTEB Chinese reranking slice:

- `FinanceMTEB/FinEvaRetrieval-reranking`
- `FinanceMTEB/DISCFinLLM-reranking`

On the L20 box, `Qwen/Qwen3-Reranker-8B` with `true_logit` scoring and per-query
RRF lexical fusion reached `0.997807` MAP average on `Reranking_zh`, above the
official `benchmark.xlsx` snapshot top average of `0.993100`.

| Benchmark | Public snapshot best | This repo | Delta |
| --- | ---: | ---: | ---: |
| `Reranking_zh` Avg. MAP | 0.993100 | 0.997807 | +0.004707 |
| `FinEvaReranking` MAP | 0.990600 | 1.000000 | +0.009400 |
| `DISCFinLLMReranking` MAP | 0.995600 | 0.995614 | +0.000014 |

Supported claim: new SOTA on the public FinanceMTEB `Reranking_zh` benchmark
snapshot. Official leaderboard inclusion is pending maintainer review.

See `RESULTS.md` for exact scores, commands, and environment details. The
public snapshot comparison is recorded in
`reports/public_reranking_zh_snapshot_comparison.md`.

## Method

The achieved SOTA path is:

1. Reproduce the official leaderboard snapshot.
2. Run a zero-shot Qwen3 reranker with train-only score-mode and rank-fusion selection.
3. Freeze the selected strategy, then evaluate on untouched test splits.

Final frozen setup:

- Base model: `Qwen/Qwen3-Reranker-8B`
- Inference: 4-bit on NVIDIA L20 46 GB
- Score mode: raw `true` token logit
- Fusion: per-query RRF with lexical features
- Selection policy: train-only search/CV, frozen test evaluation

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/finmteb_sota/` | Dataset loading, scoring, metrics, lexical features, score caching |
| `scripts/search_blend_strategy.py` | Train-only score-mode and RRF strategy search |
| `scripts/eval_blend_strategy.py` | Frozen strategy evaluation on test |
| `scripts/validate_blend_strategy_cv.py` | Train-only CV audit |
| `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json` | Current best test result |
| `reports/public_reranking_zh_snapshot_comparison.md` | Public leaderboard snapshot comparison |

## CI/CD

GitHub Actions CI runs on every push and pull request to `main`:

- Python 3.10 and 3.12
- `ruff check .`
- `python -m compileall -q src scripts tests`
- `python -m pytest -q`

The CI badge at the top of this README tracks `.github/workflows/ci.yml`.

The release workflow runs on `v*` tags or manual dispatch:

- builds wheel and source distribution
- builds a clean source zip without caches or score-cache files
- uploads artifacts
- publishes a GitHub Release for version tags

## Setup

Use Python 3.10, 3.11, or 3.12 on the GPU machine.

```bash
cd /Users/yinxiaogou/Documents/resume/finmteb-sota-reranker
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[train]"
```

Login to Hugging Face if you plan to push the adapter:

```bash
huggingface-cli login
```

## Leaderboard Snapshot

```bash
python scripts/finmteb_leaderboard_snapshot.py
```

Current target from the official `benchmark.xlsx` snapshot:

- `Reranking_zh` top average is about `0.9931`.
- `FinEvaReranking` is saturated at `1.0000`.
- `DISCFinLLMReranking` top visible score is about `0.9956`.

The current result beats the official visible Chinese reranking average, with
`DISCFinLLMReranking` test MAP `0.995614`.

## Current Best Zero-Shot Run

Search score mode and rank-fusion blends on train only:

```bash
python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --keep-candidates 25 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json
```

Audit the frozen choice on train-only CV:

```bash
python scripts/validate_blend_strategy_cv.py \
  --tasks zh \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json
```

Then freeze the train-selected strategy for test:

```bash
python scripts/eval_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --strategy-file reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json
```

## Previous Baseline

The earlier instruction/alpha search is still useful as a simple reproduction
baseline.

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_train_search.json
```

Then freeze the selected instruction and alpha for test:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_reranker_8b_train_search.json \
  --output reports/qwen3_reranker_8b_test.json
```

If you want only alpha tuning with a fixed instruction:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --lexical-grid \
  --output reports/qwen3_reranker_8b_train_tune.json
```

If memory is tight, use `Qwen/Qwen3-Reranker-4B` first.

## QLoRA Training

```bash
accelerate launch scripts/train_qwen3_reranker_lora.py \
  --config configs/l20_qwen3_reranker_8b.yaml
```

Evaluate the adapter:

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_8b_lora_train_search.json
```

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_8b_lora_train_search.json \
  --output reports/qwen3_8b_lora_test.json
```

## Guardrails

- Do not tune on test splits.
- Tune the lexical blend on train or validation only, then freeze it.
- Report per-task MAP/MRR/nDCG, not just the average.
- Save every command and commit hash before publishing.
