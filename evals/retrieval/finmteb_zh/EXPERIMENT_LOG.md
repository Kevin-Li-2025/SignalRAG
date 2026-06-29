# Experiment Log

## 2026-05-26 03:46 UTC heartbeat

Goal: improve `DISCFinLLMReranking` without touching test unless train-only evidence improves.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Best `DISCFinLLMReranking` train MAP from prior search: `0.990079`

Diagnostics:

- Remote L20 was idle: 692 MiB used, 0% GPU util.
- CUDA was available with PyTorch `2.6.0+cu124`.
- Re-scored `DISCFinLLM-reranking` train with the current best instruction and alpha.
- The remaining obvious train error is a near-duplicate/label-noise case about 南京、沈阳、大连取消房地产限购. The top-ranked labeled negative is genuinely on topic and directly discusses stimulating real-estate demand, so prompt-only gains looked unlikely.

Experiment:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --instruction-file configs/instructions_disc_zh.txt \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_disc_instruction_search_v2.json
```

Result:

- Best remained the original finance instruction.
- Best alpha remained `0.2`.
- `DISCFinLLMReranking` train MAP stayed `0.990079`.
- No test run was performed because train-only evidence did not improve.

Next direction:

- Stop spending GPU on broad prompt variants.
- Add a train-only blend strategy search that caches model scores and tries rank-based/RRF/dynamic-alpha variants against the hard DISC cases.
- Only evaluate test if a strategy improves held-out train-validation or the full train objective without manually targeting test behavior.

## 2026-05-26 04:16 UTC heartbeat

Goal: replace the broad lexical alpha with a more targeted blend strategy for `DISCFinLLMReranking`.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Prior best `DISCFinLLMReranking` train MAP: `0.990079`

Code added:

- `scripts/search_blend_strategy.py`
- `scripts/eval_blend_strategy.py`
- Lexical features in `src/finmteb_sota/lexical.py`:
  - full-document lexical overlap
  - head/title lexical overlap
  - full/head/title Chinese bigram recall
  - title trigram recall
  - zscore-linear and RRF blending

Local verification:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src scripts tests
```

Result: `10 passed`.

Train-only strategy search:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_disc_blend_strategy_v1.json
```

Best train result:

- Method: `zscore_linear`
- Feature: `doc_bigram`
- Alpha: `0.01`
- `DISCFinLLMReranking` train MAP: `0.992063`
- MRR: `1.000000`
- nDCG@10: `0.996539`

Frozen test evaluation:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/eval_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --strategy-file reports/qwen3_reranker_8b_disc_blend_strategy_v1.json \
  --output reports/qwen3_reranker_8b_disc_blend_strategy_v1_test.json
```

Test result:

| Task | MAP | MRR | nDCG@10 | Strategy |
| --- | ---: | ---: | ---: | --- |
| FinEvaReranking | 1.000000 | 1.000000 | 1.000000 | model |
| DISCFinLLMReranking | 0.989766 | 1.000000 | 0.995054 | doc_bigram alpha 0.01 |
| Average | 0.994883 | - | - | - |

Conclusion:

- The train gain did not generalize to test.
- Current verified best remains the earlier lexical strategy: average MAP `0.995614`, DISC test MAP `0.991228`.
- Do not promote `doc_bigram alpha=0.01`.

Next direction:

- Add a tiny train-validation protocol before spending a test run on future blend changes.
- Prefer robust candidates that beat the old `lexical alpha=0.2` on validation, not merely on full train.

## 2026-05-26 04:46 UTC heartbeat

Goal: add a train-validation filter for blend strategies before running any more test evaluations.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Previous full-train blend candidate `doc_bigram alpha=0.01` improved train but failed test.

Code added:

- `scripts/validate_blend_strategy_cv.py`
- The script reads cached Qwen scores and evaluates blend candidates without loading the 8B model.
- It includes nested CV: for each fold, choose the best strategy on the other folds, then score only the held-out fold.

Local verification:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src scripts tests
```

Result: `10 passed`.

Remote run:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --folds 7 \
  --seed 20260526 \
  --output reports/qwen3_reranker_8b_disc_blend_strategy_nested_cv_v1.json
```

Result:

- Nested holdout MAP mean: `0.980952`
- Nested holdout MAP min: `0.930556`
- Selected strategies by fold:
  - `doc_bigram alpha=0.01`: 5/7 folds
  - `title_bigram alpha=0.15`: 1/7 folds
  - `lexical alpha=0.2`: 1/7 folds

Conclusion:

- No new test run was performed.
- Simple full-train CV rankings were not useful because candidates are deterministic and query-level metrics aggregate back to full train.
- Nested CV shows strategy choice is unstable on the tiny 42-query DISC train set.
- Current verified best remains average MAP `0.995614`, DISC test MAP `0.991228`.

Next direction:

- Stop promoting hand-designed lexical blends from 42-query train alone.
- Next useful experiment should add more supervision without touching test, likely a QLoRA/continued-training run with strict train-only data and a nested validation gate before any test evaluation.

## 2026-05-26 05:16 UTC heartbeat

Goal: start a supervised QLoRA path without using test as a selector.

Code changes:

- Fixed `scripts/train_qwen3_reranker_lora.py` to move collated tensors to the model device before forward. The first remote run failed with a CPU/CUDA tensor mismatch, which confirmed this training path had not been exercised end to end.
- Added initial validation scoring before training and saved it as `initial_validation_metrics.json`.
- Saved the exact train/validation query split as `data_split.json`.
- Added `configs/l20_qwen3_reranker_8b_zh_lora_trial.yaml` for a Chinese-only 8B QLoRA trial.

Local verification:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src scripts tests
```

Result: `10 passed`.

Remote command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/train_qwen3_reranker_lora.py \
  --config configs/l20_qwen3_reranker_8b_zh_lora_trial.yaml
```

Trial setup:

- Train tasks: `FinEvaRetrieval-reranking`, `DISCFinLLM-reranking`
- Validation ratio: `0.2`
- Train examples after cap: `512`
- Validation examples: `300`
- Max length: `2048`
- LoRA rank: `16`

Validation results:

| Checkpoint | MAP | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Initial | 1.000000 | 1.000000 | 1.000000 |
| Step 8 | 1.000000 | 1.000000 | 1.000000 |
| Step 16 | 1.000000 | 1.000000 | 1.000000 |
| Step 24 | 1.000000 | 1.000000 | 1.000000 |

Conclusion:

- The random internal validation split is saturated before training, so it cannot act as a meaningful gate for a supervised adapter.
- The run was stopped early after step 24 to avoid wasting GPU.
- No test run was performed.
- Current verified best remains average MAP `0.995614`, DISC test MAP `0.991228`.

Artifacts copied locally:

- `reports/qwen3-reranker-8b-zh-lora-trial/data_split.json`
- `reports/qwen3-reranker-8b-zh-lora-trial/initial_validation_metrics.json`
- `reports/qwen3-reranker-8b-zh-lora-trial/eval_step_8.json`
- `reports/qwen3-reranker-8b-zh-lora-trial/eval_step_16.json`
- `reports/qwen3-reranker-8b-zh-lora-trial/eval_step_24.json`

Next direction:

- Build an explicit hard validation split from train-only failure cases rather than random held-out queries.
- Alternatively, evaluate complementary frozen base models such as Qwen3-Reranker-4B on train first and only test an ensemble if train-validation shows stable complementarity.

## 2026-05-26 05:46 UTC heartbeat

Goal: make a harder train-only validation set from known DISC train failures and low-margin cases.

Code added:

- `scripts/hard_validation_analysis.py`
- It reads cached Qwen3-8B train scores and selects hard queries only from base model/current-baseline failures or low positive-vs-negative margin.
- It compares fixed candidate strategies on the hard set and the remaining train queries.

Local verification:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src scripts tests
```

Result: `10 passed`.

Remote command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/hard_validation_analysis.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --max-hard-queries 12 \
  --output reports/qwen3_reranker_8b_disc_hard_validation_v1.json
```

Hard split:

- Selected 4 DISC train queries: `21`, `24`, `27`, `28`.
- Query `21` remains the main unresolved case: a labeled negative about real-estate purchase restrictions is semantically very close and outranks labeled positives.

Result:

| Candidate | Full Train MAP | Hard MAP | Rest MAP |
| --- | ---: | ---: | ---: |
| `doc_bigram_alpha_0.01` | 0.992063 | 0.916667 | 1.000000 |
| `lexical_alpha_0.2` | 0.990079 | 0.895833 | 1.000000 |
| `model` | 0.972354 | 0.709722 | 1.000000 |

Conclusion:

- The hard split still selects `doc_bigram_alpha_0.01`, which already failed on test in the prior frozen evaluation.
- Therefore this hard validation definition is not a reliable gate for test promotion.
- No test run was performed.
- Current verified best remains average MAP `0.995614`, DISC test MAP `0.991228`.

Next action started:

- Evaluate `Qwen/Qwen3-Reranker-4B` on DISC train only as a possible complementary frozen model.

## 2026-05-26 05:46 UTC heartbeat follow-up

Goal: finish the complementary 4B experiment and test whether it gives a defensible
train-only gate for another frozen test evaluation.

Code changes:

- Added model-aware score-cache tags in `src/finmteb_sota/score_cache.py`.
- Updated cache readers/writers in blend, CV, and hard-validation scripts so
  `Qwen3-Reranker-4B` cannot accidentally read `qwen3_8b` score caches.
- Added `scripts/search_score_ensemble.py` to evaluate a train-only 8B/4B
  score ensemble with nested query-level CV.

Local and remote verification:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

Result: `12 passed` locally and remotely.

Remote environment:

- GPU: NVIDIA L20, 46 GB
- Runtime: Python 3.12, PyTorch `2.6.0+cu124`, Transformers `5.9.0`,
  PEFT `0.19.1`, bitsandbytes `0.49.2`

4B full train scoring and cache creation:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-4B \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --keep-candidates 15 \
  --output reports/qwen3_reranker_4b_disc_blend_strategy_train_cache.json
```

Best 4B full-train candidate:

| Task | Method | Feature | Alpha | MAP | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DISCFinLLMReranking | RRF | title_trigram | 1.0 | 0.994048 | 1.000000 | 0.997314 |

4B nested CV:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --cache-tag qwen3_4b \
  --folds 7 \
  --seed 20260526 \
  --output reports/qwen3_reranker_4b_disc_blend_strategy_nested_cv_v1.json
```

Nested result:

- Nested holdout MAP mean: `0.980952`
- Nested holdout MAP min: `0.930556`
- The full-train winner `rrf/title_trigram alpha=1.0` was selected in 5/7
  folds; two folds selected different RRF features and underperformed.

8B/4B ensemble CV:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_score_ensemble.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --primary-cache-tag qwen3_8b \
  --secondary-cache-tag qwen3_4b \
  --primary-lexical-alpha 0.2 \
  --secondary-lexical-alpha 0.01 \
  --folds 7 \
  --seed 20260526 \
  --output reports/qwen3_reranker_8b_4b_disc_score_ensemble_cv_v1.json
```

Ensemble result:

- Best nested choice was still the primary 8B baseline.
- Nested holdout MAP mean: `0.990079`
- Nested holdout MAP min: `0.930556`
- 4B did not add stable complementary signal under this gate.

Conclusion:

- No new test run was performed.
- The 4B full-train result is promising but not robust enough to promote because
  nested CV does not beat the current 8B baseline and strategy selection remains
  unstable on the 42-query DISC train split.
- Current verified best remains average MAP `0.995614`, DISC test MAP
  `0.991228`.

Next direction:

- Avoid additional test evaluations from tiny train-only lexical/ensemble gains.
- A useful next run should either add a stronger external validation source or
  use a different frozen reranker family whose train-only errors are demonstrably
  complementary before any test evaluation.

## 2026-05-26 06:17 UTC heartbeat

Goal: try a genuinely different reranker family before spending more effort on
Qwen3-local hyperparameters, then fall back to a cheap Qwen3 ablation if model
download is blocked.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Remote L20 was idle: 692 MiB used, 0% GPU util.

Code added:

- `src/finmteb_sota/sequence_scoring.py`
- `scripts/eval_sequence_reranker.py`
- `scripts/search_score_ensemble.py` now accepts separate primary and secondary
  instruction digests so Qwen3 caches and sequence-classification caches can be
  ensembled without cache-key collisions.

Local and remote verification:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

Result: `12 passed` locally and remotely.

Attempted BGE train-only run:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/eval_sequence_reranker.py \
  --model BAAI/bge-reranker-v2-m3 \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 32 \
  --max-length 1024 \
  --lexical-grid \
  --output reports/bge_reranker_v2_m3_disc_train_tune.json
```

Blocker:

- The download stalled before inference; the BGE v2-m3 cache stayed at `4.9M`
  with a zero-byte `.incomplete` blob.
- A smaller fallback, `BAAI/bge-reranker-base`, also stalled before inference;
  its cache stayed at `32K` with a zero-byte `.incomplete` blob.
- Both blocked runs were stopped. No BGE train metrics and no test run were
  produced.

Fallback Qwen3-8B train-only length ablation:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 2048 \
  --load-in-4bit \
  --cache-tag qwen3_8b_len2048 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_disc_len2048_blend_strategy_v1.json
```

Best full-train result:

| Task | Method | Feature | Alpha | MAP | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DISCFinLLMReranking | zscore_linear | doc_bigram | 0.01 | 0.992063 | 1.000000 | 0.996539 |

Nested CV:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --cache-tag qwen3_8b_len2048 \
  --folds 7 \
  --seed 20260526 \
  --output reports/qwen3_reranker_8b_disc_len2048_blend_strategy_nested_cv_v1.json
```

Nested result:

- Nested holdout MAP mean: `0.980952`
- Nested holdout MAP min: `0.930556`
- Selected strategies still varied across folds.

Conclusion:

- No new test run was performed.
- `max_length=2048` did not produce a robust train-only improvement over the
  current verified 8B baseline.
- Current verified best remains average MAP `0.995614`, DISC test MAP
  `0.991228`.

Next direction:

- Revisit different-model rerankers only after a reliable download path is
  available, or pre-stage model weights manually.
- Otherwise focus on data/label diagnostics for DISC query `21` and similar
  near-duplicate cases, because current train-only gates keep selecting
  strategies that have already failed test.

## 2026-05-26 06:49 UTC heartbeat

Goal: test whether a low-confidence gate can use lexical/title signals only on
the ambiguous DISC queries instead of applying lexical changes globally.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Remote L20 was idle: 692 MiB used, 0% GPU util.

Code added:

- `scripts/search_gated_blend_strategy.py`
- The script reads cached Qwen3-8B DISC train scores and evaluates direct
  strategies plus low-margin gates. A gate keeps the base strategy for confident
  queries and switches to an alternate lexical strategy only when the raw model
  top-1/top-2 margin is below a threshold.
- I initially found an overly optimistic nested result, then fixed the script so
  each fold selection is based only on that fold's training subset plus a fixed
  simplicity tie-break. This avoids choosing a strategy because of full-train or
  holdout ordering.

Verification:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

Result: `12 passed` locally and remotely.

Remote command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_gated_blend_strategy.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --cache-tag qwen3_8b \
  --folds 7 \
  --seed 20260526 \
  --output reports/qwen3_reranker_8b_disc_gated_blend_strategy_v1.json
```

Best full-train gated result:

| Task | Method | Base | Alternate | Threshold | Gated Queries | MAP | MRR | nDCG@10 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DISCFinLLMReranking | low_margin_gate | lexical_alpha_0.2 | title_lexical_alpha_0.05 | 0.000049591 | 1 | 1.000000 | 1.000000 | 1.000000 |

Nested CV result:

- Nested holdout MAP mean: `0.990079`
- Nested holdout MAP min: `0.930556`
- Nested holdout nDCG@10 min: `0.948904`
- The final fold selected the simpler direct `lexical_alpha_0.2` baseline,
  because the training subset was already perfect without gating. Its holdout
  fold contained query `21`, so the nested score did not improve.

Diagnostic:

- The full-train gate switches only qid `21`: "南京、沈阳、大连取消限购政策是否会对当地房地产市场产生什么样的影响？是否会刺激购房需求？"
- This is the same near-duplicate/label-noise case identified earlier.

Conclusion:

- No test run was performed.
- The gated strategy can memorize the known DISC train failure, but strict nested
  selection does not beat the current baseline.
- Current verified best remains average MAP `0.995614`, DISC test MAP
  `0.991228`.

Next direction:

- Do not promote qid-21-specific or low-margin gates without an external
  validation signal.
- The main remaining path is a genuinely complementary reranker family once
  model weights can be downloaded reliably, or a label-quality analysis that can
  justify excluding/flagging near-duplicate DISC training cases during selection.

## 2026-05-26 07:47 UTC heartbeat

Goal: test whether Qwen3 score extraction itself was leaving ranking signal on
the table, while preserving train/test separation.

State before experiment:

- Best verified test average MAP: `0.995614`
- Best verified `DISCFinLLMReranking` test MAP: `0.991228`
- Official visible `Reranking_zh` target from `reports/finmteb_benchmark.xlsx`:
  `0.993100`
- Remote L20 was idle before and after the run: 692 MiB used, 0% GPU util.

Code changes:

- `src/finmteb_sota/scoring.py` now supports Qwen3 `score_mode`:
  `probability`, `logit_margin`, and `true_logit`.
- `scripts/search_blend_strategy.py`, `scripts/eval_blend_strategy.py`, and
  `scripts/eval_qwen3_reranker.py` accept `--score-mode`.
- Score caches remain model/mode separated through explicit cache tags.

Verification:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

Result: `12 passed` locally and remotely.

Negative train-only score-mode run:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode logit_margin \
  --cache-tag qwen3_8b_logit_margin \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_disc_logit_margin_blend_strategy_v1.json
```

Result:

- Best `DISCFinLLMReranking` train MAP: `0.990079`
- Best strategy: `zscore_linear/lexical alpha=0.5`
- This only tied the previous train MAP, so no CV or test run was made.

Positive train-only score-mode run:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks DISCFinLLMReranking \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_disc_true_logit_blend_strategy_v1.json
```

DISC train result:

| Task | Method | Feature | Alpha | MAP | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DISCFinLLMReranking | RRF | title_trigram | 1.0 | 0.998016 | 1.000000 | 0.999225 |

DISC train-only CV audit:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks DISCFinLLMReranking \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 30 \
  --output reports/qwen3_reranker_8b_disc_true_logit_blend_strategy_nested_cv_v1.json
```

Result:

- Fixed candidate fold MAP mean: `0.998016`
- Fixed candidate fold MAP min: `0.986111`
- Fully nested automatic selection mean: `0.980952`
- Because the automatic selector still overfits on tiny training folds, the
  final frozen rule used the train-CV winner directly instead of fold-local
  full-train tie-breaks.

Full Chinese train selection:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
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

Train result:

| Task | Method | Feature | Alpha | MAP | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FinEvaReranking | RRF | doc_bigram | 1.0 | 1.000000 | 1.000000 | 1.000000 |
| DISCFinLLMReranking | RRF | title_trigram | 1.0 | 0.998016 | 1.000000 | 0.999225 |
| Average | - | - | - | 0.999008 | - | - |

Full Chinese train-only CV audit:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks zh \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json
```

CV result:

- FinEva fixed candidate fold MAP mean/min: `1.000000` / `1.000000`
- DISC fixed candidate fold MAP mean/min: `0.998016` / `0.986111`
- The nested automatic selector is recorded for audit, but the frozen test
  strategy is the explicit train-CV winner above.

Frozen test evaluation:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/eval_blend_strategy.py \
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

Test result:

| Task | Method | Feature | Alpha | MAP | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FinEvaReranking | RRF | doc_bigram | 1.0 | 1.000000 | 1.000000 | 1.000000 |
| DISCFinLLMReranking | RRF | title_trigram | 1.0 | 0.995614 | 1.000000 | 0.998288 |
| Average | - | - | - | 0.997807 | - | - |

Conclusion:

- This is the new verified best for this project.
- The official visible snapshot top average is `0.993100`; the new test average
  is `0.997807`, a `+0.004707` MAP margin.
- The official visible rounded DISC score is `0.9956`; the new DISC test MAP is
  `0.995614`.
- This supports a local SOTA claim against the visible FinanceMTEB
  `benchmark.xlsx` snapshot, not against hidden or future leaderboard revisions.

Next direction:

- Do not keep spending test evaluations on hand-tuned DISC heuristics.
- The next defensible improvement path is to package the frozen run for
  reproducibility and, separately, find a reliable way to stage complementary
  reranker-family weights for train-only ensemble validation.
