# Experiment Log

## 2026-05-26

Sources checked:

- Official project page: https://hq-bench.github.io/coreb-page/
- Official GitHub repo: https://github.com/hq-bench/coreb
- Official dataset: https://huggingface.co/datasets/hq-bench/coreb
- Official model: https://huggingface.co/hq-bench/coreb-code-reranker

Protocol:

- Use `release_v2602` for tuning and validation.
- Use `release_v2603` for frozen test evaluation only.
- Primary metric is query-count-weighted `nDCG@10` across `text2code`,
  `code2code`, and `code2text`.
- Public target from the official project page: retrieval overall `nDCG@10`
  best is `0.639` on v202602 and `0.624` on v202603; official reranker reports
  positive deltas after reranking top-128 on v202603.

Current implementation:

- Local JSONL export for remote offline runs.
- Deterministic BM25/code-aware retrieval.
- Qwen3-Reranker-8B true-logit reranking with optional RRF fusion.
- C2LLM dense retrieval runner with cached embeddings.
- Weighted RRF fusion for BM25/dense/candidate runs.

Baseline results:

- `release_v2602`, BM25/code-aware, `top_k=128`, `eval_k=10`:
  weighted `nDCG@10=0.325560`, `Recall@10=0.464686`, `MAP@10=0.270715`.
- `release_v2603`, BM25/code-aware, `top_k=128`, `eval_k=10`:
  weighted `nDCG@10=0.226204`, `Recall@10=0.355099`, `MAP@10=0.175828`.
- Candidate ceiling check with `eval_k=128`:
  - `release_v2602`: weighted `Recall@128=0.667591`.
  - `release_v2603`: weighted `Recall@128=0.632692`.

Interpretation:

- BM25 top-128 recall is too low for SOTA; even a strong reranker is capped by
  missing positives. The next necessary step is a stronger dense first-stage
  retriever, not more BM25-only tuning.
- Official CoREB sources report `codefuse-ai/C2LLM-7B` and GemEmb-2 as the
  strongest public retrieval baselines, and `hq-bench/coreb-code-reranker` as
  the only reranker with positive deltas on all three tasks.

Remote L20 actions:

- Verified SSH to `hhai-zijun`, NVIDIA L20 46GB.
- Verified Python environment:
  `torch 2.6.0+cu124`, `transformers 5.9.0`, `datasets 4.8.5`.
- Verified cached models:
  `Qwen/Qwen3-Reranker-8B` and `Qwen/Qwen3-Reranker-4B`.
- Qwen3 smoke run passed:
  `release_v2602`, `text2code`, `top_n=16`, `limit_queries=8`,
  `nDCG@10=0.420066`, `Recall@10=1.0`, `MAP@10=0.225`.
- Invalidated one early full run because the old cache key could reuse the
  smoke-run score cache. Fixed by hashing instruction text into score-cache
  filenames and restarted the validation run with task-specific instructions.
- Started background validation run:
  `qwen3_release_v2602_all_top16_task_alpha0`, using BM25 candidates,
  `top_n=16`, `instruction_mode=task`, `score_mode=true_logit`, 4-bit load.
- Started background model download through `hf-mirror.com`:
  `codefuse-ai/C2LLM-7B` to `/home/hhai/models/codefuse-ai_C2LLM-7B`.

## 2026-05-26 heartbeat 18:44Z

State check:

- Official CoREB project page rechecked for the public benchmark target.
- Remote GPU was active on `hhai-zijun`; the only substantial CoREB GPU process
  was the validation rerank job.
- C2LLM-7B download continued in the background and had reached roughly
  9.4 GiB. The mirror connection hit one read timeout, but the HF downloader was
  retrying/resuming.

Actions:

- Inspected the running Qwen3 validation job and found `batch_size=16` used more
  memory and lower throughput than `batch_size=8`.
- Interrupted the `batch_size=16` attempt before meaningful progress and
  restarted the same validation experiment with `batch_size=8`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/rerank_qwen3.py \
  --candidate-run reports/bm25_release_v2602.json \
  --split release_v2602 \
  --tasks all \
  --top-n 16 \
  --batch-size 8 \
  --max-length 4096 \
  --score-mode true_logit \
  --load-in-4bit \
  --instruction-mode task \
  --fusion-alpha 0.0 \
  --output reports/qwen3_release_v2602_all_top16_task_alpha0.json
```

Current best verified result remains the BM25 baseline:

- `release_v2602`: weighted `nDCG@10=0.325560`, `Recall@10=0.464686`,
  `MAP@10=0.270715`.
- `release_v2603`: weighted `nDCG@10=0.226204`, `Recall@10=0.355099`,
  `MAP@10=0.175828`.

No SOTA claim: the validation rerank and dense C2LLM first-stage retrieval are
still in progress.

## 2026-05-26 user status check

Issue:

- The previous monolithic Qwen3 validation process stopped without writing a
  final JSON report. The log ended around `999/2330` scoring batches with no
  Python traceback or CUDA error, so the most likely cause is an external
  process kill/interruption rather than a handled script exception.
- The C2LLM-7B download process also stopped with
  `httpx.RemoteProtocolError: peer closed connection without sending complete
  message body`.
- The old Qwen3 script only wrote score cache after a full task completed, so
  partial progress from the interrupted full run was not recoverable.

Fixes applied:

- Updated `scripts/rerank_qwen3.py` to write score-cache files atomically after
  each `--cache-flush-pairs` chunk and to skip already-scored pairs on resume.
- Updated the heartbeat automation prompt to prefer resumable per-task or
  chunked runs instead of long monolithic jobs.
- Restarted Qwen3 reranking as a per-task, resumable `text2code` validation run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/rerank_qwen3.py \
  --candidate-run reports/bm25_release_v2602.json \
  --split release_v2602 \
  --tasks text2code \
  --top-n 16 \
  --batch-size 8 \
  --cache-flush-pairs 128 \
  --max-length 4096 \
  --score-mode true_logit \
  --load-in-4bit \
  --instruction-mode task \
  --fusion-alpha 0.0 \
  --output reports/qwen3_release_v2602_text2code_top16_task_alpha0.json
```

- Confirmed the new cache is growing:
  `text2code_release_v2602_Qwen--Qwen3-Reranker-8B_true_logit_task_0076cde142_top16.json`
  had 640 scored pairs after the first check.
- Restarted C2LLM-7B download inside a 20-attempt retry loop with single-worker
  downloads:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
/home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
-m huggingface_hub.cli.hf download codefuse-ai/C2LLM-7B \
  --repo-type model \
  --local-dir /home/hhai/models/codefuse-ai_C2LLM-7B \
  --max-workers 1
```

Current state:

- GPU is active again on the resumable Qwen3 `text2code` run.
- C2LLM-7B local directory has reached roughly 15 GiB and is still downloading.
- Current verified best remains BM25; no SOTA claim yet.

## 2026-05-26 progress check

- C2LLM-7B download completed successfully at
  `/home/hhai/models/codefuse-ai_C2LLM-7B`; all four safetensor shards and
  custom model files are present.
- Qwen3 `text2code` validation is still running on the L20.
- Resumable score cache had reached 1,280 pairs for
  `text2code_release_v2602_Qwen--Qwen3-Reranker-8B_true_logit_task_0076cde142_top16.json`.
  Total expected `text2code` top-16 pairs are 18,640.
- A dense retrieval waiter was started. It waits for the Qwen3 `text2code`
  process to release the GPU, then runs:

```bash
PYTHONPATH=src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2602 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode none \
  --output reports/dense_c2llm7b_release_v2602_none_top128.json
```

- Current verified best remains the BM25 baseline:
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.

### Heartbeat check 2026-05-26T19:26Z

- Remote host `hhai@100.111.150.63`, project
  `/home/hhai/projects/coreb-sota-reranker`.
- L20 is healthy and occupied by the resumable Qwen3 `text2code`
  validation run: GPU memory `22615 MiB`, utilization `95%`.
- Active Qwen3 command remains the frozen validation candidate reranker:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/rerank_qwen3.py \
  --candidate-run reports/bm25_release_v2602.json \
  --split release_v2602 \
  --tasks text2code \
  --top-n 16 \
  --batch-size 8 \
  --cache-flush-pairs 128 \
  --max-length 4096 \
  --score-mode true_logit \
  --load-in-4bit \
  --instruction-mode task \
  --fusion-alpha 0.0 \
  --output reports/qwen3_release_v2602_text2code_top16_task_alpha0.json
```

- Score cache reached 3,584 / 18,640 pairs (`19.23%`) at remote time
  `2026-05-27T03:27:01+08:00`.
- Final Qwen output
  `reports/qwen3_release_v2602_text2code_top16_task_alpha0.json` is not
  written yet, so no metric change is claimed.
- Dense C2LLM retrieval remains queued behind this process via waiter PID
  `1576141`; duplicate waiters were already removed.
- Current verified best remains unchanged: BM25
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.

### Manual progress check 2026-05-26T19:43Z

- Remote time: `2026-05-27T03:43:53+08:00`.
- Active process remains Qwen3 true-logit reranking for
  `release_v2602/text2code`, `top-n=16`, `instruction-mode=task`,
  `fusion-alpha=0.0`.
- L20 status: `41797 MiB` GPU memory in use, `100%` utilization.
- Resumable score cache reached 8,704 / 18,640 pairs (`46.70%`).
- Final Qwen output is still pending:
  `reports/qwen3_release_v2602_text2code_top16_task_alpha0.json`.
- Dense C2LLM retrieval remains queued behind the Qwen process, with no output
  yet at `reports/dense_c2llm7b_release_v2602_none_top128.json`.
- Verified best remains BM25:
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.

### Official C2C anchor-exclusion protocol fix 2026-05-26T21:34Z

- Rechecked the official CoREB project page and official runner code.
- Official project page snapshot:
  `https://hq-bench.github.io/coreb-page/`.
  Public `v202603` retrieval leaderboard reports query-count-weighted overall
  `nDCG@10`:
  - `GemEmb-2`: overall `0.624`, text-to-code `0.434`,
    code-to-code `0.698`, code-to-text `0.784`.
  - `C2LLM-7B`: overall `0.615`, text-to-code `0.443`,
    code-to-code `0.659`, code-to-text `0.766`.
- Official runner protocol detail from `hq-bench/coreb`:
  `code2code` queries carry an anchor code item in the shared corpus. The
  official evaluation removes that anchor before computing metrics; otherwise
  every model wastes rank 1 on the query's own code, which is not a positive.
- Local fix:
  - `src/coreb_sota/data.py`: `TaskData.exclude_doc_ids` now captures
    `meta_anchor_code_id` for C2C.
  - `src/coreb_sota/metrics.py`: `evaluate_run(...)` accepts query-level
    exclusions.
  - `scripts/run_dense_retrieval.py`, `scripts/fuse_runs.py`,
    `scripts/rerank_qwen3.py`, and `scripts/run_bm25_baseline.py` now evaluate
    with the shared official exclusion map.
  - `scripts/rerank_qwen3.py` also removes excluded anchors from C2C rerank
    candidate construction, so top-N budget is not wasted.
  - Added `scripts/recompute_report_metrics.py` to recompute stored reports
    without rerunning embeddings.
  - Added regression tests in `tests/test_protocol.py`.
- Verification:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts
```

- Test result: `3 passed`.
- Recomputed stored reports on the remote L20 project with official
  anchor-exclusion protocol:

| report | split | overall nDCG@10 | text2code | code2code | code2text |
| --- | --- | ---: | ---: | ---: | ---: |
| `dense_c2llm7b_release_v2602_none_top128_official.json` | validation | `0.639597` | `0.435450` | `0.658734` | `0.824319` |
| `dense_c2llm7b_release_v2602_query_top128_official.json` | validation | `0.639867` | `0.440546` | `0.664071` | `0.819487` |
| `dense_c2llm7b_release_v2602_query_doc_top128_official.json` | validation | `0.639279` | `0.435067` | `0.677232` | `0.821557` |
| `dense_c2llm7b_release_v2602_coreb_task_top128_official.json` | validation | `0.638760` | `0.433182` | `0.675174` | `0.822496` |
| `dense_c2llm7b_release_v2602_task_selective_query_qd_none_official.json` | validation | `0.643077` | `0.440546` | `0.677232` | `0.824319` |
| `dense_c2llm7b_release_v2603_none_top128_official.json` | frozen test/public snapshot | `0.633174` | `0.444714` | `0.657871` | `0.803820` |

- Clean frozen test interpretation:
  `release_v2603` C2LLM dense-none was chosen before test inspection. After
  correcting the evaluation protocol to match official C2C anchor exclusion,
  the frozen test/public-snapshot retrieval result is `0.633174` weighted
  `nDCG@10`, above the official project-page retrieval leaderboard best
  `GemEmb-2` at `0.624`.
- Scope of claim:
  this supports a public-snapshot retrieval SOTA claim under the project-page
  weighted metric. It is not yet a full reranker SOTA claim; next experiments
  target reranking gains on validation first.
- Dense3 RRF validation grid over `none/query/query_doc` was also checked.
  Best was `rrf_1_3_1_k5.json`, overall `0.632086` before protocol fix and
  still below the task-selective dense candidate after official C2C filtering.

### Validation reranker follow-up 2026-05-26T21:34Z

- Started a focused validation-only Qwen3 rerank run on `code2code`, because
  C2C is where official rerankers usually recover the most and the query count
  is small enough for a fast first check.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/rerank_qwen3.py \
  --candidate-run reports/dense_c2llm7b_release_v2602_task_selective_query_qd_none_official.json \
  --split release_v2602 \
  --tasks code2code \
  --model Qwen/Qwen3-Reranker-8B \
  --top-n 32 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 4096 \
  --score-mode true_logit \
  --load-in-4bit \
  --instruction-mode task \
  --fusion-alpha 0 \
  --output reports/qwen3_release_v2602_code2code_dense_top32_task_alpha0_official.json
```

- Remote PID: `1583635`.
- This is validation only. It should be used to decide whether a rerank strategy
  is worth freezing for a future public snapshot, not to retroactively tune the
  already observed `release_v2603` test result.

### Qwen3 validation rerank result 2026-05-26T21:51Z

- `Qwen/Qwen3-Reranker-8B`, 4-bit, task instruction, Qwen3 prompt style,
  `code2code`, validation `release_v2602`, dense top-32 candidates.
- Pure rerank (`fusion-alpha=0`) was harmful:
  `nDCG@10=0.491575`, `Recall@10=0.944280`, `MAP@10=0.321127`.
  Baseline dense `code2code` for the selected candidate was `0.677232`.
- Reused the cached 5,408 pair scores to sweep fusion alphas and top-N without
  additional GPU work. Best C2C validation setting:
  - `top_n=10`, `fusion_alpha=4`
  - `code2code nDCG@10=0.686176`, `Recall@10=0.998028`,
    `MAP@10=0.555587`
- New validation-only best combined report:
  `reports/qwen3_fused_release_v2602_t2cquery_c2ctop10alpha4_c2tnone_official.json`.
- Combined validation metrics:
  - Overall: `nDCG@10=0.643658`, `Recall@10=0.817313`,
    `MAP@10=0.577935`.
  - `text2code`: C2LLM `query`, `nDCG@10=0.440546`.
  - `code2code`: C2LLM `query_doc` top-10 + Qwen3 alpha-4 fusion,
    `nDCG@10=0.686176`.
  - `code2text`: C2LLM `none`, `nDCG@10=0.824319`.
- This improves the prior validation best
  `dense_c2llm7b_release_v2602_task_selective_query_qd_none_official.json`
  from `0.643077` to `0.643658`.
- Do not evaluate this post-test-tuned combination on `release_v2603`; the clean
  test claim remains the frozen C2LLM dense-none official result
  `0.633174`.

### Official CoREB reranker model blocker 2026-05-26T21:51Z

- Tried to cache `hq-bench/coreb-code-reranker` on the remote machine using
  `huggingface_hub.snapshot_download`.
- Blocker: remote network is unavailable:
  `httpx.ConnectError: [Errno 101] Network is unreachable`.
- Implemented and synced `--prompt-style coreb` support in `scripts/rerank_qwen3.py`
  and `src/coreb_sota/qwen3.py`, so the official reranker can be run with its
  documented prompt once the model is preloaded into the remote HF cache or
  copied to `/home/hhai/models`.

### Small-grid dense RRF validation update 2026-05-26T21:57Z

- Heartbeat check found no active remote CoREB processes; L20 was idle
  (`692 MiB`, `0%` utilization).
- Attempted a broad dense instruction RRF grid over `text2code` and `code2text`.
  The first implementation stored every fused run in memory and reached high
  remote memory use, so PID `1586196` was stopped and replaced with a bounded
  small-grid implementation that keeps only the current best candidate.
- Small-grid command class:
  - Inputs: C2LLM dense `none`, `query`, `query_doc`, `coreb_task` official
    reports on validation `release_v2602`.
  - Protocol: official C2C anchor exclusion.
  - Evaluation only on `release_v2602`.
- Best `text2code` dense RRF:
  `reports/dense_rrf_smallgrid_release_v2602_text2code_official.json`
  - Weights: `none=1`, `query=3`, `query_doc=1`, `coreb_task=0`,
    `rrf_k=5`.
  - `nDCG@10=0.440865`, `Recall@10=0.769306`, `MAP@10=0.317769`.
  - This is a small improvement over dense `query` alone
    (`nDCG@10=0.440546`).
- Best `code2text` dense RRF remained C2LLM `none` alone:
  `nDCG@10=0.824319`.
- New validation-only best combined report:
  `reports/smallgrid_rrf_qwen3_fused_release_v2602_best_official.json`.
- Combined validation metrics:
  - Overall: `nDCG@10=0.643800`, `Recall@10=0.817553`,
    `MAP@10=0.578019`.
  - `text2code`: dense RRF small-grid, `nDCG@10=0.440865`.
  - `code2code`: Qwen3 top-10 alpha-4 fusion, `nDCG@10=0.686176`.
  - `code2text`: C2LLM `none`, `nDCG@10=0.824319`.
- This improves the validation best from `0.643658` to `0.643800`.
- Still no new `release_v2603` evaluation: this is post-test validation tuning,
  and the clean public-snapshot test claim remains the frozen
  `dense_c2llm7b_release_v2603_none_top128_official.json` result
  (`nDCG@10=0.633174`).

### Manual progress check 2026-05-26T20:13Z

- Remote time: `2026-05-27T04:13:07+08:00`.
- Qwen3 true-logit reranking for `release_v2602/text2code` remains active.
- L20 status: `17563 MiB` GPU memory in use, `100%` utilization.
- Resumable score cache reached 15,872 / 18,640 pairs (`85.15%`).
- Final Qwen output is still pending:
  `reports/qwen3_release_v2602_text2code_top16_task_alpha0.json`.
- Dense C2LLM retrieval remains queued behind Qwen and has not started yet.
- Verified best remains BM25:
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.

### Qwen3 text2code validation result and dense restart 2026-05-26T20:30Z

- Qwen3 true-logit reranking finished for `release_v2602/text2code`, using
  BM25 top-16 candidates and `instruction-mode=task`.
- Output:
  `reports/qwen3_release_v2602_text2code_top16_task_alpha0.json`.
- Validation metrics:
  - Qwen3 `text2code`: `nDCG@10=0.273330`,
    `Recall@10=0.430038`, `MAP@10=0.201127`.
  - BM25 `text2code`: `nDCG@10=0.215141`,
    `Recall@10=0.366489`, `MAP@10=0.147938`.
- Interpretation: Qwen3 is a positive validation signal for the weak
  `text2code` task, but this is not an all-task average and is not a test/SOTA
  claim.
- The queued C2LLM dense retrieval initially failed because the C2LLM remote
  model imports `deepspeed.utils.zero_to_fp32` at module import time, and the
  shared venv does not include DeepSpeed. Added a lightweight compatibility
  shim under `compat/deepspeed/` for this unused inference path.
- A second C2LLM load attempt failed because C2LLM's config defaults to
  FlashAttention2 and the remote environment does not include `flash-attn`.
  Updated `scripts/run_dense_retrieval.py` to force
  `attn_implementation="eager"`.
- A third C2LLM load attempt tried to fetch the tokenizer from
  `codefuse-ai/C2LLM-7B` because `tokenizer_name_or_path` in config points at
  the repo id. Updated the dense script to redirect that field to the local
  model directory when the model path is local.
- A fourth C2LLM load attempt failed on a Transformers custom-model
  compatibility issue (`all_tied_weights_keys` missing). Updated the dense
  script to load the dynamic C2LLM class directly and attach the expected
  empty attribute before `from_pretrained`.
- Smoke test passed: C2LLM loads locally and exposes `encode`.
- Restarted dense retrieval:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=compat:src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2602 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode none \
  --output reports/dense_c2llm7b_release_v2602_none_top128.json
```

- Dense PID `1579550` is active on the L20. At
  `2026-05-27T04:30:06+08:00`, GPU utilization was `100%`, memory use was
  `23051 MiB`, and the log showed `Batches: 3/146` for the first encode pass.

### Heartbeat check 2026-05-26T20:33Z

- Remote time: `2026-05-27T04:33:24+08:00`.
- Dense C2LLM validation run remains active as PID `1579550`.
- L20 status: `23107 MiB` GPU memory in use, `100%` utilization.
- Dense output is still pending:
  `reports/dense_c2llm7b_release_v2602_none_top128.json`.
- Progress observed in the dense log:
  - `text2code` query embeddings completed (`146/146` batches).
  - `text2code` corpus embeddings completed (`209/209` batches).
  - `text2code` top-k scoring completed (`1165/1165` queries).
  - `code2code` query embeddings completed (`22/22` batches).
  - `code2code` corpus embeddings started and reached at least `22/209`
    batches.
- Current verified best remains BM25 for all-task validation until dense
  results and fusion scores are written:
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.

### Dense C2LLM validation result and next instruction run 2026-05-26T20:42Z

- Dense C2LLM validation run completed:
  `reports/dense_c2llm7b_release_v2602_none_top128.json`.
- Runtime: `494.88s` on the L20, `batch-size=8`, `max-length=2048`,
  `instruction-mode=none`.
- Validation metrics:
  - Overall: `nDCG@10=0.631248`, `Recall@10=0.817063`,
    `MAP@10=0.562577`.
  - `text2code`: `nDCG@10=0.435450`, `Recall@10=0.768212`,
    `MAP@10=0.311419`.
  - `code2code`: `nDCG@10=0.530096`, `Recall@10=0.998028`,
    `MAP@10=0.361421`.
  - `code2text`: `nDCG@10=0.824319`, `Recall@10=0.837795`,
    `MAP@10=0.819738`.
- This is the current best validation result and substantially improves over
  BM25 validation overall `nDCG@10=0.325560`.
- Ran BM25+dense weighted RRF grid over `release_v2602` with `rrf_k` in
  `{10,30,60,100}` and several BM25/dense weights. Best fusion was:
  `reports/fuse_bm25_dense_release_v2602_w0.25_1_rrf10.json` with
  `nDCG@10=0.629997`, `MAP@10=0.561528`, `Recall@10=0.809307`, which is
  slightly below dense alone.
- Interpretation: current frozen validation leader is dense C2LLM
  `instruction-mode=none`; do not evaluate the test split yet until the
  remaining validation instruction-mode check finishes.
- Started a validation-only C2LLM instruction experiment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=compat:src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2602 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode query_doc \
  --output reports/dense_c2llm7b_release_v2602_query_doc_top128.json
```

- Query-doc instruction PID `1580456` is active. At
  `2026-05-27T04:42:01+08:00`, L20 utilization was `100%`, memory was
  `23097 MiB`, and the first encode pass had reached `14/146` batches.

### Freeze decision and first test evaluation 2026-05-26T20:55Z

- C2LLM `query_doc` validation completed:
  `reports/dense_c2llm7b_release_v2602_query_doc_top128.json`.
- Query-doc validation overall: `nDCG@10=0.630301`,
  `Recall@10=0.815153`, `MAP@10=0.561840`.
- Query-doc was slightly below C2LLM `instruction-mode=none`
  (`nDCG@10=0.631248`, `MAP@10=0.562577`), and BM25+dense RRF was also below
  dense alone.
- Frozen validation choice for test:
  `C2LLM-7B dense retrieval`, `instruction-mode=none`, `top-k=128`,
  `eval-k=10`, `batch-size=8`, `max-length=2048`, `dtype=bf16`,
  `attn_implementation=eager`, local tokenizer redirect, deepspeed import shim.
- Started the single frozen `release_v2603` test/public-snapshot evaluation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=compat:src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2603 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode none \
  --output reports/dense_c2llm7b_release_v2603_none_top128.json
```

- Test PID `1581048` is active. At `2026-05-27T04:55:43+08:00`, L20
  utilization was `100%`, memory was `23237 MiB`, and the first encode pass
  had reached `11/141` batches.

### Frozen release_v2603 test result 2026-05-26T21:04Z

- Frozen test run completed:
  `reports/dense_c2llm7b_release_v2603_none_top128.json`.
- Runtime: `511.45s` on the L20.
- Test/public-snapshot metrics:
  - Overall: `nDCG@10=0.619127`, `Recall@10=0.821540`,
    `MAP@10=0.543348`.
  - `text2code`: `nDCG@10=0.444714`, `Recall@10=0.756894`,
    `MAP@10=0.326034`.
  - `code2code`: `nDCG@10=0.526447`, `Recall@10=0.995803`,
    `MAP@10=0.352679`.
  - `code2text`: `nDCG@10=0.803820`, `Recall@10=0.841667`,
    `MAP@10=0.790889`.
- Baseline comparison on the same `release_v2603` split:
  BM25 overall `nDCG@10=0.226204`, `Recall@10=0.355099`,
  `MAP@10=0.175828`.
- Interpretation: strong verified test improvement over BM25. Do not claim SOTA
  until this number is compared against the official CoREB public leaderboard /
  published v202603 snapshot.

### Post-test validation-only improvement pass 2026-05-26T21:09Z

- Important hygiene note: the frozen `release_v2603` test result has already
  been observed. Further tuning must be treated as validation-only unless
  evaluated on a new official/public snapshot. Selection below uses
  `release_v2602` metrics only.
- Built a task-selective validation candidate from existing dense runs:
  - `text2code`: C2LLM `instruction-mode=none`.
  - `code2code`: C2LLM `instruction-mode=query_doc`.
  - `code2text`: C2LLM `instruction-mode=none`.
- Output:
  `reports/dense_c2llm7b_release_v2602_task_selective_none_qd.json`.
- Validation metrics improved slightly over dense-none:
  `nDCG@10=0.631819`, `Recall@10=0.817063`, `MAP@10=0.563238`.
  Dense-none was `nDCG@10=0.631248`, `MAP@10=0.562577`.
- Started another validation-only instruction experiment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=compat:src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2602 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode query \
  --output reports/dense_c2llm7b_release_v2602_query_top128.json
```

- Query-instruction PID `1581776` is active. At
  `2026-05-27T05:09:24+08:00`, L20 utilization was `100%`, memory was
  `23097 MiB`, and the first encode pass had reached `11/146` batches.

### Query instruction result and per-task best update 2026-05-26T21:19Z

- C2LLM `query` validation completed:
  `reports/dense_c2llm7b_release_v2602_query_top128.json`.
- Query validation overall: `nDCG@10=0.631283`,
  `Recall@10=0.814625`, `MAP@10=0.563035`.
- Per-task signal:
  - `text2code`: query improved to `nDCG@10=0.440546`, above none
    `0.435450` and query-doc `0.435067`.
  - `code2code`: query-doc remains best at `0.538893`.
  - `code2text`: none remains best at `0.824319`.
- Built a stronger task-selective validation candidate:
  `reports/dense_c2llm7b_release_v2602_task_selective_query_qd_none.json`.
- New validation best:
  `nDCG@10=0.634099`, `Recall@10=0.817313`, `MAP@10=0.565995`.
- Started another validation-only instruction experiment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=compat:src /home/hhai/projects/finmteb-sota-reranker/.venv/bin/python \
scripts/run_dense_retrieval.py \
  --model /home/hhai/models/codefuse-ai_C2LLM-7B \
  --split release_v2602 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode coreb_task \
  --output reports/dense_c2llm7b_release_v2602_coreb_task_top128.json
```

- CoreB-task instruction PID `1582435` is active. At
  `2026-05-27T05:19:19+08:00`, L20 utilization was `95%`, memory was
  `23373 MiB`, and the first encode pass had reached `7/146` batches.

### Heartbeat check 2026-05-26T19:57Z

- Remote time: `2026-05-27T03:57:28+08:00`.
- Qwen3 true-logit reranking for `release_v2602/text2code` is still running
  normally on the L20.
- L20 status: `41797 MiB` GPU memory in use, `97%` utilization.
- Resumable score cache reached 11,904 / 18,640 pairs (`63.86%`).
- Final Qwen output is still pending:
  `reports/qwen3_release_v2602_text2code_top16_task_alpha0.json`.
- Dense C2LLM retrieval is still queued behind the active Qwen process.
- Verified best remains BM25:
  `release_v2602 nDCG@10=0.325560`, `release_v2603 nDCG@10=0.226204`.
