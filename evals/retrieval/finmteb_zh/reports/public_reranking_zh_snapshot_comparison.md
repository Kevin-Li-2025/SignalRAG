# Public Reranking_zh Snapshot Comparison

Date checked: 2026-05-26

This report compares the current project result against the public official
FinanceMTEB leaderboard snapshot exposed by the Hugging Face Space.

## Sources

- Official Space: https://huggingface.co/spaces/FinanceMTEB/FinMTEB
- Space files: https://huggingface.co/spaces/FinanceMTEB/FinMTEB/tree/main
- Leaderboard implementation:
  https://huggingface.co/spaces/FinanceMTEB/FinMTEB/blob/main/app.py
- Public snapshot file:
  https://huggingface.co/spaces/FinanceMTEB/FinMTEB/blob/main/benchmark.xlsx
- Local verified result:
  `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json`

The Space implementation reads `benchmark.xlsx`, computes per-task `Avg.`, and
sorts by that average. The current `benchmark.xlsx` file is the public snapshot
used for this comparison.

## Public Snapshot Top Result

The downloaded `benchmark.xlsx` `Reranking_zh` sheet has the following visible
top row:

| Rank | Model | FinEvaReranking MAP | DISCFinLLMReranking MAP | Reranking_zh Avg. MAP |
| ---: | --- | ---: | ---: | ---: |
| 1 | `bge-large-zh-v1.5` | 0.990600 | 0.995600 | 0.993100 |

## Current Project Result

Frozen method:

- Base model: `Qwen/Qwen3-Reranker-8B`
- Inference: 4-bit
- Score mode: raw `true` token logit
- Train-selected strategy:
  - FinEvaReranking: `rrf/doc_bigram alpha=1.0`
  - DISCFinLLMReranking: `rrf/title_trigram alpha=1.0`
- Test selection policy: train-only strategy selection, then frozen test
  evaluation.

| Task | MAP | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| FinEvaReranking | 1.000000 | 1.000000 | 1.000000 |
| DISCFinLLMReranking | 0.995614 | 1.000000 | 0.998288 |
| Reranking_zh Average | 0.997807 | - | - |

## Delta

| Metric | Public snapshot best | Current result | Delta |
| --- | ---: | ---: | ---: |
| Reranking_zh Avg. MAP | 0.993100 | 0.997807 | +0.004707 |
| FinEvaReranking MAP | 0.990600 | 1.000000 | +0.009400 |
| DISCFinLLMReranking MAP | 0.995600 | 0.995614 | +0.000014 |

## Claim Boundary

Supported claim:

> New SOTA on the public FinanceMTEB `Reranking_zh` benchmark snapshot, with
> 0.997807 average MAP versus the published visible best 0.993100.

Do not overclaim:

- This is not yet "official leaderboard #1" until the FinMTEB maintainers accept
  and list the submission.
- The strongest claim is the `Reranking_zh` average. The DISC-only margin is
  very small and should be described as matching/slightly exceeding the rounded
  public snapshot value.

