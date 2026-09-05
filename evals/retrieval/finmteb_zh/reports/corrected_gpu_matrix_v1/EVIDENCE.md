# Corrected FinanceMTEB Chinese reranking GPU matrix

Status: complete.

The matrix contains four independent RTX 4090 arms: BF16 and bitsandbytes NF4
for `FinEvaReranking` and `DISCFinLLMReranking`. Strategy selection used train
only; the test split was frozen before final evaluation. Each arm passed the
candidate-order audit for seeds 20260905, 2234, 314159, and 8675309.

| Precision | Task | Queries / pairs | MAP | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | FinEvaReranking | 53 / 582 | 0.990566 | 0.990566 | 0.993036 |
| BF16 | DISCFinLLMReranking | 19 / 242 | 1.000000 | 1.000000 | 1.000000 |
| NF4 | FinEvaReranking | 53 / 582 | 1.000000 | 1.000000 | 1.000000 |
| NF4 | DISCFinLLMReranking | 19 / 242 | 1.000000 | 1.000000 | 1.000000 |

The macro MAP is 0.995283 for BF16 and 1.0 for NF4. These small, saturated test
splits do not establish general NF4 superiority or a public-leaderboard SOTA.

`corrected_gpu_matrix_summary.json` has SHA-256
`52367765027a2acc7f51595a4e559a92196fb07fac1afd9a93f2b1ad7bea9566`,
matching the sealed remote artifact. `SHA256SUMS` covers every committed result.
Raw score caches are deliberately excluded because they reproduce dataset
candidate content; strict cache-coverage and order-invariance evidence is
retained in the committed audits and frozen outputs.
