# CoREB Retrieval Snapshot

This is a lightweight migration of
[`Kevin-Li-2025/coreb-retrieval-sota`](https://github.com/Kevin-Li-2025/coreb-retrieval-sota)
into SignalRAG.

The migrated files preserve the code, scripts, tests, experiment log, and report
documentation needed to understand and rerun the CoREB retrieval harness. Large
checked-in datasets and report JSON artifacts remain in the archived source
repository instead of being duplicated here.

## Result Summary

The source project reported a clean `release_v2603` public-snapshot retrieval
result using the official query-weighted `nDCG@10` protocol:

| Split | Overall nDCG@10 | Recall@10 | MAP@10 |
| --- | ---: | ---: | ---: |
| `release_v2603` | `0.633174` | `0.821668` | `0.561064` |

Per-task `nDCG@10`:

| Task | nDCG@10 |
| --- | ---: |
| `text2code` | `0.444714` |
| `code2code` | `0.657871` |
| `code2text` | `0.803820` |

## Heavy Artifacts

The original repository includes large files under `data/coreb/` and `reports/`.
Those are intentionally not duplicated in SignalRAG. Use the archived source
repository if you need the exact checked-in report JSON artifacts.
