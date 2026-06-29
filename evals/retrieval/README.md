# Retrieval Evaluations

This directory collects retrieval benchmark work that supports SignalRAG's
broader evidence, reranking, and citation-verification story.

- [`finmteb_zh/`](finmteb_zh/) is the migrated Chinese FinanceMTEB reranking
  snapshot and Qwen3 reranker experimentation harness.
- [`coreb/`](coreb/) is a lightweight migration of the CoREB code-retrieval
  harness. Large checked-in report and dataset artifacts remain available in the
  archived source repository to avoid bloating SignalRAG.
- [`arabic/`](arabic/) is the migrated Arabic retrieval and embedding benchmark
  lab for MIRACL/MTEB-style experiments.

## Local Checks

Run each migrated project from the appropriate root so its original relative
paths keep working:

```bash
PYTHONPATH=evals/retrieval/finmteb_zh/src \
  python -m pytest -q evals/retrieval/finmteb_zh/tests

PYTHONPATH=evals/retrieval/coreb/src \
  python -m pytest -q evals/retrieval/coreb/tests

cd evals/retrieval/arabic
PYTHONPATH=src:. python -m pytest -q tests
```
