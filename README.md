# SignalRAG

[![CI](https://github.com/yinli-systems/signal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/yinli-systems/signal-rag/actions/workflows/ci.yml)

Web-search RAG workbench for provider routing, source extraction, citation
verification, corrective retrieval, and extractive fallback.

SignalRAG is a research and evaluation repository, not a hosted search product.
The runnable application lives in `fast_rag/`; migrated retrieval benchmarks are
kept as isolated subprojects under `evals/`.

## What Is Implemented

- concurrent search across configured providers and HTML fallbacks;
- query planning for lookup, comparison, freshness, and multi-step research;
- reciprocal-rank fusion, source-aware passage ranking, and context packing;
- cited answer generation with claim-level citation checks;
- corrective retrieval and an API-free extractive fallback;
- conservative page, planner, and response caches;
- a Chromium search URL and Manifest V3 extension.

## Evidence Snapshot

The checked-in results are local runs from 2026-05-12. Both suites contain 50
hand-curated queries; they are regression evidence, not representative production
traffic or public leaderboard results.

| Suite | Expected-source recall | Used-source recall | Citation coverage | Supported claims | Average / p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Golden regression | 0.9067 | 0.7567 | 0.8139 | 0.7906 | 15.4s / 41.3s |
| Realistic short-query | 0.9733 | 0.9467 | 0.8818 | 0.8656 | 7.4s / 11.7s |

Artifacts are stored in [`benchmark_results/`](benchmark_results/). The realistic
suite intentionally removes include-domain allowlists, but it remains curated and
should not be described as anonymized user traffic.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m fast_rag.app
```

Open `http://127.0.0.1:8000`. Without API keys, SignalRAG uses HTML search
fallbacks and extractive answer generation.

Optional providers are configured through environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
export DEEPSEEK_API_KEY="..."
export BRAVE_API_KEY="..."
export LLM_PROVIDER="auto"  # auto, openai, or deepseek
```

Do not commit real credentials. Provider-free operation is the default smoke path.

## API

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does ChatGPT search cite sources?",
    "mode": "pro",
    "max_results": 10,
    "include_domains": ["openai.com", "help.openai.com"],
    "recency": "year",
    "citation_verifier": "auto"
  }'
```

Modes:

- `fast`: shallow retrieval with short timeouts;
- `pro`: balanced multi-query retrieval and verification;
- `deep`: multi-step retrieval with a returned `research_trace`.

The response includes the query plan, retrieved sources, cited answer, claim checks,
and cache metadata.

## Architecture

```text
request
  -> query planner
  -> provider fan-out
  -> deduplication and reciprocal-rank fusion
  -> page extraction and passage ranking
  -> context packing
  -> answer generation or extractive fallback
  -> claim-level citation verification
```

The search-engine-compatible browser endpoint is:

```text
http://127.0.0.1:8000/engine?q=%s&mode=pro
```

Chromium extension assets live in [`extensions/signalrag-chromium/`](extensions/signalrag-chromium/).

## Evaluate

Retrieval smoke evaluation:

```bash
python -m fast_rag.eval --mode fast --top-k 5
python -m fast_rag.eval --mode pro --top-k 8
```

End-to-end regression suites:

```bash
python -m fast_rag.benchmark \
  --api-base http://127.0.0.1:8000 \
  --suite extended \
  --clear-response-cache \
  --output benchmark_results/extended.json

python -m fast_rag.benchmark \
  --api-base http://127.0.0.1:8000 \
  --suite realistic \
  --clear-response-cache \
  --output benchmark_results/realistic.json
```

The evaluator reports retrieval recall, source use, answer-term coverage, citation
coverage, claim support, fallback rate, and latency. Paid-provider results are not
directly comparable with provider-free fallback runs.

## Test

```bash
python -m pytest tests/
```

Tests cover the main application package. The subprojects under `evals/` keep their
own commands and artifacts.

## Repository Map

| Path | Purpose |
| --- | --- |
| `fast_rag/` | Runnable API, retrieval pipeline, caches, and evaluators |
| `tests/` | Main-package tests |
| `benchmark_results/` | Checked-in web-search benchmark snapshots |
| `evals/` | Arabic retrieval, CoREB, FinanceMTEB, and claim benchmarks |
| `tools/` | Auxiliary retrieval tools, including CodeGraph |
| `extensions/` | Chromium integration |
| `docs/repo-map.md` | Detailed ownership and navigation map |

## Limitations

- Search quality depends on provider availability and web-page extractability.
- The checked-in suites are small and hand-curated.
- Citation verification is a diagnostic signal, not a proof of factual correctness.
- Cache-hit latency must be reported separately from cold retrieval latency.
- Production deployment needs stronger abuse controls, observability, and a supported
  search provider.

## Page-Fetch Safety Boundary

Search results are untrusted input. Before fetching a page, the runnable app
now requires an HTTP(S) URL with a public-looking DNS name or globally routable
IP literal. It rejects credentials in URLs, dotless/local hostnames, and
loopback, private, link-local, reserved, multicast, or unspecified IP literals.
Redirects are followed explicitly for at most five hops, and every redirect
target is checked before the next request. Unsafe or failed fetches degrade to
the provider snippet rather than aborting the whole retrieval.

This is a meaningful default guard, not a complete production SSRF sandbox:
DNS rebinding and organization-specific egress policy require a resolver-aware
transport or outbound proxy at deployment time.

## Improvement Priorities

1. Enforce DNS/IP policy at connection time through a controlled resolver or
   egress proxy, then add redirect and rebinding integration tests.
2. Replace HTML search fallbacks with supported provider APIs for stable,
   observable production operation.
3. Expand the benchmark beyond hand-curated queries and report cold, warm, and
   paid-provider runs as separate cohorts.
4. Calibrate citation-support thresholds against independently labeled claims;
   the current verifier remains a diagnostic rather than a fact checker.

## License

MIT. See [LICENSE](LICENSE).
