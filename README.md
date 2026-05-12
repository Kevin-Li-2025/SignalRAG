# SignalRAG

A fast web-search RAG workbench inspired by ChatGPT Search:

- rewrites the user question into targeted search queries
- searches multiple providers concurrently
- fetches and extracts source pages in parallel
- uses a lightweight query planner before answer generation
- exposes search controls for domains, recency, locale, and citation verifier choice
- evaluates retrieval quality with a CRAG-style corrective pass
- runs multi-step retrieval in Deep Research mode
- integrates with Chromium through a local search URL and unpacked extension
- reranks passages with hybrid lexical scoring and source-quality signals
- generates a cited answer from retrieved evidence and returns claim-level citation checks
- falls back to extractive answers when no LLM API key is configured

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m fast_rag.app
```

Open http://127.0.0.1:8000.

## Chromium Integration

SignalRAG exposes a search-engine-compatible URL:

```text
http://127.0.0.1:8000/engine?q=%s&mode=pro
```

It also includes a Manifest V3 extension in
`extensions/signalrag-chromium` with:

- `sr` omnibox keyword search.
- selected-text context menu search.
- browser side panel search.
- extension options for the local API URL and default mode.

Load it from `chrome://extensions` with Developer mode and "Load unpacked".
See `extensions/signalrag-chromium/README.md`.

## Optional Accuracy Upgrades

Set these environment variables before starting the server:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_PLANNER_MODEL="deepseek-v4-flash"
export DEEPSEEK_VERIFIER_MODEL="deepseek-v4-flash"
export BRAVE_API_KEY="..."
```

Without API keys, the app still works using DuckDuckGo HTML search and an extractive cited answer.

If both DeepSeek and OpenAI keys are present, DeepSeek is used first by default. Override with:

```bash
export LLM_PROVIDER="openai"   # or deepseek / auto
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"how does ChatGPT search work",
    "mode":"pro",
    "max_results":10,
    "include_domains":["openai.com","help.openai.com"],
    "exclude_domains":["medium.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Modes:

- `fast`: low latency, fewer pages, shorter timeouts.
- `pro`: balanced mode for fresher, comparative, or multi-hop questions.
- `deep`: Deep Research mode. It builds several focused research steps, runs them
  in parallel, dedupes the evidence, and returns `research_trace`.

The API response includes `query_plan`, which reports the planner's inferred intent,
freshness need, search depth, and DeepSeek reasoning effort:

- `none`: thinking disabled for simple lookups.
- `high`: thinking enabled for comparisons, recommendations, API/code guidance, multi-hop synthesis, or uncertainty.
- `max`: thinking enabled with max effort only for deep research, long-horizon tasks, formal proof, or many constraints.

The request can include:

- `include_domains` / `exclude_domains`: allowlist or denylist domains.
- `recency`: `any`, `day`, `week`, `month`, or `year`.
- `country` / `language`: two-letter locale hints for supported providers.
- `citation_verifier`: `auto`, `lexical`, or `deepseek`.

The response includes:

- `crag`: retrieval quality before and, if needed, after a corrective search.
- `research_trace`: the per-step trace used by Deep Research mode.
- `claim_citations`: per-claim citation trace. With DeepSeek configured, `auto`
  uses a judge model for supported/weak/contradicted/insufficient decisions;
  otherwise it falls back to the fast lexical verifier.

## Example Runs

These examples were run locally on 2026-05-12 with DeepSeek enabled,
DuckDuckGo search, and no Brave API key. They are the best demo cases because
they use official sources, produce inline citations, and exercise different
parts of the retrieval stack.

| Use case | Query | Mode | Observed result |
| --- | --- | --- | --- |
| Fast official API lookup | `DeepSeek API chat completion base URL model name and first API call` | `fast` | 2 official DeepSeek citations, 3 supported claims, ~4.0s |
| Product search explanation | `How does ChatGPT search work and how does it cite sources?` | `pro` | 2 official OpenAI citations, 3 supported claims, ~9.2s |
| API docs with source controls | `OpenAI web search API citations and domain filtering` | `pro` | OpenAI developer citation, planner chose `high` reasoning, ~12.6s |
| Deep Research trace | `Explain ChatGPT search for Enterprise and Edu data sharing and source citations.` | `deep` | 2 official OpenAI citations, 3 research steps, 5 supported claims, ~19.2s |

### Fast official API lookup

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"DeepSeek API chat completion base URL model name and first API call",
    "mode":"fast",
    "max_results":8,
    "include_domains":["api-docs.deepseek.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it shows the lightweight planner choosing
`reasoning_effort: none`, keeps latency low, and returns only official
DeepSeek API documentation as citations.

### ChatGPT Search citation behavior

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"How does ChatGPT search work and how does it cite sources?",
    "mode":"pro",
    "max_results":10,
    "include_domains":["openai.com","help.openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it exercises official-source prioritization,
answer citations, and claim-level verification against OpenAI Help Center and
OpenAI announcement pages.

### API docs with source controls

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"OpenAI web search API citations and domain filtering",
    "mode":"pro",
    "max_results":10,
    "include_domains":["developers.openai.com","platform.openai.com","openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it shows include-domain controls, freshness-aware
planning for API documentation, and citation grounding from developer docs.

### Deep Research trace

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Explain ChatGPT search for Enterprise and Edu data sharing and source citations.",
    "mode":"deep",
    "max_results":12,
    "include_domains":["help.openai.com","openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it runs Deep Research mode, returns a multi-step
`research_trace`, and verifies claims across workspace policy and citation
behavior sources.

## Recall Evaluation

```bash
python -m fast_rag.eval --mode fast --top-k 5
python -m fast_rag.eval --mode pro --top-k 8
python -m fast_rag.eval --mode deep --top-k 10
```

The evaluator reports recall@k, hit rate, MRR, and latency over a small set of known-answer web-search cases.

## Design Notes

Accuracy comes from grounding every answer in retrieved passages, checking retrieval quality before generation, and returning only used citations by default. Speed comes from short timeouts, request concurrency, page caching, early reranking, and the planner choosing the cheapest mode that fits the query. For production, use a paid search API such as Brave or Tavily, add an embedding or cross-encoder reranker, and persist traces for evaluation.
