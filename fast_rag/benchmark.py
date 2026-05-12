from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    query: str
    mode: str = "pro"
    lens: str = "official"
    max_results: int = 10
    include_domains: tuple[str, ...] = field(default_factory=tuple)
    expected_url_parts: tuple[str, ...] = field(default_factory=tuple)
    expected_terms: tuple[str, ...] = field(default_factory=tuple)
    citation_verifier: str = "auto"
    recency: str = "any"
    country: str = "us"
    language: str = "en"

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "query": data["query"],
            "mode": data["mode"],
            "lens": data["lens"],
            "max_results": data["max_results"],
            "include_domains": list(data["include_domains"]),
            "citation_verifier": data["citation_verifier"],
            "recency": data["recency"],
            "country": data["country"],
            "language": data["language"],
        }


QUICK_CASES = (
    BenchmarkCase(
        name="chatgpt_search_citations",
        query="How does ChatGPT search work and cite sources?",
        mode="pro",
        include_domains=("openai.com", "help.openai.com"),
        expected_url_parts=(
            "help.openai.com/en/articles/9237897",
            "openai.com/index/introducing-chatgpt-search",
        ),
        expected_terms=("query", "sources", "citations"),
    ),
    BenchmarkCase(
        name="openai_web_search_api",
        query="OpenAI web search API citations and domain filtering",
        mode="pro",
        include_domains=("developers.openai.com", "openai.com"),
        expected_url_parts=("developers.openai.com/api/docs/guides/tools-web-search",),
        expected_terms=("domain", "sources", "citations"),
    ),
    BenchmarkCase(
        name="deepseek_api_quickstart",
        query="DeepSeek API chat completion base URL model name and first API call",
        mode="fast",
        include_domains=("api-docs.deepseek.com",),
        expected_url_parts=(
            "api-docs.deepseek.com",
            "api-docs.deepseek.com/api/create-chat-completion",
        ),
        expected_terms=("api.deepseek.com", "deepseek", "model"),
    ),
    BenchmarkCase(
        name="deepseek_thinking_mode",
        query="Explain DeepSeek thinking mode, reasoning_effort high max, and when to disable thinking.",
        mode="deep",
        include_domains=("api-docs.deepseek.com",),
        expected_url_parts=(
            "api-docs.deepseek.com/guides/thinking_mode",
            "api-docs.deepseek.com/quick_start/pricing",
        ),
        expected_terms=("thinking", "reasoning_effort", "high", "max"),
    ),
    BenchmarkCase(
        name="context_window_compression",
        query="What is the best way to compress context windows for RAG without losing key evidence?",
        mode="deep",
        include_domains=("microsoft.com", "anthropic.com", "arxiv.org"),
        expected_url_parts=(
            "microsoft.com/en-us/research/project/llmlingua/longllmlingua",
            "anthropic.com/news/contextual-retrieval",
            "arxiv.org/abs/2307.03172",
        ),
        expected_terms=("LongLLMLingua", "contextual", "lost"),
    ),
    BenchmarkCase(
        name="chatgpt_enterprise_edu",
        query="Explain ChatGPT search for Enterprise and Edu data sharing and source citations.",
        mode="deep",
        include_domains=("help.openai.com", "openai.com"),
        expected_url_parts=(
            "help.openai.com/en/articles/10093903",
            "help.openai.com/en/articles/9237897",
        ),
        expected_terms=("Enterprise", "Edu", "citations"),
    ),
)


def _case_variants(
    prefix: str,
    queries: tuple[str, ...],
    *,
    mode: str,
    include_domains: tuple[str, ...],
    expected_url_parts: tuple[str, ...],
    expected_terms: tuple[str, ...],
    lens: str = "official",
    max_results: int = 10,
) -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(
            name=f"{prefix}_{index:02d}",
            query=query,
            mode=mode,
            lens=lens,
            max_results=max_results,
            include_domains=include_domains,
            expected_url_parts=expected_url_parts,
            expected_terms=expected_terms,
        )
        for index, query in enumerate(queries, start=1)
    )


EXTENDED_CASES = (
    *_case_variants(
        "chatgpt_search",
        (
            "How does ChatGPT search work and cite sources?",
            "How ChatGPT search cites its sources",
            "Explain ChatGPT search query rewriting and citations",
            "What sources does ChatGPT search use when answering?",
            "ChatGPT search source links and query rewriting overview",
            "How does ChatGPT decide to search the web and show citations?",
            "ChatGPT search citations desktop web source panel",
            "What is OpenAI ChatGPT search and how are sources cited?",
            "ChatGPT search answer citations and web source links",
            "Summarize how ChatGPT search uses web results with citations",
        ),
        mode="pro",
        max_results=10,
        include_domains=("openai.com", "help.openai.com"),
        expected_url_parts=(
            "help.openai.com/en/articles/9237897",
            "openai.com/index/introducing-chatgpt-search",
        ),
        expected_terms=("query", "sources", "citations"),
    ),
    *_case_variants(
        "openai_web_search_api",
        (
            "OpenAI web search API citations and domain filtering",
            "OpenAI Responses API web search sources and allowed domains",
            "How does OpenAI web search API return citations and sources?",
            "OpenAI web_search tool filters allowed domains blocked domains",
            "OpenAI API web search source citations guide",
            "Domain filtering in OpenAI web search API citations",
            "How to use OpenAI web search tool with source citations",
            "OpenAI web search API sources field and domain controls",
        ),
        mode="pro",
        max_results=10,
        include_domains=("developers.openai.com", "platform.openai.com", "openai.com"),
        expected_url_parts=("developers.openai.com/api/docs/guides/tools-web-search",),
        expected_terms=("domain", "sources", "citations"),
    ),
    *_case_variants(
        "deepseek_api",
        (
            "DeepSeek API chat completion base URL model name and first API call",
            "DeepSeek API base URL chat completion model name quickstart",
            "How to make the first DeepSeek API chat completion call",
            "DeepSeek API OpenAI compatible base URL and model",
            "DeepSeek chat completions endpoint base URL and current models",
            "DeepSeek API quickstart model deepseek chat completion",
            "What base URL should I use for DeepSeek API chat completions?",
            "DeepSeek first API call chat completion endpoint and model",
        ),
        mode="fast",
        max_results=8,
        include_domains=("api-docs.deepseek.com",),
        expected_url_parts=(
            "api-docs.deepseek.com",
            "api-docs.deepseek.com/api/create-chat-completion",
        ),
        expected_terms=("api.deepseek.com", "deepseek", "model"),
    ),
    *_case_variants(
        "deepseek_thinking",
        (
            "Explain DeepSeek thinking mode, reasoning_effort high max, and when to disable thinking.",
            "How should DeepSeek thinking mode be used for high and max reasoning?",
            "DeepSeek thinking enabled disabled reasoning_effort guide",
            "When should a RAG system disable DeepSeek thinking mode?",
            "DeepSeek reasoning_effort high versus max for agent tasks",
            "DeepSeek thinking mode output tokens and pricing considerations",
            "How to configure DeepSeek thinking mode in chat completions",
        ),
        mode="deep",
        max_results=10,
        include_domains=("api-docs.deepseek.com",),
        expected_url_parts=(
            "api-docs.deepseek.com/guides/thinking_mode",
            "api-docs.deepseek.com/quick_start/pricing",
        ),
        expected_terms=("thinking", "high", "max"),
    ),
    *_case_variants(
        "context_window",
        (
            "What is the best way to compress context windows for RAG without losing key evidence?",
            "LongLLMLingua context compression for long context RAG",
            "How does contextual retrieval reduce RAG retrieval failures?",
            "Lost in the middle context window compression RAG evidence placement",
            "Compare LongLLMLingua and contextual retrieval for RAG context packing",
            "How should RAG systems pack evidence to avoid lost in the middle?",
            "Contextual retrieval and context compression for cited AI search",
        ),
        mode="deep",
        max_results=10,
        include_domains=("microsoft.com", "anthropic.com", "arxiv.org"),
        expected_url_parts=(
            "microsoft.com/en-us/research/project/llmlingua/longllmlingua",
            "anthropic.com/news/contextual-retrieval",
            "arxiv.org/abs/2307.03172",
        ),
        expected_terms=("LongLLMLingua", "contextual", "lost"),
    ),
    *_case_variants(
        "chatgpt_enterprise_edu",
        (
            "Explain ChatGPT search for Enterprise and Edu data sharing and source citations.",
            "ChatGPT search Enterprise Edu data sharing source citations",
            "How do Enterprise and Edu workspaces handle ChatGPT search citations?",
            "ChatGPT search for Enterprise and Edu admin controls and sources",
            "What should admins know about ChatGPT search Enterprise Edu sources?",
        ),
        mode="deep",
        max_results=10,
        include_domains=("help.openai.com", "openai.com"),
        expected_url_parts=(
            "help.openai.com/en/articles/10093903",
            "help.openai.com/en/articles/9237897",
        ),
        expected_terms=("Enterprise", "Edu", "citations"),
    ),
    *_case_variants(
        "source_trust",
        (
            "Which sources are most trustworthy for RAG citations: government academic official docs?",
            "How should AI search rank government academic and official documentation sources?",
            "RAG source trust tiers government academic standards official documentation",
            "What source types should a cited search engine trust most?",
            "Source credibility for RAG answers official docs academic government standards",
        ),
        mode="pro",
        max_results=10,
        include_domains=("developers.google.com", "cancer.gov", "nih.gov", "scribbr.com"),
        expected_url_parts=(
            "developers.google.com/search/docs/fundamentals/creating-helpful-content",
            "cancer.gov/about-cancer/managing-care/using-trusted-resources",
        ),
        expected_terms=("trust", "government", "academic"),
    ),
)

REALISTIC_CASE_DATA = (
    ("chrome_default_search", "how do i change chrome default search engine", "fast", "web", ("support.google.com/chrome",), ("default", "search engine")),
    ("mac_screenshot_shortcut", "mac screenshot shortcut", "fast", "web", ("support.apple.com",), ("screenshot", "shift")),
    ("windows_screenshot_shortcut", "windows screenshot shortcut", "fast", "web", ("support.microsoft.com",), ("screenshot", "snipping")),
    ("git_undo_commit", "git undo last commit but keep changes", "pro", "web", ("git-scm.com",), ("reset", "commit")),
    ("git_rebase_merge", "git rebase vs merge", "pro", "web", ("git-scm.com",), ("rebase", "merge")),
    ("python_read_json", "python read json file", "fast", "web", ("docs.python.org",), ("json", "load")),
    ("python_venv_activate", "python venv activate", "fast", "web", ("docs.python.org",), ("venv", "activate")),
    ("pandas_read_csv_dtype", "pandas read csv dtype", "fast", "web", ("pandas.pydata.org",), ("read_csv", "dtype")),
    ("numpy_random_seed", "numpy random seed generator", "fast", "web", ("numpy.org",), ("random", "seed")),
    ("fastapi_upload_file", "fastapi upload file example", "fast", "web", ("fastapi.tiangolo.com",), ("UploadFile", "File")),
    ("docker_compose_env", "docker compose env file syntax", "fast", "web", ("docs.docker.com",), ("environment", "env")),
    ("npm_global_install", "npm install package globally", "fast", "web", ("docs.npmjs.com",), ("global", "install")),
    ("github_actions_cache_pnpm", "github actions cache pnpm", "pro", "web", ("docs.github.com",), ("cache", "pnpm")),
    ("aws_s3_presigned_url", "aws s3 presigned url boto3", "pro", "web", ("docs.aws.amazon.com",), ("presigned", "S3")),
    ("postgres_index_concurrently", "postgres create index concurrently lock", "pro", "web", ("postgresql.org/docs",), ("CONCURRENTLY", "lock")),
    ("react_useeffect_cleanup", "react useeffect cleanup function", "fast", "web", ("react.dev",), ("cleanup", "effect")),
    ("typescript_optional_chaining", "typescript optional chaining nullish coalescing", "fast", "web", ("typescriptlang.org",), ("optional", "nullish")),
    ("oauth_pkce", "oauth pkce what problem does it solve", "pro", "web", ("datatracker.ietf.org", "oauth.net"), ("PKCE", "code")),
    ("dns_cname_a_record", "dns cname vs a record", "pro", "web", ("cloudflare.com/learning/dns",), ("CNAME", "A record")),
    ("robots_disallow_all", "robots txt block all crawlers", "fast", "web", ("developers.google.com/search/docs",), ("robots.txt", "Disallow")),
    ("sitemap_xml", "sitemap xml where to put it", "fast", "web", ("developers.google.com/search/docs",), ("sitemap", "XML")),
    ("wcag_contrast_ratio", "wcag contrast ratio normal text", "fast", "web", ("w3.org",), ("contrast", "4.5")),
    ("nist_password_guidelines", "nist password length guidelines", "pro", "web", ("pages.nist.gov", "nist.gov"), ("password", "length")),
    ("owasp_broken_access", "owasp top 10 broken access control", "fast", "web", ("owasp.org",), ("Broken Access Control", "OWASP")),
    ("mitre_phishing", "mitre attack phishing technique", "fast", "web", ("attack.mitre.org",), ("Phishing", "Initial Access")),
    ("irs_standard_deduction", "irs standard deduction 2024 single", "fast", "web", ("irs.gov",), ("standard deduction", "2024")),
    ("quarterly_taxes_due", "when are 2025 quarterly taxes due", "pro", "web", ("irs.gov",), ("estimated tax", "2025")),
    ("tsa_liquids_rule", "tsa liquids rule carry on", "fast", "web", ("tsa.gov",), ("3.4", "quart")),
    ("freeze_credit", "freeze credit free ftc", "fast", "web", ("consumer.ftc.gov",), ("freeze", "credit")),
    ("escrow_mortgage", "what is escrow mortgage", "fast", "web", ("consumerfinance.gov",), ("escrow", "mortgage")),
    ("federal_funds_rate", "federal funds rate meaning", "fast", "web", ("federalreserve.gov",), ("federal funds", "rate")),
    ("cpi_inflation", "cpi inflation meaning bls", "fast", "web", ("bls.gov",), ("Consumer Price Index", "inflation")),
    ("sec_10k", "sec 10-k what is it", "fast", "web", ("sec.gov", "investor.gov"), ("10-K", "annual")),
    ("coffee_caffeine_fda", "how much caffeine in coffee fda", "fast", "web", ("fda.gov",), ("caffeine", "coffee")),
    ("vitamin_d_deficiency", "symptoms vitamin d deficiency nih", "pro", "web", ("ods.od.nih.gov", "nih.gov"), ("vitamin D", "deficiency")),
    ("blood_pressure_lifestyle", "lower blood pressure lifestyle changes cdc", "pro", "web", ("cdc.gov",), ("blood pressure", "lifestyle")),
    ("apa_website_no_author", "apa cite website no author", "fast", "web", ("owl.purdue.edu",), ("no author", "APA")),
    ("mla_youtube", "mla cite youtube video", "fast", "web", ("owl.purdue.edu",), ("YouTube", "MLA")),
    ("columbus_county", "what county is columbus city in", "fast", "web", ("wikipedia.org/wiki/Columbus,_Ohio", "franklincountyohio.gov"), ("Franklin", "county")),
    ("northern_lights", "what causes northern lights", "pro", "web", ("nasa.gov", "noaa.gov", "spaceweather.gov"), ("charged particles", "aurora")),
    ("sky_blue", "why is the sky blue", "fast", "web", ("nasa.gov", "noaa.gov"), ("scattering", "blue")),
    ("earthquake_magnitude_intensity", "earthquake magnitude vs intensity", "pro", "web", ("usgs.gov",), ("magnitude", "intensity")),
    ("hurricane_watch_warning", "hurricane watch vs warning", "fast", "web", ("weather.gov",), ("watch", "warning")),
    ("lost_middle", "lost in the middle long context paper", "deep", "academic", ("arxiv.org/abs/2307.03172",), ("lost", "middle")),
    ("anthropic_contextual_retrieval", "anthropic contextual retrieval bm25 embeddings", "deep", "web", ("anthropic.com/news/contextual-retrieval",), ("contextual", "BM25")),
    ("ragas_faithfulness", "ragas faithfulness metric", "pro", "web", ("docs.ragas.io",), ("faithfulness", "claims")),
    ("perplexity_citations_api", "perplexity api citations", "pro", "web", ("docs.perplexity.ai",), ("citations", "search")),
    ("deepseek_base_url", "deepseek api base url openai compatible", "fast", "web", ("api-docs.deepseek.com",), ("api.deepseek.com", "OpenAI")),
    ("chatgpt_search_sources", "chatgpt search sources openai help", "pro", "web", ("help.openai.com/en/articles/9237897",), ("sources", "search")),
    ("openai_web_search_domains", "openai web search api allowed domains", "pro", "web", ("developers.openai.com",), ("domains", "citations")),
)

REALISTIC_CASES = tuple(
    BenchmarkCase(
        name=name,
        query=query,
        mode=mode,
        lens=lens,
        max_results=12 if mode == "deep" else 10 if mode == "pro" else 8,
        expected_url_parts=expected_url_parts,
        expected_terms=expected_terms,
    )
    for name, query, mode, lens, expected_url_parts, expected_terms in REALISTIC_CASE_DATA
)

SUITES = {
    "quick": QUICK_CASES,
    "extended": EXTENDED_CASES,
    "realistic": REALISTIC_CASES,
}

CASES = QUICK_CASES


def post_json(api_base: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def evaluate_response(case: BenchmarkCase, response: dict[str, Any], wall_ms: int) -> dict[str, Any]:
    used = response.get("used_citations") or response.get("citations") or []
    candidates = response.get("candidate_citations") or []
    claim_citations = response.get("claim_citations") or []
    answer = str(response.get("answer") or "")
    all_urls = [str(item.get("url") or "") for item in [*used, *candidates]]
    used_urls = [str(item.get("url") or "") for item in used]

    matched_expected = [
        expected
        for expected in case.expected_url_parts
        if any(_url_matches(url, expected) for url in all_urls)
    ]
    matched_used = [
        expected
        for expected in case.expected_url_parts
        if any(_url_matches(url, expected) for url in used_urls)
    ]
    term_hits = [term for term in case.expected_terms if term.lower() in answer.lower()]

    claims_with_citations = sum(1 for claim in claim_citations if claim.get("citation_ids"))
    supported_claims = sum(1 for claim in claim_citations if claim.get("status") == "supported")
    weak_claims = sum(1 for claim in claim_citations if claim.get("status") == "weak")
    contradicted_claims = sum(1 for claim in claim_citations if claim.get("status") == "contradicted")
    review_claims = sum(
        1
        for claim in claim_citations
        if claim.get("status") in {"insufficient", "needs_review", "missing_citation", "unsupported"}
    )
    total_claims = len(claim_citations)
    meta = response.get("meta") or {}
    elapsed_ms = meta["elapsed_ms"] if meta.get("elapsed_ms") is not None else wall_ms

    return {
        "name": case.name,
        "mode": response.get("mode"),
        "answer_mode": response.get("answer_mode"),
        "reasoning_effort": (response.get("query_plan") or {}).get("reasoning_effort"),
        "expected_source_recall": _ratio(len(matched_expected), len(case.expected_url_parts))
        if case.expected_url_parts
        else None,
        "used_source_recall": _ratio(len(matched_used), len(case.expected_url_parts)) if case.expected_url_parts else None,
        "answer_term_coverage": _ratio(len(term_hits), len(case.expected_terms)),
        "citation_coverage": _ratio(claims_with_citations, total_claims),
        "supported_claim_rate": _ratio(supported_claims, total_claims),
        "weak_claim_rate": _ratio(weak_claims, total_claims),
        "review_claim_rate": _ratio(review_claims, total_claims),
        "contradicted_claims": contradicted_claims,
        "used_citations": len(used),
        "claims": total_claims,
        "crag_status": meta.get("crag_status"),
        "crag_corrected": meta.get("crag_corrected"),
        "research_steps": meta.get("research_steps", 0),
        "context_compression_ratio": (meta.get("context_packing") or {}).get("compression_ratio"),
        "cache_hit": bool(meta.get("cache_hit")),
        "elapsed_ms": elapsed_ms,
        "wall_ms": wall_ms,
        "matched_expected": matched_expected,
        "matched_used": matched_used,
        "term_hits": term_hits,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(results),
        "source_scored_cases": sum(1 for item in results if item.get("expected_source_recall") is not None),
        "expected_source_recall": _mean_defined(results, "expected_source_recall"),
        "used_source_recall": _mean_defined(results, "used_source_recall"),
        "answer_term_coverage": _mean(results, "answer_term_coverage"),
        "citation_coverage": _mean(results, "citation_coverage"),
        "supported_claim_rate": _mean(results, "supported_claim_rate"),
        "weak_claim_rate": _mean(results, "weak_claim_rate"),
        "review_claim_rate": _mean(results, "review_claim_rate"),
        "crag_sufficient_rate": _ratio(sum(1 for item in results if item.get("crag_status") == "sufficient"), len(results)),
        "fallback_rate": _ratio(sum(1 for item in results if item.get("answer_mode") == "extractive"), len(results)),
        "cache_hit_rate": _ratio(sum(1 for item in results if item.get("cache_hit")), len(results)),
        "avg_elapsed_ms": round(statistics.mean(item["elapsed_ms"] for item in results)) if results else 0,
        "p95_elapsed_ms": _percentile([item["elapsed_ms"] for item in results], 0.95),
    }


def run_benchmark(api_base: str, timeout: float, cases: tuple[BenchmarkCase, ...]) -> dict[str, Any]:
    results = []
    for case in cases:
        started = time.perf_counter()
        response = post_json(api_base, case.payload(), timeout)
        wall_ms = round((time.perf_counter() - started) * 1000)
        results.append(evaluate_response(case, response, wall_ms))
    return {
        "api_base": api_base,
        "summary": summarize(results),
        "results": results,
    }


def _url_matches(url: str, expected_part: str) -> bool:
    return expected_part.rstrip("/") in url.rstrip("/")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _mean(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return round(statistics.mean(float(item.get(key) or 0.0) for item in results), 4)


def _mean_defined(results: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in results if item.get(key) is not None]
    if not values:
        return 0.0
    return round(statistics.mean(values), 4)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end SignalRAG quality benchmark.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--clear-response-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.clear_response_cache:
        clear_response_cache()
    cases = SUITES[args.suite]
    report = run_benchmark(args.api_base, args.timeout, cases)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


def clear_response_cache() -> None:
    try:
        with sqlite3.connect(settings.cache_path) as conn:
            conn.execute("DELETE FROM responses")
            conn.commit()
    except sqlite3.Error:
        return


if __name__ == "__main__":
    main()
