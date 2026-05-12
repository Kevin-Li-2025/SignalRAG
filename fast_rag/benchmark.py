from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "query": data["query"],
            "mode": data["mode"],
            "lens": data["lens"],
            "max_results": data["max_results"],
            "include_domains": list(data["include_domains"]),
            "citation_verifier": data["citation_verifier"],
            "country": "us",
            "language": "en",
        }


CASES = (
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

    return {
        "name": case.name,
        "mode": response.get("mode"),
        "answer_mode": response.get("answer_mode"),
        "reasoning_effort": (response.get("query_plan") or {}).get("reasoning_effort"),
        "expected_source_recall": _ratio(len(matched_expected), len(case.expected_url_parts)),
        "used_source_recall": _ratio(len(matched_used), len(case.expected_url_parts)),
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
        "elapsed_ms": meta.get("elapsed_ms") or wall_ms,
        "wall_ms": wall_ms,
        "matched_expected": matched_expected,
        "matched_used": matched_used,
        "term_hits": term_hits,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(results),
        "expected_source_recall": _mean(results, "expected_source_recall"),
        "used_source_recall": _mean(results, "used_source_recall"),
        "answer_term_coverage": _mean(results, "answer_term_coverage"),
        "citation_coverage": _mean(results, "citation_coverage"),
        "supported_claim_rate": _mean(results, "supported_claim_rate"),
        "weak_claim_rate": _mean(results, "weak_claim_rate"),
        "review_claim_rate": _mean(results, "review_claim_rate"),
        "crag_sufficient_rate": _ratio(sum(1 for item in results if item.get("crag_status") == "sufficient"), len(results)),
        "fallback_rate": _ratio(sum(1 for item in results if item.get("answer_mode") == "extractive"), len(results)),
        "avg_elapsed_ms": round(statistics.mean(item["elapsed_ms"] for item in results)) if results else 0,
        "p95_elapsed_ms": _percentile([item["elapsed_ms"] for item in results], 0.95),
    }


def run_benchmark(api_base: str, timeout: float) -> dict[str, Any]:
    results = []
    for case in CASES:
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(args.api_base, args.timeout)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
