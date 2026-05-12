from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from time import perf_counter

from .cache import PageCache
from .config import settings
from .rank import rank_evidence
from .search import retrieve_documents


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_url_parts: tuple[str, ...]


CASES = (
    EvalCase(
        query="ChatGPT search how it works OpenAI",
        expected_url_parts=(
            "help.openai.com/en/articles/9237897",
            "openai.com/index/introducing-chatgpt-search",
        ),
    ),
    EvalCase(
        query="OpenAI web search API citations sources",
        expected_url_parts=("developers.openai.com/api/docs/guides/tools-web-search",),
    ),
    EvalCase(
        query="ChatGPT search Enterprise Edu data sharing Bing",
        expected_url_parts=("help.openai.com/en/articles/10093903",),
    ),
    EvalCase(
        query="DeepSeek API chat completions base URL current models",
        expected_url_parts=(
            "api-docs.deepseek.com",
            "api-docs.deepseek.com/api/create-chat-completion",
        ),
    ),
)


def _matches(url: str, expected_part: str) -> bool:
    return expected_part.rstrip("/") in url.rstrip("/")


async def evaluate_case(case: EvalCase, mode: str, top_k: int, max_results: int, cache: PageCache) -> dict:
    started = perf_counter()
    docs, meta = await retrieve_documents(case.query, mode, max_results, cache)
    evidence = rank_evidence(case.query, docs, limit=max(top_k, 10))
    top = evidence[:top_k]
    urls = [item.url for item in top]

    matched: list[str] = []
    ranks: list[int] = []
    for expected in case.expected_url_parts:
        for rank, url in enumerate(urls, start=1):
            if _matches(url, expected):
                matched.append(expected)
                ranks.append(rank)
                break

    recall = len(set(matched)) / len(case.expected_url_parts)
    reciprocal_rank = 1 / min(ranks) if ranks else 0.0
    elapsed_ms = round((perf_counter() - started) * 1000)
    return {
        "query": case.query,
        "expected": list(case.expected_url_parts),
        "matched": sorted(set(matched)),
        "recall_at_k": round(recall, 4),
        "hit": bool(matched),
        "mrr": round(reciprocal_rank, 4),
        "elapsed_ms": elapsed_ms,
        "retrieval": meta,
        "top_urls": urls,
    }


async def run_eval(mode: str, top_k: int, max_results: int) -> dict:
    cache = PageCache(settings.cache_path)
    results = []
    for case in CASES:
        results.append(await evaluate_case(case, mode, top_k, max_results, cache))

    return {
        "mode": mode,
        "top_k": top_k,
        "cases": len(results),
        "recall_at_k": round(sum(item["recall_at_k"] for item in results) / len(results), 4),
        "hit_rate": round(sum(1 for item in results if item["hit"]) / len(results), 4),
        "mrr": round(sum(item["mrr"] for item in results) / len(results), 4),
        "avg_elapsed_ms": round(sum(item["elapsed_ms"] for item in results) / len(results)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall for Fast Search RAG.")
    parser.add_argument("--mode", choices=["fast", "pro", "deep"], default="fast")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-results", type=int, default=12)
    args = parser.parse_args()
    summary = asyncio.run(run_eval(args.mode, args.top_k, args.max_results))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
