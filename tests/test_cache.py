from fast_rag.cache import SmartResponseCache, normalize_query, query_fingerprint_tokens


def _request(query: str, **overrides: object) -> dict:
    data = {
        "query": query,
        "mode": "pro",
        "lens": "official",
        "max_results": 10,
        "include_domains": ["help.openai.com", "openai.com"],
        "exclude_domains": [],
        "recency": "any",
        "country": "us",
        "language": "en",
        "citation_verifier": "auto",
    }
    data.update(overrides)
    return data


def test_normalize_query_ignores_case_spacing_and_punctuation() -> None:
    assert normalize_query("  DeepSeek API: base URL? ") == "deepseek api base url"


def test_query_fingerprint_tokens_stem_light_variants() -> None:
    assert query_fingerprint_tokens("chatgpt search cites sources") == {
        "chatgpt",
        "search",
        "cite",
        "source",
    }


def test_smart_response_cache_hits_canonical_equivalent_request(tmp_path) -> None:
    cache = SmartResponseCache(tmp_path / "cache.sqlite")
    response = {"answer": "cached", "meta": {"elapsed_ms": 123, "cache_hit": False}}
    cache.set(_request("DeepSeek API base URL?", include_domains=["openai.com", "help.openai.com"]), response)

    hit = cache.get(_request(" deepseek api base url ", include_domains=["help.openai.com", "openai.com"]))

    assert hit is not None
    assert hit.strategy == "exact"
    assert hit.response["answer"] == "cached"


def test_smart_response_cache_fuzzy_hits_safe_query_rewording(tmp_path) -> None:
    cache = SmartResponseCache(tmp_path / "cache.sqlite")
    response = {"answer": "cached", "meta": {"elapsed_ms": 123, "cache_hit": False}}
    cache.set(_request("How ChatGPT search cite sources"), response)

    hit = cache.get(_request("How ChatGPT search cites its sources"))

    assert hit is not None
    assert hit.strategy == "fuzzy"
    assert hit.score >= 0.86


def test_smart_response_cache_avoids_fuzzy_for_different_numbers(tmp_path) -> None:
    cache = SmartResponseCache(tmp_path / "cache.sqlite")
    cache.set(_request("OpenAI web search API limits 2025"), {"answer": "cached", "meta": {}})

    assert cache.get(_request("OpenAI web search API limits 2026")) is None


def test_smart_response_cache_avoids_fuzzy_for_fresh_queries(tmp_path) -> None:
    cache = SmartResponseCache(tmp_path / "cache.sqlite")
    cache.set(_request("latest OpenAI web search API citations"), {"answer": "cached", "meta": {}})

    assert cache.get(_request("current OpenAI web search API citations")) is None
