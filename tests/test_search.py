from fast_rag.models import Document, SearchResult
from fast_rag.search import (
    SearchFilters,
    dedupe_documents,
    dedupe_results,
    filter_results,
    normalize_search_filters,
    rewrite_queries,
    seed_results,
)


def test_rewrite_queries_prioritizes_openai_official_sources() -> None:
    queries = rewrite_queries("ChatGPT search how it works OpenAI", "fast")
    assert queries[0].startswith("site:help.openai.com")
    assert any("site:help.openai.com" in query for query in queries)
    assert queries[-1] == "ChatGPT search how it works OpenAI"


def test_rewrite_queries_pro_adds_developer_and_official_queries() -> None:
    queries = rewrite_queries("OpenAI web search API citations", "pro")
    assert any(query.startswith("site:developers.openai.com") for query in queries)


def test_rewrite_queries_pro_adds_official_query_for_general_topics() -> None:
    queries = rewrite_queries("RAG recall evaluation citations", "pro")
    assert any(query.endswith("official source") for query in queries)


def test_rewrite_queries_applies_domain_controls() -> None:
    filters = SearchFilters(include_domains=("docs.example.com",), exclude_domains=("medium.com",))
    queries = rewrite_queries("vector database recall", "pro", filters)
    assert any(query.startswith("site:docs.example.com") for query in queries)
    assert all("-site:medium.com" in query for query in queries)


def test_normalize_search_filters_cleans_domains_and_recency() -> None:
    filters = normalize_search_filters(
        ["https://www.OpenAI.com/docs", "bad value"],
        ["medium.com"],
        "week",
        "US",
        "EN",
    )
    assert filters.include_domains == ("openai.com",)
    assert filters.exclude_domains == ("medium.com",)
    assert filters.recency == "week"
    assert filters.country == "us"
    assert filters.language == "en"


def test_filter_results_respects_include_and_exclude_domains() -> None:
    results = filter_results(
        [
            SearchResult(title="A", url="https://docs.example.com/a"),
            SearchResult(title="B", url="https://blog.example.com/b"),
            SearchResult(title="C", url="https://medium.com/c"),
        ],
        SearchFilters(include_domains=("example.com",), exclude_domains=("blog.example.com",)),
    )
    assert [item.url for item in results] == ["https://docs.example.com/a"]


def test_seed_results_adds_chatgpt_search_official_sources() -> None:
    seeds = seed_results("ChatGPT search how it works")
    urls = {item.url for item in seeds}
    assert "https://help.openai.com/en/articles/9237897-chatgpt-search" in urls
    assert "https://openai.com/index/introducing-chatgpt-search/" in urls


def test_seed_results_adds_deepseek_official_sources() -> None:
    seeds = seed_results("DeepSeek API chat completions base URL")
    urls = {item.url for item in seeds}
    assert "https://api-docs.deepseek.com/" in urls
    assert "https://api-docs.deepseek.com/api/create-chat-completion" in urls


def test_dedupe_results_drops_duckduckgo_ads() -> None:
    results = dedupe_results(
        [
            SearchResult(
                title="ad",
                url="https://duckduckgo.com/y.js?ad_domain=example.com",
            ),
            SearchResult(title="real", url="https://example.com/page"),
        ]
    )
    assert [item.url for item in results] == ["https://example.com/page"]


def test_dedupe_documents_prefers_official_redirect_result() -> None:
    docs = dedupe_documents(
        [
            Document(url="https://help.openai.com/page", title="A", text="short", provider="duckduckgo"),
            Document(url="https://help.openai.com/page", title="B", text="long official text", provider="official"),
        ]
    )
    assert len(docs) == 1
    assert docs[0].provider == "official"
