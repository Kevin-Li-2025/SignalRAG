from fast_rag.models import Document, SearchResult
from fast_rag.search import (
    SearchFilters,
    _decode_bing_url,
    _decode_yahoo_url,
    _parse_bing_results,
    _parse_yahoo_results,
    dedupe_documents,
    dedupe_results,
    filter_results,
    fuse_results,
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


def test_rewrite_queries_applies_lens_queries() -> None:
    filters = SearchFilters(lens="academic")
    queries = rewrite_queries("corrective RAG evaluation", "deep", filters)
    assert any(query.startswith("site:arxiv.org") for query in queries)
    assert any("filetype:pdf" in query for query in queries)


def test_rewrite_queries_applies_domain_controls() -> None:
    filters = SearchFilters(include_domains=("docs.example.com",), exclude_domains=("medium.com",))
    queries = rewrite_queries("vector database recall", "pro", filters)
    assert any(query.startswith("site:docs.example.com") for query in queries)
    assert all("-site:medium.com" in query for query in queries)


def test_normalize_search_filters_cleans_domains_and_recency() -> None:
    filters = normalize_search_filters(
        ["https://www.OpenAI.com/docs", "bad value"],
        ["medium.com", ".edu"],
        "week",
        "US",
        "EN",
        "forums",
    )
    assert filters.include_domains == ("openai.com",)
    assert filters.exclude_domains == ("medium.com", ".edu")
    assert filters.recency == "week"
    assert filters.country == "us"
    assert filters.language == "en"
    assert filters.lens == "forums"


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


def test_filter_results_supports_tld_lenses() -> None:
    results = filter_results(
        [
            SearchResult(title="A", url="https://cs.stanford.edu/a"),
            SearchResult(title="B", url="https://example.com/b"),
        ],
        SearchFilters(include_domains=(".edu",)),
    )
    assert [item.url for item in results] == ["https://cs.stanford.edu/a"]


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


def test_seed_results_adds_deepseek_thinking_sources() -> None:
    seeds = seed_results("DeepSeek thinking reasoning effort")
    urls = {item.url for item in seeds}
    assert "https://api-docs.deepseek.com/guides/thinking_mode" in urls
    assert "https://api-docs.deepseek.com/quick_start/pricing" in urls


def test_seed_results_adds_context_compression_sources() -> None:
    seeds = seed_results("context window compression LongLLMLingua lost in the middle")
    urls = {item.url for item in seeds}
    assert "https://www.microsoft.com/en-us/research/project/llmlingua/longllmlingua/" in urls
    assert "https://arxiv.org/abs/2307.03172" in urls


def test_seed_results_adds_source_trust_sources() -> None:
    seeds = seed_results("source trust tiers government academic official documentation")
    urls = {item.url for item in seeds}
    assert "https://developers.google.com/search/docs/fundamentals/creating-helpful-content" in urls
    assert "https://www.cancer.gov/about-cancer/managing-care/using-trusted-resources" in urls


def test_decode_bing_redirect_url() -> None:
    url = _decode_bing_url(
        "https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9kb2NzLnB5dGhvbi5vcmcvMy9saWJyYXJ5L2pzb24uaHRtbA&ntb=1"
    )
    assert url == "https://docs.python.org/3/library/json.html"


def test_parse_bing_results_extracts_links_and_snippets() -> None:
    results = _parse_bing_results(
        """
        <html><body>
          <li class="b_algo">
            <h2><a href="https://example.com/a">Example A</a></h2>
            <div class="b_caption"><p>Snippet A</p></div>
          </li>
        </body></html>
        """,
        limit=5,
    )
    assert results == [
        SearchResult(title="Example A", url="https://example.com/a", snippet="Snippet A", provider="bing", rank=1)
    ]


def test_decode_yahoo_redirect_url() -> None:
    url = _decode_yahoo_url(
        "https://r.search.yahoo.com/_ylt=x/RV=2/RE=1/RO=10/RU=https%3a%2f%2fdocs.python.org%2f3%2flibrary%2fjson.html/RK=2/RS=x"
    )
    assert url == "https://docs.python.org/3/library/json.html"


def test_parse_yahoo_results_extracts_links_and_snippets() -> None:
    results = _parse_yahoo_results(
        """
        <html><body>
          <div data-yga='{"yModuleName":"Sr"}'>
            <div class="compTitle options-toggle">
              <a href="https://r.search.yahoo.com/x/RV=2/RU=https%3a%2f%2fexample.com%2fa/RK=2/RS=x">
                Example A
              </a>
            </div>
            <p>Snippet A</p>
          </div>
        </body></html>
        """,
        limit=5,
    )
    assert results == [
        SearchResult(title="Example A", url="https://example.com/a", snippet="Example A Snippet A", provider="yahoo", rank=1)
    ]


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


def test_fuse_results_rewards_urls_found_by_multiple_queries() -> None:
    results = fuse_results(
        [
            [
                SearchResult(title="Single", url="https://single.example.com", rank=1),
                SearchResult(title="Repeated A", url="https://repeat.example.com", rank=2),
            ],
            [
                SearchResult(title="Repeated B", url="https://repeat.example.com#section", rank=1),
            ],
        ]
    )
    assert [item.url for item in results[:2]] == [
        "https://repeat.example.com",
        "https://single.example.com",
    ]
    assert results[0].title == "Repeated A"


def test_fuse_results_keeps_seeded_official_sources_high() -> None:
    results = fuse_results(
        [[SearchResult(title="Web", url="https://web.example.com", rank=1)]],
        seeds=[SearchResult(title="Official", url="https://docs.example.com", provider="official", rank=0)],
    )
    assert results[0].url == "https://docs.example.com"
    assert results[0].provider == "official"


def test_dedupe_documents_prefers_official_redirect_result() -> None:
    docs = dedupe_documents(
        [
            Document(url="https://help.openai.com/page", title="A", text="short", provider="duckduckgo"),
            Document(url="https://help.openai.com/page", title="B", text="long official text", provider="official"),
        ]
    )
    assert len(docs) == 1
    assert docs[0].provider == "official"
