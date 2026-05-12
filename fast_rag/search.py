from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from .cache import PageCache
from .config import settings
from .extract import clean_text, extract_html
from .models import Document, SearchResult


YEAR_RE = re.compile(r"\b(20\d{2})\b")
LATEST_HINT_RE = re.compile(r"latest|today|current|now|recent|new|最新|今天|当前|最近|今年", re.IGNORECASE)
OPENAI_HINT_RE = re.compile(r"\bopenai\b|\bchatgpt\b", re.IGNORECASE)
RECENCY_TO_DDG = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}
RECENCY_TO_BRAVE = {
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
}
VALID_RECENCY = {"any", *RECENCY_TO_DDG}


@dataclass(frozen=True)
class SearchFilters:
    include_domains: tuple[str, ...] = field(default_factory=tuple)
    exclude_domains: tuple[str, ...] = field(default_factory=tuple)
    recency: str = "any"
    country: str = ""
    language: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["include_domains"] = list(self.include_domains)
        data["exclude_domains"] = list(self.exclude_domains)
        return data


@dataclass(frozen=True)
class SearchProfile:
    search_timeout: float
    fetch_timeout: float
    query_limit: int
    result_limit: int
    page_limit: int


PROFILES = {
    "fast": SearchProfile(
        search_timeout=2.8,
        fetch_timeout=3.2,
        query_limit=3,
        result_limit=8,
        page_limit=7,
    ),
    "pro": SearchProfile(
        search_timeout=4.0,
        fetch_timeout=4.8,
        query_limit=4,
        result_limit=12,
        page_limit=10,
    ),
    "deep": SearchProfile(
        search_timeout=5.5,
        fetch_timeout=6.5,
        query_limit=5,
        result_limit=16,
        page_limit=14,
    ),
}


def normalize_search_filters(
    include_domains: list[str] | tuple[str, ...] | None = None,
    exclude_domains: list[str] | tuple[str, ...] | None = None,
    recency: str | None = None,
    country: str | None = None,
    language: str | None = None,
) -> SearchFilters:
    include = _dedupe_domains(include_domains or ())[:20]
    exclude = _dedupe_domains(exclude_domains or ())[:20]
    include = tuple(domain for domain in include if domain not in exclude)
    clean_recency = (recency or "any").lower().strip()
    if clean_recency not in VALID_RECENCY:
        clean_recency = "any"
    clean_country = re.sub(r"[^a-z]", "", (country or "").lower())[:2]
    clean_language = re.sub(r"[^a-z]", "", (language or "").lower())[:2]
    return SearchFilters(
        include_domains=include,
        exclude_domains=tuple(exclude),
        recency=clean_recency,
        country=clean_country,
        language=clean_language,
    )


def rewrite_queries(query: str, mode: str, filters: SearchFilters | None = None) -> list[str]:
    query = clean_text(query)
    filters = filters or SearchFilters()
    queries = []
    if OPENAI_HINT_RE.search(query):
        queries.extend(
            [
                f"site:help.openai.com {query}",
                f"site:openai.com {query}",
            ]
        )
        if mode in {"pro", "deep"}:
            queries.append(f"site:developers.openai.com {query}")
    for domain in filters.include_domains[:4]:
        queries.append(f"site:{domain} {query}")
    queries.append(query)
    if LATEST_HINT_RE.search(query) and not YEAR_RE.search(query):
        queries.append(f"{query} 2026")
    if mode in {"pro", "deep"}:
        queries.append(f"{query} official source")
    if mode == "deep":
        queries.append(f"{query} explained")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        item = _apply_query_filters(item, filters)
        lowered = item.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            deduped.append(item)
    return deduped[: PROFILES.get(mode, PROFILES["fast"]).query_limit]


def _dedupe_domains(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    domains: list[str] = []
    seen: set[str] = set()
    for value in values:
        domain = _clean_domain(value)
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return tuple(domains)


def _clean_domain(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc or parsed.path
    host = host.split("/", 1)[0].split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", host):
        return ""
    return host


def _apply_query_filters(query: str, filters: SearchFilters) -> str:
    if not filters.exclude_domains:
        return query
    denylist = " ".join(f"-site:{domain}" for domain in filters.exclude_domains[:8])
    return f"{query} {denylist}".strip()


def seed_results(query: str) -> list[SearchResult]:
    lowered = query.lower()
    seeds: list[SearchResult] = []
    if "chatgpt" in lowered and "search" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="ChatGPT Search | OpenAI Help Center",
                    url="https://help.openai.com/en/articles/9237897-chatgpt-search",
                    snippet="OpenAI help article about how ChatGPT search works, sources, query rewriting, and location.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Introducing ChatGPT search | OpenAI",
                    url="https://openai.com/index/introducing-chatgpt-search/",
                    snippet="OpenAI product announcement describing ChatGPT search and the search model.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="ChatGPT search for Enterprise and Edu | OpenAI Help Center",
                    url="https://help.openai.com/en/articles/10093903-chatgpt-search-for-enterprise-and-edu",
                    snippet="OpenAI help article about ChatGPT search behavior, data sharing, and workspace controls.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "openai" in lowered and "web search" in lowered or "api" in lowered and "search" in lowered:
        seeds.append(
            SearchResult(
                title="Web search | OpenAI API",
                url="https://developers.openai.com/api/docs/guides/tools-web-search",
                snippet="OpenAI API documentation for web search tool calls, citations, sources, and domain filtering.",
                provider="official",
                rank=0,
            )
        )
    if "deepseek" in lowered and "api" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Your First API Call | DeepSeek API Docs",
                    url="https://api-docs.deepseek.com/",
                    snippet="DeepSeek official API quickstart with base URL and current model names.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Create Chat Completion | DeepSeek API Docs",
                    url="https://api-docs.deepseek.com/api/create-chat-completion",
                    snippet="DeepSeek official chat completions endpoint documentation.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    return seeds


def _decode_duck_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


class SearchProviders:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(
        self,
        query: str,
        limit: int,
        timeout: float,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        filters = filters or SearchFilters()
        tasks = []
        if settings.brave_api_key:
            tasks.append(self._brave(query, limit, timeout, filters))
        tasks.append(self._duckduckgo(query, limit, timeout, filters))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined: list[SearchResult] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            combined.extend(result)
        return combined[:limit]

    async def _brave(
        self,
        query: str,
        limit: int,
        timeout: float,
        filters: SearchFilters,
    ) -> list[SearchResult]:
        params = {"q": query, "count": min(limit, 20), "text_decorations": "false"}
        if filters.country:
            params["country"] = filters.country.upper()
        if filters.language:
            params["search_lang"] = filters.language
        if filters.recency in RECENCY_TO_BRAVE:
            params["freshness"] = RECENCY_TO_BRAVE[filters.recency]
        response = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={"X-Subscription-Token": settings.brave_api_key or ""},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for rank, item in enumerate(data.get("web", {}).get("results", []), start=1):
            results.append(
                SearchResult(
                    title=clean_text(item.get("title", "")),
                    url=item.get("url", ""),
                    snippet=clean_text(item.get("description", "")),
                    provider="brave",
                    rank=rank,
                )
            )
        return results

    async def _duckduckgo(
        self,
        query: str,
        limit: int,
        timeout: float,
        filters: SearchFilters,
    ) -> list[SearchResult]:
        params = {"q": query}
        if filters.recency in RECENCY_TO_DDG:
            params["df"] = RECENCY_TO_DDG[filters.recency]
        if filters.country and filters.language:
            params["kl"] = f"{filters.country}-{filters.language}"
        response = await self.client.get(
            "https://html.duckduckgo.com/html/",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for rank, block in enumerate(soup.select(".result"), start=1):
            link = block.select_one(".result__a")
            if not link:
                continue
            url = _decode_duck_url(link.get("href", ""))
            if not url.startswith(("http://", "https://")):
                continue
            snippet_el = block.select_one(".result__snippet")
            results.append(
                SearchResult(
                    title=clean_text(link.get_text(" ")),
                    url=url,
                    snippet=clean_text(snippet_el.get_text(" ") if snippet_el else ""),
                    provider="duckduckgo",
                    rank=rank,
                )
            )
            if len(results) >= limit:
                break
        return results


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        clean_url = result.url.split("#", 1)[0].rstrip("/")
        if not clean_url or clean_url in seen or is_noise_url(clean_url):
            continue
        seen.add(clean_url)
        result.url = clean_url
        deduped.append(result)
    return deduped


def filter_results(results: list[SearchResult], filters: SearchFilters | None = None) -> list[SearchResult]:
    if not filters:
        return results
    return [result for result in results if url_allowed(result.url, filters)]


def filter_documents(docs: list[Document], filters: SearchFilters | None = None) -> list[Document]:
    if not filters:
        return docs
    return [doc for doc in docs if url_allowed(doc.url, filters)]


def url_allowed(url: str, filters: SearchFilters) -> bool:
    domain = _domain_for_url(url)
    if not domain:
        return False
    if filters.include_domains and not any(_domain_matches(domain, allowed) for allowed in filters.include_domains):
        return False
    if filters.exclude_domains and any(_domain_matches(domain, blocked) for blocked in filters.exclude_domains):
        return False
    return True


def _domain_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def is_noise_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("duckduckgo.com") and parsed.path.endswith("/y.js"):
        return True
    if "ad_domain=" in parsed.query or "ad_provider=" in parsed.query:
        return True
    return False


def dedupe_documents(docs: list[Document], filters: SearchFilters | None = None) -> list[Document]:
    by_url: dict[str, Document] = {}
    for doc in docs:
        clean_url = doc.url.split("#", 1)[0].rstrip("/")
        if not clean_url or is_noise_url(clean_url):
            continue
        if filters and not url_allowed(clean_url, filters):
            continue
        doc.url = clean_url
        existing = by_url.get(clean_url)
        if not existing:
            by_url[clean_url] = doc
            continue
        if existing.provider != "official" and doc.provider == "official":
            by_url[clean_url] = doc
        elif len(doc.text) > len(existing.text) and existing.provider != "official":
            by_url[clean_url] = doc
    return list(by_url.values())


def should_fetch(url: str) -> bool:
    lower = url.lower()
    blocked_ext = (
        ".7z",
        ".avi",
        ".dmg",
        ".doc",
        ".docx",
        ".exe",
        ".gif",
        ".jpg",
        ".jpeg",
        ".mp3",
        ".mp4",
        ".png",
        ".ppt",
        ".pptx",
        ".rar",
        ".svg",
        ".webp",
        ".xls",
        ".xlsx",
        ".zip",
    )
    return not lower.endswith(blocked_ext)


async def fetch_document(
    client: httpx.AsyncClient,
    cache: PageCache,
    result: SearchResult,
    timeout: float,
) -> Document:
    if not should_fetch(result.url):
        return Document(
            url=result.url,
            title=result.title,
            text=result.snippet,
            snippet=result.snippet,
            provider=result.provider,
        )

    cached = cache.get(result.url)
    if cached:
        status, content_type, body = cached
        title, text = extract_html(body, result.title) if "html" in (content_type or "") else (result.title, body)
        return Document(
            url=result.url,
            title=title or result.title,
            text=text or result.snippet,
            snippet=result.snippet,
            provider=result.provider,
            status=status,
            fetched_from_cache=True,
        )

    try:
        response = await client.get(result.url, timeout=timeout, follow_redirects=True)
        content_type = response.headers.get("content-type", "").lower()
        body = response.text[:800_000]
        cache.set(str(response.url), response.status_code, content_type, body)
        title, text = extract_html(body, result.title) if "html" in content_type else (result.title, clean_text(body))
        return Document(
            url=str(response.url).split("#", 1)[0].rstrip("/"),
            title=title or result.title,
            text=text or result.snippet,
            snippet=result.snippet,
            provider=result.provider,
            status=response.status_code,
        )
    except Exception:
        return Document(
            url=result.url,
            title=result.title,
            text=result.snippet,
            snippet=result.snippet,
            provider=result.provider,
        )


async def retrieve_documents(
    query: str,
    mode: str,
    max_results: int,
    cache: PageCache,
    filters: SearchFilters | None = None,
    extra_queries: list[str] | None = None,
    page_limit_override: int | None = None,
) -> tuple[list[Document], dict]:
    profile = PROFILES.get(mode, PROFILES["fast"])
    filters = filters or SearchFilters()
    started = perf_counter()
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        provider = SearchProviders(client)
        queries = rewrite_queries(query, mode, filters)
        if extra_queries:
            for item in extra_queries:
                cleaned = clean_text(item)
                if cleaned:
                    queries.append(_apply_query_filters(cleaned, filters))
        queries = _dedupe_queries(queries)[: profile.query_limit + len(extra_queries or [])]
        search_tasks = [
            provider.search(item, limit=profile.result_limit, timeout=profile.search_timeout, filters=filters)
            for item in queries
        ]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)
        raw_results: list[SearchResult] = []
        for batch in search_batches:
            if isinstance(batch, Exception):
                continue
            raw_results.extend(batch)
        page_limit = min(profile.page_limit, page_limit_override or profile.page_limit)
        results = filter_results(dedupe_results(seed_results(query) + raw_results), filters)[
            : max(max_results, page_limit)
        ]

        semaphore_limit = 10 if mode == "deep" else 9 if mode == "pro" else 7
        semaphore = asyncio.Semaphore(semaphore_limit)

        async def guarded_fetch(result: SearchResult) -> Document:
            async with semaphore:
                return await fetch_document(client, cache, result, profile.fetch_timeout)

        docs = await asyncio.gather(
            *(guarded_fetch(result) for result in results[:page_limit]),
            return_exceptions=True,
        )

    documents = dedupe_documents(
        [doc for doc in docs if isinstance(doc, Document) and clean_text(doc.text)],
        filters,
    )
    meta = {
        "queries": queries,
        "filters": filters.to_dict(),
        "raw_results": len(raw_results),
        "deduped_results": len(results),
        "documents": len(documents),
        "elapsed_ms": round((perf_counter() - started) * 1000),
    }
    return documents, meta


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        lowered = query.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            deduped.append(query)
    return deduped
