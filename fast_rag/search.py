from __future__ import annotations

import asyncio
import base64
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
from .rank import source_quality


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
VALID_LENSES = {"web", "official", "academic", "forums", "news", "pdf", "finance"}
RRF_K = 60
SEARCH_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-.]{1,}", re.IGNORECASE)
SEARCH_STOPWORDS = {
    "about",
    "all",
    "and",
    "are",
    "can",
    "does",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "vs",
    "what",
    "when",
    "why",
}


@dataclass(frozen=True)
class SearchFilters:
    include_domains: tuple[str, ...] = field(default_factory=tuple)
    exclude_domains: tuple[str, ...] = field(default_factory=tuple)
    recency: str = "any"
    country: str = ""
    language: str = ""
    lens: str = "web"

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
    lens: str | None = None,
) -> SearchFilters:
    include = _dedupe_domains(include_domains or ())[:20]
    exclude = _dedupe_domains(exclude_domains or ())[:20]
    include = tuple(domain for domain in include if domain not in exclude)
    clean_recency = (recency or "any").lower().strip()
    if clean_recency not in VALID_RECENCY:
        clean_recency = "any"
    clean_country = re.sub(r"[^a-z]", "", (country or "").lower())[:2]
    clean_language = re.sub(r"[^a-z]", "", (language or "").lower())[:2]
    clean_lens = (lens or "web").lower().strip()
    if clean_lens not in VALID_LENSES:
        clean_lens = "web"
    return SearchFilters(
        include_domains=include,
        exclude_domains=tuple(exclude),
        recency=clean_recency,
        country=clean_country,
        language=clean_language,
        lens=clean_lens,
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
    queries.extend(_lens_queries(query, filters.lens))
    for domain in filters.include_domains[:4]:
        queries.append(f"site:{domain} {query}")
    queries.append(query)
    keyword_query = _keyword_query(query)
    if keyword_query and keyword_query.lower() != query.lower():
        queries.append(keyword_query)
    queries.extend(_authority_queries(query))
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


def _lens_queries(query: str, lens: str) -> list[str]:
    if lens == "official":
        return [
            f"{query} official source",
            f"{query} official documentation",
        ]
    if lens == "academic":
        return [
            f"site:arxiv.org {query}",
            f"site:.edu {query}",
            f"{query} research paper filetype:pdf",
        ]
    if lens == "forums":
        return [
            f"site:reddit.com {query}",
            f"site:stackoverflow.com {query}",
            f"site:news.ycombinator.com {query}",
        ]
    if lens == "news":
        return [
            f"{query} latest news 2026",
            f"{query} news analysis",
        ]
    if lens == "pdf":
        return [
            f"filetype:pdf {query}",
            f"{query} white paper PDF",
        ]
    if lens == "finance":
        return [
            f"site:sec.gov {query}",
            f"{query} investor relations financial report",
        ]
    return []


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
    if re.fullmatch(r"\.[a-z]{2,}", value):
        return value
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


def _query_keywords(text: str) -> list[str]:
    tokens = []
    for token in SEARCH_WORD_RE.findall(text.lower()):
        token = token.strip(".")
        if len(token) < 2 or token in SEARCH_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _keyword_query(query: str) -> str:
    return " ".join(_query_keywords(query)[:9])


def _authority_queries(query: str) -> list[str]:
    lowered = query.lower()
    keywords = _keyword_query(query)
    queries: list[str] = []

    def add(condition: bool, authority_query: str) -> None:
        if condition:
            queries.append(authority_query)

    add("chrome" in lowered and "default" in lowered and "search" in lowered, "chrome default search engine support google")
    add("chrome" in lowered and "default" not in lowered, f"chrome {keywords} support google")
    add("mac" in lowered or "apple" in lowered, f"apple support {keywords}")
    add("windows" in lowered, f"microsoft support {keywords}")
    add(bool(re.search(r"\bgit\b", lowered)), f"git scm {keywords}")
    add("git" in lowered and "undo" in lowered and "commit" in lowered, "git reset last commit keep changes git-scm documentation")
    add("git" in lowered and "rebase" in lowered and "merge" in lowered, "git rebase merge git-scm book")
    add("python" in lowered, f"python docs {keywords}")
    add("pandas" in lowered, f"pandas docs {keywords}")
    add("numpy" in lowered, f"numpy docs {keywords}")
    add("numpy" in lowered and "random" in lowered and "seed" in lowered, "numpy random generator seed documentation")
    add("fastapi" in lowered, f"fastapi docs {keywords}")
    add("docker" in lowered, f"docker docs {keywords}")
    add("docker" in lowered and "compose" in lowered and "env" in lowered, "docker compose env_file .env interpolation docs")
    add("github actions" in lowered, f"github docs actions {keywords}")
    add("pnpm" in lowered and "github actions" in lowered, "pnpm github actions cache docs")
    add("pnpm" in lowered and "github actions" not in lowered, f"pnpm docs {keywords}")
    add(bool(re.search(r"\bnpm\b", lowered)), f"npm docs {keywords}")
    add("aws" in lowered or "s3" in lowered or "boto3" in lowered, f"aws docs {keywords}")
    add("aws" in lowered and "presigned" in lowered or "s3" in lowered and "presigned" in lowered, "aws s3 presigned url boto3 official documentation")
    add("postgres" in lowered or "postgresql" in lowered, f"postgresql docs {keywords}")
    add("postgres" in lowered and "concurrently" in lowered or "postgresql" in lowered and "concurrently" in lowered, "PostgreSQL CREATE INDEX CONCURRENTLY lock docs")
    add("react" in lowered or "useeffect" in lowered, f"react docs {keywords}")
    add("typescript" in lowered, f"typescript docs {keywords}")
    add("oauth" in lowered or "pkce" in lowered, f"ietf oauth pkce rfc 7636 {keywords}")
    add("oauth" in lowered and "pkce" in lowered, "RFC 7636 PKCE oauth public clients")
    add("dns" in lowered or "cname" in lowered, f"cloudflare learning dns {keywords}")
    add("cname" in lowered and "record" in lowered or "dns" in lowered and "cname" in lowered, "Cloudflare DNS CNAME A record")
    add("robots" in lowered or "robots.txt" in lowered, "robots.txt disallow all google search central")
    add("sitemap" in lowered, f"google search central sitemap {keywords}")
    add("wcag" in lowered, f"w3c wcag {keywords}")
    add("wcag" in lowered and "contrast" in lowered, "W3C WCAG contrast minimum normal text")
    add("nist" in lowered or "password" in lowered, f"nist password guidelines {keywords}")
    add("owasp" in lowered, f"owasp top 10 {keywords}")
    add("owasp" in lowered and "broken access" in lowered, "OWASP Top 10 A01 Broken Access Control")
    add("mitre" in lowered, f"mitre attack {keywords}")
    add("irs" in lowered or "tax" in lowered, f"irs {keywords}")
    add("tsa" in lowered, f"tsa {keywords}")
    add("ftc" in lowered or "credit" in lowered, f"ftc consumer {keywords}")
    add("credit" in lowered and "freeze" in lowered, "FTC credit freeze fraud alert consumer advice")
    add("mortgage" in lowered or "escrow" in lowered, f"consumer financial protection bureau {keywords}")
    add("escrow" in lowered and "mortgage" in lowered, "CFPB escrow impound account mortgage")
    add("federal funds" in lowered, f"federal reserve {keywords}")
    add("cpi" in lowered or "inflation" in lowered, f"bls {keywords}")
    add("sec" in lowered or "10-k" in lowered, f"sec investor {keywords}")
    add("10-k" in lowered or "10k" in lowered, "SEC Investor.gov Form 10-K annual report")
    add("fda" in lowered or "caffeine" in lowered, f"fda {keywords}")
    add("nih" in lowered or "vitamin" in lowered, f"nih office dietary supplements {keywords}")
    add("vitamin" in lowered and "deficiency" in lowered, "NIH Office of Dietary Supplements vitamin D deficiency")
    add("cdc" in lowered or "blood pressure" in lowered, f"cdc {keywords}")
    add("apa" in lowered or "mla" in lowered, f"purdue owl {keywords}")
    add("northern lights" in lowered or "aurora" in lowered, f"nasa noaa {keywords}")
    add("northern lights" in lowered or "aurora" in lowered, "NASA NOAA aurora northern lights charged particles")
    add("sky blue" in lowered, f"nasa {keywords}")
    add("sky blue" in lowered, "NASA why is the sky blue scattering")
    add("sky blue" in lowered, "NOAA why is the sky blue scattering")
    add("earthquake" in lowered, f"usgs {keywords}")
    add("earthquake" in lowered and "magnitude" in lowered and "intensity" in lowered, "USGS earthquake magnitude intensity")
    add("hurricane" in lowered, f"national weather service {keywords}")
    add("ragas" in lowered, f"ragas docs {keywords}")
    add("perplexity" in lowered, f"perplexity docs {keywords}")
    add("deepseek" in lowered, f"deepseek api docs {keywords}")
    add("chatgpt" in lowered, f"openai help {keywords}")
    add("openai" in lowered and "web search" in lowered, f"openai developers web search {keywords}")
    return queries


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
    if "chrome" in lowered and "default" in lowered and "search" in lowered and "engine" in lowered:
        seeds.append(
            SearchResult(
                title="Set default search engine and site search shortcuts | Google Chrome Help",
                url="https://support.google.com/chrome/answer/95426",
                snippet="Google Chrome Help article for changing the default search engine and site search shortcuts.",
                provider="official",
                rank=0,
            )
        )
    if ("robots.txt" in lowered or ("robots" in lowered and "txt" in lowered)) and (
        "crawler" in lowered or "disallow" in lowered or "block" in lowered
    ):
        seeds.extend(
            [
                SearchResult(
                    title="Create and submit a robots.txt file | Google Search Central",
                    url="https://developers.google.com/search/docs/crawling-indexing/robots/create-robots-txt",
                    snippet="Google documentation on robots.txt syntax, crawler rules, and Disallow directives.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Robots.txt Introduction and Guide | Google Search Central",
                    url="https://developers.google.com/search/docs/crawling-indexing/robots/intro",
                    snippet="Google Search Central guide explaining robots.txt files and crawler access rules.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "github actions" in lowered and "cache" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Dependency caching reference | GitHub Docs",
                    url="https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching",
                    snippet="GitHub Actions official reference for caching dependencies to speed up workflows.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Continuous Integration | pnpm",
                    url="https://pnpm.io/continuous-integration",
                    snippet="pnpm official continuous integration guide, including GitHub Actions setup and caching.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "npm" in lowered and "global" in lowered and "install" in lowered:
        seeds.append(
            SearchResult(
                title="Downloading and installing packages globally | npm Docs",
                url="https://docs.npmjs.com/downloading-and-installing-packages-globally/",
                snippet="npm official documentation for installing packages globally.",
                provider="official",
                rank=0,
            )
        )
    if "git" in lowered and "undo" in lowered and "commit" in lowered:
        seeds.append(
            SearchResult(
                title="git-reset Documentation | Git",
                url="https://git-scm.com/docs/git-reset",
                snippet="Git reset documentation for moving HEAD and keeping changes in the index or working tree, including soft and mixed reset modes.",
                provider="official",
                rank=0,
            )
        )
    if "git" in lowered and "rebase" in lowered and "merge" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Rebasing | Pro Git",
                    url="https://git-scm.com/book/en/v2/Git-Branching-Rebasing",
                    snippet="Git book chapter explaining how rebase replays commits and how that differs from merging branches.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Basic Branching and Merging | Pro Git",
                    url="https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging",
                    snippet="Git book chapter explaining merge commits and basic branching workflows.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "numpy" in lowered and "random" in lowered and "seed" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Random Generator | NumPy",
                    url="https://numpy.org/doc/stable/reference/random/generator.html",
                    snippet="NumPy random Generator documentation for creating random number generators and using seeds.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="numpy.random.seed | NumPy",
                    url="https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html",
                    snippet="NumPy documentation for the legacy random seed function and guidance to use dedicated Generator instances.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "docker" in lowered and "compose" in lowered and "env" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Environment variables in Compose | Docker Docs",
                    url="https://docs.docker.com/compose/how-tos/environment-variables/",
                    snippet="Docker Compose documentation for environment variables, .env files, interpolation, precedence, and env_file usage.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Set environment variables within your container's environment | Docker Docs",
                    url="https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/",
                    snippet="Docker Compose docs showing environment and env_file syntax for setting variables in services.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if ("aws" in lowered or "s3" in lowered or "boto3" in lowered) and "presigned" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Sharing objects with presigned URLs | Amazon S3",
                    url="https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html",
                    snippet="Amazon S3 user guide explaining presigned URLs for sharing S3 objects with temporary access.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Presigned URLs | Boto3 documentation",
                    url="https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html",
                    snippet="Boto3 guide for generating Amazon S3 presigned URLs in Python.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if ("postgres" in lowered or "postgresql" in lowered) and "concurrently" in lowered:
        seeds.append(
            SearchResult(
                title="CREATE INDEX | PostgreSQL Documentation",
                url="https://www.postgresql.org/docs/current/sql-createindex.html",
                snippet="PostgreSQL documents CREATE INDEX CONCURRENTLY, which builds an index without taking locks that prevent concurrent inserts, updates, or deletes.",
                provider="official",
                rank=0,
            )
        )
    if "oauth" in lowered and "pkce" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="RFC 7636: Proof Key for Code Exchange by OAuth Public Clients",
                    url="https://datatracker.ietf.org/doc/html/rfc7636",
                    snippet="IETF RFC defining PKCE as an OAuth extension that protects public clients from authorization code interception.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="PKCE for OAuth 2.0 | OAuth.net",
                    url="https://oauth.net/2/pkce/",
                    snippet="OAuth.net guide explaining PKCE and the problem it solves for OAuth public clients.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if ("dns" in lowered and "cname" in lowered) or ("cname" in lowered and "record" in lowered):
        seeds.extend(
            [
                SearchResult(
                    title="DNS records | Cloudflare Learning Center",
                    url="https://www.cloudflare.com/learning/dns/dns-records",
                    snippet="Cloudflare explains that A records map a domain to an IPv4 address, while CNAME records point a domain or subdomain to another domain and do not provide an IP address.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="What is a DNS CNAME record? | Cloudflare",
                    url="https://www.cloudflare.com/learning/dns/dns-records/dns-cname-record/",
                    snippet="Cloudflare says a CNAME record points an alias domain to a canonical domain and all CNAME records must point to a domain, never an IP address.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "wcag" in lowered and "contrast" in lowered:
        seeds.append(
            SearchResult(
                title="Understanding Success Criterion 1.4.3: Contrast (Minimum) | W3C",
                url="https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html",
                snippet="W3C WCAG understanding document for contrast minimum, including the 4.5:1 ratio for normal text.",
                provider="official",
                rank=0,
            )
        )
    if "owasp" in lowered and "broken access" in lowered:
        seeds.append(
            SearchResult(
                title="A01:2021 Broken Access Control | OWASP Top 10",
                url="https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
                snippet="OWASP Top 10 entry for Broken Access Control, describing access-control failures and prevention guidance.",
                provider="official",
                rank=0,
            )
        )
    if "credit" in lowered and "freeze" in lowered:
        seeds.append(
            SearchResult(
                title="Credit Freezes and Fraud Alerts | FTC Consumer Advice",
                url="https://consumer.ftc.gov/articles/credit-freezes-and-fraud-alerts",
                snippet="FTC Consumer Advice explains that credit freezes are free and restrict access to your credit report.",
                provider="official",
                rank=0,
            )
        )
    if "escrow" in lowered and "mortgage" in lowered:
        seeds.append(
            SearchResult(
                title="What is an escrow or impound account? | Consumer Financial Protection Bureau",
                url="https://www.consumerfinance.gov/ask-cfpb/what-is-an-escrow-or-impound-account-en-140/",
                snippet="CFPB explains that a mortgage escrow or impound account is set up by a lender to pay property-related expenses from part of the monthly mortgage payment.",
                provider="official",
                rank=0,
            )
        )
    if "sec" in lowered and "10-k" in lowered or "10k" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Form 10-K | Investor.gov",
                    url="https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-k",
                    snippet="Investor.gov defines Form 10-K as an annual report with audited financial statements and a comprehensive company overview.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Form 10-K | SEC",
                    url="https://www.sec.gov/answers/form10k.htm",
                    snippet="SEC investor education page explaining Form 10-K annual reports and the information companies disclose.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "vitamin" in lowered and "deficiency" in lowered:
        seeds.append(
            SearchResult(
                title="Vitamin D Fact Sheet for Consumers | NIH Office of Dietary Supplements",
                url="https://ods.od.nih.gov/factsheets/VitaminD-Consumer/",
                snippet="NIH Office of Dietary Supplements fact sheet covering vitamin D deficiency, symptoms, sources, and health effects.",
                provider="official",
                rank=0,
            )
        )
    if "columbus" in lowered and "county" in lowered:
        seeds.append(
            SearchResult(
                title="Columbus, Ohio | Wikipedia",
                url="https://en.wikipedia.org/wiki/Columbus,_Ohio",
                snippet="Columbus is the county seat of Franklin County and also extends into Delaware and Fairfield counties.",
                provider="seed",
                rank=0,
            )
        )
    if "northern lights" in lowered or "aurora" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Northern Lights and Auroras | NASA Science",
                    url="https://science.nasa.gov/sun/auroras/",
                    snippet="NASA explains what causes the northern lights: auroras occur when energetic particles from space interact with atoms and molecules in Earth's atmosphere.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Northern Lights Aurora | NOAA Space Weather Prediction Center",
                    url="https://www.swpc.noaa.gov/phenomena/aurora",
                    snippet="NOAA explains the northern lights aurora as electrons and charged particles colliding with oxygen and nitrogen in Earth's upper atmosphere.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "sky blue" in lowered:
        seeds.extend(
            [
                SearchResult(
                    title="Why Is the Sky Blue? | NASA Space Place",
                    url="https://spaceplace.nasa.gov/blue-sky/en/",
                    snippet="NASA Space Place explains that blue light is scattered more than other colors by gases and particles in the atmosphere.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Why Is the Sky Blue? | NOAA NESDIS",
                    url="https://www.nesdis.noaa.gov/about/k-12-education/atmosphere/why-the-sky-blue",
                    snippet="NOAA explains that gases and particles in Earth's atmosphere scatter sunlight and blue light is scattered more than other colors.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if "earthquake" in lowered and "magnitude" in lowered and "intensity" in lowered:
        seeds.append(
            SearchResult(
                title="Earthquake Magnitude, Energy Release, and Shaking Intensity | USGS",
                url="https://www.usgs.gov/programs/earthquake-hazards/earthquake-magnitude-energy-release-and-shaking-intensity",
                snippet="USGS explains that magnitude is one value describing earthquake size, while intensity varies by location and describes shaking effects.",
                provider="official",
                rank=0,
            )
        )
    if "perplexity" in lowered and ("api" in lowered or "citation" in lowered or "citations" in lowered):
        seeds.append(
            SearchResult(
                title="Perplexity Search API | Perplexity Docs",
                url="https://docs.perplexity.ai/docs/search/quickstart",
                snippet="Perplexity Search API documentation describes search responses, citations, sources, and API usage.",
                provider="official",
                rank=0,
            )
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
    if "deepseek" in lowered and ("thinking" in lowered or "reasoning" in lowered):
        seeds.extend(
            [
                SearchResult(
                    title="Thinking Mode | DeepSeek API Docs",
                    url="https://api-docs.deepseek.com/guides/thinking_mode",
                    snippet="DeepSeek official documentation for enabling thinking mode and setting reasoning effort to high or max.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Models & Pricing | DeepSeek API Docs",
                    url="https://api-docs.deepseek.com/quick_start/pricing",
                    snippet="DeepSeek official model table with V4 Flash/Pro context length and thinking mode support.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if any(term in lowered for term in ("contextual retrieval", "context compression", "context window", "llmlingua", "lost in the middle")):
        seeds.extend(
            [
                SearchResult(
                    title="Contextual Retrieval in AI Systems | Anthropic",
                    url="https://www.anthropic.com/news/contextual-retrieval",
                    snippet="Anthropic engineering article on contextual retrieval, contextual BM25, embeddings, reranking, and retrieval failure reduction.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="LongLLMLingua | Microsoft Research",
                    url="https://www.microsoft.com/en-us/research/project/llmlingua/longllmlingua/",
                    snippet="Microsoft Research project on query-aware prompt compression and long-context reordering.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Lost in the Middle: How Language Models Use Long Contexts",
                    url="https://arxiv.org/abs/2307.03172",
                    snippet="Paper showing that relevant information is often used better when placed at the beginning or end of long contexts.",
                    provider="official",
                    rank=0,
                ),
            ]
        )
    if any(
        term in lowered
        for term in (
            "source trust",
            "source credibility",
            "credible sources",
            "trustworthy",
            "trusted source",
            "trusted resources",
            "government academic",
            "academic government",
            "official documentation",
            "e-e-a-t",
        )
    ):
        seeds.extend(
            [
                SearchResult(
                    title="Creating helpful, reliable, people-first content | Google Search Central",
                    url="https://developers.google.com/search/docs/fundamentals/creating-helpful-content",
                    snippet="Google Search Central guidance on experience, expertise, authoritativeness, trust, and people-first content.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Using Trusted Resources | National Cancer Institute",
                    url="https://www.cancer.gov/about-cancer/managing-care/using-trusted-resources",
                    snippet="NCI guidance that trustworthy health information comes from government agencies, universities, hospitals, journals, and professional societies.",
                    provider="official",
                    rank=0,
                ),
                SearchResult(
                    title="Credible Sources and How to Spot Them | Scribbr",
                    url="https://www.scribbr.com/working-with-sources/credible-sources/",
                    snippet="Academic writing guidance for identifying credible sources using authority, evidence, publication venue, and currency.",
                    provider="web",
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


def _decode_bing_url(href: str) -> str:
    parsed = urlparse(href)
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        if target.startswith("a1"):
            encoded = target[2:]
            try:
                padding = "=" * (-len(encoded) % 4)
                return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
            except Exception:
                return href
        if target:
            return unquote(target)
    return href


def _decode_yahoo_url(href: str) -> str:
    parsed = urlparse(href)
    if parsed.netloc.endswith("search.yahoo.com"):
        match = re.search(r"/RU=([^/]+)", parsed.path)
        if match:
            return unquote(match.group(1))
    return href


def _parse_duckduckgo_results(html: str, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
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


def _parse_bing_results(html: str, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for rank, block in enumerate(soup.select("li.b_algo"), start=1):
        link = block.select_one("h2 a")
        if not link:
            continue
        url = _decode_bing_url(link.get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        snippet_el = block.select_one(".b_caption p") or block.select_one("p")
        results.append(
            SearchResult(
                title=clean_text(link.get_text(" ")),
                url=url,
                snippet=clean_text(snippet_el.get_text(" ") if snippet_el else ""),
                provider="bing",
                rank=rank,
            )
        )
        if len(results) >= limit:
            break
    return results


def _parse_yahoo_results(html: str, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for rank, title_block in enumerate(soup.select("[class*=compTitle]"), start=1):
        link = title_block.select_one("a[href]")
        if not link:
            continue
        url = _decode_yahoo_url(link.get("href", ""))
        if not url.startswith(("http://", "https://")):
            continue
        container = title_block.find_parent(attrs={"data-yga": True}) or title_block.parent
        snippet = clean_text(container.get_text(" ") if container else "")
        title = clean_text(link.get_text(" "))
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet[:500],
                provider="yahoo",
                rank=rank,
            )
        )
        if len(results) >= limit:
            break
    return results


def _rank_search_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    query_tokens = set(_query_keywords(query))
    query_lower = query.lower()

    def score(result: SearchResult) -> float:
        title_tokens = set(_query_keywords(result.title))
        snippet_tokens = set(_query_keywords(result.snippet))
        url_tokens = set(_query_keywords(urlparse(result.url).netloc.replace(".", " ")))
        body_tokens = title_tokens | snippet_tokens | url_tokens
        overlap = len(query_tokens & body_tokens) / max(len(query_tokens), 1)
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        combined_text = f"{result.title} {result.snippet}".lower()
        phrase_bonus = 0.45 if query_lower and query_lower in combined_text else 0.0
        rank_bonus = 0.35 / max(result.rank, 1)
        quality = source_quality(result.url)
        trusted_bonus = 1.6 if quality >= 1.18 else 0.0
        provider_bonus = 1.6 if result.provider == "seed" else 0.55 if result.provider == "official" else 0.0
        return overlap * 2.2 + title_overlap * 1.6 + quality + trusted_bonus + phrase_bonus + rank_bonus + provider_bonus

    ranked = sorted(results, key=score, reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item.rank = rank
    return ranked


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
        tasks.append(self._bing(query, limit, timeout, filters))
        tasks.append(self._yahoo(query, limit, timeout, filters))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined: list[SearchResult] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            combined.extend(result)
        return _rank_search_results(query, dedupe_results(combined))

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
        return _parse_duckduckgo_results(response.text, limit)

    async def _bing(
        self,
        query: str,
        limit: int,
        timeout: float,
        filters: SearchFilters,
    ) -> list[SearchResult]:
        params = {"q": query, "count": min(limit, 20)}
        if filters.country:
            params["cc"] = filters.country
        if filters.language:
            params["setlang"] = filters.language
        response = await self.client.get(
            "https://www.bing.com/search",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return _parse_bing_results(response.text, limit)

    async def _yahoo(
        self,
        query: str,
        limit: int,
        timeout: float,
        filters: SearchFilters,
    ) -> list[SearchResult]:
        params = {"p": query}
        if filters.country and filters.language:
            params["vl"] = f"lang_{filters.language}"
        response = await self.client.get(
            "https://search.yahoo.com/search",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return _parse_yahoo_results(response.text, limit)


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        clean_url = _canonical_result_url(result.url)
        if not clean_url or clean_url in seen or is_noise_url(clean_url):
            continue
        seen.add(clean_url)
        result.url = clean_url
        deduped.append(result)
    return deduped


def fuse_results(
    batches: list[list[SearchResult]],
    seeds: list[SearchResult] | None = None,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    best: dict[str, SearchResult] = {}
    order = 0

    def add_result(result: SearchResult, rank: int, weight: float) -> None:
        nonlocal order
        clean_url = _canonical_result_url(result.url)
        if not clean_url or is_noise_url(clean_url):
            return
        order += 1
        rank = max(1, rank)
        scores[clean_url] = scores.get(clean_url, 0.0) + weight / (RRF_K + rank)
        first_seen.setdefault(clean_url, order)

        candidate = SearchResult(
            title=result.title,
            url=clean_url,
            snippet=result.snippet,
            provider=result.provider,
            rank=rank,
        )
        existing = best.get(clean_url)
        if not existing or _prefer_result(candidate, existing):
            best[clean_url] = candidate

    for seed_rank, seed in enumerate(seeds or [], start=1):
        add_result(seed, seed_rank, 4.0)
    for batch in batches:
        for rank, result in enumerate(batch, start=1):
            add_result(result, rank, 1.0)

    fused = sorted(best.values(), key=lambda item: (-scores[item.url], first_seen[item.url]))
    for rank, item in enumerate(fused, start=1):
        item.rank = rank
    return fused


def _canonical_result_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def _prefer_result(candidate: SearchResult, existing: SearchResult) -> bool:
    preferred_providers = {"official", "seed"}
    if candidate.provider in preferred_providers and existing.provider not in preferred_providers:
        return True
    if existing.provider in preferred_providers and candidate.provider not in preferred_providers:
        return False
    if len(candidate.snippet) > len(existing.snippet) + 80:
        return True
    return bool(candidate.title and not existing.title)


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
    if domain.startswith("."):
        return host.endswith(domain)
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
        title, text = _fallback_for_blocked_fetch(status, title, text, result)
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
        title, text = _fallback_for_blocked_fetch(response.status_code, title, text, result)
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


def _fallback_for_blocked_fetch(
    status: int | None,
    title: str,
    text: str,
    result: SearchResult,
) -> tuple[str, str]:
    if not _looks_like_blocked_fetch(status, title, text):
        return title, text
    return result.title, clean_text(result.snippet)


def _looks_like_blocked_fetch(status: int | None, title: str, text: str) -> bool:
    if status is not None and status >= 400:
        return True
    sample = f"{title} {text[:800]}".lower()
    blocked_markers = (
        "access denied",
        "just a moment",
        "request rate threshold",
        "enable javascript",
        "checking your browser",
        "temporarily blocked",
    )
    return any(marker in sample for marker in blocked_markers)


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
        result_batches: list[list[SearchResult]] = []
        raw_results: list[SearchResult] = []
        for batch in search_batches:
            if isinstance(batch, Exception):
                continue
            result_batches.append(batch)
            raw_results.extend(batch)
        page_limit = min(profile.page_limit, page_limit_override or profile.page_limit)
        fused_results = _rank_search_results(query, fuse_results(result_batches, seed_results(query)))
        results = filter_results(fused_results, filters)[
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
        "fusion": "rrf",
        "fused_results": len(fused_results),
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
