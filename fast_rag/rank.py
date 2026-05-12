from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

from .extract import split_passages
from .models import Document, Evidence


WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "is",
    "it",
    "its",
    "of",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "why",
    "work",
    "works",
}

TRUSTED_SUFFIXES = (
    ".gov",
    ".mil",
    ".edu",
    ".ac.uk",
    ".edu.au",
    ".ac.jp",
    ".gov.uk",
    ".gc.ca",
)
GOVERNMENT_AND_INSTITUTIONAL_DOMAINS = {
    "cancer.gov",
    "cdc.gov",
    "data.gov",
    "ec.europa.eu",
    "ed.gov",
    "ema.europa.eu",
    "epa.gov",
    "europa.eu",
    "fda.gov",
    "federalreserve.gov",
    "ftc.gov",
    "imf.org",
    "irs.gov",
    "nasa.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "nist.gov",
    "noaa.gov",
    "oecd.org",
    "sec.gov",
    "treasury.gov",
    "un.org",
    "usda.gov",
    "who.int",
    "worldbank.org",
}
ACADEMIC_AND_RESEARCH_DOMAINS = {
    "aclanthology.org",
    "arxiv.org",
    "bmj.com",
    "cell.com",
    "jamanetwork.com",
    "nejm.org",
    "nature.com",
    "pubmed.ncbi.nlm.nih.gov",
    "science.org",
    "sciencedirect.com",
    "springer.com",
}
STANDARDS_AND_SECURITY_DOMAINS = {
    "cisa.gov",
    "csrc.nist.gov",
    "ietf.org",
    "iso.org",
    "mitre.org",
    "owasp.org",
    "w3.org",
}
OFFICIAL_DOCS_DOMAINS = {
    "api-docs.deepseek.com",
    "boto3.amazonaws.com",
    "cloud.google.com",
    "cloudflare.com",
    "developer.mozilla.org",
    "developers.google.com",
    "developers.openai.com",
    "docs.docker.com",
    "docs.anthropic.com",
    "docs.aws.amazon.com",
    "docs.github.com",
    "docs.llamaindex.ai",
    "docs.microsoft.com",
    "docs.npmjs.com",
    "docs.perplexity.ai",
    "docs.python.org",
    "docs.ragas.io",
    "docs.tavily.com",
    "fastapi.tiangolo.com",
    "git-scm.com",
    "help.openai.com",
    "learn.microsoft.com",
    "nodejs.org",
    "numpy.org",
    "npmjs.com",
    "openai.com",
    "pandas.pydata.org",
    "platform.openai.com",
    "pnpm.io",
    "postgresql.org",
    "react.dev",
    "support.apple.com",
    "support.google.com",
    "support.microsoft.com",
    "typescriptlang.org",
}
MEDICAL_REFERENCE_DOMAINS = {
    "cancer.gov",
    "clevelandclinic.org",
    "mayoclinic.org",
    "medlineplus.gov",
    "msdmanuals.com",
}
NEWS_WIRE_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "reuters.com",
}
REFERENCE_DOMAINS = {
    "britannica.com",
    "wikipedia.org",
}
LOW_SIGNAL_DOMAINS = {
    "medium.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "substack.com",
}
TRUST_TIER_WEIGHTS = {
    "government": 1.24,
    "academic": 1.22,
    "standards": 1.21,
    "official_docs": 1.19,
    "medical": 1.18,
    "news_wire": 1.12,
    "reference": 1.05,
    "low_signal": 0.84,
    "general": 1.0,
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = [match.group(0) for match in WORD_RE.finditer(text) if match.group(0) not in STOPWORDS]
    cjk = CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend("".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1)))
    return tokens


def query_phrases(query: str) -> list[str]:
    words = [match.group(0).lower() for match in WORD_RE.finditer(query)]
    phrases = []
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[index : index + size]
            if all(word in STOPWORDS for word in phrase_words):
                continue
            phrases.append(" ".join(phrase_words))
    return phrases


def domain_for(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def source_quality(url: str) -> float:
    return TRUST_TIER_WEIGHTS[source_trust_tier(url)]


def source_trust_tier(url: str) -> str:
    domain = domain_for(url)
    if _domain_in(domain, LOW_SIGNAL_DOMAINS):
        return "low_signal"
    if any(domain.endswith(suffix) for suffix in TRUSTED_SUFFIXES) or _domain_in(domain, GOVERNMENT_AND_INSTITUTIONAL_DOMAINS):
        return "government"
    if _domain_in(domain, ACADEMIC_AND_RESEARCH_DOMAINS):
        return "academic"
    if _domain_in(domain, STANDARDS_AND_SECURITY_DOMAINS):
        return "standards"
    if _domain_in(domain, OFFICIAL_DOCS_DOMAINS):
        return "official_docs"
    if _domain_in(domain, MEDICAL_REFERENCE_DOMAINS):
        return "medical"
    if _domain_in(domain, NEWS_WIRE_DOMAINS):
        return "news_wire"
    if _domain_in(domain, REFERENCE_DOMAINS):
        return "reference"
    return "general"


def _domain_in(domain: str, candidates: set[str]) -> bool:
    return domain in candidates or any(domain.endswith("." + item) for item in candidates)


def contextual_passage_text(title: str, url: str, snippet: str, passage: str) -> str:
    parts = [
        f"Source title: {title}",
        f"Source domain: {domain_for(url)}",
    ]
    if snippet:
        parts.append(f"Source snippet: {snippet}")
    parts.append(passage)
    return ". ".join(part for part in parts if part)


def _bm25(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], avg_len: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    k1 = 1.35
    b = 0.72
    score = 0.0
    for token in set(query_tokens):
        freq = tf.get(token, 0)
        if not freq:
            continue
        denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1))
        score += idf.get(token, 0.0) * (freq * (k1 + 1)) / denom
    return score


def rank_evidence(query: str, docs: list[Document], limit: int = 8) -> list[Evidence]:
    query_tokens = tokenize(query)
    query_lower = query.lower().strip()
    phrases = query_phrases(query)
    candidates: list[tuple[Document, str, list[str]]] = []

    for doc in docs:
        for passage in split_passages(doc.text):
            passage_tokens = tokenize(contextual_passage_text(doc.title, doc.url, doc.snippet, passage))
            if passage_tokens:
                candidates.append((doc, passage, passage_tokens))

    if not candidates:
        return []

    doc_freq: Counter[str] = Counter()
    for _, _, tokens in candidates:
        doc_freq.update(set(tokens))
    total = len(candidates)
    idf = {
        token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    avg_len = sum(len(tokens) for _, _, tokens in candidates) / max(total, 1)

    scored: list[Evidence] = []
    seen_passages: set[str] = set()
    for doc, passage, tokens in candidates:
        fingerprint = f"{doc.url}:{passage[:220]}".lower()
        if fingerprint in seen_passages:
            continue
        seen_passages.add(fingerprint)

        lexical = _bm25(query_tokens, tokens, idf, avg_len)
        title_hit = _bm25(query_tokens, tokenize(doc.title), idf, avg_len) * 0.16
        snippet_hit = _bm25(query_tokens, tokenize(doc.snippet), idf, avg_len) * 0.10
        passage_lower = passage.lower()
        title_lower = doc.title.lower()
        snippet_lower = doc.snippet.lower()
        phrase_bonus = 1.0 if query_lower and query_lower in passage_lower else 0.0
        phrase_bonus += sum(0.75 for phrase in phrases if phrase in passage_lower)
        phrase_bonus += sum(0.08 for phrase in phrases if phrase in title_lower)
        phrase_bonus += sum(0.05 for phrase in phrases if phrase in snippet_lower)
        quality = source_quality(doc.url)
        provider_boost = 1.28 if doc.provider == "official" else 1.0
        score = (lexical + title_hit + snippet_hit + phrase_bonus) * quality * provider_boost
        if "search" in query_tokens and "search" not in tokens:
            score *= 0.42
        if len(set(query_tokens) & set(tokens)) < min(2, len(set(query_tokens))):
            score *= 0.55
        if score <= 0:
            continue
        scored.append(
            Evidence(
                id=0,
                title=doc.title,
                url=doc.url,
                passage=passage,
                score=score,
                provider=doc.provider,
                signals={
                    "lexical": round(lexical, 4),
                    "title": round(title_hit, 4),
                    "snippet": round(snippet_hit, 4),
                    "quality": round(quality, 4),
                    "trust_tier": source_trust_tier(doc.url),
                    "contextual_bm25": 1.0,
                },
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    top = _diverse_top(scored, limit)
    for idx, evidence in enumerate(top, start=1):
        evidence.id = idx
    return top


def _diverse_top(scored: list[Evidence], limit: int) -> list[Evidence]:
    selected: list[Evidence] = []
    url_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    for item in scored:
        domain = domain_for(item.url)
        if url_counts[item.url] >= 1 or domain_counts[domain] >= 3:
            continue
        selected.append(item)
        url_counts[item.url] += 1
        domain_counts[domain] += 1
        if len(selected) >= limit:
            return selected

    for item in scored:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected
