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

HIGH_TRUST_SUFFIXES = (".gov", ".edu", ".ac.uk")
HIGH_TRUST_DOMAINS = {
    "openai.com",
    "help.openai.com",
    "developers.openai.com",
    "platform.openai.com",
    "wikipedia.org",
    "reuters.com",
    "apnews.com",
    "api-docs.deepseek.com",
    "nature.com",
    "science.org",
    "who.int",
    "cdc.gov",
    "fda.gov",
}
LOW_SIGNAL_DOMAINS = {
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "medium.com",
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
    domain = domain_for(url)
    if any(domain.endswith(suffix) for suffix in HIGH_TRUST_SUFFIXES):
        return 1.18
    if domain in HIGH_TRUST_DOMAINS or any(domain.endswith("." + d) for d in HIGH_TRUST_DOMAINS):
        return 1.12
    if domain in LOW_SIGNAL_DOMAINS or any(domain.endswith("." + d) for d in LOW_SIGNAL_DOMAINS):
        return 0.86
    return 1.0


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
            passage_tokens = tokenize(passage)
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
        fingerprint = passage[:220].lower()
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
