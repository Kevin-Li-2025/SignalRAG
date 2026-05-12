from __future__ import annotations

import copy
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{0,}", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
FRESHNESS_RE = re.compile(r"\b(latest|today|current|now|recent|new|202[5-9])\b|最新|今天|当前|最近|今年", re.IGNORECASE)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "does",
    "for",
    "in",
    "is",
    "its",
    "me",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "this",
    "to",
    "you",
}
INTENT_PATTERNS = {
    "compare": re.compile(r"\b(compare|versus|vs|difference|tradeoff|trade-off)\b|比较|区别|权衡", re.IGNORECASE),
    "how": re.compile(r"\b(how|guide|tutorial|steps|implement|use|build)\b|怎么|如何|实现|步骤", re.IGNORECASE),
    "why": re.compile(r"\b(why|cause|causes|reason)\b|为什么|原因", re.IGNORECASE),
    "recommend": re.compile(r"\b(best|recommend|choose|which|should)\b|推荐|选择|哪个好", re.IGNORECASE),
    "code": re.compile(r"\b(api|sdk|code|python|javascript|typescript|curl|json)\b|代码|接口", re.IGNORECASE),
}


@dataclass(frozen=True)
class CacheLookup:
    response: dict[str, Any]
    strategy: str
    score: float
    source_query: str
    age_seconds: int


class PageCache:
    def __init__(self, path: Path, ttl_seconds: int = 60 * 60 * 24) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                status INTEGER,
                content_type TEXT,
                body TEXT,
                created_at REAL
            )
            """
        )
        self._conn.commit()

    def get(self, url: str) -> tuple[int | None, str | None, str] | None:
        row = self._conn.execute(
            "SELECT status, content_type, body, created_at FROM pages WHERE url = ?",
            (url,),
        ).fetchone()
        if not row:
            return None
        status, content_type, body, created_at = row
        if time.time() - float(created_at) > self.ttl_seconds:
            return None
        return status, content_type, body

    def set(self, url: str, status: int | None, content_type: str | None, body: str) -> None:
        self._conn.execute(
            """
            INSERT INTO pages (url, status, content_type, body, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                status = excluded.status,
                content_type = excluded.content_type,
                body = excluded.body,
                created_at = excluded.created_at
            """,
            (url, status, content_type, body, time.time()),
        )
        self._conn.commit()


class SmartResponseCache:
    def __init__(
        self,
        path: Path,
        ttl_seconds: int = 60 * 60,
        fresh_ttl_seconds: int = 5 * 60,
        max_items: int = 512,
        fuzzy_threshold: float = 0.86,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.fresh_ttl_seconds = fresh_ttl_seconds
        self.max_items = max_items
        self.fuzzy_threshold = fuzzy_threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                scope_key TEXT,
                query TEXT,
                normalized_query TEXT,
                query_tokens TEXT,
                intent_tags TEXT,
                number_tokens TEXT,
                request_json TEXT,
                response_json TEXT,
                created_at REAL,
                last_hit_at REAL,
                hit_count INTEGER DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_responses_scope ON responses(scope_key)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at)")
        self._conn.commit()

    def get(self, request: dict[str, Any]) -> CacheLookup | None:
        normalized = normalize_request(request)
        cache_key = _cache_key(normalized)
        now = time.time()
        exact = self._lookup_key(cache_key, now)
        if exact:
            return exact
        if _fuzzy_disabled(normalized):
            return None
        return self._lookup_fuzzy(normalized, now)

    def set(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        normalized = normalize_request(request)
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO responses (
                cache_key, scope_key, query, normalized_query, query_tokens,
                intent_tags, number_tokens, request_json, response_json,
                created_at, last_hit_at, hit_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(cache_key) DO UPDATE SET
                scope_key = excluded.scope_key,
                query = excluded.query,
                normalized_query = excluded.normalized_query,
                query_tokens = excluded.query_tokens,
                intent_tags = excluded.intent_tags,
                number_tokens = excluded.number_tokens,
                request_json = excluded.request_json,
                response_json = excluded.response_json,
                created_at = excluded.created_at
            """,
            (
                _cache_key(normalized),
                _scope_key(normalized),
                normalized["query"],
                normalized["normalized_query"],
                json.dumps(normalized["query_tokens"], ensure_ascii=False),
                json.dumps(normalized["intent_tags"], ensure_ascii=False),
                json.dumps(normalized["number_tokens"], ensure_ascii=False),
                json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                json.dumps(response, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        self._conn.commit()
        self._evict_old_rows()

    def _lookup_key(self, cache_key: str, now: float) -> CacheLookup | None:
        row = self._conn.execute(
            """
            SELECT query, response_json, created_at FROM responses
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        query, response_json, created_at = row
        age = int(now - float(created_at))
        if age > self.ttl_seconds:
            self._conn.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
            self._conn.commit()
            return None
        self._record_hit(cache_key, now)
        return CacheLookup(
            response=json.loads(response_json),
            strategy="exact",
            score=1.0,
            source_query=query,
            age_seconds=age,
        )

    def _lookup_fuzzy(self, normalized: dict[str, Any], now: float) -> CacheLookup | None:
        query_tokens = set(normalized["query_tokens"])
        if len(query_tokens) < 4:
            return None
        best: tuple[float, str, str, str, float] | None = None
        rows = self._conn.execute(
            """
            SELECT cache_key, query, query_tokens, intent_tags, number_tokens, response_json, created_at, request_json
            FROM responses
            WHERE scope_key = ?
            ORDER BY last_hit_at DESC
            LIMIT 80
            """,
            (_scope_key(normalized),),
        ).fetchall()
        for cache_key, query, token_json, intent_json, number_json, response_json, created_at, request_json in rows:
            age = int(now - float(created_at))
            cached_request = json.loads(request_json)
            ttl = self._ttl_for(cached_request)
            if age > ttl:
                continue
            cached_tokens = set(json.loads(token_json))
            score = _jaccard(query_tokens, cached_tokens)
            if score < self.fuzzy_threshold:
                continue
            if set(json.loads(intent_json)) != set(normalized["intent_tags"]):
                continue
            if set(json.loads(number_json)) != set(normalized["number_tokens"]):
                continue
            if int(cached_request.get("max_results") or 0) < int(normalized.get("max_results") or 0):
                continue
            if best is None or score > best[0]:
                best = (score, cache_key, query, response_json, created_at)
        if not best:
            return None
        score, cache_key, query, response_json, created_at = best
        self._record_hit(cache_key, now)
        return CacheLookup(
            response=json.loads(response_json),
            strategy="fuzzy",
            score=round(score, 4),
            source_query=query,
            age_seconds=int(now - float(created_at)),
        )

    def _ttl_for(self, normalized: dict[str, Any]) -> int:
        if normalized.get("recency") in {"day", "week"} or FRESHNESS_RE.search(str(normalized.get("query") or "")):
            return self.fresh_ttl_seconds
        return self.ttl_seconds

    def _record_hit(self, cache_key: str, now: float) -> None:
        self._conn.execute(
            """
            UPDATE responses
            SET last_hit_at = ?, hit_count = COALESCE(hit_count, 0) + 1
            WHERE cache_key = ?
            """,
            (now, cache_key),
        )
        self._conn.commit()

    def _evict_old_rows(self) -> None:
        self._conn.execute(
            """
            DELETE FROM responses
            WHERE cache_key NOT IN (
                SELECT cache_key FROM responses
                ORDER BY last_hit_at DESC, created_at DESC
                LIMIT ?
            )
            """,
            (self.max_items,),
        )
        self._conn.commit()


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    query = str(request.get("query") or "")
    normalized_query = normalize_query(query)
    query_tokens = query_fingerprint_tokens(normalized_query)
    return {
        "query": query.strip(),
        "normalized_query": normalized_query,
        "query_tokens": sorted(query_tokens),
        "intent_tags": sorted(_intent_tags(query)),
        "number_tokens": sorted(re.findall(r"\b\d+(?:\.\d+)?\b", query)),
        "mode": str(request.get("mode") or "fast"),
        "lens": str(request.get("lens") or "web"),
        "max_results": int(request.get("max_results") or 8),
        "include_domains": sorted(_clean_list(request.get("include_domains") or [])),
        "exclude_domains": sorted(_clean_list(request.get("exclude_domains") or [])),
        "recency": str(request.get("recency") or "any"),
        "country": str(request.get("country") or "").lower().strip(),
        "language": str(request.get("language") or "").lower().strip(),
        "citation_verifier": str(request.get("citation_verifier") or "auto"),
    }


def normalize_query(query: str) -> str:
    query = query.lower().strip()
    query = PUNCT_RE.sub(" ", query)
    query = SPACE_RE.sub(" ", query)
    return query.strip()


def query_fingerprint_tokens(normalized_query: str) -> set[str]:
    tokens = {
        _stem_token(token)
        for token in WORD_RE.findall(normalized_query)
        if token and token not in QUERY_STOPWORDS
    }
    cjk = CJK_RE.findall(normalized_query)
    tokens.update(cjk)
    tokens.update("".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1)))
    return tokens


def _stem_token(token: str) -> str:
    if token in {"does", "was", "has"}:
        return token
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def copy_cached_response(lookup: CacheLookup, request: dict[str, Any]) -> dict[str, Any]:
    response = copy.deepcopy(lookup.response)
    response["query"] = str(request.get("query") or response.get("query") or "")
    meta = response.setdefault("meta", {})
    meta["cache_hit"] = True
    meta["cache_strategy"] = lookup.strategy
    meta["cache_similarity"] = lookup.score
    meta["cache_age_seconds"] = lookup.age_seconds
    meta["cache_source_query"] = lookup.source_query
    meta["elapsed_ms"] = 0
    return response


def _cache_key(normalized: dict[str, Any]) -> str:
    key_data = {key: value for key, value in normalized.items() if key != "query"}
    return json.dumps(key_data, ensure_ascii=False, sort_keys=True)


def _scope_key(normalized: dict[str, Any]) -> str:
    scope = {
        "mode": normalized["mode"],
        "lens": normalized["lens"],
        "include_domains": normalized["include_domains"],
        "exclude_domains": normalized["exclude_domains"],
        "recency": normalized["recency"],
        "country": normalized["country"],
        "language": normalized["language"],
        "citation_verifier": normalized["citation_verifier"],
    }
    return json.dumps(scope, ensure_ascii=False, sort_keys=True)


def _fuzzy_disabled(normalized: dict[str, Any]) -> bool:
    return normalized.get("recency") in {"day", "week"} or bool(FRESHNESS_RE.search(str(normalized.get("query") or "")))


def _clean_list(values: list[Any]) -> list[str]:
    return [str(value).lower().strip() for value in values if str(value).strip()]


def _intent_tags(query: str) -> set[str]:
    tags = {tag for tag, pattern in INTENT_PATTERNS.items() if pattern.search(query)}
    return tags or {"lookup"}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
