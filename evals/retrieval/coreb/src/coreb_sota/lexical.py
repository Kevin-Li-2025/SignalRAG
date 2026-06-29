from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from coreb_sota.data import c2c_anchor_id

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%]=?|[{}()[\\].,;:]")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def split_identifier(token: str) -> list[str]:
    pieces: list[str] = []
    for part in token.replace("-", "_").split("_"):
        pieces.extend(CAMEL_RE.split(part))
    return [p.lower() for p in pieces if p]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        if re.match(r"^[a-zA-Z_][A-Za-z0-9_]*$", raw):
            tokens.extend(split_identifier(raw))
    return [tok for tok in tokens if tok.strip()]


@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_len: dict[str, int]
    avgdl: float
    postings: dict[str, list[tuple[str, int]]]
    idf: dict[str, float]
    k1: float = 0.9
    b: float = 0.4

    @classmethod
    def build(cls, corpus: dict[str, dict[str, Any]], k1: float = 0.9, b: float = 0.4) -> "BM25Index":
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        doc_len: dict[str, int] = {}
        doc_ids = list(corpus)
        for doc_id, doc in corpus.items():
            tokens = tokenize(doc.get("text", ""))
            if doc.get("language"):
                tokens.extend([f"lang:{doc['language']}", str(doc["language"]).lower()])
            counts = Counter(tokens)
            doc_len[doc_id] = len(tokens)
            for token, tf in counts.items():
                postings[token].append((doc_id, tf))

        n_docs = len(doc_ids)
        avgdl = sum(doc_len.values()) / n_docs if n_docs else 0.0
        idf = {
            token: math.log(1 + (n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for token, docs in postings.items()
        }
        return cls(doc_ids=doc_ids, doc_len=doc_len, avgdl=avgdl, postings=dict(postings), idf=idf, k1=k1, b=b)

    def search(self, query: str, top_k: int, exclude: set[str] | None = None) -> dict[str, float]:
        exclude = exclude or set()
        query_counts = Counter(tokenize(query))
        scores: dict[str, float] = defaultdict(float)
        for token, qtf in query_counts.items():
            if token not in self.postings:
                continue
            token_idf = self.idf[token]
            for doc_id, tf in self.postings[token]:
                if doc_id in exclude:
                    continue
                length = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1e-9))
                scores[doc_id] += token_idf * (tf * (self.k1 + 1) / denom) * (1 + math.log1p(qtf))
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])


def c2c_exclusions(query_row: dict[str, Any]) -> set[str]:
    anchor = c2c_anchor_id(query_row)
    return {anchor} if anchor else set()
