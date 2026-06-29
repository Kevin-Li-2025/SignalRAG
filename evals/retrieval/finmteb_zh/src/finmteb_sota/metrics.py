from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RankedQuery:
    query_id: str
    labels: list[int]
    scores: list[float]


def average_precision(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in ranked if label > 0)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def reciprocal_rank(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)[:k]
    ideal = sorted(labels, reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        total = 0.0
        for idx, value in enumerate(values, start=1):
            total += (2**value - 1) / math.log2(idx + 1)
        return total

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg([label for _, label in ranked]) / ideal_dcg


def reranking_metrics(queries: list[RankedQuery], ndcg_k: int = 10) -> dict[str, float]:
    if not queries:
        return {"map": 0.0, "mrr": 0.0, f"ndcg@{ndcg_k}": 0.0}
    return {
        "map": mean(average_precision(query.labels, query.scores) for query in queries),
        "mrr": mean(reciprocal_rank(query.labels, query.scores) for query in queries),
        f"ndcg@{ndcg_k}": mean(ndcg_at_k(query.labels, query.scores, ndcg_k) for query in queries),
    }

