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
    _validate_lengths(labels, scores)
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in ranked if label > 0)
    if positives == 0:
        return 0.0
    precision_sum = 0.0
    seen = hits = 0
    for block in _tie_blocks(ranked):
        block_size = len(block)
        block_hits = sum(label > 0 for _, label in block)
        if block_hits:
            for offset in range(1, block_size + 1):
                prior_in_block = (
                    (offset - 1) * (block_hits - 1) / (block_size - 1)
                    if block_size > 1 else 0.0
                )
                precision_sum += (block_hits / block_size) * (
                    hits + 1 + prior_in_block
                ) / (seen + offset)
        hits += block_hits
        seen += block_size
    return precision_sum / positives


def reciprocal_rank(labels: list[int], scores: list[float]) -> float:
    _validate_lengths(labels, scores)
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    seen = 0
    for block in _tie_blocks(ranked):
        block_size = len(block)
        block_hits = sum(label > 0 for _, label in block)
        if block_hits:
            denominator = math.comb(block_size, block_hits)
            return sum(
                math.comb(block_size - offset, block_hits - 1)
                / denominator
                / (seen + offset)
                for offset in range(1, block_size - block_hits + 2)
            )
        seen += block_size
    return 0.0


def ndcg_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    _validate_lengths(labels, scores)
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    ideal = sorted(labels, reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        total = 0.0
        for idx, value in enumerate(values, start=1):
            total += (2**value - 1) / math.log2(idx + 1)
        return total

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    expected_dcg = 0.0
    seen = 0
    for block in _tie_blocks(ranked):
        if seen >= k:
            break
        mean_gain = sum(2**label - 1 for _, label in block) / len(block)
        for rank in range(seen + 1, min(seen + len(block), k) + 1):
            expected_dcg += mean_gain / math.log2(rank + 1)
        seen += len(block)
    return expected_dcg / ideal_dcg


def _validate_lengths(labels: list[int], scores: list[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores must have identical length: {len(labels)} != {len(scores)}"
        )


def _tie_blocks(ranked: list[tuple[float, int]]) -> list[list[tuple[float, int]]]:
    blocks: list[list[tuple[float, int]]] = []
    for item in ranked:
        if not blocks or item[0] != blocks[-1][0][0]:
            blocks.append([item])
        else:
            blocks[-1].append(item)
    return blocks


def reranking_metrics(queries: list[RankedQuery], ndcg_k: int = 10) -> dict[str, float]:
    if not queries:
        return {"map": 0.0, "mrr": 0.0, f"ndcg@{ndcg_k}": 0.0}
    return {
        "map": mean(average_precision(query.labels, query.scores) for query in queries),
        "mrr": mean(reciprocal_rank(query.labels, query.scores) for query in queries),
        f"ndcg@{ndcg_k}": mean(ndcg_at_k(query.labels, query.scores, ndcg_k) for query in queries),
    }
