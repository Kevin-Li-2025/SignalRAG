from __future__ import annotations

import math
from typing import Any


def positives(qrels_for_query: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, rel in qrels_for_query.items() if rel >= 2}


def ranked_doc_ids(scores: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def ndcg_at_k(qrels_for_query: dict[str, int], docs: list[str], k: int) -> float:
    pos = positives(qrels_for_query)
    if not pos:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(docs[:k], start=1):
        gain = 1.0 if doc_id in pos else 0.0
        if gain:
            dcg += gain / math.log2(rank + 1)
    ideal_hits = min(len(pos), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(qrels_for_query: dict[str, int], docs: list[str], k: int) -> float:
    pos = positives(qrels_for_query)
    if not pos:
        return 0.0
    return len(pos.intersection(docs[:k])) / len(pos)


def average_precision_at_k(qrels_for_query: dict[str, int], docs: list[str], k: int) -> float:
    pos = positives(qrels_for_query)
    if not pos:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(docs[:k], start=1):
        if doc_id in pos:
            hits += 1
            total += hits / rank
    return total / min(len(pos), k)


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
    exclude_doc_ids: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    map_values: list[float] = []
    missing = 0
    for query_id, query_qrels in qrels.items():
        scores = run.get(query_id, {})
        if not scores:
            missing += 1
        docs = ranked_doc_ids(scores)
        if exclude_doc_ids:
            excluded = exclude_doc_ids.get(query_id)
            if excluded:
                docs = [doc_id for doc_id in docs if doc_id not in excluded]
        ndcg_values.append(ndcg_at_k(query_qrels, docs, k))
        recall_values.append(recall_at_k(query_qrels, docs, k))
        map_values.append(average_precision_at_k(query_qrels, docs, k))
    n = len(qrels)
    return {
        "queries": n,
        "missing_queries": missing,
        f"ndcg@{k}": sum(ndcg_values) / n if n else 0.0,
        f"recall@{k}": sum(recall_values) / n if n else 0.0,
        f"map@{k}": sum(map_values) / n if n else 0.0,
    }


def weighted_average(task_metrics: dict[str, dict[str, Any]], metric: str) -> float:
    total_queries = sum(int(row["queries"]) for row in task_metrics.values())
    if not total_queries:
        return 0.0
    return sum(float(row[metric]) * int(row["queries"]) for row in task_metrics.values()) / total_queries
