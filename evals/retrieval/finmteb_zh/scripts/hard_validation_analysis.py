#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_feature_by_group, lexical_feature_values
from finmteb_sota.metrics import RankedQuery, average_precision, reranking_metrics
from finmteb_sota.qwen3 import DEFAULT_INSTRUCTION
from finmteb_sota.score_cache import load_score_cache
from finmteb_sota.tasks import RerankingTask, resolve_tasks


def flatten_records(records: list[RerankRecord]) -> tuple[list[str], list[str], list[int], list[str]]:
    queries: list[str] = []
    docs: list[str] = []
    labels: list[int] = []
    qids: list[str] = []
    for record in records:
        for positive in record.positives:
            queries.append(record.query)
            docs.append(positive)
            labels.append(1)
            qids.append(record.query_id)
        for negative in record.negatives:
            queries.append(record.query)
            docs.append(negative)
            labels.append(0)
            qids.append(record.query_id)
    return queries, docs, labels, qids


def load_scores(
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_dir: Path,
    cache_tag: str,
) -> list[float]:
    scores, _ = load_score_cache(cache_dir, task, split, instruction, cache_tag)
    return scores


def feature_matrix(queries: list[str], docs: list[str]) -> dict[str, list[float]]:
    matrix: dict[str, list[float]] = {}
    for query, doc in zip(queries, docs):
        for name, value in lexical_feature_values(query, doc).items():
            matrix.setdefault(name, []).append(value)
    return matrix


def group_scores(qids: list[str], labels: list[int], scores: list[float]) -> list[RankedQuery]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for qid, label, score in zip(qids, labels, scores):
        q_labels, q_scores = grouped.setdefault(qid, ([], []))
        q_labels.append(label)
        q_scores.append(score)
    return [
        RankedQuery(query_id=qid, labels=query_labels, scores=query_scores)
        for qid, (query_labels, query_scores) in grouped.items()
    ]


def subset_metrics(
    qids: list[str],
    labels: list[int],
    scores: list[float],
    keep_qids: set[str],
) -> dict[str, float]:
    kept_qids: list[str] = []
    kept_labels: list[int] = []
    kept_scores: list[float] = []
    for qid, label, score in zip(qids, labels, scores):
        if qid in keep_qids:
            kept_qids.append(qid)
            kept_labels.append(label)
            kept_scores.append(score)
    return reranking_metrics(group_scores(kept_qids, kept_labels, kept_scores))


def per_query_ap(qids: list[str], labels: list[int], scores: list[float]) -> dict[str, float]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for qid, label, score in zip(qids, labels, scores):
        q_labels, q_scores = grouped.setdefault(qid, ([], []))
        q_labels.append(label)
        q_scores.append(score)
    return {
        qid: average_precision(q_labels, q_scores)
        for qid, (q_labels, q_scores) in grouped.items()
    }


def min_positive_margin(qids: list[str], labels: list[int], scores: list[float]) -> dict[str, float]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for qid, label, score in zip(qids, labels, scores):
        q_labels, q_scores = grouped.setdefault(qid, ([], []))
        q_labels.append(label)
        q_scores.append(score)
    margins: dict[str, float] = {}
    for qid, (q_labels, q_scores) in grouped.items():
        positives = [score for label, score in zip(q_labels, q_scores) if label > 0]
        negatives = [score for label, score in zip(q_labels, q_scores) if label <= 0]
        if not positives or not negatives:
            margins[qid] = 0.0
        else:
            margins[qid] = min(positives) - max(negatives)
    return margins


def candidate_scores(
    qids: list[str],
    model_scores: list[float],
    features: dict[str, list[float]],
) -> dict[str, list[float]]:
    candidates = {
        "model": model_scores,
        "lexical_alpha_0.2": blend_feature_by_group(qids, model_scores, features["lexical"], 0.2),
        "doc_bigram_alpha_0.01": blend_feature_by_group(
            qids, model_scores, features["doc_bigram"], 0.01
        ),
        "doc_bigram_alpha_0.05": blend_feature_by_group(
            qids, model_scores, features["doc_bigram"], 0.05
        ),
        "title_bigram_alpha_0.15": blend_feature_by_group(
            qids, model_scores, features["title_bigram"], 0.15
        ),
        "title_lexical_alpha_0.05": blend_feature_by_group(
            qids, model_scores, features["title_lexical"], 0.05
        ),
    }
    return candidates


def choose_hard_qids(
    records: list[RerankRecord],
    qids: list[str],
    labels: list[int],
    base_scores: list[float],
    baseline_scores: list[float],
    max_hard: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    base_ap = per_query_ap(qids, labels, base_scores)
    baseline_ap = per_query_ap(qids, labels, baseline_scores)
    base_margin = min_positive_margin(qids, labels, base_scores)
    baseline_margin = min_positive_margin(qids, labels, baseline_scores)
    rows = []
    for record in records:
        qid = record.query_id
        hard_reason = []
        if base_ap[qid] < 1.0:
            hard_reason.append("base_ap_lt_1")
        if baseline_ap[qid] < 1.0:
            hard_reason.append("baseline_ap_lt_1")
        if min(base_margin[qid], baseline_margin[qid]) < 0.01:
            hard_reason.append("low_margin")
        rows.append(
            {
                "query_id": qid,
                "query": record.query,
                "base_ap": base_ap[qid],
                "baseline_ap": baseline_ap[qid],
                "base_margin": base_margin[qid],
                "baseline_margin": baseline_margin[qid],
                "hard_reason": hard_reason,
            }
        )

    hard_rows = [row for row in rows if row["hard_reason"]]
    hard_rows.sort(
        key=lambda row: (
            min(row["base_ap"], row["baseline_ap"]),
            min(row["base_margin"], row["baseline_margin"]),
        )
    )
    selected = hard_rows[:max_hard]
    return {row["query_id"] for row in selected}, selected


def evaluate_task(task: RerankingTask, args: argparse.Namespace) -> dict[str, Any]:
    records = load_reranking_records(task.dataset_id, split=args.split)
    queries, docs, labels, qids = flatten_records(records)
    model_scores = load_scores(
        task,
        args.split,
        args.instruction,
        args.cache_dir,
        args.cache_tag,
    )
    features = feature_matrix(queries, docs)
    candidates = candidate_scores(qids, model_scores, features)
    hard_qids, hard_rows = choose_hard_qids(
        records,
        qids,
        labels,
        candidates["model"],
        candidates["lexical_alpha_0.2"],
        args.max_hard_queries,
    )
    rest_qids = {record.query_id for record in records} - hard_qids

    evaluations = []
    for name, scores in candidates.items():
        hard_metrics = subset_metrics(qids, labels, scores, hard_qids)
        rest_metrics = subset_metrics(qids, labels, scores, rest_qids)
        full_metrics = reranking_metrics(group_scores(qids, labels, scores))
        evaluations.append(
            {
                "candidate": name,
                "full_metrics": full_metrics,
                "hard_metrics": hard_metrics,
                "rest_metrics": rest_metrics,
                "score": mean([hard_metrics["map"], rest_metrics["map"]]),
            }
        )
    evaluations.sort(
        key=lambda item: (
            item["hard_metrics"]["map"],
            item["rest_metrics"]["map"],
            item["full_metrics"]["map"],
        ),
        reverse=True,
    )
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": args.split,
        "num_queries": len(records),
        "hard_query_count": len(hard_qids),
        "hard_queries": hard_rows,
        "evaluations": evaluations,
        "best": evaluations[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["DISCFinLLMReranking"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-tag", default="qwen3_8b")
    parser.add_argument("--max-hard-queries", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/hard_validation_analysis.json"))
    args = parser.parse_args()

    results = [evaluate_task(task, args) for task in resolve_tasks(args.tasks)]
    payload = {
        "split": args.split,
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
