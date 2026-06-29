#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_feature_by_group, lexical_feature_values
from finmteb_sota.metrics import RankedQuery, reranking_metrics
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


def feature_matrix(queries: list[str], docs: list[str]) -> dict[str, list[float]]:
    matrix: dict[str, list[float]] = {}
    for query, doc in zip(queries, docs):
        for name, value in lexical_feature_values(query, doc).items():
            matrix.setdefault(name, []).append(value)
    return matrix


def folds_for_records(records: list[RerankRecord], n_folds: int, seed: int) -> list[set[str]]:
    qids = [record.query_id for record in records]
    random.Random(seed).shuffle(qids)
    folds = [set() for _ in range(n_folds)]
    for idx, qid in enumerate(qids):
        folds[idx % n_folds].add(qid)
    return folds


def top_margin_by_qid(qids: list[str], scores: list[float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for qid, score in zip(qids, scores):
        grouped.setdefault(qid, []).append(score)
    margins: dict[str, float] = {}
    for qid, q_scores in grouped.items():
        sorted_scores = sorted(q_scores, reverse=True)
        if len(sorted_scores) < 2:
            margins[qid] = float("inf")
        else:
            margins[qid] = sorted_scores[0] - sorted_scores[1]
    return margins


def apply_gate(
    qids: list[str],
    base_scores: list[float],
    alternate_scores: list[float],
    margins: dict[str, float],
    threshold: float,
) -> list[float]:
    return [
        alternate if margins[qid] <= threshold else base
        for qid, base, alternate in zip(qids, base_scores, alternate_scores)
    ]


def qid_thresholds(qids: list[str], margins: dict[str, float]) -> list[float]:
    values = sorted({margins[qid] for qid in set(qids)})
    return [-1.0, *values]


def candidate_scores(
    qids: list[str],
    model_scores: list[float],
    features: dict[str, list[float]],
) -> dict[str, list[float]]:
    return {
        "model": model_scores,
        "lexical_alpha_0.2": blend_feature_by_group(qids, model_scores, features["lexical"], 0.2),
        "doc_bigram_alpha_0.01": blend_feature_by_group(
            qids, model_scores, features["doc_bigram"], 0.01
        ),
        "title_bigram_alpha_0.15": blend_feature_by_group(
            qids, model_scores, features["title_bigram"], 0.15
        ),
        "title_trigram_alpha_0.5": blend_feature_by_group(
            qids, model_scores, features["title_trigram"], 0.5
        ),
        "title_lexical_alpha_0.05": blend_feature_by_group(
            qids, model_scores, features["title_lexical"], 0.05
        ),
    }


def search_candidates(
    qids: list[str],
    labels: list[int],
    model_scores: list[float],
    candidates: dict[str, list[float]],
) -> list[dict[str, Any]]:
    margins = top_margin_by_qid(qids, model_scores)
    thresholds = qid_thresholds(qids, margins)
    results: list[dict[str, Any]] = []

    for name, scores in candidates.items():
        results.append(
            {
                "candidate_id": len(results),
                "method": "direct",
                "base": name,
                "alternate": None,
                "threshold": None,
                "gated_query_count": 0,
                "scores": scores,
                "full_train_metrics": reranking_metrics(group_scores(qids, labels, scores)),
            }
        )

    for base_name, base_scores in candidates.items():
        for alternate_name, alternate_scores in candidates.items():
            if base_name == alternate_name:
                continue
            for threshold in thresholds:
                scores = apply_gate(qids, base_scores, alternate_scores, margins, threshold)
                gated_qids = {qid for qid, margin in margins.items() if margin <= threshold}
                results.append(
                    {
                        "candidate_id": len(results),
                        "method": "low_margin_gate",
                        "base": base_name,
                        "alternate": alternate_name,
                        "threshold": threshold,
                        "gated_query_count": len(gated_qids),
                        "scores": scores,
                        "full_train_metrics": reranking_metrics(group_scores(qids, labels, scores)),
                    }
                )

    results.sort(
        key=lambda item: (
            item["full_train_metrics"]["map"],
            item["full_train_metrics"]["mrr"],
            item["full_train_metrics"]["ndcg@10"],
        ),
        reverse=True,
    )
    return results


def strip_scores(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "scores"}


def nested_selection_key(candidate: dict[str, Any], fold_idx: int) -> tuple[float, float, float, int, int, int]:
    metrics = candidate["fold_train_metrics"][fold_idx]
    is_direct = 1 if candidate["method"] == "direct" else 0
    return (
        metrics["map"],
        metrics["mrr"],
        metrics["ndcg@10"],
        is_direct,
        -int(candidate["gated_query_count"]),
        -int(candidate["candidate_id"]),
    )


def evaluate_task(task: RerankingTask, args: argparse.Namespace) -> dict[str, Any]:
    records = load_reranking_records(task.dataset_id, split=args.split)
    queries, docs, labels, qids = flatten_records(records)
    model_scores, cache_path = load_score_cache(
        args.cache_dir,
        task,
        args.split,
        args.instruction,
        args.cache_tag,
    )
    features = feature_matrix(queries, docs)
    candidates = search_candidates(qids, labels, model_scores, candidate_scores(qids, model_scores, features))
    folds = folds_for_records(records, args.folds, args.seed)
    all_qids = {record.query_id for record in records}

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = candidate["scores"]
        fold_metrics = [subset_metrics(qids, labels, scores, fold) for fold in folds]
        fold_train_metrics = [
            subset_metrics(qids, labels, scores, all_qids - fold)
            for fold in folds
        ]
        fold_maps = [item["map"] for item in fold_metrics]
        evaluated.append(
            {
                **strip_scores(candidate),
                "validation_map_mean": mean(fold_maps),
                "validation_map_min": min(fold_maps),
                "validation_map_std": pstdev(fold_maps),
                "fold_metrics": fold_metrics,
                "fold_train_metrics": fold_train_metrics,
            }
        )

    evaluated.sort(
        key=lambda item: (
            item["validation_map_mean"],
            item["validation_map_min"],
            item["full_train_metrics"]["map"],
        ),
        reverse=True,
    )

    nested_selections: list[dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        selected = max(
            evaluated,
            key=lambda item: nested_selection_key(item, fold_idx),
        )
        selected_scores = next(
            candidate["scores"]
            for candidate in candidates
            if selected["candidate_id"] == candidate["candidate_id"]
        )
        nested_selections.append(
            {
                "fold": fold_idx,
                "holdout_qids": sorted(fold),
                "selected": {
                    **{
                        key: selected[key]
                        for key in ("method", "base", "alternate", "threshold", "gated_query_count")
                    },
                    "train_metrics": selected["fold_train_metrics"][fold_idx],
                    "holdout_metrics": subset_metrics(qids, labels, selected_scores, fold),
                },
            }
        )
    nested_maps = [
        item["selected"]["holdout_metrics"]["map"] for item in nested_selections
    ]
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": args.split,
        "cache": str(cache_path),
        "num_queries": len(records),
        "num_pairs": len(model_scores),
        "folds": args.folds,
        "seed": args.seed,
        "best": evaluated[0],
        "nested_selection": {
            "validation_map_mean": mean(nested_maps),
            "validation_map_min": min(nested_maps),
            "validation_map_std": pstdev(nested_maps),
            "selections": nested_selections,
        },
        "candidates": evaluated[: args.keep_candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["DISCFinLLMReranking"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-tag", default="qwen3_8b")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--keep-candidates", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/gated_blend_strategy.json"))
    args = parser.parse_args()

    results = [evaluate_task(task, args) for task in resolve_tasks(args.tasks)]
    payload = {
        "split": args.split,
        "average_validation_map": sum(
            item["nested_selection"]["validation_map_mean"] for item in results
        )
        / len(results),
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
