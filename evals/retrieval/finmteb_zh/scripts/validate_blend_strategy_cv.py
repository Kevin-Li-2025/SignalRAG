#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_feature_by_group, lexical_feature_values, rrf_blend_by_group
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


def load_scores(
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_dir: Path,
    cache_tag: str,
) -> list[float]:
    try:
        scores, _ = load_score_cache(cache_dir, task, split, instruction, cache_tag)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Run search_blend_strategy.py once first with --cache-tag {cache_tag}."
        ) from exc
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
    filtered_qids: list[str] = []
    filtered_labels: list[int] = []
    filtered_scores: list[float] = []
    for qid, label, score in zip(qids, labels, scores):
        if qid in keep_qids:
            filtered_qids.append(qid)
            filtered_labels.append(label)
            filtered_scores.append(score)
    return reranking_metrics(group_scores(filtered_qids, filtered_labels, filtered_scores))


def candidate_grid(features: dict[str, list[float]]) -> list[dict[str, Any]]:
    alpha_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    rrf_alpha_grid = [0.1, 0.2, 0.5, 1.0, 2.0]
    candidates: list[dict[str, Any]] = [{"method": "model", "feature": None, "alpha": 0.0}]
    for feature_name in features:
        for alpha in alpha_grid:
            candidates.append(
                {"method": "zscore_linear", "feature": feature_name, "alpha": alpha}
            )
        for alpha in rrf_alpha_grid:
            candidates.append({"method": "rrf", "feature": feature_name, "alpha": alpha})
    return candidates


def apply_candidate(
    qids: list[str],
    model_scores: list[float],
    features: dict[str, list[float]],
    candidate: dict[str, Any],
) -> list[float]:
    method = candidate["method"]
    feature_name = candidate.get("feature")
    alpha = float(candidate.get("alpha", 0.0))
    if method == "model" or feature_name is None:
        return model_scores
    if method == "zscore_linear":
        return blend_feature_by_group(qids, model_scores, features[feature_name], alpha)
    if method == "rrf":
        return rrf_blend_by_group(qids, model_scores, features[feature_name], alpha, k=60.0)
    raise KeyError(f"Unknown candidate method: {method}")


def folds_for_records(records: list[RerankRecord], n_folds: int, seed: int) -> list[set[str]]:
    qids = [record.query_id for record in records]
    random.Random(seed).shuffle(qids)
    folds = [set() for _ in range(n_folds)]
    for idx, qid in enumerate(qids):
        folds[idx % n_folds].add(qid)
    return folds


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
    folds = folds_for_records(records, args.folds, args.seed)

    results: list[dict[str, Any]] = []
    for candidate in candidate_grid(features):
        scores = apply_candidate(qids, model_scores, features, candidate)
        fold_metrics = [subset_metrics(qids, labels, scores, fold) for fold in folds]
        fold_train_metrics = [
            subset_metrics(
                qids,
                labels,
                scores,
                set(record.query_id for record in records) - fold,
            )
            for fold in folds
        ]
        fold_maps = [item["map"] for item in fold_metrics]
        results.append(
            {
                **candidate,
                "full_train_metrics": reranking_metrics(group_scores(qids, labels, scores)),
                "validation_map_mean": mean(fold_maps),
                "validation_map_min": min(fold_maps),
                "validation_map_std": pstdev(fold_maps),
                "fold_metrics": fold_metrics,
                "fold_train_metrics": fold_train_metrics,
            }
        )

    results.sort(
        key=lambda item: (
            item["validation_map_mean"],
            item["validation_map_min"],
            item["full_train_metrics"]["map"],
        ),
        reverse=True,
    )
    nested_selections: list[dict[str, Any]] = []
    for fold_idx in range(len(folds)):
        selected = max(
            results,
            key=lambda item: (
                item["fold_train_metrics"][fold_idx]["map"],
                item["fold_train_metrics"][fold_idx]["mrr"],
                item["fold_train_metrics"][fold_idx]["ndcg@10"],
            ),
        )
        nested_selections.append(
            {
                "fold": fold_idx,
                "selected": {
                    "method": selected["method"],
                    "feature": selected["feature"],
                    "alpha": selected["alpha"],
                    "train_metrics": selected["fold_train_metrics"][fold_idx],
                    "holdout_metrics": selected["fold_metrics"][fold_idx],
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
        "num_queries": len(records),
        "folds": args.folds,
        "seed": args.seed,
        "best": results[0],
        "nested_selection": {
            "validation_map_mean": mean(nested_maps),
            "validation_map_min": min(nested_maps),
            "validation_map_std": pstdev(nested_maps),
            "selections": nested_selections,
        },
        "candidates": results[: args.keep_candidates],
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
    parser.add_argument("--output", type=Path, default=Path("reports/blend_strategy_cv.json"))
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
