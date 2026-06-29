#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_feature_by_group, lexical_feature_values, zscore
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


def folds_for_records(records: list[RerankRecord], n_folds: int, seed: int) -> list[set[str]]:
    qids = [record.query_id for record in records]
    random.Random(seed).shuffle(qids)
    folds = [set() for _ in range(n_folds)]
    for idx, qid in enumerate(qids):
        folds[idx % n_folds].add(qid)
    return folds


def feature_values(
    queries: list[str],
    docs: list[str],
    feature_name: str,
) -> list[float]:
    values: list[float] = []
    for query, doc in zip(queries, docs):
        features = lexical_feature_values(query, doc)
        if feature_name not in features:
            raise KeyError(f"Unknown lexical feature: {feature_name}")
        values.append(features[feature_name])
    return values


def apply_lexical_alpha(
    qids: list[str],
    scores: list[float],
    feature_scores: list[float],
    alpha: float,
) -> list[float]:
    if alpha == 0.0:
        return scores
    return blend_feature_by_group(qids, scores, feature_scores, alpha)


def blend_models_by_group(
    qids: list[str],
    primary_scores: list[float],
    secondary_scores: list[float],
    alpha: float,
) -> list[float]:
    grouped: dict[str, list[int]] = {}
    for idx, qid in enumerate(qids):
        grouped.setdefault(qid, []).append(idx)

    blended = [0.0 for _ in primary_scores]
    for indices in grouped.values():
        primary_norm = zscore([primary_scores[idx] for idx in indices])
        secondary_norm = zscore([secondary_scores[idx] for idx in indices])
        for local_idx, global_idx in enumerate(indices):
            blended[global_idx] = primary_norm[local_idx] + alpha * secondary_norm[local_idx]
    return blended


def candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {"method": "primary", "alpha": 0.0},
        {"method": "secondary", "alpha": 1.0},
    ]
    for alpha in [-1.0, -0.8, -0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        candidates.append({"method": "zscore_linear", "alpha": alpha})
    return candidates


def apply_candidate(
    qids: list[str],
    primary_scores: list[float],
    secondary_scores: list[float],
    candidate: dict[str, Any],
) -> list[float]:
    method = candidate["method"]
    if method == "primary":
        return primary_scores
    if method == "secondary":
        return secondary_scores
    if method == "zscore_linear":
        return blend_models_by_group(qids, primary_scores, secondary_scores, float(candidate["alpha"]))
    raise KeyError(f"Unknown candidate method: {method}")


def evaluate_candidate(
    records: list[RerankRecord],
    qids: list[str],
    labels: list[int],
    scores: list[float],
    folds: list[set[str]],
) -> dict[str, Any]:
    all_qids = {record.query_id for record in records}
    fold_metrics = [subset_metrics(qids, labels, scores, fold) for fold in folds]
    fold_train_metrics = [
        subset_metrics(qids, labels, scores, all_qids - fold)
        for fold in folds
    ]
    fold_maps = [item["map"] for item in fold_metrics]
    return {
        "full_train_metrics": reranking_metrics(group_scores(qids, labels, scores)),
        "validation_map_mean": mean(fold_maps),
        "validation_map_min": min(fold_maps),
        "validation_map_std": pstdev(fold_maps),
        "fold_metrics": fold_metrics,
        "fold_train_metrics": fold_train_metrics,
    }


def evaluate_task(task: RerankingTask, args: argparse.Namespace) -> dict[str, Any]:
    records = load_reranking_records(task.dataset_id, split=args.split)
    queries, docs, labels, qids = flatten_records(records)
    primary_raw, primary_cache = load_score_cache(
        args.cache_dir,
        task,
        args.split,
        args.primary_instruction,
        args.primary_cache_tag,
    )
    secondary_raw, secondary_cache = load_score_cache(
        args.cache_dir,
        task,
        args.split,
        args.secondary_instruction,
        args.secondary_cache_tag,
    )
    if len(primary_raw) != len(secondary_raw) or len(primary_raw) != len(labels):
        raise ValueError("Primary, secondary, and label lengths must match.")

    lexical = feature_values(queries, docs, args.lexical_feature)
    primary_scores = apply_lexical_alpha(qids, primary_raw, lexical, args.primary_lexical_alpha)
    secondary_scores = apply_lexical_alpha(qids, secondary_raw, lexical, args.secondary_lexical_alpha)
    folds = folds_for_records(records, args.folds, args.seed)

    results: list[dict[str, Any]] = []
    for candidate in candidate_grid():
        scores = apply_candidate(qids, primary_scores, secondary_scores, candidate)
        results.append(
            {
                **candidate,
                **evaluate_candidate(records, qids, labels, scores, folds),
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
        "num_pairs": len(labels),
        "primary_cache": str(primary_cache),
        "secondary_cache": str(secondary_cache),
        "primary": {
            "cache_tag": args.primary_cache_tag,
            "lexical_feature": args.lexical_feature,
            "lexical_alpha": args.primary_lexical_alpha,
        },
        "secondary": {
            "cache_tag": args.secondary_cache_tag,
            "lexical_feature": args.lexical_feature,
            "lexical_alpha": args.secondary_lexical_alpha,
        },
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
    parser.add_argument("--primary-instruction")
    parser.add_argument("--secondary-instruction", default="")
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--primary-cache-tag", default="qwen3_8b")
    parser.add_argument("--secondary-cache-tag", default="qwen3_4b")
    parser.add_argument("--lexical-feature", default="lexical")
    parser.add_argument("--primary-lexical-alpha", type=float, default=0.2)
    parser.add_argument("--secondary-lexical-alpha", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--keep-candidates", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/score_ensemble_search.json"))
    args = parser.parse_args()
    args.primary_instruction = args.primary_instruction or args.instruction

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
