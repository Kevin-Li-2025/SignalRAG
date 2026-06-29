#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import (
    blend_feature_by_group,
    lexical_feature_values,
    rrf_blend_by_group,
)
from finmteb_sota.metrics import RankedQuery, reranking_metrics
from finmteb_sota.qwen3 import DEFAULT_INSTRUCTION
from finmteb_sota.scoring import Qwen3RerankerScorer
from finmteb_sota.score_cache import load_score_cache, model_cache_tag, write_score_cache
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


def load_or_score(
    scorer: Qwen3RerankerScorer,
    task: RerankingTask,
    split: str,
    instruction: str,
    queries: list[str],
    docs: list[str],
    batch_size: int,
    max_length: int,
    cache_dir: Path,
    cache_tag: str,
    score_mode: str,
) -> list[float]:
    try:
        scores, _ = load_score_cache(cache_dir, task, split, instruction, cache_tag)
        return scores
    except FileNotFoundError:
        pass

    scores = scorer.score(
        queries=queries,
        documents=docs,
        instruction=instruction,
        batch_size=batch_size,
        max_length=max_length,
        score_mode=score_mode,
    )
    write_score_cache(cache_dir, task, split, instruction, cache_tag, scores)
    return scores


def feature_matrix(queries: list[str], docs: list[str]) -> dict[str, list[float]]:
    matrix: dict[str, list[float]] = {}
    for query, doc in zip(queries, docs):
        for name, value in lexical_feature_values(query, doc).items():
            matrix.setdefault(name, []).append(value)
    return matrix


def score_candidates(
    qids: list[str],
    labels: list[int],
    model_scores: list[float],
    features: dict[str, list[float]],
) -> list[dict[str, Any]]:
    alpha_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    rrf_alpha_grid = [0.1, 0.2, 0.5, 1.0, 2.0]
    candidates: list[dict[str, Any]] = [
        {
            "method": "model",
            "feature": None,
            "alpha": 0.0,
            "metrics": reranking_metrics(group_scores(qids, labels, model_scores)),
        }
    ]

    for feature_name, feature_scores in features.items():
        for alpha in alpha_grid:
            scores = blend_feature_by_group(qids, model_scores, feature_scores, alpha)
            candidates.append(
                {
                    "method": "zscore_linear",
                    "feature": feature_name,
                    "alpha": alpha,
                    "metrics": reranking_metrics(group_scores(qids, labels, scores)),
                }
            )
        for alpha in rrf_alpha_grid:
            scores = rrf_blend_by_group(qids, model_scores, feature_scores, alpha, k=60.0)
            candidates.append(
                {
                    "method": "rrf",
                    "feature": feature_name,
                    "alpha": alpha,
                    "metrics": reranking_metrics(group_scores(qids, labels, scores)),
                }
            )

    return sorted(candidates, key=lambda item: item["metrics"]["map"], reverse=True)


def evaluate_task(
    task: RerankingTask,
    args: argparse.Namespace,
    scorer: Qwen3RerankerScorer,
) -> dict[str, Any]:
    records = load_reranking_records(task.dataset_id, split=args.split)
    queries, docs, labels, qids = flatten_records(records)
    model_scores = load_or_score(
        scorer=scorer,
        task=task,
        split=args.split,
        instruction=args.instruction,
        queries=queries,
        docs=docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        cache_tag=args.cache_tag,
        score_mode=args.score_mode,
    )
    candidates = score_candidates(qids, labels, model_scores, feature_matrix(queries, docs))
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": args.split,
        "num_queries": len(records),
        "num_pairs": len(model_scores),
        "instruction": args.instruction,
        "best": candidates[0],
        "candidates": candidates[: args.keep_candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tasks", nargs="+", default=["DISCFinLLMReranking"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-tag")
    parser.add_argument(
        "--score-mode",
        choices=["probability", "logit_margin", "true_logit"],
        default="probability",
    )
    parser.add_argument("--keep-candidates", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("reports/blend_strategy_search.json"))
    args = parser.parse_args()
    args.cache_tag = args.cache_tag or model_cache_tag(args.model)

    scorer = Qwen3RerankerScorer(
        model_name=args.model,
        adapter=args.adapter,
        load_in_4bit=args.load_in_4bit,
        bf16=not args.fp16,
    )
    results = [evaluate_task(task, args, scorer) for task in resolve_tasks(args.tasks)]
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "average_map": sum(item["best"]["metrics"]["map"] for item in results) / len(results),
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
