#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_feature_by_group, lexical_feature_values, rrf_blend_by_group
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


def load_strategies(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategies: dict[str, dict[str, Any]] = {}
    for item in payload.get("tasks", []):
        strategy = item.get("best", item)
        strategies[item["dataset"]] = strategy
        strategies[item["leaderboard_name"]] = strategy
    return strategies


def apply_strategy(
    qids: list[str],
    model_scores: list[float],
    features: dict[str, list[float]],
    strategy: dict[str, Any] | None,
) -> tuple[list[float], dict[str, Any]]:
    if not strategy:
        return model_scores, {"method": "model", "feature": None, "alpha": 0.0}

    method = strategy.get("method", "model")
    feature_name = strategy.get("feature")
    alpha = float(strategy.get("alpha", 0.0))
    if method == "model" or not feature_name:
        return model_scores, {"method": "model", "feature": None, "alpha": 0.0}
    if feature_name not in features:
        raise KeyError(f"Unknown feature in strategy: {feature_name}")
    if method == "zscore_linear":
        return (
            blend_feature_by_group(qids, model_scores, features[feature_name], alpha),
            {"method": method, "feature": feature_name, "alpha": alpha},
        )
    if method == "rrf":
        return (
            rrf_blend_by_group(qids, model_scores, features[feature_name], alpha, k=60.0),
            {"method": method, "feature": feature_name, "alpha": alpha},
        )
    raise KeyError(f"Unknown strategy method: {method}")


def evaluate_task(
    task: RerankingTask,
    args: argparse.Namespace,
    scorer: Qwen3RerankerScorer,
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = load_reranking_records(task.dataset_id, split=args.split)
    queries, docs, labels, qids = flatten_records(records)
    strategy = strategies.get(task.dataset_id, strategies.get(task.leaderboard_name))
    instruction = args.instruction
    model_scores = load_or_score(
        scorer=scorer,
        task=task,
        split=args.split,
        instruction=instruction,
        queries=queries,
        docs=docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        cache_tag=args.cache_tag,
        score_mode=args.score_mode,
    )
    scores, applied_strategy = apply_strategy(qids, model_scores, feature_matrix(queries, docs), strategy)
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": args.split,
        "num_queries": len(records),
        "num_pairs": len(model_scores),
        "instruction": instruction,
        "strategy": applied_strategy,
        "metrics": reranking_metrics(group_scores(qids, labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tasks", nargs="+", default=["zh"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--strategy-file", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, default=Path("reports/eval_blend_strategy.json"))
    args = parser.parse_args()
    args.cache_tag = args.cache_tag or model_cache_tag(args.model)

    scorer = Qwen3RerankerScorer(
        model_name=args.model,
        adapter=args.adapter,
        load_in_4bit=args.load_in_4bit,
        bf16=not args.fp16,
    )
    strategies = load_strategies(args.strategy_file)
    results = [
        evaluate_task(task, args, scorer, strategies)
        for task in resolve_tasks(args.tasks)
    ]
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "strategy_file": str(args.strategy_file),
        "average_map": sum(item["metrics"]["map"] for item in results) / len(results),
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
