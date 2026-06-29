#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_scores_by_group, lexical_score
from finmteb_sota.metrics import RankedQuery, reranking_metrics
from finmteb_sota.score_cache import load_score_cache, model_cache_tag, write_score_cache
from finmteb_sota.sequence_scoring import SequenceClassificationRerankerScorer
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
        labels_for_qid, scores_for_qid = grouped.setdefault(qid, ([], []))
        labels_for_qid.append(label)
        scores_for_qid.append(score)
    return [
        RankedQuery(query_id=qid, labels=query_labels, scores=query_scores)
        for qid, (query_labels, query_scores) in grouped.items()
    ]


def lexical_values(queries: list[str], docs: list[str]) -> list[float]:
    return [lexical_score(query, doc) for query, doc in zip(queries, docs)]


def evaluate_records(
    records: list[RerankRecord],
    model_scores: list[float],
    alpha: float,
) -> dict[str, float]:
    queries, docs, labels, qids = flatten_records(records)
    if alpha:
        final_scores = blend_scores_by_group(qids, model_scores, lexical_values(queries, docs), alpha)
    else:
        final_scores = model_scores
    return reranking_metrics(group_scores(qids, labels, final_scores))


def tune_alpha(
    records: list[RerankRecord],
    model_scores: list[float],
    grid: list[float],
) -> tuple[float, dict[str, float]]:
    best_alpha = 0.0
    best_metrics = evaluate_records(records, model_scores, 0.0)
    for alpha in grid:
        metrics = evaluate_records(records, model_scores, alpha)
        if metrics["map"] > best_metrics["map"]:
            best_alpha = alpha
            best_metrics = metrics
    return best_alpha, best_metrics


def load_or_score(
    scorer: SequenceClassificationRerankerScorer,
    task: RerankingTask,
    split: str,
    queries: list[str],
    docs: list[str],
    batch_size: int,
    max_length: int,
    cache_dir: Path,
    cache_tag: str,
) -> list[float]:
    try:
        scores, _ = load_score_cache(cache_dir, task, split, "", cache_tag)
        return scores
    except FileNotFoundError:
        pass

    scores = scorer.score(
        queries=queries,
        documents=docs,
        batch_size=batch_size,
        max_length=max_length,
    )
    write_score_cache(cache_dir, task, split, "", cache_tag, scores)
    return scores


def evaluate_task(
    task: RerankingTask,
    split: str,
    scorer: SequenceClassificationRerankerScorer,
    batch_size: int,
    max_length: int,
    lexical_grid: bool,
    alpha_override: float | None,
    cache_dir: Path,
    cache_tag: str,
) -> dict[str, object]:
    records = load_reranking_records(task.dataset_id, split=split)
    queries, docs, labels, qids = flatten_records(records)
    del labels, qids
    scores = load_or_score(
        scorer=scorer,
        task=task,
        split=split,
        queries=queries,
        docs=docs,
        batch_size=batch_size,
        max_length=max_length,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )

    alpha = alpha_override if alpha_override is not None else 0.0
    tuned_metrics = None
    if alpha_override is None and lexical_grid and split != "test":
        grid = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
        alpha, tuned_metrics = tune_alpha(records, scores, grid)

    metrics = evaluate_records(records, scores, alpha)
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": split,
        "num_queries": len(records),
        "num_pairs": len(scores),
        "alpha": alpha,
        "alpha_source": "override" if alpha_override is not None else "tuned" if tuned_metrics else "none",
        "metrics": metrics,
        "tuned_metrics": tuned_metrics,
    }


def load_alpha_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[str, float] = {}
    for item in payload.get("tasks", []):
        alpha = float(item.get("alpha", 0.0))
        overrides[item["dataset"]] = alpha
        overrides[item["leaderboard_name"]] = alpha
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=["zh"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-tag")
    parser.add_argument("--lexical-grid", action="store_true")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alpha-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/sequence_reranker_eval.json"))
    args = parser.parse_args()
    cache_tag = args.cache_tag or model_cache_tag(args.model)

    scorer = SequenceClassificationRerankerScorer(
        model_name=args.model,
        bf16=not args.fp16,
    )
    alpha_overrides = load_alpha_overrides(args.alpha_file)
    results = []
    for task in resolve_tasks(args.tasks):
        alpha_override = args.alpha
        if alpha_override is None:
            alpha_override = alpha_overrides.get(task.dataset_id, alpha_overrides.get(task.leaderboard_name))
        results.append(
            evaluate_task(
                task=task,
                split=args.split,
                scorer=scorer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                lexical_grid=args.lexical_grid,
                alpha_override=alpha_override,
                cache_dir=args.cache_dir,
                cache_tag=cache_tag,
            )
        )

    average_map = sum(item["metrics"]["map"] for item in results) / len(results)
    payload = {
        "model": args.model,
        "cache_tag": cache_tag,
        "average_map": average_map,
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
