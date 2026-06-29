#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_scores_by_group, lexical_score
from finmteb_sota.metrics import RankedQuery, reranking_metrics
from finmteb_sota.scoring import Qwen3RerankerScorer
from finmteb_sota.tasks import resolve_tasks


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


def metrics_for_scores(
    qids: list[str],
    labels: list[int],
    scores: list[float],
) -> dict[str, float]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for qid, label, score in zip(qids, labels, scores):
        q_labels, q_scores = grouped.setdefault(qid, ([], []))
        q_labels.append(label)
        q_scores.append(score)
    return reranking_metrics(
        [
            RankedQuery(query_id=qid, labels=query_labels, scores=query_scores)
            for qid, (query_labels, query_scores) in grouped.items()
        ]
    )


def tune_alpha(
    queries: list[str],
    docs: list[str],
    labels: list[int],
    qids: list[str],
    model_scores: list[float],
    grid: list[float],
) -> tuple[float, dict[str, float]]:
    lexical_scores = [lexical_score(query, doc) for query, doc in zip(queries, docs)]
    best_alpha = 0.0
    best_metrics = metrics_for_scores(qids, labels, model_scores)
    for alpha in grid:
        scores = blend_scores_by_group(qids, model_scores, lexical_scores, alpha)
        metrics = metrics_for_scores(qids, labels, scores)
        if metrics["map"] > best_metrics["map"]:
            best_alpha = alpha
            best_metrics = metrics
    return best_alpha, best_metrics


def read_instructions(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--instruction-file", type=Path, default=Path("configs/instructions_zh.txt"))
    parser.add_argument("--tasks", nargs="+", default=["zh"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/instruction_alpha_search.json"))
    args = parser.parse_args()

    instructions = read_instructions(args.instruction_file)
    scorer = Qwen3RerankerScorer(
        model_name=args.model,
        adapter=args.adapter,
        load_in_4bit=args.load_in_4bit,
        bf16=not args.fp16,
    )
    grid = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    results = []
    for task in resolve_tasks(args.tasks):
        records = load_reranking_records(task.dataset_id, split=args.split)
        queries, docs, labels, qids = flatten_records(records)
        candidates = []
        for instruction in tqdm(instructions, desc=task.leaderboard_name):
            scores = scorer.score(
                queries=queries,
                documents=docs,
                instruction=instruction,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            alpha, metrics = tune_alpha(queries, docs, labels, qids, scores, grid)
            candidates.append({"instruction": instruction, "alpha": alpha, "metrics": metrics})
        best = max(candidates, key=lambda item: item["metrics"]["map"])
        results.append(
            {
                "dataset": task.dataset_id,
                "leaderboard_name": task.leaderboard_name,
                "split": args.split,
                "num_queries": len(records),
                "best": best,
                "candidates": candidates,
            }
        )

    average_map = sum(item["best"]["metrics"]["map"] for item in results) / len(results)
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "average_map": average_map,
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
