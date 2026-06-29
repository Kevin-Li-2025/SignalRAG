#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from finmteb_sota.data import RerankRecord, load_reranking_records
from finmteb_sota.lexical import blend_scores_by_group, lexical_score
from finmteb_sota.metrics import RankedQuery, reranking_metrics
from finmteb_sota.qwen3 import DEFAULT_INSTRUCTION
from finmteb_sota.scoring import Qwen3RerankerScorer
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


def score_with_qwen3(
    scorer: Qwen3RerankerScorer,
    queries: list[str],
    docs: list[str],
    instruction: str,
    batch_size: int,
    max_length: int,
    score_mode: str,
) -> list[float]:
    return scorer.score(
        queries=queries,
        documents=docs,
        instruction=instruction,
        batch_size=batch_size,
        max_length=max_length,
        score_mode=score_mode,
    )


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


def evaluate_task(
    task: RerankingTask,
    split: str,
    scorer: Qwen3RerankerScorer,
    instruction: str,
    batch_size: int,
    max_length: int,
    lexical_grid: bool,
    alpha_override: float | None,
    score_mode: str,
) -> dict[str, object]:
    records = load_reranking_records(task.dataset_id, split=split)
    queries, docs, labels, qids = flatten_records(records)
    del labels, qids
    scores = score_with_qwen3(
        scorer=scorer,
        queries=queries,
        docs=docs,
        instruction=instruction,
        batch_size=batch_size,
        max_length=max_length,
        score_mode=score_mode,
    )

    alpha = alpha_override if alpha_override is not None else 0.0
    tuned_metrics = None
    if alpha_override is None and lexical_grid and split != "test":
        grid = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
        alpha, tuned_metrics = tune_alpha(records, scores, grid)

    metrics = evaluate_records(records, scores, alpha)
    return {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": split,
        "num_queries": len(records),
        "num_pairs": len(scores),
        "instruction": instruction,
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


def load_tuning_overrides(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[str, dict[str, object]] = {}
    for item in payload.get("tasks", []):
        best = item.get("best", item)
        override = {
            "alpha": float(best.get("alpha", 0.0)),
            "instruction": best.get("instruction"),
        }
        overrides[item["dataset"]] = override
        overrides[item["leaderboard_name"]] = override
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tasks", nargs="+", default=["zh"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--lexical-grid", action="store_true")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alpha-file", type=Path)
    parser.add_argument("--tuning-file", type=Path)
    parser.add_argument(
        "--score-mode",
        choices=["probability", "logit_margin", "true_logit"],
        default="probability",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/eval.json"))
    args = parser.parse_args()

    resolved_tasks = resolve_tasks(args.tasks)
    scorer = Qwen3RerankerScorer(
        model_name=args.model,
        adapter=args.adapter,
        load_in_4bit=args.load_in_4bit,
        bf16=not args.fp16,
    )
    alpha_overrides = load_alpha_overrides(args.alpha_file)
    tuning_overrides = load_tuning_overrides(args.tuning_file)
    results = []
    for task in tqdm(resolved_tasks, desc="tasks"):
        tuning = tuning_overrides.get(task.dataset_id, tuning_overrides.get(task.leaderboard_name, {}))
        alpha_override = args.alpha
        if alpha_override is None:
            alpha_override = tuning.get("alpha")
        if alpha_override is None:
            alpha_override = alpha_overrides.get(task.dataset_id, alpha_overrides.get(task.leaderboard_name))
        instruction = str(tuning.get("instruction") or args.instruction)
        results.append(
            evaluate_task(
                task=task,
                split=args.split,
                scorer=scorer,
                instruction=instruction,
                batch_size=args.batch_size,
                max_length=args.max_length,
                lexical_grid=args.lexical_grid,
                alpha_override=alpha_override,
                score_mode=args.score_mode,
            )
        )

    average_map = sum(item["metrics"]["map"] for item in results) / len(results)
    payload = {"model": args.model, "adapter": args.adapter, "average_map": average_map, "tasks": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
