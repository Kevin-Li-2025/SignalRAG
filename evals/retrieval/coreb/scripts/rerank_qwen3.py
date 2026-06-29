from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from coreb_sota.data import doc_text, load_task, query_text, task_names
from coreb_sota.metrics import evaluate_run, weighted_average
from coreb_sota.qwen3 import DEFAULT_INSTRUCTION, TASK_INSTRUCTIONS
from coreb_sota.scoring import Qwen3RerankerScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank CoREB candidates with Qwen3 true-logit scoring.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--split", default="release_v2603")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--score-mode", choices=["probability", "logit_margin", "true_logit"], default="true_logit")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--instruction-mode", choices=["generic", "task"], default="generic")
    parser.add_argument("--prompt-style", choices=["qwen3", "coreb"], default="qwen3")
    parser.add_argument("--fusion-alpha", type=float, default=0.0, help="RRF weight for original candidate rank.")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-flush-pairs", type=int, default=128)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/qwen3_rerank_release_v2603.json"))
    return parser.parse_args()


def instruction_for(args: argparse.Namespace, task_name: str) -> str:
    if args.instruction_mode == "task":
        return TASK_INSTRUCTIONS[task_name]
    return args.instruction


def cache_path(args: argparse.Namespace, task_name: str) -> Path:
    model_tag = args.model.replace("/", "--")
    instruction_hash = hashlib.sha1(instruction_for(args, task_name).encode("utf-8")).hexdigest()[:10]
    return args.cache_dir / (
        f"{task_name}_{args.split}_{model_tag}_{args.score_mode}_"
        f"{args.instruction_mode}_{args.prompt_style}_{instruction_hash}_top{args.top_n}.json"
    )


def reciprocal_rank(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def rerank_scores(
    original_scores: dict[str, float],
    reranker_scores: dict[str, float],
    fusion_alpha: float,
    rrf_k: int,
) -> dict[str, float]:
    rerank_order = [doc_id for doc_id, _ in sorted(reranker_scores.items(), key=lambda x: x[1], reverse=True)]
    original_order = [doc_id for doc_id, _ in sorted(original_scores.items(), key=lambda x: x[1], reverse=True)]
    original_rank = {doc_id: idx + 1 for idx, doc_id in enumerate(original_order)}
    scores: dict[str, float] = {}
    for idx, doc_id in enumerate(rerank_order, start=1):
        score = reciprocal_rank(idx, rrf_k)
        if fusion_alpha:
            score += fusion_alpha * reciprocal_rank(original_rank[doc_id], rrf_k)
        scores[doc_id] = score
    return scores


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    candidate_report = json.loads(args.candidate_run.read_text())
    candidate_runs: dict[str, dict[str, dict[str, float]]] = candidate_report["runs"]

    scorer: Qwen3RerankerScorer | None = None
    if not args.cache_only:
        scorer = Qwen3RerankerScorer(args.model, load_in_4bit=args.load_in_4bit)

    task_metrics: dict[str, dict[str, Any]] = {}
    output_runs: dict[str, dict[str, dict[str, float]]] = {}
    score_files: dict[str, str] = {}

    for task_name in task_names(args.tasks):
        task = load_task(args.data_dir, args.split, task_name)
        candidate_run = candidate_runs[task_name]
        query_ids = list(task.queries)
        if args.limit_queries:
            query_ids = query_ids[: args.limit_queries]

        score_file = cache_path(args, task_name)
        score_files[task_name] = str(score_file)
        if score_file.exists():
            pair_scores = json.loads(score_file.read_text())
        elif args.cache_only:
            raise FileNotFoundError(f"Missing score cache: {score_file}")
        else:
            pair_scores = {}

        if not args.cache_only:
            assert scorer is not None
            task_instruction = instruction_for(args, task_name)
            queries: list[str] = []
            docs: list[str] = []
            pair_keys: list[str] = []
            for query_id in query_ids:
                excluded = task.exclude_doc_ids.get(query_id, set())
                ranked = [
                    item
                    for item in sorted(candidate_run[query_id].items(), key=lambda x: x[1], reverse=True)
                    if item[0] not in excluded
                ][: args.top_n]
                for doc_id, _ in ranked:
                    key = f"{query_id}\t{doc_id}"
                    if key in pair_scores:
                        continue
                    pair_keys.append(key)
                    queries.append(query_text(task, query_id))
                    docs.append(doc_text(task, doc_id))
            flush_size = max(args.batch_size, args.cache_flush_pairs)
            for start in range(0, len(pair_keys), flush_size):
                end = min(start + flush_size, len(pair_keys))
                scores = scorer.score(
                    queries[start:end],
                    docs[start:end],
                    instruction=task_instruction,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    score_mode=args.score_mode,
                    prompt_style=args.prompt_style,
                    show_progress=False,
                )
                pair_scores.update(dict(zip(pair_keys[start:end], scores)))
                write_json_atomic(score_file, pair_scores)

        reranked_run: dict[str, dict[str, float]] = {}
        for query_id in query_ids:
            excluded = task.exclude_doc_ids.get(query_id, set())
            original = dict(
                item
                for item in sorted(candidate_run[query_id].items(), key=lambda x: x[1], reverse=True)
                if item[0] not in excluded
            )
            original = dict(list(original.items())[: args.top_n])
            scored = {
                doc_id: float(pair_scores[f"{query_id}\t{doc_id}"])
                for doc_id in original
                if f"{query_id}\t{doc_id}" in pair_scores
            }
            reranked_run[query_id] = rerank_scores(original, scored, args.fusion_alpha, args.rrf_k)

        task_qrels = {query_id: task.qrels[query_id] for query_id in query_ids}
        task_exclusions = {
            query_id: task.exclude_doc_ids[query_id]
            for query_id in query_ids
            if query_id in task.exclude_doc_ids
        }
        metrics = evaluate_run(task_qrels, reranked_run, k=args.eval_k, exclude_doc_ids=task_exclusions)
        task_metrics[task_name] = metrics
        output_runs[task_name] = reranked_run

    metric = f"ndcg@{args.eval_k}"
    report = {
        "method": "qwen3_true_logit_rerank",
        "model": args.model,
        "split": args.split,
        "candidate_run": str(args.candidate_run),
        "top_n": args.top_n,
        "eval_k": args.eval_k,
        "score_mode": args.score_mode,
        "fusion_alpha": args.fusion_alpha,
        "rrf_k": args.rrf_k,
        "instruction_mode": args.instruction_mode,
        "prompt_style": args.prompt_style,
        "instructions": {task_name: instruction_for(args, task_name) for task_name in task_names(args.tasks)},
        "elapsed_seconds": time.perf_counter() - started,
        "score_files": score_files,
        "task_metrics": task_metrics,
        "overall": {
            metric: weighted_average(task_metrics, metric),
            f"recall@{args.eval_k}": weighted_average(task_metrics, f"recall@{args.eval_k}"),
            f"map@{args.eval_k}": weighted_average(task_metrics, f"map@{args.eval_k}"),
        },
        "runs": output_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
