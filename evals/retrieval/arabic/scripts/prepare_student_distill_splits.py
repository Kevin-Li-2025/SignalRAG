#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_margin_mse_teacher_subset import prepare_row, row_quality
from rerank_teacher_jsonl import mean, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic train/held-out MarginMSE distillation splits "
            "from reranker-scored MIRACL teacher rows."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-jsonl", default="student_margin_mse_train.jsonl")
    parser.add_argument("--eval-jsonl", default="student_margin_mse_heldout.jsonl")
    parser.add_argument("--max-train-rows", type=int, default=1_000)
    parser.add_argument("--max-eval-rows", type=int, default=200)
    parser.add_argument("--heldout-ratio", type=float, default=0.10)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--min-best-margin", type=float, default=1.0)
    parser.add_argument("--min-all-margin", type=float, default=-1.0)
    parser.add_argument(
        "--label-transform",
        choices=["raw", "tanh"],
        default="tanh",
        help="Transform teacher margins before writing pos_scores/neg_scores.",
    )
    parser.add_argument("--margin-temperature", type=float, default=4.0)
    parser.add_argument("--seed", default="20260605-student-distill-v58")
    parser.add_argument(
        "--allow-nontrain-source",
        action="store_true",
        help="Allow rows whose source split metadata is present and not train.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_fraction(value: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def source_split(row: dict[str, Any]) -> str | None:
    source = row.get("source")
    if isinstance(source, dict) and source.get("split") is not None:
        return str(source["split"])
    return None


def quality_sort_key(item: tuple[dict[str, Any], dict[str, float]]) -> tuple[float, float, str]:
    row, quality = item
    return (
        quality["best_margin"],
        quality["mean_margin"],
        str(row.get("query_id", "")),
    )


def read_eligible_rows(
    args: argparse.Namespace,
) -> tuple[list[tuple[dict[str, Any], dict[str, float]]], dict[str, Any]]:
    eligible: list[tuple[dict[str, Any], dict[str, float]]] = []
    total_rows = 0
    rows_with_scores = 0
    failed_best_margin = 0
    failed_all_margin = 0
    nontrain_rows = 0

    with Path(args.input_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            split = source_split(row)
            if split and split != "train":
                nontrain_rows += 1
                if not args.allow_nontrain_source:
                    raise ValueError(
                        f"{args.input_jsonl}:{line_number} has source.split={split!r}; "
                        "student distillation should not train from dev/test rows"
                    )
            quality = row_quality(row)
            if quality is None:
                continue
            rows_with_scores += 1
            if quality["best_margin"] < args.min_best_margin:
                failed_best_margin += 1
                continue
            if quality["all_margin"] < args.min_all_margin:
                failed_all_margin += 1
                continue
            eligible.append((row, quality))

    stats = {
        "input_jsonl": args.input_jsonl,
        "total_rows": total_rows,
        "rows_with_scores": rows_with_scores,
        "eligible_rows": len(eligible),
        "nontrain_source_rows_seen": nontrain_rows,
        "filters": {
            "min_best_margin": args.min_best_margin,
            "min_all_margin": args.min_all_margin,
            "failed_best_margin": failed_best_margin,
            "failed_all_margin_after_best_margin": failed_all_margin,
        },
    }
    return eligible, stats


def split_rows(
    rows: list[tuple[dict[str, Any], dict[str, float]]],
    *,
    heldout_ratio: float,
    seed: str,
    max_train_rows: int,
    max_eval_rows: int,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, float]]],
    list[tuple[dict[str, Any], dict[str, float]]],
]:
    if not 0.0 < heldout_ratio < 1.0:
        raise ValueError("--heldout-ratio must be between 0 and 1")

    train: list[tuple[dict[str, Any], dict[str, float]]] = []
    heldout: list[tuple[dict[str, Any], dict[str, float]]] = []
    for item in rows:
        row, _quality = item
        query_key = str(row.get("query_id") or row.get("query"))
        if stable_fraction(query_key, seed) < heldout_ratio:
            heldout.append(item)
        else:
            train.append(item)

    train.sort(key=quality_sort_key, reverse=True)
    heldout.sort(key=quality_sort_key, reverse=True)
    if max_train_rows:
        train = train[:max_train_rows]
    if max_eval_rows:
        heldout = heldout[:max_eval_rows]
    if not train:
        raise ValueError("No train rows selected")
    if not heldout:
        raise ValueError("No held-out rows selected; increase --heldout-ratio or --max-eval-rows")
    return train, heldout


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_selected(
    selected: list[tuple[dict[str, Any], dict[str, float]]],
    output_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    qualities = [quality for _row, quality in selected]
    margins = [
        margin
        for row in output_rows
        for margin in row.get("target_margins", [])
    ]
    query_ids = [str(row.get("query_id") or row.get("query")) for row, _quality in selected]
    return {
        "rows": len(output_rows),
        "expanded_margin_mse_triples": sum(len(row.get("neg", [])) for row in output_rows),
        "unique_queries": len(set(query_ids)),
        "quality": {
            "mean_best_margin": mean([quality["best_margin"] for quality in qualities]),
            "mean_all_margin": mean([quality["all_margin"] for quality in qualities]),
            "mean_mean_margin": mean([quality["mean_margin"] for quality in qualities]),
            "min_best_margin": min((quality["best_margin"] for quality in qualities), default=None),
            "min_all_margin": min((quality["all_margin"] for quality in qualities), default=None),
        },
        "target_margin_stats": {
            "mean": mean(margins),
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
    }


def prepare_splits(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible, input_stats = read_eligible_rows(args)
    train_selected, eval_selected = split_rows(
        eligible,
        heldout_ratio=args.heldout_ratio,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
    )
    prepare_kwargs = {
        "negatives_per_query": args.negatives_per_query,
        "label_transform": args.label_transform,
        "margin_temperature": args.margin_temperature,
    }
    train_rows = [prepare_row(row, **prepare_kwargs) for row, _quality in train_selected]
    eval_rows = [prepare_row(row, **prepare_kwargs) for row, _quality in eval_selected]
    train_query_ids = {str(row.get("query_id") or row.get("query")) for row in train_rows}
    eval_query_ids = {str(row.get("query_id") or row.get("query")) for row in eval_rows}
    overlap = train_query_ids.intersection(eval_query_ids)
    if overlap:
        raise ValueError(f"Train/eval query overlap detected: {sorted(overlap)[:5]}")

    summary = {
        **input_stats,
        "splitter": {
            "seed": args.seed,
            "heldout_ratio": args.heldout_ratio,
            "max_train_rows": args.max_train_rows,
            "max_eval_rows": args.max_eval_rows,
            "query_overlap": 0,
        },
        "distillation_target": {
            "loss": "MarginMSELoss",
            "negatives_per_query": args.negatives_per_query,
            "label_transform": args.label_transform,
            "margin_temperature": args.margin_temperature,
            "teacher": "MIRACL train BGE-M3 hybrid r100 candidates rescored by BGE reranker v2-m3",
        },
        "train": summarize_selected(train_selected, train_rows),
        "heldout": summarize_selected(eval_selected, eval_rows),
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
    }
    return train_rows, eval_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_jsonl
    eval_path = output_dir / args.eval_jsonl
    summary_path = output_dir / "student_distill_split_summary.json"
    if not args.force:
        for path in (train_path, eval_path, summary_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to overwrite")

    train_rows, eval_rows, summary = prepare_splits(args)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    summary["outputs"] = {
        "train_jsonl": train_path.name,
        "eval_jsonl": eval_path.name,
        "summary_json": summary_path.name,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
