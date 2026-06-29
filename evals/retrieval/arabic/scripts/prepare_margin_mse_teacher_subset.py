#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_teacher_jsonl import mean, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a filtered, score-normalized teacher JSONL for MarginMSE "
            "smoke training from reranker-scored MIRACL teacher rows."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-jsonl", default="teacher_train_margin_mse.jsonl")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--min-best-margin", type=float, default=1.0)
    parser.add_argument("--min-all-margin", type=float, default=0.0)
    parser.add_argument(
        "--label-transform",
        choices=["raw", "tanh"],
        default="tanh",
        help="Transform teacher margins before writing pos_scores/neg_scores.",
    )
    parser.add_argument(
        "--margin-temperature",
        type=float,
        default=4.0,
        help="Temperature for tanh margin transform.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda index: scores[index])


def transformed_margin(margin: float, transform: str, temperature: float) -> float:
    if transform == "raw":
        return margin
    if temperature <= 0:
        raise ValueError("--margin-temperature must be positive")
    return math.tanh(margin / temperature)


def row_quality(row: dict[str, Any]) -> dict[str, float] | None:
    pos_scores = [float(score) for score in row.get("pos_scores", [])]
    neg_scores = [float(score) for score in row.get("neg_scores", [])]
    if not pos_scores or not neg_scores:
        return None
    return {
        "best_pos": max(pos_scores),
        "min_pos": min(pos_scores),
        "max_neg": max(neg_scores),
        "mean_pos": mean(pos_scores) or 0.0,
        "mean_neg": mean(neg_scores) or 0.0,
        "best_margin": max(pos_scores) - max(neg_scores),
        "all_margin": min(pos_scores) - max(neg_scores),
        "mean_margin": (mean(pos_scores) or 0.0) - (mean(neg_scores) or 0.0),
    }


def prepare_row(
    row: dict[str, Any],
    *,
    negatives_per_query: int,
    label_transform: str,
    margin_temperature: float,
) -> dict[str, Any]:
    pos_scores = [float(score) for score in row["pos_scores"]]
    neg_scores = [float(score) for score in row["neg_scores"]]
    pos_idx = best_index(pos_scores)
    negative_order = sorted(range(len(neg_scores)), key=lambda index: neg_scores[index], reverse=True)
    negative_order = negative_order[:negatives_per_query]

    pos_score = pos_scores[pos_idx]
    margins = [
        transformed_margin(pos_score - neg_scores[index], label_transform, margin_temperature)
        for index in negative_order
    ]
    output_row = {
        "query_id": row.get("query_id"),
        "query": row["query"],
        "pos_doc_ids": [row.get("pos_doc_ids", [None] * len(pos_scores))[pos_idx]],
        "pos": [row["pos"][pos_idx]],
        "pos_scores": [1.0],
        "neg_doc_ids": [row.get("neg_doc_ids", [None] * len(neg_scores))[index] for index in negative_order],
        "neg": [row["neg"][index] for index in negative_order],
        "neg_scores": [1.0 - margin for margin in margins],
        "neg_sources": [
            row.get("neg_sources", ["unknown"] * len(neg_scores))[index]
            for index in negative_order
        ],
        "positive": row["pos"][pos_idx],
        "negative": row["neg"][negative_order[0]],
        "original_reranker_pos_score": pos_score,
        "original_reranker_neg_scores": [neg_scores[index] for index in negative_order],
        "target_margins": margins,
        "teacher": {
            "source": "prepare_margin_mse_teacher_subset.py",
            "label_transform": label_transform,
            "margin_temperature": margin_temperature,
            "source_teacher": row.get("reranker_teacher", {}),
        },
        "source": row.get("source", {}),
    }
    if "bge_m3_hybrid_pos_scores" in row:
        output_row["bge_m3_hybrid_pos_score"] = row["bge_m3_hybrid_pos_scores"][pos_idx]
    if "bge_m3_hybrid_neg_scores" in row:
        output_row["bge_m3_hybrid_neg_scores"] = [
            row["bge_m3_hybrid_neg_scores"][index] for index in negative_order
        ]
    return output_row


def load_and_filter_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[tuple[dict[str, Any], dict[str, float]]] = []
    total_rows = 0
    rows_with_scores = 0
    failed_best_margin = 0
    failed_all_margin = 0
    with Path(args.input_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
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
            selected.append((row, quality))

    selected.sort(
        key=lambda item: (
            item[1]["best_margin"],
            item[1]["mean_margin"],
            str(item[0].get("query_id", "")),
        ),
        reverse=True,
    )
    if args.max_rows:
        selected = selected[: args.max_rows]

    output_rows = [
        prepare_row(
            row,
            negatives_per_query=args.negatives_per_query,
            label_transform=args.label_transform,
            margin_temperature=args.margin_temperature,
        )
        for row, _quality in selected
    ]
    qualities = [quality for _row, quality in selected]
    margins = [
        margin
        for row in output_rows
        for margin in row["target_margins"]
    ]
    summary = {
        "input_jsonl": args.input_jsonl,
        "total_rows": total_rows,
        "rows_with_scores": rows_with_scores,
        "eligible_rows": len(qualities),
        "output_rows": len(output_rows),
        "expanded_margin_mse_triples": sum(len(row["neg"]) for row in output_rows),
        "filters": {
            "max_rows": args.max_rows,
            "negatives_per_query": args.negatives_per_query,
            "min_best_margin": args.min_best_margin,
            "min_all_margin": args.min_all_margin,
            "failed_best_margin": failed_best_margin,
            "failed_all_margin_after_best_margin": failed_all_margin,
        },
        "label_transform": {
            "name": args.label_transform,
            "margin_temperature": args.margin_temperature,
        },
        "selected_quality": {
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
        "raw_training_data_committed": False,
    }
    return output_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / args.output_jsonl
    if output_jsonl.exists() and not args.force:
        raise FileExistsError(f"{output_jsonl} exists; pass --force to overwrite")

    output_rows, summary = load_and_filter_rows(args)
    if not output_rows:
        raise ValueError("No rows matched the requested filters")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["output_jsonl"] = output_jsonl.name
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
