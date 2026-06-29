#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_miracl_candidates import (
    build_cross_encoder,
    build_flag_llm_reranker,
    build_flag_reranker,
    build_qwen3_causal_reranker,
    build_sequence_classifier,
    score_pairs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score an existing MIRACL teacher JSONL with a reranker teacher. "
            "The output JSONL preserves the source rows but replaces pos_scores/"
            "neg_scores with reranker scores for distillation."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-jsonl", default="teacher_train_reranker.jsonl")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--reranker-backend",
        choices=[
            "flag",
            "flag-llm",
            "sequence-classification",
            "cross-encoder",
            "qwen3-causal",
        ],
        default="sequence-classification",
    )
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt",
        default="Given an Arabic search query, retrieve relevant Arabic passages that answer the query.",
        help="Prompt/instruction for instruction-aware CrossEncoder or Qwen3 rerankers.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, query_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" not in row or "pos" not in row or "neg" not in row:
                raise ValueError(f"{path}:{line_number} must contain query, pos, and neg")
            rows.append(row)
            if query_limit and len(rows) >= query_limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def build_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        query = str(row["query"])
        for pos_index, text in enumerate(row.get("pos", [])):
            items.append(
                {
                    "row_index": row_index,
                    "kind": "pos",
                    "item_index": pos_index,
                    "query": query,
                    "text": str(text),
                }
            )
        for neg_index, text in enumerate(row.get("neg", [])):
            items.append(
                {
                    "row_index": row_index,
                    "kind": "neg",
                    "item_index": neg_index,
                    "query": query,
                    "text": str(text),
                }
            )
    return items


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    if x_mean is None or y_mean is None:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_var * y_var)
    return numerator / denom if denom else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos_scores = [
        float(score)
        for row in rows
        for score in row.get("pos_scores", [])
    ]
    neg_scores = [
        float(score)
        for row in rows
        for score in row.get("neg_scores", [])
    ]
    best_pos_margins: list[float] = []
    mean_pos_minus_mean_neg: list[float] = []
    pos_above_max_neg = 0
    all_pos_above_all_neg = 0
    rows_with_negatives = 0
    source_counts: dict[str, int] = {}

    original_margins: list[float] = []
    reranker_margins_for_original: list[float] = []
    for row in rows:
        for source in row.get("neg_sources", []):
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
        pos = [float(score) for score in row.get("pos_scores", [])]
        neg = [float(score) for score in row.get("neg_scores", [])]
        if not pos or not neg:
            continue
        rows_with_negatives += 1
        margin = max(pos) - max(neg)
        best_pos_margins.append(margin)
        mean_pos_minus_mean_neg.append((mean(pos) or 0.0) - (mean(neg) or 0.0))
        if margin > 0:
            pos_above_max_neg += 1
        if min(pos) > max(neg):
            all_pos_above_all_neg += 1

        old_pos = [float(score) for score in row.get("bge_m3_hybrid_pos_scores", [])]
        old_neg = [float(score) for score in row.get("bge_m3_hybrid_neg_scores", [])]
        if old_pos and old_neg:
            original_margins.append(max(old_pos) - max(old_neg))
            reranker_margins_for_original.append(margin)

    return {
        "rows_with_scores": rows_with_negatives,
        "positive_pairs": len(pos_scores),
        "negative_pairs": len(neg_scores),
        "selected_negative_sources": source_counts,
        "mean_positive_score": mean(pos_scores),
        "mean_negative_score": mean(neg_scores),
        "mean_positive_minus_negative": (
            (mean(pos_scores) or 0.0) - (mean(neg_scores) or 0.0)
            if pos_scores and neg_scores
            else None
        ),
        "mean_best_positive_minus_best_negative": mean(best_pos_margins),
        "mean_row_mean_positive_minus_mean_negative": mean(mean_pos_minus_mean_neg),
        "positive_above_max_negative_rows": pos_above_max_neg,
        "positive_above_max_negative_rate": (
            pos_above_max_neg / rows_with_negatives if rows_with_negatives else None
        ),
        "all_positives_above_all_negatives_rows": all_pos_above_all_neg,
        "all_positives_above_all_negatives_rate": (
            all_pos_above_all_neg / rows_with_negatives if rows_with_negatives else None
        ),
        "hybrid_margin_vs_reranker_margin_pearson": pearson(
            original_margins,
            reranker_margins_for_original,
        ),
    }


def build_reranker(args: argparse.Namespace):
    if args.reranker_backend == "flag":
        return build_flag_reranker(args)
    if args.reranker_backend == "flag-llm":
        return build_flag_llm_reranker(args)
    if args.reranker_backend == "sequence-classification":
        return build_sequence_classifier(args)
    if args.reranker_backend == "qwen3-causal":
        return build_qwen3_causal_reranker(args)
    return build_cross_encoder(args)


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / args.output_jsonl
    if output_jsonl.exists() and not args.force:
        raise FileExistsError(f"{output_jsonl} exists; pass --force to overwrite")

    rows = load_rows(input_jsonl, args.query_limit)
    items = build_items(rows)
    print(
        f"[teacher-rerank] rows={len(rows)} pairs={len(items)} "
        f"model={args.reranker_model} backend={args.reranker_backend}",
        flush=True,
    )

    for row in rows:
        row["bge_m3_hybrid_pos_scores"] = [float(value) for value in row.get("pos_scores", [])]
        row["bge_m3_hybrid_neg_scores"] = [float(value) for value in row.get("neg_scores", [])]
        row["pos_scores"] = [None] * len(row.get("pos", []))
        row["neg_scores"] = [None] * len(row.get("neg", []))
        row["reranker_teacher"] = {
            "model": args.reranker_model,
            "backend": args.reranker_backend,
            "max_length": args.max_length,
            "source": "rerank_teacher_jsonl.py",
        }

    reranker = build_reranker(args)
    started = time.monotonic()
    scored = 0
    chunk_size = max(args.batch_size, args.score_chunk_size)
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        scores = score_pairs(
            reranker,
            args,
            [(item["query"], item["text"]) for item in chunk],
        )
        for item, score in zip(chunk, scores, strict=True):
            target = rows[item["row_index"]]
            if item["kind"] == "pos":
                target["pos_scores"][item["item_index"]] = float(score)
            else:
                target["neg_scores"][item["item_index"]] = float(score)
        scored += len(chunk)
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = scored / elapsed
        eta = (len(items) - scored) / max(rate, 1e-6)
        print(
            f"[teacher-rerank-progress] pairs={scored}/{len(items)} "
            f"rate={rate:.2f} pairs/s eta={eta / 60:.1f}m",
            flush=True,
        )

    for row in rows:
        if any(score is None for score in row["pos_scores"] + row["neg_scores"]):
            raise RuntimeError(f"Missing reranker score for query_id={row.get('query_id')}")
        row["positive"] = row["pos"][0] if row.get("pos") else ""
        row["negative"] = row["neg"][0] if row.get("neg") else ""

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - started
    summary = {
        "experiment": "miracl-ar-train-reranker-teacher-jsonl",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": output_jsonl.name,
        "raw_training_data_committed": False,
        "query_limit": args.query_limit,
        "query_count": len(rows),
        "reranker_model": args.reranker_model,
        "reranker_backend": args.reranker_backend,
        "batch_size": args.batch_size,
        "score_chunk_size": args.score_chunk_size,
        "max_length": args.max_length,
        "score_stats": summarize_rows(rows),
        "timings": {
            "elapsed_seconds": elapsed,
            "pairs_scored": len(items),
            "pairs_per_second": len(items) / max(elapsed, 1e-6),
        },
        "note": (
            "Output JSONL replaces pos_scores/neg_scores with reranker scores and "
            "preserves original BGE-M3 hybrid scores under bge_m3_hybrid_* keys. "
            "Keep JSONL out of git because it contains generated training text."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
