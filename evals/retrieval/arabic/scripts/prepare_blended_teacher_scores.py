#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_teacher_score_distribution import analyze_rows
from rerank_teacher_jsonl import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend reranker teacher scores with BGE-M3 hybrid scores inside each "
            "teacher row using v56-style per-query min-max normalization."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--reranker-weight",
        type=float,
        default=0.65,
        help="Weight for reranker scores after row-level min-max normalization.",
    )
    parser.add_argument(
        "--score-scale",
        type=float,
        default=1.0,
        help="Scale blended scores before writing pos_scores/neg_scores.",
    )
    parser.add_argument(
        "--missing-hybrid",
        choices=["skip", "reranker"],
        default="skip",
        help="Skip rows with incomplete hybrid scores or fall back to reranker-only normalized scores.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def float_list(values: Any) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def blended_scores(
    reranker_scores: list[float],
    hybrid_scores: list[float],
    *,
    reranker_weight: float,
    score_scale: float,
) -> list[float]:
    if len(reranker_scores) != len(hybrid_scores):
        raise ValueError("reranker_scores and hybrid_scores must have the same length")
    reranker_norm = minmax(reranker_scores)
    hybrid_norm = minmax(hybrid_scores)
    hybrid_weight = 1.0 - reranker_weight
    return [
        score_scale * ((reranker_weight * reranker) + (hybrid_weight * hybrid))
        for reranker, hybrid in zip(reranker_norm, hybrid_norm, strict=True)
    ]


def hybrid_scores_for_row(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    pos = float_list(row.get("bge_m3_hybrid_pos_scores"))
    neg = float_list(row.get("bge_m3_hybrid_neg_scores"))
    return pos, neg


def blend_row(
    row: dict[str, Any],
    *,
    reranker_weight: float,
    score_scale: float,
    missing_hybrid: str,
) -> tuple[dict[str, Any] | None, str | None]:
    pos_scores = float_list(row.get("pos_scores"))
    neg_scores = float_list(row.get("neg_scores"))
    if not pos_scores or not neg_scores:
        return None, "missing_reranker_scores"

    hybrid_pos, hybrid_neg = hybrid_scores_for_row(row)
    expected_hybrid_count = len(pos_scores) + len(neg_scores)
    hybrid_complete = len(hybrid_pos) == len(pos_scores) and len(hybrid_neg) == len(neg_scores)
    if not hybrid_complete:
        if missing_hybrid == "skip":
            return None, "missing_hybrid_scores"
        hybrid_pos = pos_scores
        hybrid_neg = neg_scores

    reranker_all = [*pos_scores, *neg_scores]
    hybrid_all = [*hybrid_pos, *hybrid_neg]
    if len(hybrid_all) != expected_hybrid_count:
        return None, "missing_hybrid_scores"

    blended_all = blended_scores(
        reranker_all,
        hybrid_all,
        reranker_weight=reranker_weight,
        score_scale=score_scale,
    )
    output = dict(row)
    output["original_reranker_pos_scores"] = pos_scores
    output["original_reranker_neg_scores"] = neg_scores
    output["pos_scores"] = blended_all[: len(pos_scores)]
    output["neg_scores"] = blended_all[len(pos_scores) :]
    output["score_blend_teacher"] = {
        "method": "row_minmax_reranker_bge_m3_hybrid",
        "reranker_weight": reranker_weight,
        "hybrid_weight": 1.0 - reranker_weight,
        "score_scale": score_scale,
        "missing_hybrid": missing_hybrid,
        "source_script": "prepare_blended_teacher_scores.py",
    }
    return output, None


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.reranker_weight <= 1.0:
        raise ValueError("--reranker-weight must be between 0 and 1")
    if args.score_scale <= 0:
        raise ValueError("--score-scale must be positive")

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    if not args.force:
        for path in [output_jsonl, summary_json]:
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to overwrite")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped: dict[str, int] = {}
    total_rows = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(Path(args.input_jsonl)):
            total_rows += 1
            blended, reason = blend_row(
                row,
                reranker_weight=args.reranker_weight,
                score_scale=args.score_scale,
                missing_hybrid=args.missing_hybrid,
            )
            if blended is None:
                skipped[reason or "unknown"] = skipped.get(reason or "unknown", 0) + 1
                continue
            rows.append(blended)
            handle.write(json.dumps(blended, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(output_jsonl),
        "total_rows": total_rows,
        "written_rows": len(rows),
        "skipped_rows": skipped,
        "score_blend_teacher": {
            "method": "row_minmax_reranker_bge_m3_hybrid",
            "reranker_weight": args.reranker_weight,
            "hybrid_weight": 1.0 - args.reranker_weight,
            "score_scale": args.score_scale,
            "missing_hybrid": args.missing_hybrid,
        },
        "distribution": analyze_rows(rows, temperatures=[1.0, 0.5, 0.2, 0.1]) if rows else {},
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
