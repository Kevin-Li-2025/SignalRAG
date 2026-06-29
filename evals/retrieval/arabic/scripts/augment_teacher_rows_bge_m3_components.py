#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_teacher_score_distribution import analyze_rows
from rerank_miracl_candidates_bge_m3_hybrid import load_bge_model
from rerank_teacher_jsonl import write_json
from sweep_bge_m3_hybrid_weights import (
    DEFAULT_WEIGHT_SPECS,
    WeightConfig,
    fused_score,
    parse_weight_grid,
)


COMPONENT_KEYS = ("dense", "sparse", "colbert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment existing teacher JSONL rows with BGE-M3 dense/sparse/ColBERT "
            "component scores and named fusion-surface scores. This prepares "
            "student data design for multi-surface gates instead of relying on "
            "a single hybrid aggregate score."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="", help="Local model path, if pre-downloaded.")
    parser.add_argument(
        "--head-checkpoint",
        default="",
        help=(
            "Optional BGE-M3 sparse/ColBERT head checkpoint. Kept for compatibility "
            "with the shared BGE-M3 loader; v75 normally leaves this empty."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help=(
            "Fusion weight spec. Use label:dense,sparse,colbert or dense,sparse,colbert. "
            "May be repeated. Defaults to the standard diagnostic grid."
        ),
    )
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def score_pairs(
    model: object,
    pairs: list[list[str]],
    *,
    batch_size: int,
    max_passage_length: int,
) -> list[dict[str, float]]:
    components: list[dict[str, float]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        result = model.compute_score(
            batch,
            batch_size=batch_size,
            max_passage_length=max_passage_length,
            weights_for_different_modes=[1.0, 1.0, 1.0],
        )
        components.extend(
            {
                "dense": float(dense),
                "sparse": float(sparse),
                "colbert": float(colbert),
            }
            for dense, sparse, colbert in zip(
                result["dense"],
                result["sparse"],
                result["colbert"],
                strict=True,
            )
        )
    return components


def surface_scores(
    components: list[dict[str, float]],
    weight_grid: list[WeightConfig],
) -> dict[str, list[float]]:
    return {
        weights.label: [fused_score(component, weights) for component in components]
        for weights in weight_grid
    }


def split_component_scores(
    components: list[dict[str, float]],
    pos_count: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    pos_components = components[:pos_count]
    neg_components = components[pos_count:]
    return (
        {
            key: [component[key] for component in pos_components]
            for key in COMPONENT_KEYS
        },
        {
            key: [component[key] for component in neg_components]
            for key in COMPONENT_KEYS
        },
    )


def augment_row(
    row: dict[str, Any],
    components: list[dict[str, float]],
    weight_grid: list[WeightConfig],
) -> dict[str, Any]:
    pos_count = len(row.get("pos", []))
    neg_count = len(row.get("neg", []))
    if len(components) != pos_count + neg_count:
        raise ValueError("component count does not match row positive/negative count")

    pos_component_scores, neg_component_scores = split_component_scores(components, pos_count)
    all_surface_scores = surface_scores(components, weight_grid)
    surface_pos_scores = {
        label: values[:pos_count]
        for label, values in all_surface_scores.items()
    }
    surface_neg_scores = {
        label: values[pos_count:]
        for label, values in all_surface_scores.items()
    }

    output = dict(row)
    output["bge_m3_component_pos_scores"] = pos_component_scores
    output["bge_m3_component_neg_scores"] = neg_component_scores
    output["bge_m3_surface_pos_scores"] = surface_pos_scores
    output["bge_m3_surface_neg_scores"] = surface_neg_scores
    output["bge_m3_component_teacher"] = {
        "source": "augment_teacher_rows_bge_m3_components.py",
        "components": list(COMPONENT_KEYS),
        "surfaces": {
            weights.label: weights.values
            for weights in weight_grid
        },
    }
    return output


def row_pairs(row: dict[str, Any]) -> list[list[str]]:
    query = str(row.get("query", ""))
    positives = row.get("pos", [])
    negatives = row.get("neg", [])
    if not query or not positives or not negatives:
        return []
    return [[query, str(text)] for text in [*positives, *negatives]]


def surface_distribution(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    projected = []
    for row in rows:
        pos_by_surface = row.get("bge_m3_surface_pos_scores", {})
        neg_by_surface = row.get("bge_m3_surface_neg_scores", {})
        if label not in pos_by_surface or label not in neg_by_surface:
            continue
        projected.append(
            {
                **row,
                "pos_scores": pos_by_surface[label],
                "neg_scores": neg_by_surface[label],
            }
        )
    return analyze_rows(projected, temperatures=[1.0, 0.5, 0.2, 0.1])


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_passage_length <= 0:
        raise ValueError("--max-passage-length must be positive")
    if args.row_limit < 0:
        raise ValueError("--row-limit must be non-negative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    if not args.force:
        for path in [output_jsonl, summary_json]:
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to overwrite")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    weight_grid = parse_weight_grid(args.weight or DEFAULT_WEIGHT_SPECS)
    model = load_bge_model(args)

    started = time.monotonic()
    total_rows = 0
    written_rows = 0
    skipped_rows: dict[str, int] = {}
    pair_count = 0
    augmented_rows: list[dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(Path(args.input_jsonl)):
            total_rows += 1
            if args.row_limit and total_rows > args.row_limit:
                break
            pairs = row_pairs(row)
            if not pairs:
                skipped_rows["missing_query_pos_or_neg"] = (
                    skipped_rows.get("missing_query_pos_or_neg", 0) + 1
                )
                continue
            components = score_pairs(
                model,
                pairs,
                batch_size=args.batch_size,
                max_passage_length=args.max_passage_length,
            )
            augmented = augment_row(row, components, weight_grid)
            handle.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            augmented_rows.append(augmented)
            written_rows += 1
            pair_count += len(pairs)
            if written_rows % 100 == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[component-augment] rows={written_rows} pairs={pair_count} "
                    f"rate={pair_count / elapsed:.1f} pairs/s",
                    flush=True,
                )

    elapsed_seconds = time.monotonic() - started
    surface_summaries = {
        weights.label: surface_distribution(augmented_rows, weights.label)
        for weights in weight_grid
    }
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "model": args.model,
        "model_path": args.model_path or None,
        "row_limit": args.row_limit,
        "total_rows_seen": total_rows,
        "written_rows": written_rows,
        "skipped_rows": skipped_rows,
        "pairs_scored": pair_count,
        "elapsed_seconds": elapsed_seconds,
        "pairs_per_second": pair_count / max(elapsed_seconds, 1e-6),
        "weight_grid": [
            {
                "label": weights.label,
                "weights_for_different_modes": weights.values,
            }
            for weights in weight_grid
        ],
        "surface_distribution": surface_summaries,
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
        "generated_embeddings_committed": False,
        "credentials_committed": False,
    }
    write_json(summary_json, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
