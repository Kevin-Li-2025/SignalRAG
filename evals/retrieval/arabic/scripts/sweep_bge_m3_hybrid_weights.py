#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_miracl_candidates import (  # noqa: E402
    DEFAULT_REVISION,
    load_candidate_texts,
    load_relevance,
    mean_metrics,
    metrics_for_query,
    read_trec_run,
    write_json,
)
from rerank_miracl_candidates_bge_m3_hybrid import load_bge_model  # noqa: E402


DEFAULT_WEIGHT_SPECS = [
    "model_card:0.4,0.2,0.4",
    "training_default:1.0,0.3,1.0",
    "dense_heavy:0.6,0.2,0.2",
    "sparse_heavy:0.25,0.5,0.25",
    "colbert_heavy:0.25,0.15,0.6",
    "dense_sparse:0.7,0.3,0.0",
    "dense_colbert:0.5,0.0,0.5",
    "sparse_colbert:0.0,0.33,0.67",
    "colbert_only:0.0,0.0,1.0",
]


@dataclass(frozen=True)
class WeightConfig:
    label: str
    dense: float
    sparse: float
    colbert: float

    @property
    def values(self) -> list[float]:
        return [self.dense, self.sparse, self.colbert]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a fixed MIRACL candidate pool once with BGE-M3 component scores, "
            "then sweep dense/sparse/ColBERT fusion weights for base and optional "
            "student model/head checkpoints."
        )
    )
    parser.add_argument("--candidate-run-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--student-model-path",
        default="",
        help=(
            "Optional full student model directory. Use this for full-encoder "
            "or adapter-merged checkpoints; --student-head-checkpoint only swaps "
            "sparse/ColBERT heads."
        ),
    )
    parser.add_argument("--student-head-checkpoint", default="")
    parser.add_argument("--student-label", default="student")
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument(
        "--query-id-file",
        default="",
        help=(
            "Optional newline-delimited query id file. When set, the sweep uses "
            "only those candidate-run queries, preserving file order after "
            "dropping blank lines and comments."
        ),
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help=(
            "Start offset for deterministic query splitting. For example, "
            "--query-stride 2 --query-offset 0 selects the even split and "
            "--query-offset 1 selects the odd split."
        ),
    )
    parser.add_argument(
        "--query-stride",
        type=int,
        default=1,
        help="Stride for deterministic query splitting after optional query-id-file filtering.",
    )
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument(
        "--write-per-query",
        action="store_true",
        help=(
            "Write per_query_metrics.jsonl with query-level base/student metrics "
            "for each weight label. This is useful for failure analysis after a "
            "student checkpoint misses the multi-split gate."
        ),
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help=(
            "Fusion weight spec. Use label:dense,sparse,colbert or dense,sparse,colbert. "
            "May be repeated. Defaults to a compact diagnostic grid."
        ),
    )
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("_") or "weights"


def parse_weight_spec(spec: str) -> WeightConfig:
    label = ""
    values = spec
    if ":" in spec:
        label, values = spec.split(":", 1)
    parts = [part.strip() for part in values.split(",")]
    if len(parts) != 3:
        raise ValueError(f"weight spec {spec!r} must contain three comma-separated values")
    dense, sparse, colbert = (float(part) for part in parts)
    if dense < 0 or sparse < 0 or colbert < 0:
        raise ValueError(f"weight spec {spec!r} must be non-negative")
    if dense == 0 and sparse == 0 and colbert == 0:
        raise ValueError(f"weight spec {spec!r} must not be all zero")
    if not label:
        label = f"d{dense:g}_s{sparse:g}_c{colbert:g}"
    return WeightConfig(safe_label(label), dense, sparse, colbert)


def parse_weight_grid(specs: list[str]) -> list[WeightConfig]:
    seen: set[str] = set()
    configs: list[WeightConfig] = []
    for spec in specs or DEFAULT_WEIGHT_SPECS:
        config = parse_weight_spec(spec)
        if config.label in seen:
            raise ValueError(f"duplicate weight label {config.label!r}")
        seen.add(config.label)
        configs.append(config)
    return configs


def read_query_id_file(path: Path) -> list[str]:
    query_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        query_ids.append(value)
    return query_ids


def select_query_ids(
    candidate_query_ids: list[str],
    *,
    query_limit: int,
    query_id_file: str = "",
    query_offset: int = 0,
    query_stride: int = 1,
) -> list[str]:
    if query_stride <= 0:
        raise ValueError("--query-stride must be positive")
    if query_offset < 0:
        raise ValueError("--query-offset must be non-negative")

    available = set(candidate_query_ids)
    if query_id_file:
        query_ids = read_query_id_file(Path(query_id_file))
        missing = [query_id for query_id in query_ids if query_id not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"{len(missing)} query ids from {query_id_file!r} are not in the "
                f"candidate run; first missing ids: {preview}"
            )
    else:
        query_ids = sorted(candidate_query_ids)

    query_ids = query_ids[query_offset::query_stride]
    if query_limit:
        query_ids = query_ids[:query_limit]
    if not query_ids:
        raise ValueError("query selection produced no queries")
    return query_ids


def fused_score(components: dict[str, float], weights: WeightConfig) -> float:
    return (
        weights.dense * float(components["dense"])
        + weights.sparse * float(components["sparse"])
        + weights.colbert * float(components["colbert"])
    )


def ranked_doc_ids_from_components(
    rows: list[dict[str, Any]],
    scores: list[dict[str, float]],
    *,
    weights: WeightConfig,
    top_k: int,
) -> list[str]:
    ranked = sorted(
        zip(rows, scores, strict=True),
        key=lambda item: fused_score(item[1], weights),
        reverse=True,
    )
    return [str(row["docid"]) for row, _score in ranked[:top_k]]


def query_metrics_from_components(
    rows: list[dict[str, Any]],
    relevant_doc_ids: set[str],
    scores: list[dict[str, float]],
    *,
    weights: WeightConfig,
    top_k: int,
    metric_k: int,
) -> dict[str, float]:
    metrics = metrics_for_query(
        ranked_doc_ids_from_components(
            rows,
            scores,
            weights=weights,
            top_k=top_k,
        ),
        relevant_doc_ids,
        metric_k,
    )
    metrics["main_score"] = metrics["ndcg_at_10"]
    return metrics


def patch_distributed_tensor_for_peft_model_loading() -> None:
    """Let Transformers/PEFT adapter loading probe DTensor on older torch stacks."""

    try:
        import torch.distributed as dist
        from run_flagembedding_m3_no_ddp import patch_torch_distributed_tensor_for_peft
    except Exception:
        return

    patch_torch_distributed_tensor_for_peft(dist)


def metrics_from_components(
    candidates: dict[str, list[dict[str, Any]]],
    qrels: dict[str, set[str]],
    component_scores_by_query: dict[str, list[dict[str, float]]],
    query_ids: list[str],
    *,
    weights: WeightConfig,
    top_k: int,
    metric_k: int,
) -> dict[str, float]:
    per_query = []
    for query_id in query_ids:
        rows = candidates[query_id]
        scores = component_scores_by_query[query_id]
        per_query.append(
            query_metrics_from_components(
                rows,
                qrels[query_id],
                scores,
                weights=weights,
                top_k=top_k,
                metric_k=metric_k,
            )
        )
    metrics = mean_metrics(per_query)
    metrics["main_score"] = metrics["ndcg_at_10"]
    return metrics


def score_components_for_model(
    args: argparse.Namespace,
    *,
    label: str,
    model_path: str,
    head_checkpoint: str,
    candidates: dict[str, list[dict[str, Any]]],
    query_text_by_id: dict[str, str],
    candidate_text_by_id: dict[str, str],
    query_ids: list[str],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    model_args = argparse.Namespace(
        model=args.model,
        model_path=model_path,
        head_checkpoint=head_checkpoint,
        use_fp16=args.use_fp16,
    )
    if model_path:
        patch_distributed_tensor_for_peft_model_loading()
    model = load_bge_model(model_args)
    component_scores_by_query: dict[str, list[dict[str, float]]] = {}
    pair_count = 0
    started = time.monotonic()
    for query_index, query_id in enumerate(query_ids, start=1):
        rows = candidates[query_id]
        query = query_text_by_id[query_id]
        pairs = [[query, candidate_text_by_id[str(row["docid"])]] for row in rows]
        components: list[dict[str, float]] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            result = model.compute_score(
                batch,
                batch_size=args.batch_size,
                max_passage_length=args.max_passage_length,
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
        component_scores_by_query[query_id] = components
        pair_count += len(pairs)
        if query_index % 25 == 0 or query_index == len(query_ids):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[weight-sweep-components] model={label} "
                f"queries={query_index}/{len(query_ids)} pairs={pair_count} "
                f"rate={pair_count / elapsed:.1f} pairs/s",
                flush=True,
            )
    elapsed_seconds = time.monotonic() - started
    return component_scores_by_query, {
        "label": label,
        "model_path": model_path or None,
        "head_checkpoint": head_checkpoint or None,
        "component_elapsed_seconds": elapsed_seconds,
        "pairs": pair_count,
        "pairs_per_second": pair_count / max(elapsed_seconds, 1e-6),
    }


def summarize_model_results(
    *,
    model_info: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    qrels: dict[str, set[str]],
    component_scores_by_query: dict[str, list[dict[str, float]]],
    query_ids: list[str],
    weight_grid: list[WeightConfig],
    top_k: int,
    metric_k: int,
) -> dict[str, Any]:
    results = []
    for weights in weight_grid:
        metrics = metrics_from_components(
            candidates,
            qrels,
            component_scores_by_query,
            query_ids,
            weights=weights,
            top_k=top_k,
            metric_k=metric_k,
        )
        results.append(
            {
                "weight_label": weights.label,
                "weights_for_different_modes": weights.values,
                "metrics": metrics,
            }
        )
    best = max(results, key=lambda row: row["metrics"]["main_score"])
    return {**model_info, "results": results, "best": best}


def compare_models(base: dict[str, Any], student: dict[str, Any] | None) -> list[dict[str, Any]]:
    if student is None:
        return []
    base_by_label = {row["weight_label"]: row for row in base["results"]}
    comparisons = []
    for student_row in student["results"]:
        label = student_row["weight_label"]
        base_row = base_by_label[label]
        comparisons.append(
            {
                "weight_label": label,
                "weights_for_different_modes": student_row["weights_for_different_modes"],
                "base_main_score": base_row["metrics"]["main_score"],
                "student_main_score": student_row["metrics"]["main_score"],
                "delta": (
                    student_row["metrics"]["main_score"]
                    - base_row["metrics"]["main_score"]
                ),
            }
        )
    return comparisons


def per_query_comparisons(
    *,
    candidates: dict[str, list[dict[str, Any]]],
    qrels: dict[str, set[str]],
    base_components: dict[str, list[dict[str, float]]],
    student_components: dict[str, list[dict[str, float]]] | None,
    query_ids: list[str],
    weight_grid: list[WeightConfig],
    top_k: int,
    metric_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        candidate_rows = candidates[query_id]
        relevant_doc_ids = qrels[query_id]
        for weights in weight_grid:
            base_metrics = query_metrics_from_components(
                candidate_rows,
                relevant_doc_ids,
                base_components[query_id],
                weights=weights,
                top_k=top_k,
                metric_k=metric_k,
            )
            row: dict[str, Any] = {
                "query_id": query_id,
                "weight_label": weights.label,
                "weights_for_different_modes": weights.values,
                "base": {
                    "metrics": base_metrics,
                    "top_docids": ranked_doc_ids_from_components(
                        candidate_rows,
                        base_components[query_id],
                        weights=weights,
                        top_k=metric_k,
                    ),
                },
            }
            if student_components is not None:
                student_metrics = query_metrics_from_components(
                    candidate_rows,
                    relevant_doc_ids,
                    student_components[query_id],
                    weights=weights,
                    top_k=top_k,
                    metric_k=metric_k,
                )
                row["student"] = {
                    "metrics": student_metrics,
                    "top_docids": ranked_doc_ids_from_components(
                        candidate_rows,
                        student_components[query_id],
                        weights=weights,
                        top_k=metric_k,
                    ),
                }
                row["delta"] = student_metrics["main_score"] - base_metrics["main_score"]
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_grid = parse_weight_grid(args.weight)

    candidates = read_trec_run(Path(args.candidate_run_file), args.candidate_depth)
    query_ids = select_query_ids(
        list(candidates),
        query_limit=args.query_limit,
        query_id_file=args.query_id_file,
        query_offset=args.query_offset,
        query_stride=args.query_stride,
    )
    candidates = {query_id: candidates[query_id] for query_id in query_ids}
    query_text_by_id, qrels = load_relevance(
        args.subset,
        args.split,
        args.revision,
        query_ids,
    )
    needed_doc_ids = {str(row["docid"]) for rows in candidates.values() for row in rows}
    candidate_text_by_id = load_candidate_texts(
        args.subset,
        args.split,
        args.revision,
        needed_doc_ids,
    )

    base_components, base_info = score_components_for_model(
        args,
        label="base",
        model_path=args.model_path,
        head_checkpoint="",
        candidates=candidates,
        query_text_by_id=query_text_by_id,
        candidate_text_by_id=candidate_text_by_id,
        query_ids=query_ids,
    )
    base_summary = summarize_model_results(
        model_info=base_info,
        candidates=candidates,
        qrels=qrels,
        component_scores_by_query=base_components,
        query_ids=query_ids,
        weight_grid=weight_grid,
        top_k=args.top_k,
        metric_k=args.metric_k,
    )

    student_summary = None
    if args.student_model_path or args.student_head_checkpoint:
        student_components, student_info = score_components_for_model(
            args,
            label=args.student_label,
            model_path=args.student_model_path,
            head_checkpoint=args.student_head_checkpoint,
            candidates=candidates,
            query_text_by_id=query_text_by_id,
            candidate_text_by_id=candidate_text_by_id,
            query_ids=query_ids,
        )
        student_summary = summarize_model_results(
            model_info=student_info,
            candidates=candidates,
            qrels=qrels,
            component_scores_by_query=student_components,
            query_ids=query_ids,
            weight_grid=weight_grid,
            top_k=args.top_k,
            metric_k=args.metric_k,
        )

    comparisons = compare_models(base_summary, student_summary)
    best_comparison = max(comparisons, key=lambda row: row["delta"]) if comparisons else None
    per_query_metrics_file = None
    if args.write_per_query:
        per_query_metrics_file = "per_query_metrics.jsonl"
        write_jsonl(
            output_dir / per_query_metrics_file,
            per_query_comparisons(
                candidates=candidates,
                qrels=qrels,
                base_components=base_components,
                student_components=student_components if student_summary is not None else None,
                query_ids=query_ids,
                weight_grid=weight_grid,
                top_k=args.top_k,
                metric_k=args.metric_k,
            ),
        )
    summary = {
        "experiment": "bge-m3-hybrid-weight-diagnostic",
        "model": args.model,
        "model_path": args.model_path or None,
        "student_model_path": args.student_model_path or None,
        "student_head_checkpoint": args.student_head_checkpoint or None,
        "subset": args.subset,
        "split": args.split,
        "dataset_revision": args.revision,
        "candidate_run_file": args.candidate_run_file,
        "query_id_file": args.query_id_file or None,
        "query_offset": args.query_offset,
        "query_stride": args.query_stride,
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "candidate_depth": args.candidate_depth,
        "top_k": args.top_k,
        "metric_k": args.metric_k,
        "batch_size": args.batch_size,
        "max_passage_length": args.max_passage_length,
        "weight_grid": [
            {
                "label": weights.label,
                "weights_for_different_modes": weights.values,
            }
            for weights in weight_grid
        ],
        "base": base_summary,
        "student": student_summary,
        "comparisons": comparisons,
        "best_student_delta": best_comparison,
        "per_query_metrics_file": per_query_metrics_file,
        "generated_outputs_committed": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
