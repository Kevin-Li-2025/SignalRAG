#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank an existing MIRACL candidate run with BGE-M3 hybrid "
            "dense+sparse+ColBERT scoring, optionally loading a trained head checkpoint."
        )
    )
    parser.add_argument("--candidate-run-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--head-checkpoint", default="")
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--dense-weight", type=float, default=0.4)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--colbert-weight", type=float, default=0.4)
    parser.add_argument("--run-id", default="bge-m3-hybrid-candidate-rerank")
    parser.add_argument(
        "--baseline-summary-file",
        default="",
        help=(
            "Optional summary JSON containing baseline metrics for a same-candidate "
            "diagnostic gate. Supports this script's summary.json or the v52-v54 "
            "experiment record format."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default="base_bge_m3_hybrid_same_candidate",
        help="Human-readable label for the diagnostic baseline.",
    )
    parser.add_argument(
        "--gate-metric",
        default="main_score",
        help="Metric key used for the diagnostic gate, usually main_score or ndcg_at_10.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum required metric improvement over the diagnostic baseline.",
    )
    parser.add_argument(
        "--require-gate-pass",
        action="store_true",
        help="Exit non-zero after writing summary.json if the diagnostic gate fails.",
    )
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_head_checkpoint_state(
    checkpoint_path: Path,
    *,
    torch_module: Any,
    map_location: Any,
) -> dict[str, Any]:
    if checkpoint_path.is_dir():
        return {
            "sparse_linear": torch_module.load(
                checkpoint_path / "sparse_linear.pt",
                map_location=map_location,
            ),
            "colbert_linear": torch_module.load(
                checkpoint_path / "colbert_linear.pt",
                map_location=map_location,
            ),
        }
    checkpoint = torch_module.load(checkpoint_path, map_location=map_location)
    if "sparse_linear" not in checkpoint or "colbert_linear" not in checkpoint:
        raise ValueError(
            "--head-checkpoint must be a custom checkpoint with sparse_linear/"
            "colbert_linear keys or an official FlagEmbedding model directory"
        )
    return checkpoint


def load_bge_model(args: argparse.Namespace) -> Any:
    from FlagEmbedding import BGEM3FlagModel

    model_name_or_path = args.model_path or args.model
    model = BGEM3FlagModel(model_name_or_path, use_fp16=args.use_fp16)
    if args.head_checkpoint:
        import torch

        checkpoint_path = Path(args.head_checkpoint)
        device = model.model.sparse_linear.weight.device
        checkpoint = load_head_checkpoint_state(
            checkpoint_path,
            torch_module=torch,
            map_location=device,
        )
        model.model.sparse_linear.load_state_dict(checkpoint["sparse_linear"])
        model.model.colbert_linear.load_state_dict(checkpoint["colbert_linear"])
        print(f"[hybrid-candidate-rerank] loaded head checkpoint {checkpoint_path}", flush=True)
    return model


def run_metrics(
    candidates: dict[str, list[dict[str, Any]]],
    qrels: dict[str, set[str]],
    query_ids: list[str],
    *,
    metric_k: int,
) -> dict[str, float]:
    per_query = [
        metrics_for_query(
            [str(row["docid"]) for row in candidates[query_id]],
            qrels[query_id],
            metric_k,
        )
        for query_id in query_ids
    ]
    metrics = mean_metrics(per_query)
    metrics["main_score"] = metrics["ndcg_at_10"]
    return metrics


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("metrics"),
        payload.get("base_first_stage_metrics"),
        payload.get("evaluation_surface", {}).get("base_first_stage_metrics"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return {str(key): float(value) for key, value in candidate.items()}
    raise ValueError("baseline summary does not contain metrics")


def load_baseline_metrics(path: Path) -> dict[str, float]:
    return extract_metrics(json.loads(path.read_text(encoding="utf-8")))


def diagnostic_gate(
    *,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    metric_key: str,
    min_delta: float,
    baseline_label: str,
) -> dict[str, Any]:
    if metric_key not in metrics:
        raise ValueError(f"metric {metric_key!r} missing from candidate metrics")
    if metric_key not in baseline_metrics:
        raise ValueError(f"metric {metric_key!r} missing from baseline metrics")
    candidate_score = float(metrics[metric_key])
    baseline_score = float(baseline_metrics[metric_key])
    delta = candidate_score - baseline_score
    return {
        "baseline_label": baseline_label,
        "metric": metric_key,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "delta": delta,
        "min_required_delta": min_delta,
        "passed": delta >= min_delta,
    }


def write_trec_run(
    path: Path,
    ranked_by_query: dict[str, list[dict[str, Any]]],
    query_ids: list[str],
    run_id: str,
    depth: int,
) -> int:
    lines = 0
    with path.open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            for rank, row in enumerate(ranked_by_query[query_id][:depth], start=1):
                handle.write(
                    f"{query_id} Q0 {row['docid']} {rank} {float(row['score']):.10f} {run_id}\n"
                )
                lines += 1
    return lines


def rerank(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_trec_run(Path(args.candidate_run_file), args.candidate_depth)
    query_ids = sorted(candidates)[: args.query_limit] if args.query_limit else sorted(candidates)
    candidates = {query_id: candidates[query_id] for query_id in query_ids}
    query_text_by_id, qrels = load_relevance(
        args.subset,
        args.split,
        args.revision,
        query_ids,
    )
    first_stage_metrics = run_metrics(
        candidates,
        qrels,
        query_ids,
        metric_k=args.metric_k,
    )
    needed_doc_ids = {str(row["docid"]) for rows in candidates.values() for row in rows}
    candidate_text_by_id = load_candidate_texts(
        args.subset,
        args.split,
        args.revision,
        needed_doc_ids,
    )

    model = load_bge_model(args)
    weights = [args.dense_weight, args.sparse_weight, args.colbert_weight]
    ranked_by_query: dict[str, list[dict[str, Any]]] = {}
    pair_count = 0
    started = time.monotonic()
    for query_index, query_id in enumerate(query_ids, start=1):
        rows = candidates[query_id]
        query = query_text_by_id[query_id]
        pairs = [[query, candidate_text_by_id[str(row["docid"])]] for row in rows]
        scores: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            result = model.compute_score(
                batch,
                batch_size=args.batch_size,
                max_passage_length=args.max_passage_length,
                weights_for_different_modes=weights,
            )
            scores.extend(float(score) for score in result["colbert+sparse+dense"])
        ordered = sorted(
            zip(rows, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_by_query[query_id] = [
            {
                "docid": str(row["docid"]),
                "score": score,
                "first_stage_score": float(row["first_stage_score"]),
            }
            for row, score in ordered[: args.top_k]
        ]
        pair_count += len(pairs)
        if query_index % 25 == 0 or query_index == len(query_ids):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[hybrid-candidate-rerank-progress] queries={query_index}/{len(query_ids)} "
                f"pairs={pair_count} rate={pair_count / elapsed:.1f} pairs/s",
                flush=True,
            )

    elapsed_seconds = time.monotonic() - started
    metrics = run_metrics(ranked_by_query, qrels, query_ids, metric_k=args.metric_k)
    gate = None
    if args.baseline_summary_file:
        baseline_metrics = load_baseline_metrics(Path(args.baseline_summary_file))
        gate = diagnostic_gate(
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            metric_key=args.gate_metric,
            min_delta=args.min_delta,
            baseline_label=args.baseline_label,
        )
    run_file = output_dir / f"{args.subset}_{args.split}.txt"
    line_count = write_trec_run(run_file, ranked_by_query, query_ids, args.run_id, args.top_k)
    summary = {
        "experiment": "bge-m3-hybrid-candidate-rerank",
        "model": args.model,
        "model_path": args.model_path,
        "head_checkpoint": args.head_checkpoint or None,
        "candidate_run_file": args.candidate_run_file,
        "subset": args.subset,
        "split": args.split,
        "dataset_revision": args.revision,
        "query_count": len(query_ids),
        "candidate_depth": args.candidate_depth,
        "top_k": args.top_k,
        "metric_k": args.metric_k,
        "weights_for_different_modes": weights,
        "batch_size": args.batch_size,
        "max_passage_length": args.max_passage_length,
        "first_stage_metrics": first_stage_metrics,
        "metrics": metrics,
        "delta_vs_first_stage": {
            key: metrics[key] - first_stage_metrics[key]
            for key in metrics
            if key in first_stage_metrics
        },
        "diagnostic_gate": gate,
        "timings": {
            "rerank_elapsed_seconds": elapsed_seconds,
            "rerank_pairs": pair_count,
            "rerank_pairs_per_second": pair_count / max(elapsed_seconds, 1e-6),
        },
        "run_file": {"path": str(run_file), "lines": line_count},
        "generated_outputs_committed": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.require_gate_pass and gate and not gate["passed"]:
        raise SystemExit(
            f"Diagnostic gate failed: delta={gate['delta']:.6f} "
            f"< required={gate['min_required_delta']:.6f}"
        )


def main() -> None:
    rerank(parse_args())


if __name__ == "__main__":
    main()
