#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TeacherRow:
    query: str
    positive: str
    negatives: list[str]
    target_margins: list[float]
    query_id: str | None = None
    teacher_scores: list[float] | None = None
    base_scores: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the BGE-M3 encoder and distill reranker teacher margins into "
            "only the sparse and ColBERT heads."
        )
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="BAAI/bge-m3")
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dense-weight", type=float, default=0.4)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--colbert-weight", type=float, default=0.4)
    parser.add_argument(
        "--objective",
        choices=["margin_mse", "listwise_kl"],
        default="margin_mse",
        help="Training objective for positive plus hard-negative score groups.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for raw teacher scores under listwise_kl.",
    )
    parser.add_argument(
        "--base-anchor-weight",
        type=float,
        default=0.0,
        help="KL anchor weight against base BGE-M3 hybrid score distributions.",
    )
    parser.add_argument(
        "--base-anchor-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for base hybrid score anchor distributions.",
    )
    parser.add_argument(
        "--target-margin-scale",
        type=float,
        default=1.0,
        help="Scale teacher margins before computing MSE; useful when teacher logits are sharper than hybrid scores.",
    )
    parser.add_argument(
        "--head-l2-anchor-weight",
        type=float,
        default=0.0,
        help="L2 anchor against the initial sparse/ColBERT head parameters.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def softmax_distribution(scores: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        raise ValueError("scores must not be empty")
    import math

    scaled = [score / temperature for score in scores]
    max_score = max(scaled)
    exp_scores = [math.exp(score - max_score) for score in scaled]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def listwise_scores_from_row(row: dict[str, Any], negatives_count: int) -> list[float] | None:
    pos_scores = row.get("pos_scores")
    neg_scores = row.get("neg_scores")
    if not pos_scores or not neg_scores:
        return None
    return [float(pos_scores[0]), *[float(score) for score in neg_scores[:negatives_count]]]


def base_scores_from_row(row: dict[str, Any], negatives_count: int) -> list[float] | None:
    pos_score = row.get("bge_m3_hybrid_pos_score")
    if pos_score is None:
        pos_scores = row.get("bge_m3_hybrid_pos_scores")
        if pos_scores:
            pos_score = pos_scores[0]
    neg_scores = row.get("bge_m3_hybrid_neg_scores")
    if pos_score is None or not neg_scores:
        return None
    return [float(pos_score), *[float(score) for score in neg_scores[:negatives_count]]]


def fallback_scores_from_margins(target_margins: list[float]) -> list[float]:
    return [0.0, *[-float(margin) for margin in target_margins]]


def row_listwise_distribution(row: TeacherRow, temperature: float) -> list[float]:
    scores = row.teacher_scores or fallback_scores_from_margins(row.target_margins)
    return softmax_distribution(scores, temperature)


def read_teacher_rows(
    path: Path,
    *,
    max_rows: int,
    negatives_per_query: int,
) -> list[TeacherRow]:
    rows: list[TeacherRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("query") or "")
            positives = row.get("pos") or [row.get("positive")]
            negatives = row.get("neg") or ([row.get("negative")] if row.get("negative") else [])
            if not query or not positives or not negatives:
                raise ValueError(
                    f"{path}:{line_number} must include query, pos/positive, neg, "
                    "and either target_margins or pos_scores/neg_scores"
                )
            usable_negatives = [
                str(value) for value in negatives[:negatives_per_query] if value is not None
            ]
            if len(usable_negatives) < negatives_per_query:
                continue
            teacher_scores = listwise_scores_from_row(row, len(usable_negatives))
            target_margins = row.get("target_margins")
            if target_margins:
                usable_margins = [
                    float(value) for value in target_margins[: len(usable_negatives)]
                ]
            elif teacher_scores:
                usable_margins = [
                    float(teacher_scores[0]) - float(negative_score)
                    for negative_score in teacher_scores[1:]
                ]
            else:
                raise ValueError(
                    f"{path}:{line_number} must include target_margins or "
                    "pos_scores/neg_scores"
                )
            if len(usable_margins) != len(usable_negatives):
                raise ValueError(f"{path}:{line_number} has mismatched negatives/target_margins")
            base_scores = base_scores_from_row(row, len(usable_negatives))
            rows.append(
                TeacherRow(
                    query=query,
                    positive=str(positives[0]),
                    negatives=usable_negatives,
                    target_margins=usable_margins,
                    query_id=str(row["query_id"]) if row.get("query_id") is not None else None,
                    teacher_scores=teacher_scores,
                    base_scores=base_scores,
                )
            )
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"No teacher rows found in {path}")
    return rows


def batched(rows: list[TeacherRow], batch_size: int) -> Iterable[list[TeacherRow]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def training_summary(
    *,
    args: argparse.Namespace,
    rows: list[TeacherRow],
    step_losses: list[float],
    elapsed_seconds: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    margins = [margin for row in rows for margin in row.target_margins]
    return {
        "experiment": "bge-m3-head-only-distillation",
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "row_count": len(rows),
        "expanded_triples": sum(len(row.negatives) for row in rows),
        "negatives_per_query": args.negatives_per_query,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "weights_for_different_modes": [
            args.dense_weight,
            args.sparse_weight,
            args.colbert_weight,
        ],
        "objective": getattr(args, "objective", "margin_mse"),
        "teacher_temperature": getattr(args, "teacher_temperature", None),
        "base_anchor_weight": getattr(args, "base_anchor_weight", 0.0),
        "base_anchor_temperature": getattr(args, "base_anchor_temperature", None),
        "target_margin_scale": args.target_margin_scale,
        "head_l2_anchor_weight": args.head_l2_anchor_weight,
        "trainable_modules": ["sparse_linear", "colbert_linear"],
        "encoder_frozen": True,
        "target_margin_stats": {
            "mean": mean(margins),
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "rows_with_teacher_scores": sum(row.teacher_scores is not None for row in rows),
        "rows_with_base_scores": sum(row.base_scores is not None for row in rows),
        "train_loss": {
            "first": step_losses[0] if step_losses else None,
            "last": step_losses[-1] if step_losses else None,
            "mean": mean(step_losses),
            "steps": len(step_losses),
        },
        "elapsed_seconds": elapsed_seconds,
        "head_checkpoint": str(checkpoint_path),
        "raw_training_data_committed": False,
        "checkpoint_committed": False,
    }


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def encode_trainable_heads(
    base_model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = move_batch_to_device(encoded, device)
    with torch.no_grad():
        hidden_state = base_model.model(**encoded, return_dict=True).last_hidden_state
        dense_vecs = base_model._dense_embedding(hidden_state, encoded["attention_mask"])

    sparse_vecs = base_model._sparse_embedding(
        hidden_state,
        encoded["input_ids"],
        return_embedding=True,
    )
    colbert_vecs = base_model._colbert_embedding(hidden_state, encoded["attention_mask"])

    if getattr(base_model, "normalize_embeddings", False):
        dense_vecs = functional.normalize(dense_vecs, dim=-1)
        colbert_vecs = functional.normalize(colbert_vecs, dim=-1)

    return {
        "dense": dense_vecs,
        "sparse": sparse_vecs,
        "colbert": colbert_vecs,
        "attention_mask": encoded["attention_mask"],
    }


def aligned_hybrid_scores(
    query_reps: dict[str, Any],
    doc_reps: dict[str, Any],
    *,
    docs_per_query: int,
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
    temperature: float,
) -> Any:
    import torch

    query_count = query_reps["dense"].shape[0]
    dense_docs = doc_reps["dense"].view(query_count, docs_per_query, -1)
    sparse_docs = doc_reps["sparse"].view(query_count, docs_per_query, -1)
    colbert_docs = doc_reps["colbert"].view(
        query_count,
        docs_per_query,
        doc_reps["colbert"].shape[1],
        doc_reps["colbert"].shape[2],
    )

    dense_scores = torch.einsum("bd,bnd->bn", query_reps["dense"], dense_docs) / temperature
    sparse_scores = torch.einsum("bv,bnv->bn", query_reps["sparse"], sparse_docs) / temperature
    token_scores = torch.einsum("bqd,bnpd->bnqp", query_reps["colbert"], colbert_docs)
    max_token_scores = token_scores.max(dim=-1).values
    query_mask = query_reps["attention_mask"][:, 1:].float()
    query_lengths = query_mask.sum(dim=-1).clamp(min=1.0)
    colbert_scores = (
        (max_token_scores * query_mask[:, None, :]).sum(dim=-1) / query_lengths[:, None]
    ) / temperature

    return (
        dense_weight * dense_scores
        + sparse_weight * sparse_scores
        + colbert_weight * colbert_scores
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from FlagEmbedding import BGEM3FlagModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "bge_m3_head_state.pt"
    summary_path = output_dir / "train_summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_teacher_rows(
        Path(args.train_jsonl),
        max_rows=args.max_rows,
        negatives_per_query=args.negatives_per_query,
    )
    print(
        f"[head-distill] rows={len(rows)} triples={sum(len(row.negatives) for row in rows)}",
        flush=True,
    )

    flag_model = BGEM3FlagModel(args.model_path, use_fp16=args.use_fp16)
    base_model = flag_model.model.to(args.device)
    tokenizer = flag_model.tokenizer

    base_model.train()
    base_model.model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    for module_name in ("sparse_linear", "colbert_linear"):
        module = getattr(base_model, module_name)
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_parameters = [
        parameter for parameter in base_model.parameters() if parameter.requires_grad
    ]
    anchor_parameters = [parameter.detach().clone() for parameter in trainable_parameters]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    temperature = float(getattr(base_model, "temperature", 1.0) or 1.0)

    step_losses: list[float] = []
    started = time.monotonic()
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(rows)
        for batch_rows in batched(rows, args.batch_size):
            global_step += 1
            docs_per_query = 1 + args.negatives_per_query
            query_texts = [row.query for row in batch_rows]
            doc_texts = [
                document
                for row in batch_rows
                for document in [row.positive, *row.negatives]
            ]
            targets = torch.tensor(
                [row.target_margins for row in batch_rows],
                dtype=torch.float32,
                device=args.device,
            ) * args.target_margin_scale

            optimizer.zero_grad(set_to_none=True)
            query_reps = encode_trainable_heads(
                base_model,
                tokenizer,
                query_texts,
                max_length=args.max_length,
                device=args.device,
            )
            doc_reps = encode_trainable_heads(
                base_model,
                tokenizer,
                doc_texts,
                max_length=args.max_length,
                device=args.device,
            )
            scores = aligned_hybrid_scores(
                query_reps,
                doc_reps,
                docs_per_query=docs_per_query,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
                colbert_weight=args.colbert_weight,
                temperature=temperature,
            )
            if args.objective == "margin_mse":
                predicted_margins = scores[:, :1] - scores[:, 1:]
                loss = functional.mse_loss(predicted_margins, targets)
            else:
                teacher_targets = torch.tensor(
                    [
                        row_listwise_distribution(row, args.teacher_temperature)
                        for row in batch_rows
                    ],
                    dtype=torch.float32,
                    device=args.device,
                )
                loss = functional.kl_div(
                    functional.log_softmax(scores, dim=1),
                    teacher_targets,
                    reduction="batchmean",
                )
                if args.base_anchor_weight:
                    if not all(row.base_scores for row in batch_rows):
                        raise ValueError(
                            "--base-anchor-weight requires bge_m3_hybrid_* scores "
                            "for every row"
                        )
                    base_targets = torch.tensor(
                        [
                            softmax_distribution(
                                row.base_scores or [],
                                args.base_anchor_temperature,
                            )
                            for row in batch_rows
                        ],
                        dtype=torch.float32,
                        device=args.device,
                    )
                    loss = loss + args.base_anchor_weight * functional.kl_div(
                        functional.log_softmax(scores, dim=1),
                        base_targets,
                        reduction="batchmean",
                    )
            if args.head_l2_anchor_weight:
                anchor_loss = sum(
                    functional.mse_loss(parameter.float(), anchor.float())
                    for parameter, anchor in zip(trainable_parameters, anchor_parameters, strict=True)
                ) / len(trainable_parameters)
                loss = loss + args.head_l2_anchor_weight * anchor_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()

            step_losses.append(float(loss.detach().cpu()))
            if global_step % args.log_steps == 0 or global_step == 1:
                elapsed = time.monotonic() - started
                print(
                    f"[head-distill-progress] epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step} loss={step_losses[-1]:.6f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed_seconds = time.monotonic() - started
    summary = training_summary(
        args=args,
        rows=rows,
        step_losses=step_losses,
        elapsed_seconds=elapsed_seconds,
        checkpoint_path=checkpoint_path,
    )
    torch.save(
        {
            "sparse_linear": base_model.sparse_linear.state_dict(),
            "colbert_linear": base_model.colbert_linear.state_dict(),
            "summary": summary,
        },
        checkpoint_path,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
