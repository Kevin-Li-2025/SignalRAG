#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arabic_embedding_sota.config import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Arabic sentence embedding model.")
    parser.add_argument("--config", required=True, help="YAML training config.")
    return parser.parse_args()


def rows_from_jsonl(path: Path, loss_name: str = "multiple_negatives_ranking") -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "query" not in row or "positive" not in row:
                raise ValueError(f"{path}:{line_number} must include query and positive")
            if loss_name == "margin_mse":
                pos_scores = row.get("pos_scores")
                neg_scores = row.get("neg_scores")
                negatives = row.get("neg") or ([row["negative"]] if row.get("negative") else [])
                if not pos_scores or not neg_scores or not negatives:
                    raise ValueError(
                        f"{path}:{line_number} must include pos_scores, neg_scores, and neg "
                        "for margin_mse"
                    )
                positive_score = float(pos_scores[0])
                positive = (row.get("pos") or [row["positive"]])[0]
                for negative, negative_score in zip(negatives, neg_scores, strict=False):
                    rows.append(
                        {
                            "query": row["query"],
                            "positive": positive,
                            "negative": negative,
                            "label": positive_score - float(negative_score),
                        }
                    )
            else:
                rows.append(
                    {
                        "anchor": row["query"],
                        "positive": row["positive"],
                        "negative": row.get("negative"),
                    }
                )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_jsonl_pairs(path: Path, loss_name: str = "multiple_negatives_ranking"):
    from datasets import Dataset

    return Dataset.from_list(rows_from_jsonl(path, loss_name))


def config_summary(config) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "train_file": str(config.train_file),
        "eval_file": str(config.eval_file) if config.eval_file else None,
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "loss": config.loss,
        "matryoshka_dims": list(config.matryoshka_dims),
        "trust_remote_code": config.trust_remote_code,
        "save_total_limit": config.save_total_limit,
    }


def build_loss(model, config):
    from sentence_transformers import losses

    if config.loss == "margin_mse":
        return losses.MarginMSELoss(model)
    if config.loss != "multiple_negatives_ranking":
        raise ValueError(f"Unsupported loss: {config.loss}")

    base_loss = losses.MultipleNegativesRankingLoss(model)
    if config.matryoshka_dims and hasattr(losses, "MatryoshkaLoss"):
        return losses.MatryoshkaLoss(
            model,
            base_loss,
            matryoshka_dims=list(config.matryoshka_dims),
        )
    return base_loss


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)

    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    model = SentenceTransformer(config.model_name, trust_remote_code=config.trust_remote_code)
    model.max_seq_length = config.max_seq_length

    train_dataset = load_jsonl_pairs(config.train_file, config.loss)
    eval_dataset = (
        load_jsonl_pairs(config.eval_file, config.loss)
        if config.eval_file and config.eval_file.exists()
        else None
    )
    loss = build_loss(model, config)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        save_total_limit=config.save_total_limit,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=500 if eval_dataset is not None else None,
        seed=config.seed,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    train_output = trainer.train()
    model.save_pretrained(str(config.output_dir / "final"))
    summary = {
        "config": config_summary(config),
        "train_metrics": dict(train_output.metrics),
        "final_model_dir": str(config.output_dir / "final"),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
