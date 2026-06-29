#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from finmteb_sota.data import iter_pair_examples, load_reranking_records, train_validation_split
from finmteb_sota.metrics import RankedQuery, reranking_metrics
from finmteb_sota.qwen3 import DEFAULT_INSTRUCTION, qwen3_batch_tokenize, yes_no_token_ids


@dataclass
class PairExample:
    query_id: str
    query: str
    document: str
    label: int


class PairDataset:
    def __init__(self, examples: list[PairExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PairExample:
        return self.examples[idx]


class PairCollator:
    def __init__(self, tokenizer, instruction: str, max_length: int):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_length = max_length

    def __call__(self, examples: list[PairExample]) -> dict[str, Any]:
        batch = qwen3_batch_tokenize(
            tokenizer=self.tokenizer,
            queries=[item.query for item in examples],
            documents=[item.document for item in examples],
            instruction=self.instruction,
            max_length=self.max_length,
        )
        batch["labels"] = self._labels_tensor([item.label for item in examples])
        batch["query_ids"] = [item.query_id for item in examples]
        return batch

    @staticmethod
    def _labels_tensor(labels: list[int]):
        import torch

        return torch.tensor(labels, dtype=torch.long)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_examples(config: dict[str, Any]) -> tuple[list[PairExample], list[PairExample], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    negatives_per_query = int(config["negatives_per_query"])
    validation_ratio = float(config["validation_ratio"])

    train_examples: list[PairExample] = []
    validation_examples: list[PairExample] = []
    split_report: dict[str, Any] = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "negatives_per_query": negatives_per_query,
        "tasks": [],
    }

    for dataset_id in config["train_tasks"]:
        records = load_reranking_records(dataset_id, split="train")
        train_records, validation_records = train_validation_split(records, validation_ratio, seed)
        split_report["tasks"].append(
            {
                "dataset": dataset_id,
                "train_queries": len(train_records),
                "validation_queries": len(validation_records),
                "train_query_ids": [record.query_id for record in train_records],
                "validation_query_ids": [record.query_id for record in validation_records],
            }
        )
        for raw in iter_pair_examples(train_records, negatives_per_query, rng):
            train_examples.append(PairExample(**raw))
        for raw in iter_pair_examples(validation_records, negatives_per_query, rng):
            validation_examples.append(PairExample(**raw))

    rng.shuffle(train_examples)
    max_train_examples = config.get("max_train_examples")
    if max_train_examples:
        train_examples = train_examples[: int(max_train_examples)]
    split_report["train_examples"] = len(train_examples)
    split_report["validation_examples"] = len(validation_examples)
    return train_examples, validation_examples, split_report


def build_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=list(lora_config["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, tokenizer


def forward_loss(model, batch: dict[str, Any], false_id: int, true_id: int):
    import torch
    import torch.nn.functional as F

    labels = batch.pop("labels")
    batch.pop("query_ids", None)
    device = next(param.device for param in model.parameters())
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    outputs = model(**batch)
    final_logits = outputs.logits[:, -1, [false_id, true_id]]
    loss = F.cross_entropy(final_logits, labels.to(final_logits.device))
    probabilities = torch.softmax(final_logits.float(), dim=-1)[:, 1]
    return loss, probabilities.detach().cpu().tolist()


def evaluate(model, dataloader, false_id: int, true_id: int) -> dict[str, float]:
    import torch

    model.eval()
    by_qid: dict[str, tuple[list[int], list[float]]] = {}
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].detach().cpu().tolist()
            qids = list(batch["query_ids"])
            loss, scores = forward_loss(model, batch, false_id, true_id)
            del loss
            for qid, label, score in zip(qids, labels, scores):
                q_labels, q_scores = by_qid.setdefault(qid, ([], []))
                q_labels.append(int(label))
                q_scores.append(float(score))
    queries = [
        RankedQuery(query_id=qid, labels=labels, scores=scores)
        for qid, (labels, scores) in by_qid.items()
    ]
    model.train()
    return reranking_metrics(queries)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(config: dict[str, Any]) -> None:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    train_examples, validation_examples, split_report = load_examples(config)
    save_json(output_dir / "data_split.json", split_report)
    model, tokenizer = build_model_and_tokenizer(config)
    false_id, true_id = yes_no_token_ids(tokenizer)
    collator = PairCollator(
        tokenizer=tokenizer,
        instruction=config.get("instruction", DEFAULT_INSTRUCTION),
        max_length=int(config["max_length"]),
    )

    train_loader = DataLoader(
        PairDataset(train_examples),
        batch_size=int(config["per_device_batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        PairDataset(validation_examples),
        batch_size=int(config["per_device_batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_update_steps = math.ceil(
        len(train_loader)
        * int(config["epochs"])
        / int(config["gradient_accumulation_steps"])
    )
    warmup_steps = int(total_update_steps * float(config["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=not bool(config.get("bf16", True)))
    grad_accum = int(config["gradient_accumulation_steps"])
    global_step = 0
    initial_metrics = evaluate(model, validation_loader, false_id, true_id)
    save_json(output_dir / "initial_validation_metrics.json", initial_metrics)
    best_map = initial_metrics["map"]
    model.save_pretrained(output_dir / "best")
    tokenizer.save_pretrained(output_dir / "best")
    model.train()

    for epoch in range(int(config["epochs"])):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(config.get("bf16", True))):
                loss, _ = forward_loss(model, batch, false_id, true_id)
                loss = loss / grad_accum
            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(config["log_steps"]) == 0:
                    progress.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", step=global_step)

                if global_step % int(config["eval_steps"]) == 0:
                    metrics = evaluate(model, validation_loader, false_id, true_id)
                    save_json(output_dir / f"eval_step_{global_step}.json", metrics)
                    if metrics["map"] > best_map:
                        best_map = metrics["map"]
                        model.save_pretrained(output_dir / "best")
                        tokenizer.save_pretrained(output_dir / "best")

                if global_step % int(config["save_steps"]) == 0:
                    model.save_pretrained(output_dir / f"step_{global_step}")
                    tokenizer.save_pretrained(output_dir / f"step_{global_step}")

    final_metrics = evaluate(model, validation_loader, false_id, true_id)
    save_json(output_dir / "final_validation_metrics.json", final_metrics)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
