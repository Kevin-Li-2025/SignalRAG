from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    train_file: Path
    eval_file: Path | None
    output_dir: Path
    seed: int = 42
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 256
    loss: str = "multiple_negatives_ranking"
    matryoshka_dims: tuple[int, ...] = (1024, 768, 512, 256)
    use_lora: bool = False
    trust_remote_code: bool = True
    save_total_limit: int = 2


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_training_config(path: str | Path) -> TrainingConfig:
    data = read_yaml(path)
    required = ["model_name", "train_file", "output_dir"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    eval_file = data.get("eval_file")
    return TrainingConfig(
        model_name=str(data["model_name"]),
        train_file=Path(data["train_file"]),
        eval_file=Path(eval_file) if eval_file else None,
        output_dir=Path(data["output_dir"]),
        seed=int(data.get("seed", 42)),
        num_train_epochs=float(data.get("num_train_epochs", 1.0)),
        learning_rate=float(data.get("learning_rate", 2e-5)),
        warmup_ratio=float(data.get("warmup_ratio", 0.05)),
        train_batch_size=int(data.get("train_batch_size", 16)),
        eval_batch_size=int(data.get("eval_batch_size", 16)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 1)),
        max_seq_length=int(data.get("max_seq_length", 256)),
        loss=str(data.get("loss", "multiple_negatives_ranking")),
        matryoshka_dims=tuple(int(dim) for dim in data.get("matryoshka_dims", [1024, 768, 512, 256])),
        use_lora=bool(data.get("use_lora", False)),
        trust_remote_code=bool(data.get("trust_remote_code", True)),
        save_total_limit=int(data.get("save_total_limit", 2)),
    )

