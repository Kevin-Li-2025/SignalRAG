#!/usr/bin/env python
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


LORA_TARGET_PRESETS: dict[str, list[str] | str] = {
    "attention_qv": ["query", "value"],
    "attention_qkv": ["query", "key", "value"],
    "all_linear": "all-linear",
}


@dataclass(frozen=True)
class LinearModuleInfo:
    name: str
    in_features: int
    out_features: int
    base_parameters: int

    def lora_parameters(self, rank: int) -> int:
        return int(rank) * (self.in_features + self.out_features)


def target_modules_for_preset(preset: str, explicit_modules: str = "") -> list[str] | str:
    if explicit_modules:
        modules = [module.strip() for module in explicit_modules.split(",") if module.strip()]
        if not modules:
            raise ValueError("--lora-target-modules did not contain any module names")
        return modules
    if preset not in LORA_TARGET_PRESETS:
        raise ValueError(f"unknown LoRA target preset {preset!r}")
    return LORA_TARGET_PRESETS[preset]


def parameter_counts(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for _name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def is_linear_like(module: Any) -> bool:
    return (
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
    )


def iter_linear_modules(model: Any) -> Iterable[LinearModuleInfo]:
    for name, module in model.named_modules():
        if not name or not is_linear_like(module):
            continue
        in_features = int(getattr(module, "in_features"))
        out_features = int(getattr(module, "out_features"))
        yield LinearModuleInfo(
            name=name,
            in_features=in_features,
            out_features=out_features,
            base_parameters=in_features * out_features,
        )


def _matches_target(info: LinearModuleInfo, target_modules: list[str] | str) -> bool:
    if target_modules == "all-linear":
        return True
    return any(info.name == target or info.name.endswith(f".{target}") for target in target_modules)


def select_lora_modules(
    modules: Iterable[LinearModuleInfo],
    *,
    target_modules: list[str] | str,
    include_regex: str = "",
    exclude_regex: str = "",
) -> list[LinearModuleInfo]:
    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None
    selected = []
    for info in modules:
        if not _matches_target(info, target_modules):
            continue
        if include_pattern and not include_pattern.search(info.name):
            continue
        if exclude_pattern and exclude_pattern.search(info.name):
            continue
        selected.append(info)
    return selected


def build_lora_plan(
    *,
    model: Any,
    target_preset: str,
    target_modules: str = "",
    include_regex: str = "",
    exclude_regex: str = "",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> dict[str, Any]:
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")

    resolved_targets = target_modules_for_preset(target_preset, target_modules)
    linear_modules = list(iter_linear_modules(model))
    selected_modules = select_lora_modules(
        linear_modules,
        target_modules=resolved_targets,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )
    lora_parameter_count = sum(info.lora_parameters(rank) for info in selected_modules)
    base_selected_parameter_count = sum(info.base_parameters for info in selected_modules)
    counts = parameter_counts(model)
    total = max(counts["total_parameters"], 1)
    return {
        "target_preset": target_preset,
        "target_modules": resolved_targets,
        "include_regex": include_regex or None,
        "exclude_regex": exclude_regex or None,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "bias": bias,
        "task_type": task_type,
        "linear_module_count": len(linear_modules),
        "selected_module_count": len(selected_modules),
        "selected_modules": [asdict(info) for info in selected_modules],
        "base_selected_parameter_count": base_selected_parameter_count,
        "estimated_lora_trainable_parameters": lora_parameter_count,
        "estimated_lora_trainable_fraction": lora_parameter_count / total,
        **counts,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
