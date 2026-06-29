#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

try:
    from bge_m3_lora_utils import build_lora_plan, target_modules_for_preset, write_json
except ModuleNotFoundError:
    from scripts.bge_m3_lora_utils import build_lora_plan, target_modules_for_preset, write_json


def _wrap_distributed_fn(
    *,
    original: Callable[..., int],
    fallback: int,
    is_available: Callable[[], bool],
    is_initialized: Callable[[], bool],
) -> Callable[..., int]:
    def wrapped(*args: Any, **kwargs: Any) -> int:
        if not is_available() or not is_initialized():
            return fallback
        return original(*args, **kwargs)

    return wrapped


def patch_torch_distributed_for_single_process(distributed_module: Any) -> None:
    """Let FlagEmbedding's single-process data loader run without torchrun/DDP."""

    distributed_module.get_rank = _wrap_distributed_fn(
        original=distributed_module.get_rank,
        fallback=0,
        is_available=distributed_module.is_available,
        is_initialized=distributed_module.is_initialized,
    )
    distributed_module.get_world_size = _wrap_distributed_fn(
        original=distributed_module.get_world_size,
        fallback=1,
        is_available=distributed_module.is_available,
        is_initialized=distributed_module.is_initialized,
    )


class _UnavailableDTensor:
    pass


def patch_torch_distributed_tensor_for_peft(distributed_module: Any) -> None:
    """Compat shim for PEFT versions that probe torch.distributed.tensor.DTensor."""

    if hasattr(distributed_module, "tensor") and hasattr(distributed_module.tensor, "DTensor"):
        return

    try:
        tensor_module = importlib.import_module("torch.distributed.tensor")
    except Exception:
        tensor_module = getattr(distributed_module, "tensor", SimpleNamespace())
        if not hasattr(tensor_module, "DTensor"):
            tensor_module.DTensor = _UnavailableDTensor
        distributed_module.tensor = tensor_module
        return

    distributed_module.tensor = tensor_module


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lora-enable", action="store_true")
    parser.add_argument(
        "--lora-target-preset",
        choices=["attention_qv", "attention_qkv", "all_linear"],
        default="attention_qv",
    )
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--lora-include-regex", default="")
    parser.add_argument("--lora-exclude-regex", default="pooler|sparse_linear|colbert_linear")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none")
    parser.add_argument("--lora-task-type", default="FEATURE_EXTRACTION")
    parser.add_argument("--lora-report-json", default="")
    return parser.parse_known_args(argv)


def _load_peft_helpers() -> tuple[Any, Callable[..., Any]]:
    import torch.distributed as dist

    patch_torch_distributed_tensor_for_peft(dist)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LoRA mode requires the peft package. Install requirements.txt or "
            "the remote training overlay before using --lora-enable."
        ) from exc
    return (LoraConfig, TaskType, get_peft_model)


def patch_flagembedding_m3_runner_for_lora(
    runner_cls: Any,
    *,
    wrapper_args: argparse.Namespace,
    lora_config_cls: Any | None = None,
    task_type_cls: Any | None = None,
    get_peft_model_fn: Callable[..., Any] | None = None,
) -> None:
    """Inject PEFT/LoRA into the encoder returned by FlagEmbedding's M3 runner."""

    if lora_config_cls is None or task_type_cls is None or get_peft_model_fn is None:
        lora_config_cls, task_type_cls, get_peft_model_fn = _load_peft_helpers()

    original_get_model = runner_cls.get_model
    resolved_target_modules = target_modules_for_preset(
        wrapper_args.lora_target_preset,
        wrapper_args.lora_target_modules,
    )
    task_type = getattr(task_type_cls, wrapper_args.lora_task_type, wrapper_args.lora_task_type)

    def get_model_with_lora(*args: Any, **kwargs: Any) -> dict[str, Any]:
        base_model = original_get_model(*args, **kwargs)
        encoder = base_model["model"]
        before_plan = build_lora_plan(
            model=encoder,
            target_preset=wrapper_args.lora_target_preset,
            target_modules=wrapper_args.lora_target_modules,
            include_regex=wrapper_args.lora_include_regex,
            exclude_regex=wrapper_args.lora_exclude_regex,
            rank=wrapper_args.lora_r,
            alpha=wrapper_args.lora_alpha,
            dropout=wrapper_args.lora_dropout,
            bias=wrapper_args.lora_bias,
            task_type=wrapper_args.lora_task_type,
        )
        config = lora_config_cls(
            r=wrapper_args.lora_r,
            lora_alpha=wrapper_args.lora_alpha,
            target_modules=resolved_target_modules,
            lora_dropout=wrapper_args.lora_dropout,
            bias=wrapper_args.lora_bias,
            task_type=task_type,
        )
        base_model["model"] = get_peft_model_fn(encoder, config)
        after_plan = build_lora_plan(
            model=base_model["model"],
            target_preset=wrapper_args.lora_target_preset,
            target_modules=wrapper_args.lora_target_modules,
            include_regex=wrapper_args.lora_include_regex,
            exclude_regex=wrapper_args.lora_exclude_regex,
            rank=wrapper_args.lora_r,
            alpha=wrapper_args.lora_alpha,
            dropout=wrapper_args.lora_dropout,
            bias=wrapper_args.lora_bias,
            task_type=wrapper_args.lora_task_type,
        )
        report = {
            "lora_enabled": True,
            "warning": (
                "Use this with FlagEmbedding fix_encoder=False. If fix_encoder=True, "
                "FlagEmbedding freezes every non-head parameter after this hook."
            ),
            "before": before_plan,
            "after": after_plan,
            "raw_outputs_committed": False,
            "model_checkpoints_committed": False,
        }
        print("[flagembedding-m3-lora] " + json.dumps(report["before"], sort_keys=True), flush=True)
        if wrapper_args.lora_report_json:
            write_json(Path(wrapper_args.lora_report_json), report)
        return base_model

    runner_cls.get_model = staticmethod(get_model_with_lora)


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run official FlagEmbedding BGE-M3 finetuning in single-process mode.\n\n"
            "This wrapper patches uninitialized torch.distributed rank/world-size "
            "lookups to 0/1, then forwards all non-help arguments to "
            "FlagEmbedding.finetune.embedder.encoder_only.m3.\n\n"
            "Use it with the same arguments as the official FlagEmbedding M3 "
            "finetuning CLI, but do not launch it through torchrun.\n\n"
            "Wrapper-only LoRA options are stripped before forwarding to "
            "FlagEmbedding: --lora-enable, --lora-target-preset, "
            "--lora-target-modules, --lora-r, --lora-alpha, --lora-dropout, "
            "--lora-bias, --lora-task-type, and --lora-report-json."
        )
        return
    wrapper_args, forwarded_args = parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *forwarded_args]

    import torch.distributed as dist

    patch_torch_distributed_for_single_process(dist)
    patch_torch_distributed_tensor_for_peft(dist)

    if wrapper_args.lora_enable:
        from FlagEmbedding.finetune.embedder.encoder_only.m3.runner import (
            EncoderOnlyEmbedderM3Runner,
        )

        patch_flagembedding_m3_runner_for_lora(
            EncoderOnlyEmbedderM3Runner,
            wrapper_args=wrapper_args,
        )

    from FlagEmbedding.finetune.embedder.encoder_only.m3.__main__ import (
        main as flagembedding_m3_main,
    )

    flagembedding_m3_main()


if __name__ == "__main__":
    main()
