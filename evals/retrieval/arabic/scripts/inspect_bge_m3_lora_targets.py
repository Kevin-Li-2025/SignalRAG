#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bge_m3_lora_utils import build_lora_plan, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect BGE-M3 encoder linear modules and estimate PEFT/LoRA trainable "
            "parameter counts before launching adapter training."
        )
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--target-preset",
        choices=["attention_qv", "attention_qkv", "all_linear"],
        default="attention_qv",
    )
    parser.add_argument(
        "--target-modules",
        default="",
        help="Optional comma-separated PEFT target module names. Overrides --target-preset.",
    )
    parser.add_argument(
        "--include-regex",
        default="",
        help="Optional regex that selected linear module names must match.",
    )
    parser.add_argument(
        "--exclude-regex",
        default="pooler|sparse_linear|colbert_linear",
        help="Optional regex for excluding linear module names.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bias", default="none")
    parser.add_argument("--task-type", default="FEATURE_EXTRACTION")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--preview-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModel

    model_name_or_path = args.model_path or args.model
    model = AutoModel.from_pretrained(
        model_name_or_path,
        cache_dir=args.cache_dir or None,
        trust_remote_code=args.trust_remote_code,
    )
    plan = build_lora_plan(
        model=model,
        target_preset=args.target_preset,
        target_modules=args.target_modules,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        bias=args.bias,
        task_type=args.task_type,
    )
    write_json(Path(args.output_json), plan)
    preview = plan["selected_modules"][: args.preview_limit]
    print(
        "selected_modules=",
        plan["selected_module_count"],
        "estimated_lora_trainable_parameters=",
        plan["estimated_lora_trainable_parameters"],
        "fraction=",
        f"{plan['estimated_lora_trainable_fraction']:.6f}",
    )
    for item in preview:
        print(
            item["name"],
            f"{item['in_features']}->{item['out_features']}",
            "lora_params=",
            int(args.rank) * (item["in_features"] + item["out_features"]),
        )


if __name__ == "__main__":
    main()
