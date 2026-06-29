#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate embedding models on Arabic MTEB tasks.")
    parser.add_argument("--models", nargs="*", default=[], help="Hugging Face model IDs to evaluate.")
    parser.add_argument("--language", default="ara", help="MTEB language code, usually ara.")
    parser.add_argument("--tasks", nargs="*", default=[], help="Explicit MTEB task names.")
    parser.add_argument("--task-types", nargs="*", default=[], help="Optional MTEB task type filters.")
    parser.add_argument("--output-dir", default="outputs/mteb", help="Directory for MTEB result files.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--list-tasks", action="store_true", help="Print matching tasks and exit.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def get_mteb_tasks(language: str, task_names: Iterable[str], task_types: Iterable[str]):
    import mteb

    task_names = list(task_names)
    task_types = list(task_types)
    if task_names:
        return mteb.get_tasks(tasks=task_names)

    kwargs = {"languages": [language]}
    if task_types:
        kwargs["task_types"] = task_types
    return mteb.get_tasks(**kwargs)


def describe_task(task) -> dict:
    metadata = getattr(task, "metadata", None)
    if metadata is None:
        return {
            "name": getattr(task, "description", {}).get("name", task.__class__.__name__),
            "type": getattr(task, "description", {}).get("type"),
            "category": getattr(task, "description", {}).get("category"),
        }
    return {
        "name": metadata.name,
        "type": metadata.type,
        "category": metadata.category,
    }


def build_model(model_name: str, trust_remote_code: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)


def main() -> None:
    args = parse_args()
    tasks = get_mteb_tasks(args.language, args.tasks, args.task_types)
    task_descriptions = [describe_task(task) for task in tasks]

    if args.list_tasks:
        print(json.dumps(task_descriptions, indent=2, ensure_ascii=False))
        return

    if not args.models:
        raise SystemExit("Pass at least one --models value, or use --list-tasks.")

    import mteb

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "language": args.language,
        "tasks": task_descriptions,
        "models": [],
    }
    for model_name in args.models:
        model = build_model(model_name, args.trust_remote_code)
        evaluator = mteb.MTEB(tasks=tasks)
        model_output = output_dir / model_name.replace("/", "__")
        model_output.mkdir(parents=True, exist_ok=True)
        evaluator.run(model, output_folder=str(model_output), batch_size=args.batch_size)
        summary["models"].append({"model": model_name, "output_dir": str(model_output)})

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

