#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect lightweight Hugging Face model metadata.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/model_cards.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for model_id in args.models:
        response = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=30)
        response.raise_for_status()
        data = response.json()
        records.append(
            {
                "model_id": model_id,
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "pipeline_tag": data.get("pipeline_tag"),
                "tags": data.get("tags", []),
                "last_modified": data.get("lastModified"),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

