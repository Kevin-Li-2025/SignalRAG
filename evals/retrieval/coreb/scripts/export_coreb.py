from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from coreb_sota.data import write_jsonl


CONFIGS = [
    "code_corpus",
    "text_corpus",
    "text2code_queries",
    "text2code_qrels",
    "code2code_queries",
    "code2code_qrels",
    "code2text_queries",
    "code2text_qrels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CoREB HF datasets to local JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--splits", nargs="+", default=["release_v2602", "release_v2603"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        for config in CONFIGS:
            dataset = load_dataset("hq-bench/coreb", config, split=split)
            rows = [dict(row) for row in dataset]
            path = args.output_dir / split / f"{config}.jsonl"
            write_jsonl(rows, path)
            print(f"wrote {len(rows):6d} rows -> {path}")


if __name__ == "__main__":
    main()
