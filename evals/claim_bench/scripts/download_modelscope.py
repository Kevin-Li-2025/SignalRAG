from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a ModelScope snapshot and print the local path."
    )
    parser.add_argument("model_id", help="ModelScope model id, for example Qwen/Qwen2.5-32B-Instruct-AWQ.")
    parser.add_argument("--cache-dir", default=None, help="Optional ModelScope cache directory.")
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download

        path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    except Exception as modelscope_error:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        path = hf_snapshot_download(args.model_id, cache_dir=args.cache_dir)
        print(f"# ModelScope failed: {modelscope_error}", flush=True)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
