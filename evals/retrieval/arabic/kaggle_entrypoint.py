from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def concat_embedding_outputs(outputs: list[object]) -> object:
    if not outputs:
        return outputs
    first = outputs[0]
    if hasattr(first, "detach") and hasattr(first, "dim"):
        import torch

        return torch.cat(outputs, dim=0)
    if hasattr(first, "shape"):
        import numpy as np

        return np.concatenate(outputs, axis=0)
    if isinstance(first, list):
        merged: list[object] = []
        for output in outputs:
            merged.extend(output)
        return merged
    if isinstance(first, tuple):
        return tuple(
            concat_embedding_outputs([output[index] for output in outputs])
            for index in range(len(first))
        )
    if isinstance(first, dict):
        return {
            key: concat_embedding_outputs([output[key] for output in outputs])
            for key in first
        }
    return outputs


def encode_with_progress(
    model_name: str,
    method_name: str,
    method: object,
    chunk_size: int,
    inputs: object,
    **kwargs: object,
) -> object:
    if isinstance(inputs, str) or not hasattr(inputs, "__len__"):
        return method(inputs, **kwargs)

    total = len(inputs)  # type: ignore[arg-type]
    if not isinstance(inputs, (list, tuple)) or total <= chunk_size:
        print(
            f"[progress] {model_name}.{method_name}: start {total} texts",
            flush=True,
        )
        output = method(inputs, **kwargs)
        print(
            f"[progress] {model_name}.{method_name}: done {total}/{total} texts",
            flush=True,
        )
        return output

    start_time = time.monotonic()
    outputs: list[object] = []
    print(
        f"[progress] {model_name}.{method_name}: start {total} texts "
        f"chunk_size={chunk_size}",
        flush=True,
    )
    for start_index in range(0, total, chunk_size):
        end_index = min(start_index + chunk_size, total)
        chunk = inputs[start_index:end_index]  # type: ignore[index]
        outputs.append(method(chunk, **kwargs))
        done = end_index
        elapsed = max(time.monotonic() - start_time, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate else 0.0
        print(
            f"[progress] {model_name}.{method_name}: "
            f"{done}/{total} texts ({done / total:.1%}), "
            f"{rate:.1f} texts/s, eta {format_duration(eta)}",
            flush=True,
        )
    return concat_embedding_outputs(outputs)


def install_progress_logging(model: object, model_name: str, chunk_size: int) -> object:
    chunk_size = max(1, chunk_size)
    for method_name in ("encode", "encode_queries", "encode_corpus"):
        method = getattr(model, method_name, None)
        if method is None:
            continue

        def progress_method(
            inputs: object,
            *args: object,
            _method: object = method,
            _method_name: str = method_name,
            **kwargs: object,
        ) -> object:
            if args:
                return _method(inputs, *args, **kwargs)
            return encode_with_progress(
                model_name,
                _method_name,
                _method,
                chunk_size,
                inputs,
                **kwargs,
            )

        setattr(model, method_name, progress_method)
    return model


def main() -> None:
    if os.environ.get("SKIP_RUNTIME_PIP_INSTALL") != "1":
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "mteb>=1.15.0",
                "sentence-transformers>=3.0.0",
                "transformers>=4.43.0",
                "accelerate>=0.30.0",
            ]
        )

    import mteb
    import torch
    from sentence_transformers import SentenceTransformer

    experiment_name = os.environ.get(
        "EXPERIMENT_NAME", "reranking-miracl-ar-e5-baseline"
    )
    model_names = env_list(
        "MODEL_NAMES",
        ["intfloat/multilingual-e5-large-instruct"],
    )
    task_names = env_list("TASK_NAMES", ["MIRACLReranking"])
    eval_subsets = env_list("EVAL_SUBSETS", ["ar"])
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "256"))
    progress_chunk_size = int(
        os.environ.get("PROGRESS_CHUNK_SIZE", str(max(batch_size * 1024, 8192)))
    )
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/kaggle/working"))
    output_dir = output_root / f"arabic-embedding-{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary = {
        "experiment": experiment_name,
        "tasks": task_names,
        "eval_subsets": eval_subsets,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "progress_chunk_size": progress_chunk_size,
        "output_root": str(output_root),
        "models": [],
    }
    for model_name in model_names:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = mteb.get_model(model_name, device=device)
        except Exception:
            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
        if hasattr(model, "model") and hasattr(model.model, "max_seq_length"):
            model.model.max_seq_length = max_seq_length
        elif hasattr(model, "max_seq_length"):
            model.max_seq_length = max_seq_length
        model = install_progress_logging(model, model_name, progress_chunk_size)
        model_output = output_dir / model_name.replace("/", "__")
        model_output.mkdir(parents=True, exist_ok=True)
        task_records = []
        for task_name in task_names:
            torch.cuda.empty_cache()
            task_output = model_output / task_name
            task_output.mkdir(parents=True, exist_ok=True)
            try:
                tasks = mteb.get_tasks(tasks=[task_name])
                for task in tasks:
                    if hasattr(task, "hf_subsets"):
                        available_subsets = set(task.hf_subsets)
                        task.hf_subsets = [
                            subset for subset in eval_subsets if subset in available_subsets
                        ]
                    if not getattr(task, "hf_subsets", None):
                        raise ValueError(
                            f"{task_name} has no matching subsets for {eval_subsets}"
                        )
                print(
                    f"Running {model_name} / {task_name} on subsets {eval_subsets}",
                    flush=True,
                )
                evaluator = mteb.MTEB(tasks=tasks)
                evaluator.run(
                    model,
                    output_folder=str(task_output),
                    eval_subsets=eval_subsets,
                    overwrite_results=True,
                    encode_kwargs={"batch_size": batch_size},
                )
                task_records.append(
                    {
                        "task": task_name,
                        "status": "complete",
                        "eval_subsets": eval_subsets,
                        "output_dir": str(task_output),
                    }
                )
            except Exception as exc:  # Keep smoke runs useful even when one task OOMs.
                error_path = task_output / "error.txt"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                task_records.append(
                    {
                        "task": task_name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "output_dir": str(task_output),
                    }
                )
                print(f"Task failed: {model_name} / {task_name}: {exc}", flush=True)
            finally:
                torch.cuda.empty_cache()
        summary["models"].append(
            {"model": model_name, "output_dir": str(model_output), "tasks": task_records}
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        del model
        torch.cuda.empty_cache()

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
