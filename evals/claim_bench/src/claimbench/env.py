from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch


def collect_env() -> dict:
    packages = {}
    for name in ["torch", "transformers", "datasets", "accelerate"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def write_env(path: Path) -> dict:
    env = collect_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return env
