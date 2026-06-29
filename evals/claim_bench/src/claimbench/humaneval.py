from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

from .datasets import load_humaneval_split

if TYPE_CHECKING:
    from .model import GenerationConfig


STOP_PREFIXES = (
    "\n#",
    "\nif __name__",
    "\nprint(",
)
PROMPT_TEMPLATES = ("claimbench", "evalscope", "complete", "concise", "raw")


@dataclass(frozen=True)
class HumanEvalResult:
    task_id: str
    prompt: str
    completion: str
    passed: bool
    result: str


def _extract_matching_function_body(text: str, entry_point: str) -> str | None:
    lines = text.replace("\r\n", "\n").splitlines()
    pattern = re.compile(rf"^(\s*)def\s+{re.escape(entry_point)}\s*\(")
    for start, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        def_indent = len(match.group(1))
        body: list[str] = []
        main_end = len(lines)
        for offset, candidate in enumerate(lines[start + 1 :], start + 1):
            if candidate.strip() and len(candidate) - len(candidate.lstrip(" ")) <= def_indent:
                main_end = offset
                break
            body.append(candidate)
        while body and not body[0].strip():
            body.pop(0)
        if not body:
            return None
        body_indent = min(
            len(line) - len(line.lstrip(" "))
            for line in body
            if line.strip()
        )
        dedented = [line[body_indent:] if len(line) >= body_indent else line for line in body]
        extracted = "\n".join("    " + line if line.strip() else line for line in dedented).rstrip()
        support = _extract_top_level_support(lines, start, main_end, entry_point)
        if support:
            extracted = f"{extracted}\n{support}"
        return extracted.rstrip() + "\n"
    return None


def _extract_top_level_support(
    lines: list[str],
    main_start: int,
    main_end: int,
    entry_point: str,
) -> str:
    blocks: list[str] = []
    function_pattern = re.compile(rf"def\s+{re.escape(entry_point)}\s*\(")
    index = 0
    while index < len(lines):
        if main_start <= index < main_end:
            index = main_end
            continue

        line = lines[index]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if not stripped or indent != 0:
            index += 1
            continue

        if stripped.startswith(("import ", "from ")):
            blocks.append(line)
            index += 1
            continue

        if stripped.startswith(("def ", "class ")) and not function_pattern.match(stripped):
            block_start = index
            index += 1
            while index < len(lines):
                candidate = lines[index]
                candidate_indent = len(candidate) - len(candidate.lstrip(" "))
                if candidate.strip() and candidate_indent == 0:
                    break
                index += 1
            blocks.extend(lines[block_start:index])
            continue

        index += 1

    return "\n".join(blocks).rstrip()


def _ensure_indented_body(text: str) -> str:
    lines = text.splitlines()
    first = next((line for line in lines if line.strip()), "")
    if first and not first.startswith((" ", "\t")) and not first.lstrip().startswith(("def ", "class ")):
        return "\n".join("    " + line if line.strip() else line for line in lines).rstrip() + "\n"
    return text.rstrip() + "\n"


def clean_completion(text: str, entry_point: str | None = None) -> str:
    cleaned = text.replace("\r\n", "\n")
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    fence = "```"
    if fence in cleaned:
        parts = cleaned.split(fence)
        if len(parts) >= 3:
            candidate = parts[1]
            if candidate.lstrip().startswith("python"):
                candidate = candidate.lstrip()[len("python") :]
            cleaned = candidate
    if entry_point:
        full_function_body = _extract_matching_function_body(cleaned, entry_point)
        if full_function_body:
            return full_function_body
    stop_at = len(cleaned)
    for marker in STOP_PREFIXES:
        index = cleaned.find(marker)
        if index >= 0:
            stop_at = min(stop_at, index)
    return _ensure_indented_body(cleaned[:stop_at])


def build_user_prompt(row: dict, *, template: str) -> str:
    if template == "raw":
        return row["prompt"]
    if template == "evalscope":
        return (
            "Read the following function signature and docstring, and fully "
            "implement the function described. Your response should only contain "
            "the code for this function.\n"
            f"{row['prompt']}"
        )
    if template == "complete":
        return (
            "Implement the Python function below. Return only executable Python "
            "code for the solution. Include any imports or helper functions you "
            "define. Do not include explanations, tests, or markdown fences.\n\n"
            f"{row['prompt']}"
        )
    if template == "concise":
        return (
            "Fill in the implementation for this Python function. Return only "
            "the final code, keeping it short and robust for edge cases.\n\n"
            f"{row['prompt']}"
        )
    if template == "claimbench":
        return (
            "Complete the following Python function. Return only the missing "
            "function body or the complete function code. Do not include "
            "explanations.\n\n"
            f"{row['prompt']}"
        )
    raise ValueError(f"Unknown HumanEval prompt template: {template}")


def _guard_source() -> str:
    return r'''
import faulthandler
import os
import shutil
import signal
import subprocess
import sys

faulthandler.disable()
os.environ["OMP_NUM_THREADS"] = "1"

def _blocked(*args, **kwargs):
    raise PermissionError("blocked during HumanEval execution")

os.system = _blocked
os.remove = _blocked
os.rmdir = _blocked
os.unlink = _blocked
os.rename = _blocked
os.replace = _blocked
os.chdir = _blocked
shutil.rmtree = _blocked
subprocess.Popen = _blocked
subprocess.call = _blocked
subprocess.run = _blocked
subprocess.check_call = _blocked
subprocess.check_output = _blocked
if hasattr(signal, "alarm"):
    signal.alarm = _blocked
'''


def check_correctness(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    *,
    timeout_s: float,
) -> tuple[bool, str]:
    program = "\n".join(
        [
            _guard_source(),
            prompt,
            completion,
            test,
            f"check({entry_point})",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="claimbench_humaneval_") as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "passed"
    message = (proc.stderr or proc.stdout).strip().splitlines()
    return False, message[-1] if message else f"failed rc={proc.returncode}"


def run_humaneval(
    tokenizer,
    model,
    out_dir: Path,
    *,
    batch_size: int,
    generation: GenerationConfig,
    limit: int | None,
    timeout_s: float,
    prompt_template: str,
    use_chat_template: bool,
    chat_enable_thinking: bool | None = None,
    generate_fn=None,
) -> dict:
    from .model import batched, format_user_prompt, generate_texts

    dataset = list(load_humaneval_split())
    if limit is not None:
        dataset = dataset[:limit]

    if prompt_template == "claimbench" and not use_chat_template:
        prompts = [row["prompt"] for row in dataset]
    else:
        prompts = [
            format_user_prompt(
                tokenizer,
                build_user_prompt(row, template=prompt_template),
                use_chat_template=use_chat_template,
                chat_enable_thinking=chat_enable_thinking,
            )
            for row in dataset
        ]
    if generate_fn is None:
        raw_completions = generate_texts(
            tokenizer,
            model,
            prompts,
            batch_size=batch_size,
            config=generation,
        )
    else:
        raw_completions = []
        for prompt_batch in batched(prompts, batch_size):
            raw_completions.extend(generate_fn(prompt_batch, generation))
    completions = [
        clean_completion(text, row["entry_point"])
        for text, row in zip(raw_completions, dataset)
    ]

    rows: list[HumanEvalResult] = []
    for row, completion in tqdm(
        list(zip(dataset, completions)),
        desc="HumanEval check",
        unit="task",
    ):
        passed, result = check_correctness(
            row["prompt"],
            completion,
            row["test"],
            row["entry_point"],
            timeout_s=timeout_s,
        )
        rows.append(
            HumanEvalResult(
                task_id=row["task_id"],
                prompt=row["prompt"],
                completion=completion,
                passed=passed,
                result=result,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "humaneval_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    total = len(rows)
    passed = sum(1 for item in rows if item.passed)
    return {
        "task": "humaneval",
        "total": total,
        "passed": passed,
        "pass_at_1": passed / total if total else 0.0,
        "prompt_template": prompt_template,
        "use_chat_template": use_chat_template,
        "chat_enable_thinking": chat_enable_thinking,
        "predictions": str(pred_path),
    }
