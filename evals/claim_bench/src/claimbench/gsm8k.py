from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

from .datasets import load_gsm8k_splits

if TYPE_CHECKING:
    from .model import GenerationConfig


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")
QWEN25_MATH_COT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT_TEMPLATES = ("claimbench", "evalscope", "qwen25-math-cot")


@dataclass(frozen=True)
class GSM8KResult:
    index: int
    question: str
    gold: str
    prediction: str | None
    completion: str
    correct: bool
    candidate_predictions: list[str | None] | None = None
    candidate_completions: list[str] | None = None


def normalize_number(text: str | None) -> str | None:
    if not text:
        return None
    raw = (
        text.strip()
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )
    raw = raw.strip().rstrip(".")
    try:
        if "/" in raw and not raw.startswith("http"):
            value = Fraction(raw)
            return str(Decimal(value.numerator) / Decimal(value.denominator)).rstrip("0").rstrip(".")
        return str(Decimal(raw).normalize()).replace("E+", "e")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return raw


def _extract_boxed_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\\boxed\s*", text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            continue
        if text[start] != "{":
            tail = text[start:].split(maxsplit=1)[0]
            candidates.append(tail.strip())
            continue

        depth = 0
        for end in range(start, len(text)):
            char = text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start + 1 : end].strip())
                    break
    return candidates


def extract_answer(text: str, *, template: str = "claimbench") -> str | None:
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown GSM8K prompt template: {template}")
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        match = NUMBER_RE.search(tail)
        return normalize_number(match.group(0)) if match else None
    if template in {"evalscope", "qwen25-math-cot"}:
        boxed_candidates = _extract_boxed_candidates(text)
        for candidate in reversed(boxed_candidates):
            match = NUMBER_RE.search(candidate)
            if match:
                return normalize_number(match.group(0))
    lowered = text.lower()
    for marker in ["answer is", "final answer", "answer:", "therefore"]:
        if marker in lowered:
            tail = text[lowered.rfind(marker) :]
            matches = NUMBER_RE.findall(tail)
            if matches:
                return normalize_number(matches[-1])
    matches = NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def answers_match(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    if prediction == gold:
        return True
    try:
        predicted_value = Decimal(prediction)
        gold_value = Decimal(gold)
    except InvalidOperation:
        return prediction.strip() == gold.strip()
    tolerance = max(Decimal("1e-6"), abs(gold_value) * Decimal("1e-6"))
    return abs(predicted_value - gold_value) <= tolerance


def select_prediction(
    predictions: list[str | None],
    *,
    selection: str,
) -> tuple[str | None, int | None]:
    if not predictions:
        return None, None
    if selection == "first":
        return predictions[0], 0
    if selection != "majority":
        raise ValueError(f"Unknown GSM8K selection strategy: {selection}")

    counts = Counter(prediction for prediction in predictions if prediction is not None)
    if not counts:
        return None, None
    best_count = max(counts.values())
    tied = {prediction for prediction, count in counts.items() if count == best_count}
    for index, prediction in enumerate(predictions):
        if prediction in tied:
            return prediction, index
    return None, None


def _split_gsm8k_answer(answer: str) -> tuple[str, str]:
    if "####" not in answer:
        return answer.strip(), extract_answer(answer) or answer.strip()
    reasoning, final = answer.rsplit("####", 1)
    final_answer = final.strip()
    return reasoning.strip(), final_answer


def build_prompt(
    train_rows: list[dict],
    question: str,
    *,
    n_shot: int,
    template: str = "claimbench",
) -> str:
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown GSM8K prompt template: {template}")
    if template == "evalscope":
        examples = []
        for row in train_rows[:n_shot]:
            reasoning, final_answer = _split_gsm8k_answer(row["answer"])
            examples.append(
                "\n\n".join(
                    [
                        row["question"],
                        f"Reasoning:\n{reasoning}",
                        f"ANSWER: \\boxed{{{final_answer}}}",
                    ]
                )
            )
        fewshot = "\n\n".join(examples)
        return "\n\n".join(
            [
                "Here are some examples of how to solve similar problems:",
                fewshot,
                question,
                "Please reason step by step, and put your final answer within \\boxed{}.",
            ]
        )

    if template == "qwen25-math-cot":
        return question

    parts = [
        "Solve each grade school math problem. Give the reasoning, then end with '#### <answer>'.",
        "",
    ]
    for row in train_rows[:n_shot]:
        parts.extend(
            [
                f"Question: {row['question']}",
                f"Answer: {row['answer']}",
                "",
            ]
        )
    parts.extend(
        [
            f"Question: {question}",
            "Answer: Let's think step by step.",
        ]
    )
    return "\n".join(parts)


def system_prompt_for_template(template: str) -> str | None:
    if template == "qwen25-math-cot":
        return QWEN25_MATH_COT_SYSTEM_PROMPT
    return None


def run_gsm8k(
    tokenizer,
    model,
    out_dir: Path,
    *,
    batch_size: int,
    generation: GenerationConfig,
    limit: int | None,
    n_shot: int,
    prompt_template: str,
    use_chat_template: bool,
    chat_enable_thinking: bool | None = None,
    generate_fn=None,
    generate_candidates_fn=None,
    selection: str = "majority",
) -> dict:
    from .model import batched, format_user_prompt, generate_text_candidates, generate_texts

    train_split, test_split = load_gsm8k_splits()
    train_rows = list(train_split)
    test_rows = list(test_split)
    if limit is not None:
        test_rows = test_rows[:limit]

    prompts = [
        format_user_prompt(
            tokenizer,
            build_prompt(
                train_rows,
                row["question"],
                n_shot=n_shot,
                template=prompt_template,
            ),
            use_chat_template=use_chat_template,
            chat_enable_thinking=chat_enable_thinking,
            system_prompt=system_prompt_for_template(prompt_template),
        )
        for row in test_rows
    ]
    if generation.num_samples == 1:
        if generate_fn is None:
            completions = generate_texts(
                tokenizer,
                model,
                prompts,
                batch_size=batch_size,
                config=generation,
            )
        else:
            completions = []
            for prompt_batch in batched(prompts, batch_size):
                completions.extend(generate_fn(prompt_batch, generation))
        candidate_lists = [[completion] for completion in completions]
    elif generate_candidates_fn is None:
        candidate_lists = generate_text_candidates(
            tokenizer,
            model,
            prompts,
            batch_size=batch_size,
            config=generation,
        )
    else:
        candidate_lists = []
        for prompt_batch in batched(prompts, batch_size):
            candidate_lists.extend(generate_candidates_fn(prompt_batch, generation))

    rows: list[GSM8KResult] = []
    for index, (row, candidates) in enumerate(
        tqdm(list(zip(test_rows, candidate_lists)), desc="GSM8K score", unit="task")
    ):
        gold = extract_answer(row["answer"])
        candidate_predictions = [
            extract_answer(completion, template=prompt_template)
            for completion in candidates
        ]
        prediction, selected_index = select_prediction(
            candidate_predictions,
            selection=selection,
        )
        completion = candidates[selected_index or 0] if candidates else ""
        rows.append(
            GSM8KResult(
                index=index,
                question=row["question"],
                gold=gold or "",
                prediction=prediction,
                completion=completion,
                correct=answers_match(prediction, gold),
                candidate_predictions=candidate_predictions if generation.num_samples > 1 else None,
                candidate_completions=candidates if generation.num_samples > 1 else None,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "gsm8k_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    total = len(rows)
    correct = sum(1 for item in rows if item.correct)
    return {
        "task": "gsm8k",
        "total": total,
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
        "n_shot": n_shot,
        "prompt_template": prompt_template,
        "use_chat_template": use_chat_template,
        "chat_enable_thinking": chat_enable_thinking,
        "samples": generation.num_samples,
        "selection": selection,
        "predictions": str(pred_path),
    }
