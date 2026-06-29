from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: int = 1
    num_samples: int = 1


def parse_dtype(dtype: str | None) -> torch.dtype | str:
    if dtype is None or dtype == "auto":
        return "auto"
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_name: str,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    revision: str | None = None,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": parse_dtype(dtype),
    }
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "cuda" and not load_in_4bit:
        model.to(device)
    model.eval()
    return tokenizer, model


def format_user_prompt(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
    chat_enable_thinking: bool | None = None,
    system_prompt: str | None = None,
) -> str:
    if not use_chat_template:
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": prompt})

    kwargs = {
        "conversation": conversation,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if chat_enable_thinking is not None:
        kwargs["enable_thinking"] = chat_enable_thinking
    try:
        return tokenizer.apply_chat_template(**kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(**kwargs)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def generate_texts(
    tokenizer,
    model,
    prompts: list[str],
    *,
    batch_size: int,
    config: GenerationConfig,
) -> list[str]:
    return [
        candidates[0] if candidates else ""
        for candidates in generate_text_candidates(
            tokenizer,
            model,
            prompts,
            batch_size=batch_size,
            config=config,
        )
    ]


def generate_text_candidates(
    tokenizer,
    model,
    prompts: list[str],
    *,
    batch_size: int,
    config: GenerationConfig,
) -> list[list[str]]:
    if config.num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    outputs: list[list[str]] = []
    do_sample = config.temperature > 0
    for batch in batched(prompts, batch_size):
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "num_return_sequences": config.num_samples,
        }
        if do_sample:
            generation_kwargs["temperature"] = config.temperature
            generation_kwargs["top_p"] = config.top_p
            if config.top_k > 0:
                generation_kwargs["top_k"] = config.top_k
        if config.repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = config.repetition_penalty
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        decoded = [
            tokenizer.decode(row[input_width:], skip_special_tokens=True)
            for row in generated
        ]
        for offset in range(0, len(decoded), config.num_samples):
            outputs.append(decoded[offset : offset + config.num_samples])
    return outputs
