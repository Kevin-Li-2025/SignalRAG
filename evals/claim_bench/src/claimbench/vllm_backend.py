from __future__ import annotations

from .model import GenerationConfig


def _vllm_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    if normalized == "auto":
        return "auto"
    raise ValueError(f"Unsupported vLLM dtype: {dtype}")


def load_vllm(
    model_name: str,
    *,
    dtype: str,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    quantization: str | None = None,
    cpu_offload_gb: float = 0.0,
    enforce_eager: bool = False,
):
    from vllm import LLM

    kwargs = {
        "model": model_name,
        "dtype": _vllm_dtype(dtype),
        "trust_remote_code": trust_remote_code,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
    if quantization:
        kwargs["quantization"] = quantization
    if cpu_offload_gb > 0:
        kwargs["cpu_offload_gb"] = cpu_offload_gb
    if enforce_eager:
        kwargs["enforce_eager"] = True

    llm = LLM(**kwargs)
    return llm.get_tokenizer(), llm


def generate_texts_vllm(llm, prompts: list[str], *, config: GenerationConfig) -> list[str]:
    return [
        candidates[0] if candidates else ""
        for candidates in generate_text_candidates_vllm(llm, prompts, config=config)
    ]


def generate_text_candidates_vllm(
    llm,
    prompts: list[str],
    *,
    config: GenerationConfig,
) -> list[list[str]]:
    from vllm import SamplingParams

    if config.num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    do_sample = config.temperature > 0
    params = SamplingParams(
        n=config.num_samples,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature if do_sample else 0.0,
        top_p=config.top_p if do_sample else 1.0,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
    )
    outputs = llm.generate(prompts, params)
    return [[candidate.text for candidate in item.outputs] for item in outputs]
