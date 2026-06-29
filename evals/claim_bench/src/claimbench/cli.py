from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gsm8k import run_gsm8k
from .humaneval import run_humaneval
from .report import write_summary

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CLAIM_HUMANEVAL_THRESHOLD = 0.741
CLAIM_GSM8K_THRESHOLD = 0.836


def _optional_threshold(value: str) -> float | None:
    if value.lower() in {"none", "off", "skip"}:
        return None
    return float(value)


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id or local model path.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Generation backend.",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    parser.add_argument("--device", default="cuda", help="Transformers backend device.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Transformers generation batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Generation seed.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization cap.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional vLLM context length cap for fitting large quantized models.",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM maximum concurrent sequence count.",
    )
    parser.add_argument(
        "--vllm-quantization",
        default=None,
        help="Optional vLLM quantization override, for example awq or gptq.",
    )
    parser.add_argument(
        "--vllm-cpu-offload-gb",
        type=float,
        default=0.0,
        help="Optional vLLM CPU offload budget in GiB.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable vLLM CUDA graphs/torch.compile for tighter host-memory runs.",
    )


def cmd_inspect_env(args: argparse.Namespace) -> int:
    from .env import collect_env

    print(json.dumps(collect_env(), indent=2, sort_keys=True))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .env import write_env
    from .model import GenerationConfig

    out_dir: Path = args.out
    env = write_env(out_dir / "environment.json")
    generate_fn = None
    generate_candidates_fn = None
    if args.backend == "vllm":
        if args.revision is not None:
            raise ValueError("vLLM backend does not support --revision in this harness.")
        if args.load_in_4bit:
            raise ValueError("vLLM backend does not support --load-in-4bit in this harness.")
        from .vllm_backend import generate_text_candidates_vllm, generate_texts_vllm, load_vllm

        tokenizer, model = load_vllm(
            args.model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            quantization=args.vllm_quantization,
            cpu_offload_gb=args.vllm_cpu_offload_gb,
            enforce_eager=args.vllm_enforce_eager,
        )

        def generate_fn(prompts, generation):
            return generate_texts_vllm(model, prompts, config=generation)

        def generate_candidates_fn(prompts, generation):
            return generate_text_candidates_vllm(model, prompts, config=generation)

    else:
        from .model import load_model

        tokenizer, model = load_model(
            args.model,
            dtype=args.dtype,
            device=args.device,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            load_in_4bit=args.load_in_4bit,
        )

    tasks = {task.strip().lower() for task in args.tasks.split(",") if task.strip()}
    results = {}
    if "humaneval" in tasks:
        results["humaneval"] = run_humaneval(
            tokenizer,
            model,
            out_dir,
            batch_size=args.batch_size,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_code_tokens,
                temperature=args.temperature_code,
                top_p=args.top_p_code,
                top_k=args.top_k_code,
                repetition_penalty=args.repetition_penalty_code,
                seed=args.seed,
            ),
            limit=args.limit,
            timeout_s=args.humaneval_timeout,
            prompt_template=args.humaneval_template,
            use_chat_template=args.use_chat_template,
            chat_enable_thinking=args.chat_enable_thinking,
            generate_fn=generate_fn,
        )
    if "gsm8k" in tasks:
        results["gsm8k"] = run_gsm8k(
            tokenizer,
            model,
            out_dir,
            batch_size=args.batch_size,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_math_tokens,
                temperature=args.temperature_math,
                top_p=args.top_p_math,
                top_k=args.top_k_math,
                repetition_penalty=args.repetition_penalty_math,
                seed=args.seed,
                num_samples=args.gsm8k_samples,
            ),
            limit=args.limit,
            n_shot=args.gsm8k_n_shot,
            prompt_template=args.gsm8k_template,
            use_chat_template=args.use_chat_template,
            chat_enable_thinking=args.chat_enable_thinking,
            generate_fn=generate_fn,
            generate_candidates_fn=generate_candidates_fn,
            selection=args.gsm8k_selection,
        )
    unknown = tasks - {"humaneval", "gsm8k"}
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")

    settings = {
        "tasks": sorted(tasks),
        "backend": args.backend,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": args.device,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "max_new_code_tokens": args.max_new_code_tokens,
        "max_new_math_tokens": args.max_new_math_tokens,
        "temperature_code": args.temperature_code,
        "temperature_math": args.temperature_math,
        "top_p_code": args.top_p_code,
        "top_p_math": args.top_p_math,
        "top_k_code": args.top_k_code,
        "top_k_math": args.top_k_math,
        "repetition_penalty_code": args.repetition_penalty_code,
        "repetition_penalty_math": args.repetition_penalty_math,
        "gsm8k_n_shot": args.gsm8k_n_shot,
        "gsm8k_template": args.gsm8k_template,
        "gsm8k_samples": args.gsm8k_samples,
        "gsm8k_selection": args.gsm8k_selection,
        "humaneval_template": args.humaneval_template,
        "humaneval_timeout": args.humaneval_timeout,
        "use_chat_template": args.use_chat_template,
        "chat_enable_thinking": args.chat_enable_thinking,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_quantization": args.vllm_quantization,
        "vllm_cpu_offload_gb": args.vllm_cpu_offload_gb,
        "vllm_enforce_eager": args.vllm_enforce_eager,
    }
    write_summary(
        out_dir,
        model_name=args.model,
        revision=args.revision,
        settings=settings,
        env=env,
        results=results,
    )
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = summary["results"]
    failures = []
    if args.humaneval is not None:
        score = results.get("humaneval", {}).get("pass_at_1")
        if score is None or score < args.humaneval:
            failures.append(f"HumanEval {score} < {args.humaneval}")
    if args.gsm8k is not None:
        score = results.get("gsm8k", {}).get("exact_match")
        if score is None or score < args.gsm8k:
            failures.append(f"GSM8K {score} < {args.gsm8k}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    humaneval = results.get("humaneval", {}).get("pass_at_1")
    gsm8k = results.get("gsm8k", {}).get("exact_match")
    print("PASS")
    if humaneval is not None:
        print(f"HumanEval Pass@1: {humaneval * 100:.2f}%")
    if gsm8k is not None:
        print(f"GSM8K exact match: {gsm8k * 100:.2f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog="claimbench", formatter_class=formatter)
    subparsers = parser.add_subparsers(required=True)

    inspect_env = subparsers.add_parser("inspect-env", formatter_class=formatter)
    inspect_env.set_defaults(func=cmd_inspect_env)

    eval_parser = subparsers.add_parser("eval", formatter_class=formatter)
    add_common_eval_args(eval_parser)
    eval_parser.add_argument(
        "--tasks",
        default="humaneval,gsm8k",
        help="Comma-separated benchmark tasks.",
    )
    eval_parser.add_argument(
        "--max-new-code-tokens",
        type=int,
        default=384,
        help="HumanEval generation token cap.",
    )
    eval_parser.add_argument(
        "--max-new-math-tokens",
        type=int,
        default=512,
        help="GSM8K generation token cap.",
    )
    eval_parser.add_argument(
        "--temperature-code",
        type=float,
        default=0.0,
        help="HumanEval sampling temperature.",
    )
    eval_parser.add_argument(
        "--temperature-math",
        type=float,
        default=0.0,
        help="GSM8K sampling temperature.",
    )
    eval_parser.add_argument("--top-p-code", type=float, default=1.0, help="HumanEval top-p value.")
    eval_parser.add_argument("--top-p-math", type=float, default=1.0, help="GSM8K top-p value.")
    eval_parser.add_argument("--top-k-code", type=int, default=-1, help="HumanEval top-k value; -1 keeps backend default.")
    eval_parser.add_argument("--top-k-math", type=int, default=-1, help="GSM8K top-k value; -1 keeps backend default.")
    eval_parser.add_argument(
        "--repetition-penalty-code",
        type=float,
        default=1.0,
        help="HumanEval repetition penalty.",
    )
    eval_parser.add_argument(
        "--repetition-penalty-math",
        type=float,
        default=1.0,
        help="GSM8K repetition penalty.",
    )
    eval_parser.add_argument(
        "--gsm8k-n-shot",
        type=int,
        default=4,
        help="Number of GSM8K few-shot examples.",
    )
    eval_parser.add_argument(
        "--gsm8k-template",
        choices=["claimbench", "evalscope", "qwen25-math-cot"],
        default="evalscope",
        help="GSM8K prompt and answer extraction template.",
    )
    eval_parser.add_argument(
        "--gsm8k-samples",
        type=int,
        default=1,
        help="Number of sampled completions per GSM8K problem.",
    )
    eval_parser.add_argument(
        "--gsm8k-selection",
        choices=["first", "majority"],
        default="majority",
        help="How to select a final GSM8K answer when multiple samples are generated.",
    )
    eval_parser.add_argument(
        "--humaneval-template",
        choices=["claimbench", "evalscope", "complete", "concise", "raw"],
        default="claimbench",
        help="HumanEval prompt template.",
    )
    eval_parser.add_argument(
        "--humaneval-timeout",
        type=float,
        default=4.0,
        help="Per-sample HumanEval execution timeout in seconds.",
    )
    eval_parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    eval_parser.add_argument(
        "--chat-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass Qwen-style enable_thinking to chat templates when supported.",
    )
    eval_parser.set_defaults(func=cmd_eval)

    gate = subparsers.add_parser("gate", formatter_class=formatter)
    gate.add_argument("--summary", type=Path, required=True, help="Path to summary.json.")
    gate.add_argument(
        "--humaneval",
        type=_optional_threshold,
        default=CLAIM_HUMANEVAL_THRESHOLD,
        help="Minimum HumanEval Pass@1. Use none to skip this task.",
    )
    gate.add_argument(
        "--gsm8k",
        type=_optional_threshold,
        default=CLAIM_GSM8K_THRESHOLD,
        help="Minimum GSM8K exact match. Use none to skip this task.",
    )
    gate.set_defaults(func=cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
