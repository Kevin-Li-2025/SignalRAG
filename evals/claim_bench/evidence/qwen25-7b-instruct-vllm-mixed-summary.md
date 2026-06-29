# Benchmark Summary

- model: `/home/hhai/models/modelscope/Qwen/Qwen2___5-7B-Instruct`
- revision: `default`
- created_at: `2026-05-28T16:24:08.963161+00:00`

## Results

- HumanEval Pass@1: **79.88%** (131 / 164)
- GSM8K exact match: **89.99%** (1187 / 1319), 4-shot

## Settings

```json
{
  "backend": "vllm",
  "batch_size": 4,
  "device": "cuda",
  "dtype": "bfloat16",
  "gsm8k_n_shot": 4,
  "gsm8k_template": "evalscope",
  "humaneval_template": "claimbench",
  "humaneval_timeout": 4.0,
  "limit": null,
  "load_in_4bit": false,
  "max_new_code_tokens": 384,
  "max_new_math_tokens": 512,
  "seed": 1,
  "tasks": [
    "gsm8k",
    "humaneval"
  ],
  "temperature_code": 0.0,
  "temperature_math": 0.0,
  "top_p_code": 1.0,
  "top_p_math": 1.0,
  "use_chat_template": true,
  "vllm_gpu_memory_utilization": 0.9
}
```

## Environment

```json
{
  "cuda": {
    "available": true,
    "device_count": 1,
    "devices": [
      {
        "index": 0,
        "major": 8,
        "minor": 9,
        "name": "NVIDIA L20",
        "total_memory_gb": 44.521
      }
    ],
    "torch_cuda": "12.4"
  },
  "packages": {
    "accelerate": "1.13.0",
    "datasets": "3.6.0",
    "torch": "2.6.0",
    "transformers": "4.51.3"
  },
  "platform": "Linux-6.14.0-27-generic-x86_64-with-glibc2.39",
  "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]"
}
```
