# Benchmark Summary

- model: `/home/hhai/models/modelscope/Qwen/Qwen2___5-Math-7B-Instruct`
- revision: `default`
- created_at: `2026-05-28T19:35:12.398727+00:00`

## Results

- GSM8K exact match: **93.33%** (1231 / 1319), 4-shot

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
    "gsm8k"
  ],
  "temperature_code": 0.0,
  "temperature_math": 0.0,
  "top_p_code": 1.0,
  "top_p_math": 1.0,
  "use_chat_template": true,
  "vllm_cpu_offload_gb": 0.0,
  "vllm_gpu_memory_utilization": 0.88,
  "vllm_max_model_len": 4096,
  "vllm_max_num_seqs": 32,
  "vllm_quantization": null
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
