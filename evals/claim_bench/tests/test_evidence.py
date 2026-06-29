import json
from pathlib import Path

from claimbench.cli import (
    CLAIM_GSM8K_THRESHOLD,
    CLAIM_HUMANEVAL_THRESHOLD,
    DEFAULT_MODEL,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "qwen25-7b-instruct-vllm-mixed-summary.json"
CODER_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-t02-s1-summary.json"
)
QWEN3_CODER_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope-summary.json"
)
QWEN3_CODER_COMPLETE_RECOMMENDED_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-recommended-s1-summary.json"
)
MATH_EVIDENCE = ROOT / "evidence" / "qwen25-math-7b-instruct-vllm-gsm8k-summary.json"
MATH_QWEN25_COT_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-summary.json"
)
MATH_QWEN25_COT_MAJORITY8_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-majority8-summary.json"
)
MATH72_EVIDENCE = (
    ROOT
    / "evidence"
    / "qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-full-u099-l4096-eager-summary.json"
)
QWEN3_PROMPT_SWEEP = {
    "claimbench": (147, ROOT / "evidence" / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-claimbench-sweep-summary.json"),
    "complete": (150, ROOT / "evidence" / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-sweep-summary.json"),
    "concise": (148, ROOT / "evidence" / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-concise-sweep-summary.json"),
    "raw": (87, ROOT / "evidence" / "qwen3-coder-30b-a3b-fp8-vllm-humaneval-raw-sweep-summary.json"),
}


def test_checked_in_evidence_passes_claim_gate():
    summary = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    humaneval = summary["results"]["humaneval"]
    gsm8k = summary["results"]["gsm8k"]

    assert humaneval["pass_at_1"] >= CLAIM_HUMANEVAL_THRESHOLD
    assert gsm8k["exact_match"] >= CLAIM_GSM8K_THRESHOLD
    assert humaneval["passed"] == 131
    assert humaneval["total"] == 164
    assert gsm8k["correct"] == 1187
    assert gsm8k["total"] == 1319


def test_checked_in_evidence_uses_verified_protocol():
    summary = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    settings = summary["settings"]

    assert summary["model"].endswith("Qwen2___5-7B-Instruct")
    assert settings["backend"] == "vllm"
    assert settings["dtype"] == "bfloat16"
    assert settings["limit"] is None
    assert settings["temperature_code"] == 0.0
    assert settings["temperature_math"] == 0.0
    assert settings["humaneval_template"] == "claimbench"
    assert settings["humaneval_timeout"] == 4.0
    assert settings["gsm8k_template"] == "evalscope"
    assert settings["gsm8k_n_shot"] == 4
    assert settings["use_chat_template"] is True


def test_checked_in_specialist_evidence_scores():
    qwen3_coder = json.loads(QWEN3_CODER_EVIDENCE.read_text(encoding="utf-8"))
    qwen3_coder_complete = json.loads(QWEN3_CODER_COMPLETE_RECOMMENDED_EVIDENCE.read_text(encoding="utf-8"))
    coder = json.loads(CODER_EVIDENCE.read_text(encoding="utf-8"))
    math = json.loads(MATH_EVIDENCE.read_text(encoding="utf-8"))
    math_qwen25_cot = json.loads(MATH_QWEN25_COT_EVIDENCE.read_text(encoding="utf-8"))
    math_qwen25_cot_majority8 = json.loads(MATH_QWEN25_COT_MAJORITY8_EVIDENCE.read_text(encoding="utf-8"))
    math72 = json.loads(MATH72_EVIDENCE.read_text(encoding="utf-8"))

    humaneval = qwen3_coder["results"]["humaneval"]
    humaneval_complete = qwen3_coder_complete["results"]["humaneval"]
    sampled_humaneval = coder["results"]["humaneval"]
    gsm8k = math["results"]["gsm8k"]
    gsm8k_qwen25_cot = math_qwen25_cot["results"]["gsm8k"]
    gsm8k_qwen25_cot_majority8 = math_qwen25_cot_majority8["results"]["gsm8k"]
    gsm8k_72b = math72["results"]["gsm8k"]

    assert humaneval["passed"] == 151
    assert humaneval["total"] == 164
    assert humaneval["pass_at_1"] >= 0.92
    assert humaneval_complete["passed"] == 153
    assert humaneval_complete["total"] == 164
    assert humaneval_complete["pass_at_1"] >= 0.932
    assert sampled_humaneval["passed"] == 150
    assert sampled_humaneval["total"] == 164
    assert gsm8k["correct"] == 1231
    assert gsm8k["total"] == 1319
    assert gsm8k["exact_match"] >= 0.93
    assert gsm8k_qwen25_cot["correct"] == 1263
    assert gsm8k_qwen25_cot["total"] == 1319
    assert gsm8k_qwen25_cot["exact_match"] >= 0.957
    assert gsm8k_qwen25_cot_majority8["correct"] == 1265
    assert gsm8k_qwen25_cot_majority8["total"] == 1319
    assert gsm8k_qwen25_cot_majority8["exact_match"] >= 0.959
    assert gsm8k_72b["correct"] == 1245
    assert gsm8k_72b["total"] == 1319
    assert gsm8k_72b["exact_match"] >= 0.943


def test_checked_in_specialist_evidence_protocols():
    qwen3_coder = json.loads(QWEN3_CODER_EVIDENCE.read_text(encoding="utf-8"))
    qwen3_coder_complete = json.loads(QWEN3_CODER_COMPLETE_RECOMMENDED_EVIDENCE.read_text(encoding="utf-8"))
    coder = json.loads(CODER_EVIDENCE.read_text(encoding="utf-8"))
    math = json.loads(MATH_EVIDENCE.read_text(encoding="utf-8"))
    math_qwen25_cot = json.loads(MATH_QWEN25_COT_EVIDENCE.read_text(encoding="utf-8"))
    math_qwen25_cot_majority8 = json.loads(MATH_QWEN25_COT_MAJORITY8_EVIDENCE.read_text(encoding="utf-8"))
    math72 = json.loads(MATH72_EVIDENCE.read_text(encoding="utf-8"))

    assert qwen3_coder["model"].endswith("Qwen3-Coder-30B-A3B-Instruct-FP8")
    assert qwen3_coder["settings"]["backend"] == "vllm"
    assert qwen3_coder["settings"]["tasks"] == ["humaneval"]
    assert qwen3_coder["settings"]["limit"] is None
    assert qwen3_coder["settings"]["humaneval_template"] == "evalscope"
    assert qwen3_coder["settings"]["temperature_code"] == 0.0
    assert qwen3_coder["settings"]["seed"] == 1
    assert qwen3_coder_complete["model"].endswith("Qwen3-Coder-30B-A3B-Instruct-FP8")
    assert qwen3_coder_complete["settings"]["backend"] == "vllm"
    assert qwen3_coder_complete["settings"]["tasks"] == ["humaneval"]
    assert qwen3_coder_complete["settings"]["limit"] is None
    assert qwen3_coder_complete["settings"]["humaneval_template"] == "complete"
    assert qwen3_coder_complete["settings"]["max_new_code_tokens"] == 1024
    assert qwen3_coder_complete["settings"]["temperature_code"] == 0.7
    assert qwen3_coder_complete["settings"]["top_p_code"] == 0.8
    assert qwen3_coder_complete["settings"]["top_k_code"] == 20
    assert qwen3_coder_complete["settings"]["repetition_penalty_code"] == 1.05
    assert qwen3_coder_complete["settings"]["seed"] == 1
    assert coder["model"].endswith("Qwen2___5-Coder-32B-Instruct-GPTQ-Int4")
    assert coder["settings"]["backend"] == "vllm"
    assert coder["settings"]["tasks"] == ["humaneval"]
    assert coder["settings"]["limit"] is None
    assert coder["settings"]["humaneval_template"] == "evalscope"
    assert coder["settings"]["temperature_code"] == 0.2
    assert coder["settings"]["top_p_code"] == 0.95
    assert coder["settings"]["seed"] == 1
    assert coder["settings"]["vllm_quantization"] == "gptq_marlin"
    assert math["model"].endswith("Qwen2___5-Math-7B-Instruct")
    assert math["settings"]["backend"] == "vllm"
    assert math["settings"]["tasks"] == ["gsm8k"]
    assert math["settings"]["limit"] is None
    assert math["settings"]["batch_size"] == 4
    assert math_qwen25_cot["model"].endswith("Qwen2___5-Math-7B-Instruct")
    assert math_qwen25_cot["settings"]["backend"] == "vllm"
    assert math_qwen25_cot["settings"]["tasks"] == ["gsm8k"]
    assert math_qwen25_cot["settings"]["limit"] is None
    assert math_qwen25_cot["settings"]["batch_size"] == 4
    assert math_qwen25_cot["settings"]["gsm8k_template"] == "qwen25-math-cot"
    assert math_qwen25_cot["settings"]["gsm8k_n_shot"] == 0
    assert math_qwen25_cot["settings"]["max_new_math_tokens"] == 1024
    assert math_qwen25_cot["settings"]["temperature_math"] == 0.0
    assert math_qwen25_cot["settings"]["use_chat_template"] is True
    assert math_qwen25_cot_majority8["model"].endswith("Qwen2___5-Math-7B-Instruct")
    assert math_qwen25_cot_majority8["settings"]["backend"] == "vllm"
    assert math_qwen25_cot_majority8["settings"]["tasks"] == ["gsm8k"]
    assert math_qwen25_cot_majority8["settings"]["limit"] is None
    assert math_qwen25_cot_majority8["settings"]["gsm8k_template"] == "qwen25-math-cot"
    assert math_qwen25_cot_majority8["settings"]["gsm8k_n_shot"] == 0
    assert math_qwen25_cot_majority8["settings"]["gsm8k_samples"] == 8
    assert math_qwen25_cot_majority8["settings"]["gsm8k_selection"] == "majority"
    assert math_qwen25_cot_majority8["settings"]["temperature_math"] == 0.7
    assert math_qwen25_cot_majority8["settings"]["top_p_math"] == 0.8
    assert math_qwen25_cot_majority8["settings"]["top_k_math"] == 20
    assert math72["model"].endswith("Qwen2___5-Math-72B-Instruct-AWQ")
    assert math72["settings"]["backend"] == "vllm"
    assert math72["settings"]["tasks"] == ["gsm8k"]
    assert math72["settings"]["limit"] is None
    assert math72["settings"]["batch_size"] == 1
    assert math72["settings"]["vllm_quantization"] == "awq_marlin"
    assert math72["settings"]["vllm_gpu_memory_utilization"] == 0.99
    assert math72["settings"]["vllm_max_model_len"] == 4096
    assert math72["settings"]["vllm_max_num_seqs"] == 1
    assert math72["settings"]["vllm_cpu_offload_gb"] == 0.0
    assert math72["settings"]["vllm_enforce_eager"] is True


def test_checked_in_qwen3_prompt_sweep_scores():
    for template, (passed, path) in QWEN3_PROMPT_SWEEP.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        result = summary["results"]["humaneval"]
        settings = summary["settings"]

        assert summary["model"].endswith("Qwen3-Coder-30B-A3B-Instruct-FP8")
        assert result["passed"] == passed
        assert result["total"] == 164
        assert settings["tasks"] == ["humaneval"]
        assert settings["limit"] is None
        assert settings["humaneval_template"] == template
        assert settings["temperature_code"] == 0.0


def test_cli_defaults_match_verified_protocol():
    parser = build_parser()
    args = parser.parse_args(["eval", "--out", "reports/default-check"])

    assert args.model == DEFAULT_MODEL
    assert args.gsm8k_n_shot == 4
    assert args.gsm8k_template == "evalscope"
    assert args.gsm8k_samples == 1
    assert args.gsm8k_selection == "majority"
    assert args.top_k_code == -1
    assert args.top_k_math == -1
    assert args.repetition_penalty_code == 1.0
    assert args.repetition_penalty_math == 1.0
    assert args.humaneval_template == "claimbench"
    assert args.humaneval_timeout == 4.0
    assert args.vllm_max_model_len is None
    assert args.vllm_max_num_seqs is None
    assert args.vllm_quantization is None
    assert args.vllm_cpu_offload_gb == 0.0
    assert args.vllm_enforce_eager is False
    assert args.chat_enable_thinking is None


def test_cli_accepts_large_vllm_fit_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--out",
            "reports/large-vllm-check",
            "--backend",
            "vllm",
            "--vllm-max-model-len",
            "8192",
            "--vllm-max-num-seqs",
            "16",
            "--vllm-quantization",
            "awq",
            "--vllm-cpu-offload-gb",
            "8",
            "--vllm-enforce-eager",
            "--chat-enable-thinking",
            "--top-k-code",
            "20",
            "--repetition-penalty-code",
            "1.05",
        ]
    )

    assert args.vllm_max_model_len == 8192
    assert args.vllm_max_num_seqs == 16
    assert args.vllm_quantization == "awq"
    assert args.vllm_cpu_offload_gb == 8.0
    assert args.vllm_enforce_eager is True
    assert args.chat_enable_thinking is True
    assert args.top_k_code == 20
    assert args.repetition_penalty_code == 1.05


def test_gate_defaults_match_claim_thresholds():
    parser = build_parser()
    args = parser.parse_args(["gate", "--summary", str(EVIDENCE)])

    assert args.humaneval == CLAIM_HUMANEVAL_THRESHOLD
    assert args.gsm8k == CLAIM_GSM8K_THRESHOLD


def test_gate_can_skip_single_task_thresholds():
    parser = build_parser()
    args = parser.parse_args(
        [
            "gate",
            "--summary",
            str(EVIDENCE),
            "--humaneval",
            "none",
            "--gsm8k",
            "0.9",
        ]
    )

    assert args.humaneval is None
    assert args.gsm8k == 0.9
