from claimbench.gsm8k import (
    QWEN25_MATH_COT_SYSTEM_PROMPT,
    answers_match,
    build_prompt,
    extract_answer,
    normalize_number,
    run_gsm8k,
    select_prediction,
    system_prompt_for_template,
)
from claimbench.model import GenerationConfig


def test_extract_hash_answer():
    assert extract_answer("We compute it. #### 1,234") == "1234"


def test_extract_last_number():
    assert extract_answer("The answer is 42.") == "42"


def test_extract_boxed_answer_evalscope():
    assert extract_answer("So the result is \\boxed{1,234}.", template="evalscope") == "1234"


def test_extract_boxed_answer_qwen25_math_cot():
    assert extract_answer("Therefore \\boxed{18}.", template="qwen25-math-cot") == "18"


def test_fraction_normalization():
    assert normalize_number("3/2") == "1.5"


def test_decimal_answer_match_tolerates_formatting():
    assert answers_match("1.0000001", "1")


def test_build_evalscope_prompt():
    train_rows = [{"question": "A?", "answer": "We add. #### 3"}]
    prompt = build_prompt(train_rows, "B?", n_shot=1, template="evalscope")
    assert "Here are some examples of how to solve similar problems:" in prompt
    assert "ANSWER: \\boxed{3}" in prompt
    assert prompt.endswith("Please reason step by step, and put your final answer within \\boxed{}.")


def test_build_qwen25_math_cot_prompt_uses_official_system_instruction():
    prompt = build_prompt([], "What is 2+2?", n_shot=0, template="qwen25-math-cot")

    assert prompt == "What is 2+2?"
    assert system_prompt_for_template("qwen25-math-cot") == QWEN25_MATH_COT_SYSTEM_PROMPT


def test_majority_selection_ignores_missing_and_keeps_first_tie():
    prediction, index = select_prediction([None, "7", "9", "7", "9"], selection="majority")
    assert prediction == "7"
    assert index == 1


def test_first_selection_keeps_first_candidate():
    prediction, index = select_prediction([None, "7"], selection="first")
    assert prediction is None
    assert index == 0


def test_run_gsm8k_batches_external_generation(monkeypatch, tmp_path):
    train_rows = [{"question": "Example?", "answer": "Reason. #### 1"}]
    test_rows = [
        {"question": f"Question {index}?", "answer": "Reason. #### 1"}
        for index in range(5)
    ]
    calls = []

    monkeypatch.setattr(
        "claimbench.gsm8k.load_gsm8k_splits",
        lambda: (train_rows, test_rows),
    )

    def generate_fn(prompts, generation):
        calls.append(list(prompts))
        return ["Reason. #### 1" for _ in prompts]

    summary = run_gsm8k(
        None,
        None,
        tmp_path,
        batch_size=2,
        generation=GenerationConfig(max_new_tokens=8),
        limit=None,
        n_shot=1,
        prompt_template="claimbench",
        use_chat_template=False,
        generate_fn=generate_fn,
    )

    assert [len(batch) for batch in calls] == [2, 2, 1]
    assert summary["correct"] == 5
    assert summary["exact_match"] == 1.0
