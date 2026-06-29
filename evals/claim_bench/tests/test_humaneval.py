from claimbench.humaneval import build_user_prompt, check_correctness, clean_completion


def test_clean_completion_preserves_generated_helper():
    assert (
        clean_completion("    return helper(x)\n\ndef helper(x):\n    return x + 1\n")
        == "    return helper(x)\n\ndef helper(x):\n    return x + 1\n"
    )


def test_clean_completion_extracts_full_function_body():
    text = "```python\ndef add_one(x):\n    return x + 1\n```"
    assert clean_completion(text, "add_one") == "    return x + 1\n"


def test_clean_completion_preserves_support_for_full_function():
    text = "```python\nimport math\n\ndef add_one(x):\n    return math.floor(x) + 1\n```"
    assert clean_completion(text, "add_one") == "    return math.floor(x) + 1\nimport math\n"


def test_clean_completion_indents_plain_body():
    assert clean_completion("return x + 1\n") == "    return x + 1\n"


def test_build_evalscope_user_prompt():
    row = {"prompt": "def add_one(x):\n    \"\"\"Return x plus one.\"\"\"\n"}
    prompt = build_user_prompt(row, template="evalscope")
    assert prompt.startswith("Read the following function signature and docstring")
    assert prompt.endswith(row["prompt"])


def test_build_complete_user_prompt_allows_helpers():
    row = {"prompt": "def add_one(x):\n    \"\"\"Return x plus one.\"\"\"\n"}
    prompt = build_user_prompt(row, template="complete")
    assert "Include any imports or helper functions" in prompt
    assert prompt.endswith(row["prompt"])


def test_check_correctness_passes_simple_solution():
    prompt = "def add_one(x):\n"
    completion = "    return x + 1\n"
    test = "def check(fn):\n    assert fn(1) == 2\n"
    passed, result = check_correctness(prompt, completion, test, "add_one", timeout_s=1.0)
    assert passed, result
