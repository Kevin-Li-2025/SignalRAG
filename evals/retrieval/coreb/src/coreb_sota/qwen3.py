from __future__ import annotations

DEFAULT_INSTRUCTION = (
    "Given a code search query, rank candidates that solve the same programming problem "
    "or provide the matching problem statement. Prioritize algorithmic behavior, input-output "
    "constraints, language constraints, APIs, identifiers, and edge cases."
)

TASK_INSTRUCTIONS = {
    "text2code": "Given a natural language programming task, retrieve code that correctly solves or implements the task.",
    "code2code": "Given a code snippet, retrieve code that is semantically equivalent or solves the same task.",
    "code2text": "Given a code snippet, retrieve the natural language description or problem statement that best matches the code.",
}

QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Candidate is a relevant result for the Query and Instruct. "
    'The answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

COREB_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
COREB_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def format_pair(query: str, candidate: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {candidate}"


def batch_tokenize(
    tokenizer,
    queries: list[str],
    candidates: list[str],
    instruction: str,
    max_length: int,
    prompt_style: str = "qwen3",
    return_tensors: str = "pt",
):
    if prompt_style == "qwen3":
        prefix, suffix = QWEN3_PREFIX, QWEN3_SUFFIX
    elif prompt_style == "coreb":
        prefix, suffix = COREB_PREFIX, COREB_SUFFIX
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    pair_max_length = max(8, max_length - len(prefix_tokens) - len(suffix_tokens))
    pairs = [format_pair(query, candidate, instruction) for query, candidate in zip(queries, candidates)]
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=pair_max_length,
    )
    for idx, input_ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][idx] = prefix_tokens + input_ids + suffix_tokens
    return tokenizer.pad(inputs, padding=True, return_tensors=return_tensors, max_length=max_length)


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    false_id = tokenizer.convert_tokens_to_ids("no")
    true_id = tokenizer.convert_tokens_to_ids("yes")
    if false_id is None or true_id is None or false_id < 0 or true_id < 0:
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        if len(no_ids) != 1 or len(yes_ids) != 1:
            raise ValueError("Cannot resolve single-token yes/no ids for this tokenizer.")
        false_id, true_id = no_ids[0], yes_ids[0]
    return int(false_id), int(true_id)
