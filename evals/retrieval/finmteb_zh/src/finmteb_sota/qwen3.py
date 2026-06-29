from __future__ import annotations

DEFAULT_INSTRUCTION = (
    "Given a finance question, retrieve and rank passages that directly answer it. "
    "Prioritize exact entities, numbers, dates, formulas, and financial terminology."
)

QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_reranker_pair(
    query: str,
    document: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def format_qwen3_prompt(
    query: str,
    document: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    return QWEN3_PREFIX + format_reranker_pair(query, document, instruction) + QWEN3_SUFFIX


def qwen3_batch_tokenize(
    tokenizer,
    queries: list[str],
    documents: list[str],
    instruction: str,
    max_length: int,
    return_tensors: str = "pt",
):
    prefix_tokens = tokenizer.encode(QWEN3_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)
    pair_max_length = max(8, max_length - len(prefix_tokens) - len(suffix_tokens))
    pairs = [
        format_reranker_pair(query=query, document=document, instruction=instruction)
        for query, document in zip(queries, documents)
    ]
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
