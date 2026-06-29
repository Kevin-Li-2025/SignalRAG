from __future__ import annotations

def load_humaneval_split():
    from datasets import load_dataset

    try:
        return load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        return load_dataset("openai_humaneval", split="test")


def load_gsm8k_splits():
    from datasets import load_dataset

    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    return train, test
