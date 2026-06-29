from __future__ import annotations

from collections.abc import Iterable

from tqdm import tqdm


class SequenceClassificationRerankerScorer:
    def __init__(
        self,
        model_name: str,
        bf16: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        }
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def score(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        import torch

        scores: list[float] = []
        indices: Iterable[int] = range(0, len(queries), batch_size)
        with torch.no_grad():
            for start in tqdm(indices, total=(len(queries) + batch_size - 1) // batch_size, desc="pairs"):
                end = start + batch_size
                batch = self.tokenizer(
                    queries[start:end],
                    documents[start:end],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch, return_dict=True).logits.float()
                if logits.shape[-1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
        return scores
