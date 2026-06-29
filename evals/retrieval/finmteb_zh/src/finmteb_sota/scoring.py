from __future__ import annotations

from collections.abc import Iterable

from tqdm import tqdm

from finmteb_sota.qwen3 import qwen3_batch_tokenize, yes_no_token_ids


class Qwen3RerankerScorer:
    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        load_in_4bit: bool = False,
        bf16: bool = True,
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.false_id, self.true_id = yes_no_token_ids(self.tokenizer)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def score(
        self,
        queries: list[str],
        documents: list[str],
        instruction: str,
        batch_size: int,
        max_length: int,
        score_mode: str = "probability",
    ) -> list[float]:
        import torch

        if score_mode not in {"probability", "logit_margin", "true_logit"}:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        scores: list[float] = []
        indices: Iterable[int] = range(0, len(queries), batch_size)
        with torch.no_grad():
            for start in tqdm(indices, total=(len(queries) + batch_size - 1) // batch_size, desc="pairs"):
                end = start + batch_size
                batch = qwen3_batch_tokenize(
                    tokenizer=self.tokenizer,
                    queries=queries[start:end],
                    documents=documents[start:end],
                    instruction=instruction,
                    max_length=max_length,
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch).logits[:, -1, [self.false_id, self.true_id]].float()
                if score_mode == "probability":
                    batch_scores = torch.softmax(logits, dim=-1)[:, 1]
                elif score_mode == "logit_margin":
                    batch_scores = logits[:, 1] - logits[:, 0]
                else:
                    batch_scores = logits[:, 1]
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
        return scores
