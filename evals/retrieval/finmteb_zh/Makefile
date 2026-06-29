PYTHONPATH := src
PYTHON ?= python3
ACCELERATE ?= accelerate
MODEL ?= Qwen/Qwen3-Reranker-8B
ADAPTER ?= outputs/qwen3-reranker-8b-finmteb-lora
BATCH_SIZE ?= 8
MAX_LENGTH ?= 4096
HF_ENDPOINT ?= https://hf-mirror.com
HF_HUB_DOWNLOAD_TIMEOUT ?= 120
HF_ENV := HF_ENDPOINT=$(HF_ENDPOINT) HF_HUB_DOWNLOAD_TIMEOUT=$(HF_HUB_DOWNLOAD_TIMEOUT)

.PHONY: leaderboard test compile search-base eval-base train-lora search-lora eval-lora

leaderboard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/finmteb_leaderboard_snapshot.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src scripts tests

search-base:
	$(HF_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/search_instruction_alpha.py \
		--model $(MODEL) \
		--tasks zh \
		--split train \
		--batch-size $(BATCH_SIZE) \
		--max-length $(MAX_LENGTH) \
		--load-in-4bit \
		--output reports/qwen3_reranker_8b_train_search.json

eval-base:
	$(HF_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_qwen3_reranker.py \
		--model $(MODEL) \
		--tasks zh \
		--split test \
		--batch-size $(BATCH_SIZE) \
		--max-length $(MAX_LENGTH) \
		--load-in-4bit \
		--tuning-file reports/qwen3_reranker_8b_train_search.json \
		--output reports/qwen3_reranker_8b_test.json

train-lora:
	$(HF_ENV) PYTHONPATH=$(PYTHONPATH) $(ACCELERATE) launch scripts/train_qwen3_reranker_lora.py \
		--config configs/l20_qwen3_reranker_8b.yaml

search-lora:
	$(HF_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/search_instruction_alpha.py \
		--model $(MODEL) \
		--adapter $(ADAPTER) \
		--tasks zh \
		--split train \
		--batch-size $(BATCH_SIZE) \
		--max-length $(MAX_LENGTH) \
		--load-in-4bit \
		--output reports/qwen3_8b_lora_train_search.json

eval-lora:
	$(HF_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_qwen3_reranker.py \
		--model $(MODEL) \
		--adapter $(ADAPTER) \
		--tasks zh \
		--split test \
		--batch-size $(BATCH_SIZE) \
		--max-length $(MAX_LENGTH) \
		--load-in-4bit \
		--tuning-file reports/qwen3_8b_lora_train_search.json \
		--output reports/qwen3_8b_lora_test.json
