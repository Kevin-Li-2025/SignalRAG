from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankingTask:
    dataset_id: str
    leaderboard_name: str
    language: str


RERANKING_TASKS: tuple[RerankingTask, ...] = (
    RerankingTask("FinanceMTEB/FinEvaRetrieval-reranking", "FinEvaReranking", "zh"),
    RerankingTask("FinanceMTEB/DISCFinLLM-reranking", "DISCFinLLMReranking", "zh"),
    RerankingTask("FinanceMTEB/FinFact-reranking", "FinFactReranking", "en"),
    RerankingTask("FinanceMTEB/FiQA-reranking", "FiQA2018Reranking", "en"),
    RerankingTask("FinanceMTEB/HPC3-reranking", "HC3Reranking", "en"),
)

TASKS_BY_DATASET = {task.dataset_id: task for task in RERANKING_TASKS}
TASKS_BY_ALIAS = {task.leaderboard_name: task for task in RERANKING_TASKS}
ZH_TASKS = tuple(task for task in RERANKING_TASKS if task.language == "zh")


def resolve_tasks(selector: str | list[str]) -> list[RerankingTask]:
    if isinstance(selector, str):
        raw_names = [selector]
    else:
        raw_names = selector

    tasks: list[RerankingTask] = []
    for name in raw_names:
        if name == "zh":
            tasks.extend(ZH_TASKS)
        elif name == "all":
            tasks.extend(RERANKING_TASKS)
        elif name in TASKS_BY_DATASET:
            tasks.append(TASKS_BY_DATASET[name])
        elif name in TASKS_BY_ALIAS:
            tasks.append(TASKS_BY_ALIAS[name])
        else:
            raise KeyError(f"Unknown task selector: {name}")

    deduped: dict[str, RerankingTask] = {}
    for task in tasks:
        deduped[task.dataset_id] = task
    return list(deduped.values())

