from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASKS = {
    "text2code": {
        "queries": "text2code_queries",
        "qrels": "text2code_qrels",
        "corpus": "code_corpus",
        "doc_id": "code_id",
        "doc_text": "code",
    },
    "code2code": {
        "queries": "code2code_queries",
        "qrels": "code2code_qrels",
        "corpus": "code_corpus",
        "doc_id": "code_id",
        "doc_text": "code",
    },
    "code2text": {
        "queries": "code2text_queries",
        "qrels": "code2text_qrels",
        "corpus": "text_corpus",
        "doc_id": "text_id",
        "doc_text": "text",
    },
}


@dataclass(frozen=True)
class TaskData:
    name: str
    split: str
    queries: dict[str, dict[str, Any]]
    corpus: dict[str, dict[str, Any]]
    qrels: dict[str, dict[str, int]]
    exclude_doc_ids: dict[str, set[str]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_exported_config(data_dir: Path, split: str, config: str) -> list[dict[str, Any]]:
    path = data_dir / split / f"{config}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing exported CoREB file: {path}")
    return read_jsonl(path)


def normalize_query(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    query_id = row["query_id"]
    query_text = row.get("query") or row.get("code") or row.get("text") or ""
    normalized = dict(row)
    normalized["query"] = str(query_text)
    return query_id, normalized


def normalize_doc(row: dict[str, Any], doc_id_field: str, doc_text_field: str) -> tuple[str, dict[str, Any]]:
    doc_id = row[doc_id_field]
    text = row.get(doc_text_field) or row.get("text") or row.get("code") or ""
    normalized = dict(row)
    normalized["doc_id"] = doc_id
    normalized["text"] = str(text)
    return doc_id, normalized


def c2c_anchor_id(query_row: dict[str, Any]) -> str:
    anchor = query_row.get("meta_anchor_code_id", "")
    if anchor:
        return str(anchor)
    meta = query_row.get("meta")
    if isinstance(meta, dict):
        return str(meta.get("anchor_code_id", "") or "")
    if isinstance(meta, str):
        marker = "'anchor_code_id':"
        if marker in meta:
            tail = meta.split(marker, 1)[1].strip()
            quote = "'" if "'" in tail[:2] else '"'
            pieces = tail.split(quote)
            if len(pieces) >= 2:
                return pieces[1]
    return ""


def load_task(data_dir: Path, split: str, task: str) -> TaskData:
    spec = TASKS[task]
    queries = dict(
        normalize_query(row)
        for row in load_exported_config(data_dir, split, spec["queries"])
    )
    corpus = dict(
        normalize_doc(row, spec["doc_id"], spec["doc_text"])
        for row in load_exported_config(data_dir, split, spec["corpus"])
    )
    qrels: dict[str, dict[str, int]] = {}
    for row in load_exported_config(data_dir, split, spec["qrels"]):
        qid = row["query_id"]
        did = row["doc_id"]
        qrels.setdefault(qid, {})[did] = int(row["relevance"])
    exclude_doc_ids: dict[str, set[str]] = {}
    if task == "code2code":
        for query_id, query_row in queries.items():
            anchor = c2c_anchor_id(query_row)
            if anchor:
                exclude_doc_ids[query_id] = {anchor}
    return TaskData(name=task, split=split, queries=queries, corpus=corpus, qrels=qrels, exclude_doc_ids=exclude_doc_ids)


def task_names(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown CoREB task(s): {unknown}")
    return names


def query_text(task: TaskData, query_id: str) -> str:
    return task.queries[query_id]["query"]


def doc_text(task: TaskData, doc_id: str) -> str:
    return task.corpus[doc_id]["text"]
