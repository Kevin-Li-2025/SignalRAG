from coreb_sota.data import TaskData, c2c_anchor_id
from coreb_sota.metrics import evaluate_run


def test_c2c_anchor_id_from_exported_flat_meta() -> None:
    row = {"meta_anchor_code_id": "code_v202601_00173"}
    assert c2c_anchor_id(row) == "code_v202601_00173"


def test_c2c_anchor_exclusion_shifts_metrics() -> None:
    qrels = {"q1": {"positive": 2}}
    run = {"q1": {"anchor": 1.0, "positive": 0.9}}

    unfiltered = evaluate_run(qrels, run, k=1)
    filtered = evaluate_run(qrels, run, k=1, exclude_doc_ids={"q1": {"anchor"}})

    assert unfiltered["ndcg@1"] == 0.0
    assert filtered["ndcg@1"] == 1.0
    assert filtered["recall@1"] == 1.0
    assert filtered["map@1"] == 1.0


def test_task_data_carries_exclusions() -> None:
    task = TaskData(
        name="code2code",
        split="release_v2602",
        queries={},
        corpus={},
        qrels={},
        exclude_doc_ids={"q1": {"anchor"}},
    )
    assert task.exclude_doc_ids["q1"] == {"anchor"}
