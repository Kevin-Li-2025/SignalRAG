from finmteb_sota.data import RerankRecord, flatten_records


def test_flatten_records_is_deterministic_and_label_independent() -> None:
    records = [
        RerankRecord(
            query_id=f"q-{idx}",
            query=f"query {idx}",
            positives=(f"positive {idx}",),
            negatives=tuple(f"negative {idx}-{j}" for j in range(3)),
        )
        for idx in range(24)
    ]
    first = flatten_records(records)
    second = flatten_records(records)
    assert first == second
    _, _, labels, qids = first
    positive_positions = []
    for qid in dict.fromkeys(qids):
        group_labels = [
            label
            for label, candidate_qid in zip(labels, qids, strict=True)
            if candidate_qid == qid
        ]
        positive_positions.append(group_labels.index(1))
    assert len(set(positive_positions)) > 1
