import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_student_distill_splits import (
    prepare_splits,
    source_split,
    stable_fraction,
)


def make_row(query_id: str, split: str = "train") -> dict[str, object]:
    suffix = int(query_id[1:])
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "pos_doc_ids": [f"p{suffix}"],
        "pos": [f"positive {query_id}"],
        "pos_scores": [5.0 + suffix / 100.0],
        "neg_doc_ids": [f"n{suffix}a", f"n{suffix}b"],
        "neg": [f"negative a {query_id}", f"negative b {query_id}"],
        "neg_scores": [2.0, 1.0],
        "neg_sources": ["judged_negative", "unjudged_candidate"],
        "source": {"split": split},
    }


class PrepareStudentDistillSplitsTest(unittest.TestCase):
    def test_stable_fraction_is_deterministic(self):
        self.assertEqual(stable_fraction("q1", "seed"), stable_fraction("q1", "seed"))
        self.assertNotEqual(stable_fraction("q1", "seed"), stable_fraction("q1", "other"))

    def test_source_split_reads_nested_metadata(self):
        self.assertEqual(source_split({"source": {"split": "train"}}), "train")
        self.assertIsNone(source_split({"source": {}}))

    def test_prepare_splits_keeps_train_and_eval_queries_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            rows = [make_row(f"q{index}") for index in range(1, 31)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                max_train_rows=10,
                max_eval_rows=5,
                heldout_ratio=0.4,
                negatives_per_query=2,
                min_best_margin=1.0,
                min_all_margin=-1.0,
                label_transform="tanh",
                margin_temperature=4.0,
                seed="unit-test",
                allow_nontrain_source=False,
            )

            train_rows, eval_rows, summary = prepare_splits(args)

        train_queries = {row["query_id"] for row in train_rows}
        eval_queries = {row["query_id"] for row in eval_rows}
        self.assertTrue(train_queries)
        self.assertTrue(eval_queries)
        self.assertFalse(train_queries.intersection(eval_queries))
        self.assertEqual(summary["splitter"]["query_overlap"], 0)
        self.assertEqual(summary["train"]["expanded_margin_mse_triples"], 20)
        self.assertFalse(summary["raw_training_data_committed"])

    def test_prepare_splits_rejects_dev_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            input_path.write_text(json.dumps(make_row("q1", split="dev")) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                max_train_rows=10,
                max_eval_rows=5,
                heldout_ratio=0.5,
                negatives_per_query=1,
                min_best_margin=1.0,
                min_all_margin=-1.0,
                label_transform="tanh",
                margin_temperature=4.0,
                seed="unit-test",
                allow_nontrain_source=False,
            )

            with self.assertRaises(ValueError):
                prepare_splits(args)


if __name__ == "__main__":
    unittest.main()
