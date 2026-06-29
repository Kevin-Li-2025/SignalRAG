import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.augment_teacher_rows_bge_m3_components as augment
from scripts.augment_teacher_rows_bge_m3_components import (
    augment_row,
    score_pairs,
    surface_scores,
)
from scripts.sweep_bge_m3_hybrid_weights import WeightConfig


class FakeBgeM3Model:
    def compute_score(self, pairs, **_kwargs):
        return {
            "dense": [float(index + 1) for index, _pair in enumerate(pairs)],
            "sparse": [float((index + 1) * 10) for index, _pair in enumerate(pairs)],
            "colbert": [float((index + 1) * 100) for index, _pair in enumerate(pairs)],
        }


def teacher_row(query_id="q1"):
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "pos": [f"positive {query_id}"],
        "pos_doc_ids": [f"p-{query_id}"],
        "pos_scores": [1.0],
        "neg": [f"neg {query_id} 1", f"neg {query_id} 2"],
        "neg_doc_ids": [f"n-{query_id}-1", f"n-{query_id}-2"],
        "neg_scores": [0.5, 0.25],
        "source": {"split": "train"},
    }


class AugmentTeacherRowsBgeM3ComponentsTest(unittest.TestCase):
    def test_score_pairs_returns_component_dicts(self):
        components = score_pairs(
            FakeBgeM3Model(),
            [["q", "d1"], ["q", "d2"]],
            batch_size=8,
            max_passage_length=128,
        )

        self.assertEqual(
            components,
            [
                {"dense": 1.0, "sparse": 10.0, "colbert": 100.0},
                {"dense": 2.0, "sparse": 20.0, "colbert": 200.0},
            ],
        )

    def test_surface_scores_uses_named_weight_configs(self):
        components = [{"dense": 1.0, "sparse": 2.0, "colbert": 3.0}]

        scores = surface_scores(
            components,
            [WeightConfig("model_card", 0.4, 0.2, 0.4)],
        )

        self.assertAlmostEqual(scores["model_card"][0], 2.0)

    def test_augment_row_adds_component_and_surface_fields(self):
        row = teacher_row()
        components = [
            {"dense": 1.0, "sparse": 10.0, "colbert": 100.0},
            {"dense": 2.0, "sparse": 20.0, "colbert": 200.0},
            {"dense": 3.0, "sparse": 30.0, "colbert": 300.0},
        ]

        output = augment_row(
            row,
            components,
            [WeightConfig("dense_sparse", 0.7, 0.3, 0.0)],
        )

        self.assertEqual(output["bge_m3_component_pos_scores"]["dense"], [1.0])
        self.assertEqual(output["bge_m3_component_neg_scores"]["colbert"], [200.0, 300.0])
        self.assertEqual(output["bge_m3_surface_pos_scores"]["dense_sparse"], [3.7])
        self.assertEqual(output["bge_m3_surface_neg_scores"]["dense_sparse"], [7.4, 11.1])
        self.assertEqual(
            output["bge_m3_component_teacher"]["surfaces"]["dense_sparse"],
            [0.7, 0.3, 0.0],
        )

    def test_run_writes_augmented_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            output_path = root / "augmented.jsonl"
            summary_path = root / "summary.json"
            input_path.write_text(
                "".join(json.dumps(teacher_row(f"q{index}")) + "\n" for index in range(3)),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                output_jsonl=str(output_path),
                summary_json=str(summary_path),
                model="BAAI/bge-m3",
                model_path="",
                head_checkpoint="",
                batch_size=2,
                max_passage_length=128,
                row_limit=0,
                weight=["model_card:0.4,0.2,0.4", "sparse_colbert:0.0,0.33,0.67"],
                use_fp16=True,
                force=True,
            )

            with patch.object(augment, "load_bge_model", return_value=FakeBgeM3Model()):
                summary = augment.run(args)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["written_rows"], 3)
        self.assertEqual(summary["pairs_scored"], 9)
        self.assertEqual(persisted_summary["surface_distribution"]["model_card"]["rows_seen"], 3)
        self.assertFalse(summary["raw_training_data_committed"])


if __name__ == "__main__":
    unittest.main()
