import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_bge_m3_head_distill import (
    TeacherRow,
    batched,
    base_scores_from_row,
    fallback_scores_from_margins,
    listwise_scores_from_row,
    read_teacher_rows,
    row_listwise_distribution,
    softmax_distribution,
    training_summary,
)


class TrainBgeM3HeadDistillTest(unittest.TestCase):
    def test_read_teacher_rows_uses_pos_neg_and_target_margins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "query text",
                        "pos": ["positive passage"],
                        "neg": ["hard one", "hard two", "unused"],
                        "target_margins": [0.9, 0.8, 0.7],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = read_teacher_rows(path, max_rows=10, negatives_per_query=2)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].query_id, "q1")
        self.assertEqual(rows[0].positive, "positive passage")
        self.assertEqual(rows[0].negatives, ["hard one", "hard two"])
        self.assertEqual(rows[0].target_margins, [0.9, 0.8])

    def test_read_teacher_rows_preserves_teacher_and_base_score_distributions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "query text",
                        "pos": ["positive passage"],
                        "neg": ["hard one", "hard two"],
                        "target_margins": [0.9, 0.8],
                        "pos_scores": [7.0],
                        "neg_scores": [3.0, 1.0],
                        "bge_m3_hybrid_pos_score": 0.8,
                        "bge_m3_hybrid_neg_scores": [0.4, 0.1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = read_teacher_rows(path, max_rows=10, negatives_per_query=2)

        self.assertEqual(rows[0].teacher_scores, [7.0, 3.0, 1.0])
        self.assertEqual(rows[0].base_scores, [0.8, 0.4, 0.1])

    def test_read_teacher_rows_derives_margins_from_raw_teacher_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "query text",
                        "pos": ["positive passage"],
                        "neg": ["hard one", "hard two"],
                        "pos_scores": [10.0],
                        "neg_scores": [4.0, -1.0],
                        "bge_m3_hybrid_pos_scores": [0.8],
                        "bge_m3_hybrid_neg_scores": [0.4, 0.1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = read_teacher_rows(path, max_rows=10, negatives_per_query=2)

        self.assertEqual(rows[0].teacher_scores, [10.0, 4.0, -1.0])
        self.assertEqual(rows[0].target_margins, [6.0, 11.0])
        self.assertEqual(rows[0].base_scores, [0.8, 0.4, 0.1])

    def test_batched_rejects_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            list(batched([], 0))

    def test_listwise_score_helpers_use_available_scores(self):
        row = {
            "pos_scores": [8.0],
            "neg_scores": [4.0, 2.0, 1.0],
            "bge_m3_hybrid_pos_scores": [0.7],
            "bge_m3_hybrid_neg_scores": [0.3, 0.2, 0.1],
        }

        self.assertEqual(listwise_scores_from_row(row, 2), [8.0, 4.0, 2.0])
        self.assertEqual(base_scores_from_row(row, 2), [0.7, 0.3, 0.2])
        self.assertEqual(fallback_scores_from_margins([0.5, 0.25]), [0.0, -0.5, -0.25])

    def test_softmax_distribution_is_normalized_and_temperature_sensitive(self):
        sharp = softmax_distribution([4.0, 0.0], temperature=1.0)
        smooth = softmax_distribution([4.0, 0.0], temperature=4.0)

        self.assertAlmostEqual(sum(sharp), 1.0)
        self.assertAlmostEqual(sum(smooth), 1.0)
        self.assertGreater(sharp[0], smooth[0])

    def test_row_listwise_distribution_falls_back_to_margins(self):
        row = TeacherRow(
            query="q",
            positive="p",
            negatives=["n1", "n2"],
            target_margins=[0.5, 1.0],
        )

        distribution = row_listwise_distribution(row, temperature=1.0)

        self.assertAlmostEqual(sum(distribution), 1.0)
        self.assertGreater(distribution[0], distribution[1])
        self.assertGreater(distribution[1], distribution[2])

    def test_training_summary_records_checkpoint_as_uncommitted(self):
        args = argparse.Namespace(
            model_path="BAAI/bge-m3",
            train_jsonl="/tmp/train.jsonl",
            negatives_per_query=2,
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_length=128,
            dense_weight=0.4,
            sparse_weight=0.2,
            colbert_weight=0.4,
            objective="listwise_kl",
            teacher_temperature=2.0,
            base_anchor_weight=0.25,
            base_anchor_temperature=1.0,
            target_margin_scale=0.1,
            head_l2_anchor_weight=1.0,
        )
        summary = training_summary(
            args=args,
            rows=[
                TeacherRow(
                    query="q",
                    positive="p",
                    negatives=["n1", "n2"],
                    target_margins=[0.5, 0.25],
                    teacher_scores=[5.0, 2.0, 1.0],
                    base_scores=[0.8, 0.3, 0.2],
                )
            ],
            step_losses=[1.0, 0.5],
            elapsed_seconds=12.0,
            checkpoint_path=Path("/tmp/head.pt"),
        )

        self.assertEqual(summary["trainable_modules"], ["sparse_linear", "colbert_linear"])
        self.assertTrue(summary["encoder_frozen"])
        self.assertFalse(summary["checkpoint_committed"])
        self.assertEqual(summary["objective"], "listwise_kl")
        self.assertEqual(summary["rows_with_teacher_scores"], 1)
        self.assertEqual(summary["rows_with_base_scores"], 1)
        self.assertEqual(summary["base_anchor_weight"], 0.25)
        self.assertEqual(summary["target_margin_scale"], 0.1)
        self.assertEqual(summary["head_l2_anchor_weight"], 1.0)
        self.assertEqual(summary["target_margin_stats"]["mean"], 0.375)


if __name__ == "__main__":
    unittest.main()
