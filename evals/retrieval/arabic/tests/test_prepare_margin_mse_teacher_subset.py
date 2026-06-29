import unittest

from scripts.prepare_margin_mse_teacher_subset import (
    prepare_row,
    row_quality,
    transformed_margin,
)


class PrepareMarginMseTeacherSubsetTest(unittest.TestCase):
    def test_transformed_margin_uses_tanh_temperature(self):
        self.assertEqual(transformed_margin(0.0, "tanh", 4.0), 0.0)
        self.assertAlmostEqual(transformed_margin(4.0, "tanh", 4.0), 0.7615941559)
        self.assertEqual(transformed_margin(2.5, "raw", 4.0), 2.5)

    def test_row_quality_reports_best_and_all_margin(self):
        quality = row_quality({"pos_scores": [5.0, 4.0], "neg_scores": [3.0, 1.0]})

        self.assertIsNotNone(quality)
        assert quality is not None
        self.assertEqual(quality["best_margin"], 2.0)
        self.assertEqual(quality["all_margin"], 1.0)
        self.assertEqual(quality["mean_margin"], 2.5)

    def test_prepare_row_selects_best_positive_and_hardest_negatives(self):
        row = {
            "query_id": "q1",
            "query": "query",
            "pos_doc_ids": ["p-low", "p-high"],
            "pos": ["positive low", "positive high"],
            "pos_scores": [3.0, 6.0],
            "neg_doc_ids": ["n-low", "n-high", "n-mid"],
            "neg": ["negative low", "negative high", "negative mid"],
            "neg_scores": [0.5, 4.0, 2.0],
            "neg_sources": ["unjudged", "judged", "unjudged"],
            "bge_m3_hybrid_pos_scores": [0.2, 0.7],
            "bge_m3_hybrid_neg_scores": [0.1, 0.6, 0.3],
            "reranker_teacher": {"model": "teacher"},
            "source": {"split": "train"},
        }

        prepared = prepare_row(
            row,
            negatives_per_query=2,
            label_transform="tanh",
            margin_temperature=4.0,
        )

        self.assertEqual(prepared["pos_doc_ids"], ["p-high"])
        self.assertEqual(prepared["neg_doc_ids"], ["n-high", "n-mid"])
        self.assertEqual(prepared["pos_scores"], [1.0])
        self.assertAlmostEqual(
            prepared["target_margins"][0],
            transformed_margin(2.0, "tanh", 4.0),
        )
        self.assertAlmostEqual(
            prepared["neg_scores"][0],
            1.0 - prepared["target_margins"][0],
        )
        self.assertEqual(prepared["bge_m3_hybrid_pos_score"], 0.7)
        self.assertEqual(prepared["bge_m3_hybrid_neg_scores"], [0.6, 0.3])


if __name__ == "__main__":
    unittest.main()
