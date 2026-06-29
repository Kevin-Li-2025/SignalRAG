import unittest

from scripts.analyze_teacher_score_distribution import (
    analyze_rows,
    margin_bin,
    normalized_entropy,
    numeric_stats,
    quantile,
    softmax,
)


class AnalyzeTeacherScoreDistributionTest(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertEqual(quantile([1.0, 3.0], 0.5), 2.0)
        self.assertEqual(quantile([3.0, 1.0, 5.0], 0.5), 3.0)
        self.assertIsNone(quantile([], 0.5))

    def test_numeric_stats_reports_common_quantiles(self):
        stats = numeric_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["p50"], 2.5)

    def test_softmax_entropy_drops_with_lower_temperature(self):
        high_temp = normalized_entropy(softmax([4.0, 3.0, 1.0], 1.0))
        low_temp = normalized_entropy(softmax([4.0, 3.0, 1.0], 0.1))
        self.assertLess(low_temp, high_temp)

    def test_margin_bin_labels_boundaries(self):
        self.assertEqual(margin_bin(-0.1), "negative")
        self.assertEqual(margin_bin(0.0), "0_to_0.5")
        self.assertEqual(margin_bin(0.5), "0.5_to_1")
        self.assertEqual(margin_bin(4.0), "4_plus")

    def test_analyze_rows_reports_bins_entropy_and_hybrid_margin(self):
        rows = [
            {
                "query_id": "q1",
                "pos_scores": [5.0],
                "neg_scores": [4.2, 2.0, 0.0],
                "neg_sources": ["hard", "middle", "easy"],
                "bge_m3_hybrid_pos_scores": [0.9],
                "bge_m3_hybrid_neg_scores": [0.6, 0.2, 0.1],
            },
            {
                "query_id": "q2",
                "pos_scores": [4.0],
                "neg_scores": [1.5, 1.0],
                "neg_sources": ["middle", "easy"],
                "bge_m3_hybrid_pos_scores": [0.8],
                "bge_m3_hybrid_neg_scores": [0.3, 0.1],
            },
        ]

        summary = analyze_rows(rows, [1.0, 0.1])

        self.assertEqual(summary["rows_with_scores"], 2)
        self.assertEqual(summary["negative_pairs"], 5)
        self.assertEqual(summary["negative_source_counts"]["middle"], 2)
        self.assertEqual(summary["margin_bins"]["best_pos_minus_max_neg"]["0.5_to_1"], 1)
        self.assertEqual(summary["margin_bins"]["best_pos_minus_max_neg"]["2_to_4"], 1)
        entropy_high = summary["teacher_distribution_by_temperature"]["1"]["mean_normalized_entropy"]
        entropy_low = summary["teacher_distribution_by_temperature"]["0.1"]["mean_normalized_entropy"]
        self.assertLess(entropy_low, entropy_high)
        self.assertEqual(summary["teacher_vs_bge_m3_hybrid_margin"]["paired_rows"], 2)
        self.assertIn("sampling_implications", summary)


if __name__ == "__main__":
    unittest.main()
