import unittest

from scripts.analyze_student_surface_failures import (
    add_focus_from_gate,
    analyze_failures,
    parse_labeled_path,
    summarize_deltas,
)


def row(query_id, label, delta):
    return {
        "query_id": query_id,
        "weight_label": label,
        "delta": delta,
        "base": {"metrics": {"main_score": 0.5}, "top_docids": ["d1"]},
        "student": {"metrics": {"main_score": 0.5 + delta}, "top_docids": ["d2"]},
    }


class AnalyzeStudentSurfaceFailuresTest(unittest.TestCase):
    def test_parse_labeled_path_accepts_label_prefix(self):
        label, path = parse_labeled_path("slice0=/tmp/per_query.jsonl")

        self.assertEqual(label, "slice0")
        self.assertEqual(str(path), "/tmp/per_query.jsonl")

    def test_summarize_deltas_counts_passes_and_regressions(self):
        summary = summarize_deltas([0.006, -0.001, 0.002], required_delta=0.005)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["regression_count"], 1)
        self.assertAlmostEqual(summary["pass_fraction"], 1 / 3)

    def test_add_focus_from_gate_deduplicates_selected_surface(self):
        labels = add_focus_from_gate(
            ["model_card", "sparse_colbert"],
            {"selected_same_weight_label": "sparse_colbert"},
        )

        self.assertEqual(labels, ["model_card", "sparse_colbert"])

    def test_analyze_failures_reports_split_and_aggregate_surface_stats(self):
        result = analyze_failures(
            labeled_rows=[
                ("slice0", [row("q1", "model_card", 0.006), row("q1", "sparse_colbert", -0.002)]),
                ("slice1", [row("q2", "model_card", 0.001), row("q2", "sparse_colbert", 0.008)]),
            ],
            gate_summary={"gate_pass": False, "selected_same_weight_label": "sparse_colbert"},
            focus_labels=["model_card"],
            metric_key="main_score",
            required_delta=0.005,
            worst_limit=1,
        )

        self.assertFalse(result["gate_summary"]["gate_pass"])
        self.assertEqual(result["aggregate"]["query_count"], 2)
        self.assertEqual(result["aggregate"]["surface_stats"]["model_card"]["pass_count"], 1)
        self.assertEqual(result["aggregate"]["surface_stats"]["sparse_colbert"]["regression_count"], 1)
        self.assertEqual(
            result["aggregate"]["focus"]["model_card"]["worst_queries"][0]["query_id"],
            "q2",
        )


if __name__ == "__main__":
    unittest.main()
