import unittest

from scripts.prepare_failure_aware_teacher_rows import (
    allocate_counts,
    build_failure_surface_plan,
    format_counts,
    format_weights,
    parse_label_values,
    parse_surfaces,
)


class PrepareFailureAwareTeacherRowsTest(unittest.TestCase):
    def test_parse_surfaces_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            parse_surfaces("model_card,model_card")

    def test_parse_label_values_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            parse_label_values("model_card=-1")

    def test_allocate_counts_gives_each_positive_label_one_slot(self):
        counts = allocate_counts(
            {"model_card": 4.0, "training_default": 2.0, "sparse": 1.0},
            total=8,
        )

        self.assertEqual(sum(counts.values()), 8)
        self.assertGreaterEqual(counts["model_card"], counts["training_default"])
        self.assertGreaterEqual(counts["training_default"], counts["sparse"])
        self.assertGreaterEqual(min(counts.values()), 1)

    def test_build_failure_surface_plan_prioritizes_failure_surfaces(self):
        failure = {
            "required_delta": 0.005,
            "aggregate": {
                "surface_stats": {
                    "model_card": {
                        "pass_fraction": 0.10,
                        "regression_fraction": 0.08,
                        "mean_delta": 0.003,
                    },
                    "training_default": {
                        "pass_fraction": 0.05,
                        "regression_fraction": 0.04,
                        "mean_delta": 0.0,
                    },
                    "sparse_colbert": {
                        "pass_fraction": 0.14,
                        "regression_fraction": 0.05,
                        "mean_delta": 0.0047,
                    },
                }
            },
        }

        plan = build_failure_surface_plan(
            failure,
            surfaces=["model_card", "training_default", "sparse_colbert"],
            surface_priorities={
                "model_card": 1.6,
                "training_default": 1.4,
                "sparse_colbert": 0.8,
            },
            negatives_per_query=8,
        )

        self.assertEqual(sum(plan["surface_negative_counts"].values()), 8)
        self.assertGreaterEqual(
            plan["surface_negative_counts"]["model_card"],
            plan["surface_negative_counts"]["sparse_colbert"],
        )
        self.assertIn("model_card=", plan["surface_negative_counts_spec"])
        self.assertIn("training_default=", plan["surface_average_source_weights_spec"])

    def test_format_helpers_are_stable(self):
        self.assertEqual(format_counts({"a": 2, "b": 1}), "a=2,b=1")
        self.assertEqual(format_weights({"a": 2.0, "b": 0.5}), "a=2,b=0.5")


if __name__ == "__main__":
    unittest.main()
