import unittest

from scripts.check_student_diagnostic_gate import check_gate, summarize_split


def summary(
    *,
    base_best_label="base_best",
    base_best_score=0.80,
    student_best_label="student_best",
    student_best_score=0.81,
    model_card_base=0.78,
    model_card_student=0.79,
    sparse_base=0.77,
    sparse_student=0.78,
):
    rows = [
        ("model_card", model_card_base, model_card_student),
        ("sparse_heavy", sparse_base, sparse_student),
        (base_best_label, base_best_score, base_best_score),
        (student_best_label, student_best_score, student_best_score),
    ]
    base_results = [
        {"weight_label": label, "metrics": {"main_score": base_score}}
        for label, base_score, _student_score in rows
    ]
    student_results = [
        {"weight_label": label, "metrics": {"main_score": student_score}}
        for label, _base_score, student_score in rows
    ]
    return {
        "query_count": 100,
        "query_offset": 0,
        "query_stride": 2,
        "base": {
            "results": base_results,
            "best": {"weight_label": base_best_label, "metrics": {"main_score": base_best_score}},
        },
        "student": {
            "results": student_results,
            "best": {
                "weight_label": student_best_label,
                "metrics": {"main_score": student_best_score},
            },
        },
        "comparisons": [
            {
                "weight_label": label,
                "base_main_score": base_score,
                "student_main_score": student_score,
                "delta": student_score - base_score,
            }
            for label, base_score, student_score in rows
        ],
    }


class CheckStudentDiagnosticGateTest(unittest.TestCase):
    def test_summarize_split_reports_key_surfaces(self):
        split = summarize_split(
            summary(),
            metric_key="main_score",
            model_card_label="model_card",
        )

        self.assertAlmostEqual(split["best_vs_best"]["delta"], 0.01)
        self.assertAlmostEqual(split["model_card"]["delta"], 0.01)
        self.assertEqual(split["best_same_weight"]["weight_label"], "model_card")

    def test_gate_passes_only_when_tune_and_heldout_clear_all_criteria(self):
        result = check_gate(
            tune_summary=summary(model_card_student=0.786, sparse_student=0.776),
            heldout_summary=summary(model_card_student=0.786, sparse_student=0.776),
            metric_key="main_score",
            model_card_label="model_card",
            required_delta=0.005,
        )

        self.assertTrue(result["gate_pass"])
        self.assertTrue(all(result["criteria"].values()))

    def test_gate_rejects_tune_only_gain(self):
        result = check_gate(
            tune_summary=summary(model_card_student=0.786, sparse_student=0.776),
            heldout_summary=summary(
                student_best_score=0.801,
                model_card_student=0.782,
                sparse_student=0.771,
            ),
            metric_key="main_score",
            model_card_label="model_card",
            required_delta=0.005,
        )

        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["criteria"]["heldout_best_vs_best"])
        self.assertFalse(result["criteria"]["heldout_model_card"])
        self.assertFalse(result["criteria"]["heldout_tune_selected_same_weight"])


if __name__ == "__main__":
    unittest.main()
