import unittest

from scripts.check_student_multisplit_gate import (
    check_multisplit_gate,
    parse_split_arg,
    select_same_weight_label,
)


def summary(
    *,
    base_best_label="colbert_heavy",
    student_best_label="colbert_heavy",
    base_best_score=0.80,
    student_best_score=0.81,
    model_card_base=0.78,
    model_card_student=0.79,
    sparse_colbert_base=0.77,
    sparse_colbert_student=0.78,
    dense_base=0.76,
    dense_student=0.765,
):
    rows = [
        ("model_card", model_card_base, model_card_student),
        ("sparse_colbert", sparse_colbert_base, sparse_colbert_student),
        ("dense_heavy", dense_base, dense_student),
        (base_best_label, base_best_score, base_best_score),
        (student_best_label, student_best_score, student_best_score),
    ]
    deduped = {}
    for label, base_score, student_score in rows:
        deduped[label] = (base_score, student_score)
    return {
        "query_count": 50,
        "query_offset": 0,
        "query_stride": 4,
        "base": {
            "results": [
                {"weight_label": label, "metrics": {"main_score": base_score}}
                for label, (base_score, _student_score) in deduped.items()
            ],
            "best": {"weight_label": base_best_label, "metrics": {"main_score": base_best_score}},
        },
        "student": {
            "results": [
                {"weight_label": label, "metrics": {"main_score": student_score}}
                for label, (_base_score, student_score) in deduped.items()
            ],
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
            for label, (base_score, student_score) in deduped.items()
        ],
    }


class CheckStudentMultisplitGateTest(unittest.TestCase):
    def test_parse_split_arg_accepts_labeled_and_unlabeled_paths(self):
        label, path = parse_split_arg("fold0=/tmp/fold0/summary.json")
        self.assertEqual(label, "fold0")
        self.assertEqual(str(path), "/tmp/fold0/summary.json")

        label, path = parse_split_arg("/tmp/fold1/summary.json")
        self.assertEqual(label, "fold1")
        self.assertEqual(str(path), "/tmp/fold1/summary.json")

    def test_select_same_weight_label_uses_mean_across_splits(self):
        selected = select_same_weight_label(
            [
                summary(sparse_colbert_student=0.785, dense_student=0.780),
                summary(sparse_colbert_student=0.786, dense_student=0.770),
            ]
        )

        self.assertEqual(selected["selected_label"], "sparse_colbert")
        self.assertGreater(
            selected["by_label"]["sparse_colbert"]["mean_delta"],
            selected["by_label"]["dense_heavy"]["mean_delta"],
        )

    def test_multisplit_gate_passes_when_all_surfaces_are_stable(self):
        result = check_multisplit_gate(
            labeled_summaries=[
                ("fold0", summary()),
                ("fold1", summary(student_best_score=0.812, model_card_student=0.791)),
            ],
            metric_key="main_score",
            model_card_label="model_card",
            required_delta=0.005,
            min_pass_fraction=1.0,
        )

        self.assertTrue(result["gate_pass"])
        self.assertTrue(all(result["criteria"].values()))

    def test_multisplit_gate_rejects_single_split_instability(self):
        result = check_multisplit_gate(
            labeled_summaries=[
                (
                    "tune",
                    summary(
                        student_best_score=0.799,
                        model_card_student=0.782,
                        sparse_colbert_student=0.772,
                    ),
                ),
                (
                    "heldout",
                    summary(
                        student_best_score=0.812,
                        model_card_student=0.790,
                        sparse_colbert_student=0.781,
                    ),
                ),
            ],
            metric_key="main_score",
            model_card_label="model_card",
            required_delta=0.005,
            min_pass_fraction=1.0,
        )

        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["criteria"]["best_vs_best_pass_fraction"])
        self.assertFalse(result["criteria"]["model_card_pass_fraction"])
        self.assertFalse(result["criteria"]["selected_same_weight_pass_fraction"])


if __name__ == "__main__":
    unittest.main()
