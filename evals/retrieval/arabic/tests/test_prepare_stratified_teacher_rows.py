import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_stratified_teacher_rows import (
    available_negative_strata,
    choose_negatives,
    prepare_splits,
    row_bin,
    select_balanced_rows,
    validate_args,
)


def make_row(query_id: str, margins: list[float] | None = None) -> dict[str, object]:
    pos_score = 10.0
    if margins is None:
        margins = [0.1, 0.25, 0.75, 1.5, 2.5, 3.5, 5.0, 8.0]
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "pos_doc_ids": [f"p-{query_id}"],
        "pos": [f"positive {query_id}"],
        "pos_scores": [pos_score],
        "neg_doc_ids": [f"n-{query_id}-{index}" for index in range(8)],
        "neg": [f"negative {index} {query_id}" for index in range(8)],
        "neg_scores": [pos_score - margin for margin in margins],
        "neg_sources": ["hard", "hard", "hard", "middle", "middle", "middle", "easy", "easy"],
        "bge_m3_hybrid_pos_scores": [0.9],
        "bge_m3_hybrid_neg_scores": [0.5, 0.4, 0.35, 0.25, 0.2, 0.1, 0.0, -0.1],
        "source": {"split": "train"},
    }


class PrepareStratifiedTeacherRowsTest(unittest.TestCase):
    def test_choose_negatives_honors_requested_mix(self):
        row = make_row("q1")
        available = available_negative_strata(row)
        selected, diagnostics = choose_negatives(
            row,
            pos_score=10.0,
            negatives_per_query=8,
            hard_negatives=3,
            middle_negatives=3,
            easy_negatives=2,
            seed="unit",
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(available["hard"], 3)
        self.assertEqual(available["middle"], 3)
        self.assertEqual(available["easy"], 2)
        self.assertTrue(diagnostics["requested_mix_met"])
        self.assertEqual(diagnostics["selected_by_stratum"]["hard"], 3)
        self.assertEqual(diagnostics["selected_by_stratum"]["middle"], 3)
        self.assertEqual(diagnostics["selected_by_stratum"]["easy"], 2)

    def test_select_balanced_rows_round_robins_margin_bins(self):
        rows = [
            make_row("q1", margins=[-1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]),
            make_row("q2", margins=[0.25, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]),
            make_row("q3", margins=[0.75, 1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]),
            make_row("q4", margins=[1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            make_row("q5", margins=[2.5, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            make_row("q6", margins=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            make_row("q7", margins=[5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            make_row("q8", margins=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
        ]

        selected = select_balanced_rows(rows, 6, "unit")
        bins = {row_bin(row) for row in selected}

        self.assertEqual(len(selected), 6)
        self.assertIn("negative", bins)
        self.assertIn("0_to_0.5", bins)
        self.assertIn("0.5_to_1", bins)
        self.assertIn("1_to_2", bins)
        self.assertIn("2_to_4", bins)
        self.assertIn("4_plus", bins)

    def test_prepare_splits_writes_scaled_scores_and_query_disjoint_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            margin_sets = [
                [-1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                [0.25, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                [0.75, 1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                [1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [2.5, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
            rows = [
                make_row(f"q{index}", margins=margin_sets[index % len(margin_sets)])
                for index in range(40)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                output_dir=tmpdir,
                train_jsonl="train.jsonl",
                eval_jsonl="eval.jsonl",
                max_train_rows=20,
                max_eval_rows=8,
                heldout_ratio=0.3,
                negatives_per_query=8,
                hard_negatives=3,
                middle_negatives=3,
                easy_negatives=2,
                min_hard_available=1,
                min_middle_available=1,
                min_easy_available=1,
                min_best_margin=None,
                max_best_margin=None,
                score_scale=0.25,
                seed="unit-test",
                allow_nontrain_source=False,
                force=True,
            )

            train_rows, eval_rows, summary = prepare_splits(args)

        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        train_queries = {row["query_id"] for row in train_rows}
        eval_queries = {row["query_id"] for row in eval_rows}
        self.assertFalse(train_queries.intersection(eval_queries))
        self.assertEqual(summary["splitter"]["query_overlap"], 0)
        self.assertEqual(summary["score_transform"]["score_scale"], 0.25)
        self.assertEqual(
            summary["negative_mix"]["minimum_available"],
            {"hard": 1, "middle": 1, "easy": 1},
        )
        self.assertEqual(train_rows[0]["pos_scores"], [2.5])
        self.assertEqual(len(train_rows[0]["neg"]), 8)
        self.assertIn("teacher_distribution_by_temperature", summary["train"]["distribution"])
        self.assertFalse(summary["raw_training_data_committed"])

    def test_prepare_splits_filters_rows_without_minimum_mix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            rows = [
                make_row(f"easy-{index}", margins=[5.0] * 8)
                for index in range(12)
            ] + [
                make_row(
                    f"mixed-{index}",
                    margins=[0.25, 1.5, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                for index in range(12)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                output_dir=tmpdir,
                train_jsonl="train.jsonl",
                eval_jsonl="eval.jsonl",
                max_train_rows=20,
                max_eval_rows=8,
                heldout_ratio=0.4,
                negatives_per_query=8,
                hard_negatives=1,
                middle_negatives=2,
                easy_negatives=5,
                min_hard_available=1,
                min_middle_available=1,
                min_easy_available=1,
                min_best_margin=None,
                max_best_margin=None,
                score_scale=0.25,
                seed="unit-test-filter",
                allow_nontrain_source=False,
                force=True,
            )

            train_rows, eval_rows, summary = prepare_splits(args)

        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        self.assertEqual(summary["insufficient_minimum_mix_rows"], 12)
        for row in train_rows + eval_rows:
            self.assertEqual(row["neg_strata"].count("hard"), 1)
            self.assertEqual(row["neg_strata"].count("middle"), 2)
            self.assertEqual(row["neg_strata"].count("easy"), 5)

    def test_prepare_splits_filters_rows_below_min_best_margin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            rows = [
                make_row(
                    f"bad-{index}",
                    margins=[-0.25, 1.5, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                for index in range(12)
            ] + [
                make_row(
                    f"good-{index}",
                    margins=[0.25, 1.5, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                for index in range(12)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                output_dir=tmpdir,
                train_jsonl="train.jsonl",
                eval_jsonl="eval.jsonl",
                max_train_rows=20,
                max_eval_rows=8,
                heldout_ratio=0.4,
                negatives_per_query=4,
                hard_negatives=1,
                middle_negatives=1,
                easy_negatives=2,
                min_hard_available=1,
                min_middle_available=1,
                min_easy_available=2,
                min_best_margin=0.05,
                max_best_margin=None,
                score_scale=0.25,
                seed="unit-test-margin-filter",
                allow_nontrain_source=False,
                force=True,
            )

            train_rows, eval_rows, summary = prepare_splits(args)

        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        self.assertEqual(summary["insufficient_minimum_mix_rows"], 12)
        self.assertEqual(summary["negative_mix"]["min_best_margin"], 0.05)
        for row in train_rows + eval_rows:
            self.assertTrue(row["query_id"].startswith("good-"))

    def test_prepare_splits_filters_rows_above_max_best_margin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "teacher.jsonl"
            rows = [
                make_row(
                    f"band-{index}",
                    margins=[0.25, 1.5, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                for index in range(12)
            ] + [
                make_row(
                    f"easy-{index}",
                    margins=[5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                )
                for index in range(12)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_jsonl=str(input_path),
                output_dir=tmpdir,
                train_jsonl="train.jsonl",
                eval_jsonl="eval.jsonl",
                max_train_rows=20,
                max_eval_rows=8,
                heldout_ratio=0.4,
                negatives_per_query=4,
                hard_negatives=1,
                middle_negatives=1,
                easy_negatives=2,
                min_hard_available=1,
                min_middle_available=1,
                min_easy_available=2,
                min_best_margin=0.05,
                max_best_margin=0.5,
                score_scale=0.25,
                seed="unit-test-margin-band",
                allow_nontrain_source=False,
                force=True,
            )

            train_rows, eval_rows, summary = prepare_splits(args)

        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        self.assertEqual(summary["insufficient_minimum_mix_rows"], 12)
        self.assertEqual(summary["negative_mix"]["max_best_margin"], 0.5)
        for row in train_rows + eval_rows:
            self.assertTrue(row["query_id"].startswith("band-"))

    def test_validate_args_rejects_impossible_margin_band(self):
        args = argparse.Namespace(
            heldout_ratio=0.2,
            negatives_per_query=4,
            hard_negatives=1,
            middle_negatives=1,
            easy_negatives=2,
            score_scale=0.25,
            min_hard_available=0,
            min_middle_available=0,
            min_easy_available=0,
            min_best_margin=1.0,
            max_best_margin=0.5,
        )

        with self.assertRaisesRegex(ValueError, "min-best-margin"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
