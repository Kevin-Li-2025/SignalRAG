import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_surface_aware_teacher_rows import (
    classify_negative,
    minmax,
    parse_args,
    prepare_splits,
    surface_target_scores,
    validate_args,
)


def make_row(query_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "pos_doc_ids": [f"p-{query_id}"],
        "pos": [f"positive {query_id}"],
        "pos_scores": [1.0],
        "neg_doc_ids": [f"n-{query_id}-{index}" for index in range(8)],
        "neg": [f"negative {index} {query_id}" for index in range(8)],
        "neg_scores": [0.92, 0.75, 0.70, 0.90, 0.65, 0.20, 0.10, 0.05],
        "neg_sources": [
            "agreement",
            "hybrid_fp",
            "hybrid_fp",
            "target_hard",
            "middle",
            "easy",
            "easy",
            "easy",
        ],
        "original_reranker_pos_scores": [1.0],
        "original_reranker_neg_scores": [0.92, 0.75, 0.70, 0.90, 0.65, 0.20, 0.10, 0.05],
        "bge_m3_hybrid_pos_scores": [0.50],
        "bge_m3_hybrid_neg_scores": [0.46, 0.55, 0.52, 0.20, 0.10, 0.00, -0.05, -0.10],
        "source": {"split": "train"},
    }


def default_args(input_path: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_jsonl=str(input_path),
        output_dir=str(output_dir),
        train_jsonl="train.jsonl",
        eval_jsonl="eval.jsonl",
        max_train_rows=18,
        max_eval_rows=6,
        heldout_ratio=0.25,
        negatives_per_query=8,
        agreement_hard_negatives=1,
        hybrid_false_positive_negatives=2,
        target_hard_negatives=1,
        middle_negatives=1,
        easy_negatives=3,
        min_hybrid_false_positive_available=1,
        min_target_hard_available=1,
        min_target_best_margin=0.0,
        max_target_best_margin=None,
        target_hard_margin=0.15,
        hybrid_hard_margin=0.15,
        middle_margin=0.45,
        teacher_separation_margin=0.05,
        reranker_weight=1.0,
        hybrid_weight=0.0,
        existing_weight=0.0,
        score_scale=1.0,
        missing_hybrid="skip",
        seed="unit-surface-aware",
        allow_nontrain_source=False,
        force=True,
    )


class PrepareSurfaceAwareTeacherRowsTest(unittest.TestCase):
    def test_minmax_handles_constant_values(self) -> None:
        self.assertEqual(minmax([3.0, 3.0, 3.0]), [0.5, 0.5, 0.5])

    def test_surface_target_scores_blend_row_normalized_sources(self) -> None:
        row = {
            "query": "q",
            "pos": ["p"],
            "neg": ["n"],
            "pos_scores": [0.8],
            "neg_scores": [0.2],
            "original_reranker_pos_scores": [10.0],
            "original_reranker_neg_scores": [0.0],
            "bge_m3_hybrid_pos_scores": [0.0],
            "bge_m3_hybrid_neg_scores": [10.0],
        }

        result = surface_target_scores(
            row,
            reranker_weight=0.5,
            hybrid_weight=0.5,
            existing_weight=0.0,
            score_scale=1.0,
            missing_hybrid="skip",
        )

        self.assertIsNotNone(result)
        assert result is not None
        pos_scores, neg_scores, sources = result
        self.assertAlmostEqual(pos_scores[0], 0.5)
        self.assertAlmostEqual(neg_scores[0], 0.5)
        self.assertTrue(sources["hybrid_complete"])

    def test_classify_negative_prioritizes_hybrid_false_positive(self) -> None:
        label = classify_negative(
            target_margin=0.30,
            hybrid_margin=-0.10,
            target_hard_margin=0.15,
            hybrid_hard_margin=0.15,
            middle_margin=0.45,
            teacher_separation_margin=0.05,
        )

        self.assertEqual(label, "hybrid_false_positive")

    def test_prepare_splits_surface_mix_is_query_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            rows = [make_row(f"q{index}") for index in range(32)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            train_rows, heldout_rows, summary = prepare_splits(default_args(input_path, root))

        self.assertTrue(train_rows)
        self.assertTrue(heldout_rows)
        train_queries = {row["query_id"] for row in train_rows}
        heldout_queries = {row["query_id"] for row in heldout_rows}
        self.assertFalse(train_queries.intersection(heldout_queries))
        self.assertEqual(summary["splitter"]["query_overlap"], 0)
        self.assertGreaterEqual(
            summary["train"]["negative_strata_counts"]["hybrid_false_positive"],
            len(train_rows),
        )
        self.assertEqual(len(train_rows[0]["neg"]), 8)
        self.assertIn("surface_teacher", train_rows[0])
        self.assertIn("teacher_distribution_by_temperature", summary["train"]["distribution"])
        self.assertFalse(summary["raw_training_data_committed"])

    def test_prepare_splits_filters_rows_without_required_hybrid_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            rows = []
            for index in range(16):
                row = make_row(f"bad-{index}")
                row["bge_m3_hybrid_neg_scores"] = [0.10, 0.08, 0.06, 0.04, 0.02, 0.0, -0.02, -0.04]
                rows.append(row)
            rows.extend(make_row(f"good-{index}") for index in range(16))
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            train_rows, heldout_rows, summary = prepare_splits(default_args(input_path, root))

        self.assertTrue(train_rows)
        self.assertTrue(heldout_rows)
        self.assertEqual(summary["candidate_rows"], 16)
        for row in train_rows + heldout_rows:
            self.assertTrue(str(row["query_id"]).startswith("good-"))

    def test_validate_args_rejects_bad_negative_mix(self) -> None:
        args = default_args(Path("input.jsonl"), Path("out"))
        args.easy_negatives = 2

        with self.assertRaisesRegex(ValueError, "sum"):
            validate_args(args)

    def test_cli_defaults_focus_on_hybrid_false_positives(self) -> None:
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_surface_aware_teacher_rows.py",
                "--input-jsonl",
                "input.jsonl",
                "--output-dir",
                "out",
            ]
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.agreement_hard_negatives, 1)
        self.assertEqual(args.hybrid_false_positive_negatives, 7)
        self.assertEqual(args.target_hard_negatives, 0)
        self.assertEqual(args.middle_negatives, 0)
        self.assertEqual(args.easy_negatives, 0)
        validate_args(args)


if __name__ == "__main__":
    unittest.main()
