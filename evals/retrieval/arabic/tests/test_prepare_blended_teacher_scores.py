import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_blended_teacher_scores import (
    blend_row,
    blended_scores,
    main,
    minmax,
)


class PrepareBlendedTeacherScoresTest(unittest.TestCase):
    def test_minmax_handles_constant_values(self) -> None:
        self.assertEqual(minmax([3.0, 3.0]), [0.5, 0.5])

    def test_blended_scores_combines_row_normalized_sources(self) -> None:
        scores = blended_scores(
            [10.0, 0.0, 5.0],
            [0.2, 0.8, 0.1],
            reranker_weight=0.65,
            score_scale=1.0,
        )

        self.assertAlmostEqual(scores[0], 0.65 * 1.0 + 0.35 * (1.0 / 7.0))
        self.assertAlmostEqual(scores[1], 0.65 * 0.0 + 0.35 * 1.0)
        self.assertAlmostEqual(scores[2], 0.65 * 0.5 + 0.35 * 0.0)

    def test_blend_row_preserves_original_reranker_scores(self) -> None:
        row = {
            "query": "q",
            "pos": ["p"],
            "neg": ["n1", "n2"],
            "pos_scores": [10.0],
            "neg_scores": [0.0, 5.0],
            "bge_m3_hybrid_pos_scores": [0.2],
            "bge_m3_hybrid_neg_scores": [0.8, 0.1],
        }

        blended, reason = blend_row(
            row,
            reranker_weight=0.65,
            score_scale=1.0,
            missing_hybrid="skip",
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(blended)
        assert blended is not None
        self.assertEqual(blended["original_reranker_pos_scores"], [10.0])
        self.assertEqual(blended["original_reranker_neg_scores"], [0.0, 5.0])
        self.assertEqual(blended["score_blend_teacher"]["reranker_weight"], 0.65)
        self.assertAlmostEqual(blended["pos_scores"][0], 0.7)

    def test_blend_row_skips_or_falls_back_when_hybrid_missing(self) -> None:
        row = {
            "query": "q",
            "pos": ["p"],
            "neg": ["n"],
            "pos_scores": [2.0],
            "neg_scores": [1.0],
        }

        skipped, reason = blend_row(
            row,
            reranker_weight=0.65,
            score_scale=1.0,
            missing_hybrid="skip",
        )
        self.assertIsNone(skipped)
        self.assertEqual(reason, "missing_hybrid_scores")

        blended, reason = blend_row(
            row,
            reranker_weight=0.65,
            score_scale=1.0,
            missing_hybrid="reranker",
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(blended)
        assert blended is not None
        self.assertGreater(blended["pos_scores"][0], blended["neg_scores"][0])

    def test_main_writes_jsonl_and_summary(self) -> None:
        row = {
            "query": "q",
            "pos": ["p"],
            "neg": ["n1", "n2"],
            "pos_scores": [10.0],
            "neg_scores": [0.0, 5.0],
            "bge_m3_hybrid_pos_scores": [0.2],
            "bge_m3_hybrid_neg_scores": [0.8, 0.1],
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "teacher.jsonl"
            output_path = root / "blended.jsonl"
            summary_path = root / "summary.json"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "prepare_blended_teacher_scores.py",
                    "--input-jsonl",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--summary-json",
                    str(summary_path),
                ]
                with redirect_stdout(StringIO()):
                    main()
            finally:
                sys.argv = old_argv

            self.assertTrue(output_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["written_rows"], 1)
            self.assertFalse(summary["raw_training_data_committed"])


if __name__ == "__main__":
    unittest.main()
