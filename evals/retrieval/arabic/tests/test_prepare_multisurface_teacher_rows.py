import argparse
import json
import tempfile
import unittest
from pathlib import Path

import scripts.prepare_multisurface_teacher_rows as multisurface
from scripts.prepare_multisurface_teacher_rows import (
    parse_surface_counts,
    parse_surface_list,
    parse_surface_weights,
    prepare_splits,
    surface_false_positive_labels,
    target_scores_for_row,
)


def component_augmented_row(query_id="q1"):
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "pos_doc_ids": [f"p-{query_id}"],
        "pos": [f"positive {query_id}"],
        "pos_scores": [1.0],
        "neg_doc_ids": [
            f"n-{query_id}-model",
            f"n-{query_id}-colbert",
            f"n-{query_id}-sparse",
            f"n-{query_id}-training",
            f"n-{query_id}-easy",
        ],
        "neg": [
            f"model-card false positive {query_id}",
            f"colbert false positive {query_id}",
            f"sparse-colbert false positive {query_id}",
            f"training-default false positive {query_id}",
            f"easy negative {query_id}",
        ],
        "neg_scores": [0.50, 0.40, 0.30, 0.20, 0.10],
        "neg_sources": ["hard", "hard", "hard", "hard", "easy"],
        "original_reranker_pos_scores": [10.0],
        "original_reranker_neg_scores": [2.0, 1.0, 0.5, 0.1, 0.0],
        "bge_m3_surface_pos_scores": {
            "model_card": [0.50],
            "colbert_heavy": [0.60],
            "sparse_colbert": [0.55],
            "training_default": [0.60],
        },
        "bge_m3_surface_neg_scores": {
            "model_card": [0.70, 0.40, 0.10, 0.00, -0.10],
            "colbert_heavy": [0.20, 0.80, 0.10, 0.00, -0.10],
            "sparse_colbert": [0.20, 0.10, 0.90, 0.00, -0.10],
            "training_default": [0.20, 0.10, 0.00, 0.90, -0.10],
        },
        "score_blend_teacher": {"method": "test"},
        "source": {"split": "train"},
    }


def args_for(input_path, output_dir, **overrides):
    defaults = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "train_jsonl": "teacher_train_multisurface.jsonl",
        "eval_jsonl": "teacher_eval_multisurface.jsonl",
        "max_train_rows": 20,
        "max_eval_rows": 20,
        "heldout_ratio": 0.5,
        "negatives_per_query": 4,
        "surface_negative_counts": (
            "model_card=1,colbert_heavy=1,sparse_colbert=1,training_default=1"
        ),
        "min_surface_false_positive_available": 1,
        "min_distinct_surfaces_available": 1,
        "surface_hard_margin": 0.0,
        "teacher_separation_margin": 0.05,
        "teacher_hard_margin": 0.10,
        "middle_margin": 0.35,
        "reranker_weight": 0.60,
        "existing_weight": 0.25,
        "surface_average_weight": 0.15,
        "anti_regression_surfaces": "",
        "anti_regression_weight": 0.0,
        "anti_regression_margin": 0.20,
        "anti_regression_surface_hard_margin": None,
        "surface_average_source_weights": "",
        "score_scale": 1.0,
        "min_target_best_margin": 0.0,
        "max_target_best_margin": None,
        "seed": "test-multisurface",
        "allow_nontrain_source": False,
        "force": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class PrepareMultisurfaceTeacherRowsTest(unittest.TestCase):
    def test_parse_surface_counts_rejects_duplicates(self):
        self.assertEqual(
            parse_surface_counts("model_card=2,colbert_heavy=1"),
            {"model_card": 2, "colbert_heavy": 1},
        )

        with self.assertRaises(ValueError):
            parse_surface_counts("model_card=1,model_card=2")

    def test_parse_surface_weights_defaults_and_validates_labels(self):
        self.assertEqual(
            parse_surface_weights("", ["model_card", "training_default"]),
            {"model_card": 1.0, "training_default": 1.0},
        )
        self.assertEqual(
            parse_surface_weights("model_card=2.5", ["model_card", "training_default"]),
            {"model_card": 2.5, "training_default": 1.0},
        )
        with self.assertRaises(ValueError):
            parse_surface_weights("missing=1", ["model_card"])

    def test_parse_surface_list_validates_known_unique_surfaces(self):
        self.assertEqual(
            parse_surface_list("model_card,training_default", ["model_card", "training_default"]),
            ["model_card", "training_default"],
        )
        with self.assertRaises(ValueError):
            parse_surface_list("model_card,model_card", ["model_card"])
        with self.assertRaises(ValueError):
            parse_surface_list("missing", ["model_card"])

    def test_target_scores_blend_reranker_existing_and_surface_average(self):
        row = component_augmented_row()

        pos_scores, neg_scores, sources = target_scores_for_row(
            row,
            surfaces=["model_card", "colbert_heavy", "sparse_colbert", "training_default"],
            reranker_weight=0.60,
            existing_weight=0.25,
            surface_average_weight=0.15,
            surface_average_source_weights="",
            score_scale=1.0,
        )

        self.assertEqual(pos_scores, [1.0])
        self.assertEqual(len(neg_scores), 5)
        self.assertLess(max(neg_scores), pos_scores[0])
        self.assertIn("surface_pos_by_label", sources)

    def test_target_scores_support_weighted_surface_average(self):
        row = component_augmented_row()

        _pos_scores, _neg_scores, sources = target_scores_for_row(
            row,
            surfaces=["model_card", "training_default"],
            reranker_weight=0.0,
            existing_weight=0.0,
            surface_average_weight=1.0,
            surface_average_source_weights="model_card=3,training_default=1",
            score_scale=1.0,
        )

        self.assertEqual(
            sources["surface_average_source_weights"],
            {"model_card": 3.0, "training_default": 1.0},
        )

    def test_target_scores_support_anti_regression_source(self):
        row = component_augmented_row()

        pos_scores, neg_scores, sources = target_scores_for_row(
            row,
            surfaces=["model_card", "training_default"],
            reranker_weight=0.0,
            existing_weight=0.0,
            surface_average_weight=0.0,
            surface_average_source_weights="",
            anti_regression_surfaces="model_card,training_default",
            anti_regression_weight=1.0,
            anti_regression_margin=0.25,
            anti_regression_surface_hard_margin=0.0,
            score_scale=1.0,
        )

        self.assertEqual(pos_scores, [1.0])
        self.assertGreater(neg_scores[0], neg_scores[-1])
        self.assertGreater(neg_scores[3], neg_scores[-1])
        self.assertEqual(
            sources["anti_regression"]["false_positive_surfaces_by_negative"][0],
            ["model_card"],
        )
        self.assertEqual(
            sources["anti_regression"]["false_positive_surfaces_by_negative"][3],
            ["training_default"],
        )

    def test_prepare_splits_records_anti_regression_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            output_dir = root / "out"
            rows = [component_augmented_row(f"q{index}") for index in range(20)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            train_rows, _eval_rows, summary = prepare_splits(
                args_for(
                    input_path,
                    output_dir,
                    anti_regression_surfaces="model_card,training_default",
                    anti_regression_weight=0.5,
                    anti_regression_margin=0.25,
                )
            )

        self.assertEqual(
            summary["surface_target"]["anti_regression_surfaces"],
            ["model_card", "training_default"],
        )
        self.assertEqual(summary["surface_target"]["anti_regression_weight"], 0.5)
        self.assertIn("anti_regression", train_rows[0]["multisurface_teacher"])

    def test_surface_false_positive_requires_teacher_separation(self):
        row = component_augmented_row()
        pos_scores, neg_scores, sources = target_scores_for_row(
            row,
            surfaces=["model_card", "colbert_heavy"],
            reranker_weight=1.0,
            existing_weight=0.0,
            surface_average_weight=0.0,
            surface_average_source_weights="",
            score_scale=1.0,
        )

        labels = surface_false_positive_labels(
            pos_index=0,
            neg_index=0,
            score_sources=sources,
            surfaces=["model_card", "colbert_heavy"],
            surface_hard_margin=0.0,
            teacher_margin=pos_scores[0] - neg_scores[0],
            teacher_separation_margin=0.05,
        )
        blocked = surface_false_positive_labels(
            pos_index=0,
            neg_index=0,
            score_sources=sources,
            surfaces=["model_card", "colbert_heavy"],
            surface_hard_margin=0.0,
            teacher_margin=0.01,
            teacher_separation_margin=0.05,
        )

        self.assertEqual(labels, ["model_card"])
        self.assertEqual(blocked, [])

    def test_prepare_splits_writes_query_disjoint_multisurface_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            output_dir = root / "out"
            rows = [component_augmented_row(f"q{index}") for index in range(20)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            train_rows, eval_rows, summary = prepare_splits(
                args_for(input_path, output_dir)
            )

        train_queries = {row["query_id"] for row in train_rows}
        eval_queries = {row["query_id"] for row in eval_rows}
        self.assertTrue(train_queries)
        self.assertTrue(eval_queries)
        self.assertFalse(train_queries.intersection(eval_queries))
        self.assertEqual(summary["splitter"]["query_overlap"], 0)
        self.assertEqual(summary["train"]["rows"], len(train_rows))
        self.assertGreaterEqual(
            summary["train"]["selected_surface_counts"]["model_card"],
            len(train_rows),
        )
        self.assertFalse(summary["raw_training_data_committed"])

    def test_main_writes_files_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.jsonl"
            output_dir = root / "out"
            input_path.write_text(
                "".join(
                    json.dumps(component_augmented_row(f"q{index}")) + "\n"
                    for index in range(12)
                ),
                encoding="utf-8",
            )

            train_rows, eval_rows, summary = multisurface.prepare_splits(
                args_for(input_path, output_dir, max_train_rows=6, max_eval_rows=4)
            )
            output_dir.mkdir()
            multisurface.write_jsonl(output_dir / "teacher_train_multisurface.jsonl", train_rows)
            multisurface.write_jsonl(output_dir / "teacher_eval_multisurface.jsonl", eval_rows)
            multisurface.write_json(output_dir / "summary.json", summary)

            persisted = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(persisted["surface_target"]["surfaces"], [
            "model_card",
            "colbert_heavy",
            "sparse_colbert",
            "training_default",
        ])
        self.assertLessEqual(persisted["train"]["rows"], 6)


if __name__ == "__main__":
    unittest.main()
