import unittest
from argparse import Namespace
from unittest.mock import patch

import scripts.sweep_bge_m3_hybrid_weights as sweep
from scripts.sweep_bge_m3_hybrid_weights import (
    WeightConfig,
    compare_models,
    fused_score,
    metrics_from_components,
    per_query_comparisons,
    parse_weight_grid,
    parse_weight_spec,
    query_metrics_from_components,
    read_query_id_file,
    safe_label,
    score_components_for_model,
    select_query_ids,
)


class SweepBgeM3HybridWeightsTest(unittest.TestCase):
    def test_parse_weight_spec_accepts_labeled_values(self):
        config = parse_weight_spec("model-card:0.4,0.2,0.4")

        self.assertEqual(config.label, "model-card")
        self.assertEqual(config.values, [0.4, 0.2, 0.4])

    def test_parse_weight_spec_builds_label_for_unlabeled_values(self):
        config = parse_weight_spec("1,0.3,1")

        self.assertEqual(config.label, "d1_s0.3_c1")
        self.assertEqual(config.values, [1.0, 0.3, 1.0])

    def test_parse_weight_grid_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            parse_weight_grid(["x:0.4,0.2,0.4", "x:1,0,0"])

    def test_parse_weight_spec_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError):
            parse_weight_spec("bad:0,0,0")

    def test_safe_label_removes_shell_unfriendly_characters(self):
        self.assertEqual(safe_label("model card / default"), "model_card_default")

    def test_fused_score_uses_all_components(self):
        score = fused_score(
            {"dense": 2.0, "sparse": 3.0, "colbert": 5.0},
            WeightConfig("w", 0.4, 0.2, 0.4),
        )

        self.assertAlmostEqual(score, 3.4)

    def test_metrics_from_components_reranks_by_requested_weights(self):
        candidates = {
            "q1": [
                {"docid": "d1"},
                {"docid": "d2"},
            ]
        }
        qrels = {"q1": {"d2"}}
        components = {
            "q1": [
                {"dense": 10.0, "sparse": 0.0, "colbert": 0.0},
                {"dense": 0.0, "sparse": 10.0, "colbert": 0.0},
            ]
        }

        dense_metrics = metrics_from_components(
            candidates,
            qrels,
            components,
            ["q1"],
            weights=WeightConfig("dense", 1.0, 0.0, 0.0),
            top_k=2,
            metric_k=2,
        )
        sparse_metrics = metrics_from_components(
            candidates,
            qrels,
            components,
            ["q1"],
            weights=WeightConfig("sparse", 0.0, 1.0, 0.0),
            top_k=2,
            metric_k=2,
        )

        self.assertLess(dense_metrics["main_score"], sparse_metrics["main_score"])
        self.assertEqual(sparse_metrics["main_score"], 1.0)

    def test_query_metrics_from_components_records_main_score(self):
        metrics = query_metrics_from_components(
            [{"docid": "d1"}, {"docid": "d2"}],
            {"d2"},
            [
                {"dense": 1.0, "sparse": 0.0, "colbert": 0.0},
                {"dense": 2.0, "sparse": 0.0, "colbert": 0.0},
            ],
            weights=WeightConfig("dense", 1.0, 0.0, 0.0),
            top_k=2,
            metric_k=2,
        )

        self.assertEqual(metrics["main_score"], metrics["ndcg_at_10"])
        self.assertEqual(metrics["main_score"], 1.0)

    def test_per_query_comparisons_reports_delta_and_top_docs(self):
        rows = per_query_comparisons(
            candidates={"q1": [{"docid": "d1"}, {"docid": "d2"}]},
            qrels={"q1": {"d2"}},
            base_components={
                "q1": [
                    {"dense": 2.0, "sparse": 0.0, "colbert": 0.0},
                    {"dense": 1.0, "sparse": 0.0, "colbert": 0.0},
                ]
            },
            student_components={
                "q1": [
                    {"dense": 1.0, "sparse": 0.0, "colbert": 0.0},
                    {"dense": 2.0, "sparse": 0.0, "colbert": 0.0},
                ]
            },
            query_ids=["q1"],
            weight_grid=[WeightConfig("dense", 1.0, 0.0, 0.0)],
            top_k=2,
            metric_k=2,
        )

        self.assertEqual(rows[0]["query_id"], "q1")
        self.assertEqual(rows[0]["weight_label"], "dense")
        self.assertGreater(rows[0]["delta"], 0.0)
        self.assertEqual(rows[0]["base"]["top_docids"], ["d1", "d2"])
        self.assertEqual(rows[0]["student"]["top_docids"], ["d2", "d1"])

    def test_compare_models_reports_same_weight_delta(self):
        base = {
            "results": [
                {
                    "weight_label": "w",
                    "weights_for_different_modes": [0.4, 0.2, 0.4],
                    "metrics": {"main_score": 0.8},
                }
            ]
        }
        student = {
            "results": [
                {
                    "weight_label": "w",
                    "weights_for_different_modes": [0.4, 0.2, 0.4],
                    "metrics": {"main_score": 0.806},
                }
            ]
        }

        comparisons = compare_models(base, student)

        self.assertEqual(comparisons[0]["weight_label"], "w")
        self.assertAlmostEqual(comparisons[0]["delta"], 0.006)

    def test_select_query_ids_supports_deterministic_stride_split(self):
        selected = select_query_ids(
            ["q3", "q1", "q4", "q2"],
            query_limit=0,
            query_stride=2,
            query_offset=1,
        )

        self.assertEqual(selected, ["q2", "q4"])

    def test_read_query_id_file_skips_comments_and_blank_lines(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.txt"
            path.write_text("q1\n\n# held out\nq3\n", encoding="utf-8")

            self.assertEqual(read_query_id_file(path), ["q1", "q3"])

    def test_select_query_ids_preserves_file_order_and_limit(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.txt"
            path.write_text("q3\nq1\nq2\n", encoding="utf-8")

            selected = select_query_ids(
                ["q1", "q2", "q3"],
                query_limit=2,
                query_id_file=str(path),
            )

        self.assertEqual(selected, ["q3", "q1"])

    def test_select_query_ids_rejects_missing_file_ids(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.txt"
            path.write_text("q1\nq9\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                select_query_ids(["q1"], query_limit=0, query_id_file=str(path))

    def test_score_components_patches_peft_adapter_student_loading(self):
        class FakeModel:
            def compute_score(self, pairs, **_kwargs):
                return {
                    "dense": [1.0 for _pair in pairs],
                    "sparse": [2.0 for _pair in pairs],
                    "colbert": [3.0 for _pair in pairs],
                }

        calls = []

        def fake_load_bge_model(_args):
            return FakeModel()

        def fake_patch():
            calls.append("patched")

        args = Namespace(model="BAAI/bge-m3", use_fp16=True, batch_size=8, max_passage_length=128)

        with (
            patch.object(sweep, "load_bge_model", side_effect=fake_load_bge_model),
            patch.object(
                sweep,
                "patch_distributed_tensor_for_peft_model_loading",
                side_effect=fake_patch,
            ),
        ):
            scores, info = score_components_for_model(
                args,
                label="student",
                model_path="/tmp/adapter",
                head_checkpoint="",
                candidates={"q1": [{"docid": "d1"}]},
                query_text_by_id={"q1": "query"},
                candidate_text_by_id={"d1": "doc"},
                query_ids=["q1"],
            )

        self.assertEqual(calls, ["patched"])
        self.assertEqual(scores["q1"][0], {"dense": 1.0, "sparse": 2.0, "colbert": 3.0})
        self.assertEqual(info["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
