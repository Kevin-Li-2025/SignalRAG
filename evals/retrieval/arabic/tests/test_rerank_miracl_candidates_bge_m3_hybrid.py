import json
import tempfile
import unittest
from pathlib import Path

from scripts.rerank_miracl_candidates_bge_m3_hybrid import (
    diagnostic_gate,
    extract_metrics,
    load_head_checkpoint_state,
    load_baseline_metrics,
)


class DummyTorch:
    def __init__(self):
        self.loaded_paths = []

    def load(self, path, map_location=None):
        self.loaded_paths.append((Path(path).name, map_location))
        return {"loaded_from": Path(path).name}


class HybridCandidateRerankGateTest(unittest.TestCase):
    def test_extract_metrics_accepts_script_summary(self):
        metrics = extract_metrics(
            {
                "metrics": {
                    "main_score": 0.81,
                    "ndcg_at_10": 0.81,
                }
            }
        )

        self.assertEqual(metrics["main_score"], 0.81)
        self.assertEqual(metrics["ndcg_at_10"], 0.81)

    def test_extract_metrics_accepts_v52_v54_experiment_record(self):
        metrics = extract_metrics(
            {
                "evaluation_surface": {
                    "base_first_stage_metrics": {
                        "main_score": 0.7888288154070898,
                        "ndcg_at_10": 0.7888288154070898,
                    }
                }
            }
        )

        self.assertEqual(metrics["main_score"], 0.7888288154070898)

    def test_load_baseline_metrics_reads_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                json.dumps({"metrics": {"main_score": 0.8}}) + "\n",
                encoding="utf-8",
            )

            metrics = load_baseline_metrics(path)

        self.assertEqual(metrics["main_score"], 0.8)

    def test_load_head_checkpoint_state_accepts_official_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "sparse_linear.pt").write_text("sparse", encoding="utf-8")
            (checkpoint_dir / "colbert_linear.pt").write_text("colbert", encoding="utf-8")
            torch = DummyTorch()

            checkpoint = load_head_checkpoint_state(
                checkpoint_dir,
                torch_module=torch,
                map_location="cpu",
            )

        self.assertEqual(checkpoint["sparse_linear"]["loaded_from"], "sparse_linear.pt")
        self.assertEqual(checkpoint["colbert_linear"]["loaded_from"], "colbert_linear.pt")
        self.assertEqual(
            torch.loaded_paths,
            [("sparse_linear.pt", "cpu"), ("colbert_linear.pt", "cpu")],
        )

    def test_load_head_checkpoint_state_accepts_custom_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "head.pt"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            class FileTorch(DummyTorch):
                def load(self, path, map_location=None):
                    self.loaded_paths.append((Path(path).name, map_location))
                    return {
                        "sparse_linear": {"weight": "s"},
                        "colbert_linear": {"weight": "c"},
                    }

            torch = FileTorch()

            checkpoint = load_head_checkpoint_state(
                checkpoint_path,
                torch_module=torch,
                map_location="cuda",
            )

        self.assertEqual(checkpoint["sparse_linear"]["weight"], "s")
        self.assertEqual(checkpoint["colbert_linear"]["weight"], "c")
        self.assertEqual(torch.loaded_paths, [("head.pt", "cuda")])

    def test_diagnostic_gate_requires_meaningful_delta(self):
        gate = diagnostic_gate(
            metrics={"main_score": 0.794},
            baseline_metrics={"main_score": 0.788},
            metric_key="main_score",
            min_delta=0.005,
            baseline_label="base",
        )

        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["delta"], 0.006)
        self.assertEqual(gate["baseline_label"], "base")

    def test_diagnostic_gate_rejects_noise_gain(self):
        gate = diagnostic_gate(
            metrics={"main_score": 0.789180855197925},
            baseline_metrics={"main_score": 0.7888288154070898},
            metric_key="main_score",
            min_delta=0.005,
            baseline_label="base",
        )

        self.assertFalse(gate["passed"])
        self.assertLess(gate["delta"], 0.005)

    def test_diagnostic_gate_rejects_missing_metric(self):
        with self.assertRaises(ValueError):
            diagnostic_gate(
                metrics={"main_score": 0.8},
                baseline_metrics={"ndcg_at_10": 0.7},
                metric_key="main_score",
                min_delta=0.005,
                baseline_label="base",
            )


if __name__ == "__main__":
    unittest.main()
