import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from arabic_embedding_sota.config import load_training_config
from scripts.train_sentence_transformer import rows_from_jsonl


class ConfigTest(unittest.TestCase):
    def test_load_training_config(self) -> None:
        config = load_training_config(Path("configs/train_arabic_matryoshka.yaml"))
        self.assertEqual(config.model_name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(config.train_file, Path("data/train_pairs.jsonl"))
        self.assertEqual(config.matryoshka_dims, (1024, 768, 512, 256))

    def test_load_margin_mse_teacher_rows(self) -> None:
        row = {
            "query": "query",
            "positive": "positive passage",
            "negative": "negative passage",
            "pos": ["positive passage"],
            "neg": ["negative passage", "second negative"],
            "pos_scores": [0.7],
            "neg_scores": [0.2, 0.4],
        }
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "teacher.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = rows_from_jsonl(path, "margin_mse")

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["query"], "query")
        self.assertAlmostEqual(dataset[0]["label"], 0.5)
        self.assertAlmostEqual(dataset[1]["label"], 0.3)


if __name__ == "__main__":
    unittest.main()
