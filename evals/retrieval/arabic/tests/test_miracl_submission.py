import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_miracl_submission import validate_run_file
from scripts.rerank_miracl_candidates import read_trec_run
from scripts.run_miracl_bge_m3_hybrid import write_trec_run


class MiraclSubmissionTest(unittest.TestCase):
    def test_write_and_validate_trec_run(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "ar_dev.txt"
            ranked_by_query = {
                "1": [
                    {"docid": "10#0", "rank": 1, "score": 3.0},
                    {"docid": "11#0", "rank": 2, "score": 2.0},
                ],
                "2": [
                    {"docid": "20#0", "rank": 1, "score": 5.0},
                    {"docid": "21#0", "rank": 2, "score": 4.0},
                ],
            }

            write_result = write_trec_run(
                run_path,
                ranked_by_query,
                query_ids=["1", "2"],
                run_id="unit-test",
                depth=2,
            )
            validation = validate_run_file(
                run_path,
                expected_query_ids=["1", "2"],
                depth=2,
                qrels={"1": {"10#0"}, "2": {"22#0"}},
            )

        self.assertEqual(write_result["lines"], 4)
        self.assertEqual(validation["queries"], 2)
        self.assertEqual(validation["run_ids"], ["unit-test"])
        self.assertAlmostEqual(validation["metrics"]["ndcg_at_10"], 0.5)

    def test_validate_rejects_wrong_depth(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "ar_dev.txt"
            run_path.write_text("1 Q0 10#0 1 3.0 unit-test\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_run_file(
                    run_path,
                    expected_query_ids=["1"],
                    depth=2,
                    qrels=None,
                )

    def test_read_trec_run_keeps_requested_depth(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "ar_dev.txt"
            run_path.write_text(
                "\n".join(
                    [
                        "1 Q0 10#0 1 3.0 base",
                        "1 Q0 11#0 2 2.0 base",
                        "1 Q0 12#0 3 1.0 base",
                        "2 Q0 20#0 1 4.0 base",
                        "2 Q0 21#0 2 3.0 base",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = read_trec_run(run_path, depth=2)

        self.assertEqual([row["docid"] for row in candidates["1"]], ["10#0", "11#0"])
        self.assertEqual(candidates["2"][0]["first_stage_score"], 4.0)

    def test_read_trec_run_from_submission_zip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "submission.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "miracl_submission/ar_dev.txt",
                    "1 Q0 10#0 1 3.0 base\n1 Q0 11#0 2 2.0 base\n",
                )

            candidates = read_trec_run(zip_path, depth=1)

        self.assertEqual(len(candidates["1"]), 1)
        self.assertEqual(candidates["1"][0]["docid"], "10#0")


if __name__ == "__main__":
    unittest.main()
