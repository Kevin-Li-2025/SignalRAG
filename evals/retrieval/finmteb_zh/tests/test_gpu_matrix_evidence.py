import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "reports/corrected_gpu_matrix_v1"


def summarizer():
    spec = importlib.util.spec_from_file_location(
        "matrix_summary", ROOT / "scripts/summarize_finmteb_gpu_matrix.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def test_sealed_matrix_rebuilds_without_changing_results():
    result = summarizer()(PACKAGE)
    assert result == json.loads((PACKAGE / "corrected_gpu_matrix_summary.json").read_text())
    assert (
        sum(row["num_test_queries"] for row in result["arms"] if row["precision"] == "bf16") == 72
    )


def test_all_sealed_result_hashes_match():
    lines = (PACKAGE / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 26
    for line in lines:
        digest, name = line.split(maxsplit=1)
        path = (PACKAGE / name).resolve()
        assert path.is_relative_to(PACKAGE.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "file,keys,value",
    [
        ("train_strategy.json", ["split"], "test"),
        ("train_nested_cv.json", ["tasks", 0, "split"], "test"),
        ("frozen_test.json", ["tasks", 0, "strategy", "alpha"], 0.99),
        ("frozen_test.json", ["tasks", 0, "metrics", "map"], float("nan")),
        ("frozen_test.json", ["tasks", 0, "num_pairs"], 583),
        ("order_invariance.json", ["all_tasks_order_invariant"], "false"),
        ("order_invariance.json", ["tasks", 0, "seeds", 0, "seed"], 2234),
        ("order_invariance.json", ["tasks", 0, "seeds", 0, "same_candidate_coverage"], False),
        ("order_invariance.json", ["tasks", 0, "seeds", 0, "candidate_set_sha256"], "0" * 64),
        ("order_invariance.json", ["tasks", 0, "seeds", 0, "metrics", "map"], 0.1),
    ],
)
def test_inconsistent_receipts_fail_closed(tmp_path, file, keys, value):
    package = tmp_path / "matrix"
    shutil.copytree(PACKAGE, package)
    path = package / "bf16/FinEvaReranking" / file
    document = json.loads(path.read_text())
    node = document
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        summarizer()(package)
