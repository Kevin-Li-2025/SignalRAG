from __future__ import annotations

import json

import pytest

from finmteb_sota.score_cache import (
    build_candidate_ids,
    load_score_cache,
    write_score_cache,
)
from finmteb_sota.tasks import RerankingTask

TASK = RerankingTask(dataset_id="owner/data", leaderboard_name="task", language="zh")


def test_cache_reconstructs_requested_candidate_order(tmp_path) -> None:
    first_ids = build_candidate_ids(["q", "q"], ["query", "query"], ["doc-a", "doc-b"])
    path = write_score_cache(tmp_path, TASK, "test", "instruction", "model", [0.1, 0.9], first_ids)

    requested_ids = list(reversed(first_ids))
    scores, loaded_path = load_score_cache(
        tmp_path, TASK, "test", "instruction", "model", requested_ids
    )

    assert loaded_path == path
    assert scores == [0.9, 0.1]


def test_cache_coverage_mismatch_fails_closed(tmp_path) -> None:
    cached_ids = build_candidate_ids(["q"], ["query"], ["doc-a"])
    write_score_cache(tmp_path, TASK, "test", "instruction", "model", [0.1], cached_ids)
    requested_ids = build_candidate_ids(["q"], ["query"], ["doc-b"])

    with pytest.raises(ValueError, match="Candidate coverage mismatch"):
        load_score_cache(tmp_path, TASK, "test", "instruction", "model", requested_ids)


def test_legacy_positional_payload_is_rejected(tmp_path) -> None:
    candidate_ids = build_candidate_ids(["q"], ["query"], ["doc-a"])
    path = write_score_cache(
        tmp_path, TASK, "test", "instruction", "model", [0.1], candidate_ids
    )
    payload = json.loads(path.read_text())
    payload.pop("format_version")
    payload.pop("candidate_ids")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Unsupported score-cache format"):
        load_score_cache(tmp_path, TASK, "test", "instruction", "model", candidate_ids)


def test_candidate_ids_handle_repeated_identical_documents() -> None:
    candidate_ids = build_candidate_ids(["q", "q"], ["query", "query"], ["same", "same"])
    assert len(candidate_ids) == len(set(candidate_ids)) == 2
