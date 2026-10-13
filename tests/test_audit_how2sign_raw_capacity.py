"""Raw-capacity estimates stay source-disjoint and metadata-only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_how2sign_raw_capacity import audit


def _rows(name: str, video_id: str) -> list[dict[str, str]]:
    return [
        {
            "VIDEO_NAME": name,
            "VIDEO_ID": video_id,
            "SENTENCE_NAME": f"{name}_{index}",
            "START_REALIGNED": str(index * 3),
            "END_REALIGNED": str(index * 3 + 2),
            "SENTENCE": "one two three",
        }
        for index in range(4)
    ]


def test_capacity_excludes_validation_and_pre_cut_source_videos():
    selected = "raw-2-rgb_front"
    new = "extra-5-rgb_front"
    training = [
        {"VIDEO_NAME": name, "VIDEO_ID": video_id}
        for name, video_id in [
            (selected, "raw"), (new, "extra"),
            ("precut-1-rgb_front", "precut"),
        ]
    ]
    aligned = [
        *_rows(selected, "raw"),
        *_rows(new, "extra"),
        *_rows("precut-1-rgb_front", "precut"),
    ]
    precut = {"schema": "umi-studio-how2sign-training-aligned/1", "rows": [{"video_id": "precut"}]}
    current = {
        "schema": "umi-studio-how2sign-raw-training-expansion/2",
        "selected_video_count": 1,
        "videos": [{"video_name": selected}],
    }
    result = audit(training, aligned, [], precut, current)
    assert result["totals"]["candidate_videos"] == 2
    assert result["totals"]["remaining_videos"] == 1
    assert result["by_signer"]["5"]["remaining_clips"] == 4
    assert result["source_video_bytes_checked"] is False
    with pytest.raises(ValueError, match="not contained"):
        audit(training, aligned, [], precut, {
            **current, "videos": [{"video_name": "missing-2-rgb_front"}]
        })


def test_capacity_rejects_training_validation_source_overlap():
    name = "raw-2-rgb_front"
    precut = {"schema": "umi-studio-how2sign-training-aligned/1", "rows": []}
    current = {
        "schema": "umi-studio-how2sign-raw-training-expansion/2",
        "selected_video_count": 0,
        "videos": [],
    }
    with pytest.raises(ValueError, match="overlap"):
        audit(
            [{"VIDEO_NAME": name, "VIDEO_ID": "raw"}],
            _rows(name, "raw"),
            [{"VIDEO_ID": "raw"}],
            precut,
            current,
        )
