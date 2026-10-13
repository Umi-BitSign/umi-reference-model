"""Source-duration revision excludes only uncuttable publisher intervals."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from revise_how2sign_raw_expansion import revised_selection


def _fixture():
    clips = [
        {
            "sentence_name": f"example_{number}",
            "start_sec": float(number * 3),
            "end_sec": float(number * 3 + 3),
            "reference": "three signed words",
        }
        for number in range(5)
    ]
    video = {
        "video_name": "example-5-rgb_front",
        "eligible_clip_count": 5,
        "eligible_duration_sec": 15.0,
        "clips": clips,
    }
    selection = {
        "schema": "umi-studio-how2sign-raw-training-expansion/1",
        "min_clips_per_video": 4,
        "selected_video_count": 1,
        "selected_clip_count": 5,
        "selected_duration_sec": 15.0,
        "videos": [video],
    }
    receipt = {
        "video_name": "example-5-rgb_front",
        "video_probe": {"duration_sec": 13.0},
        "video_sha256": "verified-sha",
    }
    return selection, receipt


def test_excludes_interval_past_verified_video_end():
    selection, receipt = _fixture()
    revised = revised_selection(selection, [receipt], "plan-sha", "receipts-sha")
    assert revised["schema"] == "umi-studio-how2sign-raw-training-expansion/2"
    assert revised["previous_selection_sha256"] == "plan-sha"
    assert revised["selected_clip_count"] == 4
    assert revised["selected_duration_sec"] == 12.0
    assert revised["excluded_unavailable_clip_count"] == 1
    assert revised["excluded_unavailable_clips"][0]["sentence_name"] == "example_4"
    assert revised["videos"][0]["verified_video_sha256"] == "verified-sha"
    assert selection["selected_clip_count"] == 5


def test_requires_all_sources_and_keeps_minimum_clip_count():
    selection, receipt = _fixture()
    with pytest.raises(ValueError, match="every selected raw video"):
        revised_selection(selection, [], "plan-sha", "receipts-sha")
    receipt["video_probe"]["duration_sec"] = 10.0
    with pytest.raises(ValueError, match="too few available"):
        revised_selection(selection, [receipt], "plan-sha", "receipts-sha")
