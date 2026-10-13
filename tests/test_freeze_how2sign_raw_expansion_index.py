"""The raw expansion index must keep exact plan order and coverage."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_how2sign_raw_expansion_index import ordered_expansion_rows


def _inputs():
    videos = [
        {
            "video_name": "source-a",
            "video_id": "a",
            "signer": 1,
            "eligible_clip_count": 2,
            "clips": [
                {"sentence_name": "a-1", "reference": "first sentence"},
                {"sentence_name": "a-2", "reference": "second sentence"},
            ],
        },
        {
            "video_name": "source-b",
            "video_id": "b",
            "signer": 2,
            "eligible_clip_count": 1,
            "clips": [{"sentence_name": "b-1", "reference": "third sentence"}],
        },
    ]
    plan = {
        "schema": "umi-studio-how2sign-raw-training-expansion/2",
        "training_only": True,
        "selected_video_count": 2,
        "selected_clip_count": 3,
        "videos": videos,
    }
    rows = {
        video["video_name"]: [
            {
                "sentence_name": clip["sentence_name"],
                "reference": clip["reference"],
                "signer": video["signer"],
            }
            for clip in video["clips"]
        ]
        for video in videos
    }
    return plan, rows


def test_orders_verified_raw_rows():
    plan, rows = _inputs()
    assert [row["sentence_name"] for row in ordered_expansion_rows(plan, rows)] == [
        "a-1",
        "a-2",
        "b-1",
    ]


def test_rejects_missing_or_repeated_source_and_clip():
    plan, rows = _inputs()
    with pytest.raises(ValueError, match="coverage"):
        ordered_expansion_rows(plan, {"source-a": rows["source-a"]})
    repeated = deepcopy(plan)
    repeated["videos"][1]["video_id"] = "a"
    with pytest.raises(ValueError, match="coverage"):
        ordered_expansion_rows(repeated, rows)
    repeated = deepcopy(plan)
    repeated["videos"][1]["clips"][0]["sentence_name"] = "a-1"
    rows["source-b"][0]["sentence_name"] = "a-1"
    with pytest.raises(ValueError, match="selection"):
        ordered_expansion_rows(repeated, rows)


def test_rejects_changed_reference_or_partial_cache():
    plan, rows = _inputs()
    rows["source-b"][0]["reference"] = "wrong"
    with pytest.raises(ValueError, match="selection"):
        ordered_expansion_rows(plan, rows)
    rows["source-b"][0]["reference"] = "third sentence"
    rows["source-a"].pop()
    with pytest.raises(ValueError, match="coverage"):
        ordered_expansion_rows(plan, rows)
