"""Bounded raw-video expansion preserves source-video and signer splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from how2sign_split_remote_zip import SplitZipEntry
from select_how2sign_raw_expansion import select_videos


def _sources():
    training = []
    realigned = []
    entries = {}
    for video_id, signer in (("first", 2), ("second", 2), ("third", 9)):
        video_name = f"{video_id}-{signer}-rgb_front"
        training.append({"VIDEO_ID": video_id, "VIDEO_NAME": video_name})
        for number in range(4):
            realigned.append(
                {
                    "VIDEO_ID": video_id,
                    "VIDEO_NAME": video_name,
                    "SENTENCE_NAME": f"{video_id}_{number}-{signer}-rgb_front",
                    "START_REALIGNED": str(number * 4),
                    "END_REALIGNED": str(number * 4 + 3),
                    "SENTENCE": f"three signed words {number}",
                }
            )
        member = f"raw_videos/{video_name}.mp4"
        entries[member] = SplitZipEntry(member, 0, 0, 100, 100, 0, 0, 1)
    return training, realigned, entries


def test_selects_distinct_source_videos_and_keeps_included_source():
    training, realigned, entries = _sources()
    selected = select_videos(
        training,
        realigned,
        [],
        {"schema": "umi-studio-how2sign-training-aligned/1", "rows": []},
        entries,
        {"second-2-rgb_front"},
        quotas={2: 1, 9: 1},
        max_compressed_bytes=200,
    )
    assert [row["video_name"] for row in selected] == [
        "second-2-rgb_front",
        "third-9-rgb_front",
    ]
    assert sum(row["eligible_clip_count"] for row in selected) == 8
    assert all(len(row["clips"]) == 4 for row in selected)


def test_rejects_validation_overlap_and_pre_cut_included_video():
    training, realigned, entries = _sources()
    pre_cut = {"schema": "umi-studio-how2sign-training-aligned/1", "rows": []}
    with pytest.raises(ValueError, match="overlap"):
        select_videos(
            training,
            realigned,
            [{"VIDEO_ID": "first"}],
            pre_cut,
            entries,
            set(),
            quotas={2: 1, 9: 1},
            max_compressed_bytes=200,
        )
    with pytest.raises(ValueError, match="included raw video"):
        select_videos(
            training,
            realigned,
            [],
            {**pre_cut, "rows": [{"video_id": "second"}]},
            entries,
            {"second-2-rgb_front"},
            quotas={2: 1, 9: 1},
            max_compressed_bytes=200,
        )
