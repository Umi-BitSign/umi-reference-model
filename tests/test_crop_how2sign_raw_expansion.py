"""Raw expansion crops must match the frozen plan and clip bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from crop_how2sign_raw_expansion import receipt_paths, verified_crops


def _fixture(tmp_path):
    ledger = tmp_path / "ledger"
    clips = tmp_path / "clips"
    ledger.mkdir()
    clips.mkdir()
    video_name = "source-2-rgb_front"
    sentence_name = "source_0-2-rgb_front"
    video = {
        "video_name": video_name,
        "signer": 2,
        "clips": [
            {
                "sentence_name": sentence_name,
                "start_sec": 1.0,
                "end_sec": 4.0,
                "reference": "A short sentence.",
            }
        ],
    }
    selection_path = ledger / f"{video_name}-selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "umi-studio-how2sign-training-raw-realigned/1",
                "video_name": video_name,
                "source_video_sha256": "source-hash",
                "rows": video["clips"],
            }
        )
    )
    clip_path = clips / f"{sentence_name}.mp4"
    clip_path.write_bytes(b"verified crop")
    receipt_path = ledger / f"{video_name}-clip-receipts.jsonl"
    row = {
        **video["clips"][0],
        "schema": "umi-studio-how2sign-raw-realigned-clip/1",
        "video_name": video_name,
        "signer": 2,
        "selection_sha256": sha256(selection_path),
        "source_video_sha256": "source-hash",
        "clip_sha256": sha256(clip_path),
    }
    receipt_path.write_text(json.dumps(row) + "\n")
    sources = {video_name: {"video_sha256": "source-hash"}}
    return video, sources, ledger, clips, clip_path, receipt_path


def test_accepts_verified_crop_and_rejects_changed_bytes(tmp_path):
    video, sources, ledger, clips, clip_path, _ = _fixture(tmp_path)
    paths = receipt_paths(ledger)
    assert verified_crops([video], sources, paths, clips) == {video["video_name"]}
    clip_path.write_bytes(b"altered crop")
    with pytest.raises(ValueError, match="differs"):
        verified_crops([video], sources, paths, clips)


def test_rejects_plan_difference_and_duplicate_crop_receipt(tmp_path):
    video, sources, ledger, clips, _, receipt_path = _fixture(tmp_path)
    paths = receipt_paths(ledger)
    with pytest.raises(ValueError, match="differs"):
        verified_crops(
            [{**video, "clips": [{**video["clips"][0], "reference": "wrong"}]}],
            sources,
            paths,
            clips,
        )
    receipt_path.write_text(receipt_path.read_text() * 2)
    with pytest.raises(ValueError, match="differs"):
        verified_crops([video], sources, paths, clips)
