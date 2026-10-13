"""ASL STEM Wiki video receipts stay bound to selection and bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from fetch_asl_stem_wiki_videos import verified_receipts


def test_verifies_receipt_and_local_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    row = {
        "example_id": "clip",
        "participant": "P0",
        "article": "Article",
        "archive_member": "ASL_STEM_Wiki/videos/clip.mp4",
        "archive_crc32": 1,
        "archive_compressed_size": 4,
        "archive_file_size": 5,
        "archive_header_offset": 10,
    }
    selection = {
        "schema": "umi-studio-asl-stem-wiki-training-selection/1",
        "training_only": True,
        "selected_examples": 1,
        "license_sha256": "license",
        "records": [row],
    }
    receipt = {
        **row,
        "schema": "umi-studio-asl-stem-wiki-training-video/1",
        "selection_sha256": "selection",
        "license_sha256": "license",
        "video_size_bytes": 5,
        "video_sha256": sha256(video),
    }
    assert verified_receipts(
        selection, "selection", [json.dumps(receipt)], tmp_path
    ) == {"clip"}
    video.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="receipt differs"):
        verified_receipts(selection, "selection", [json.dumps(receipt)], tmp_path)
