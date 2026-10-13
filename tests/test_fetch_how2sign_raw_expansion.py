"""Raw expansion downloads must match frozen archive members and bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from fetch_how2sign_raw_expansion import verified_downloads


def _fixture(tmp_path):
    video = {
        "video_name": "source-2-rgb_front",
        "video_id": "source",
        "signer": 2,
        "archive_member": "raw_videos/source-2-rgb_front.mp4",
        "archive_disk": 0,
        "archive_header_offset": 123,
        "archive_compressed_size": 5,
        "archive_file_size": 5,
        "archive_crc32": 7,
        "eligible_clip_count": 4,
    }
    selection = {
        "schema": "umi-studio-how2sign-raw-training-expansion/1",
        "training_only": True,
        "source_url": "publisher",
        "archive_manifest_sha256": "manifest",
        "training_metadata_sha256": "train",
        "validation_metadata_sha256": "validation",
        "license_sha256": "license",
        "selected_video_count": 1,
        "selected_clip_count": 4,
        "selected_compressed_bytes": 5,
        "videos": [video],
    }
    path = tmp_path / "source-2-rgb_front.mp4"
    path.write_bytes(b"video")
    receipt = {
        **video,
        "schema": "umi-studio-how2sign-raw-training-video/1",
        "source_url": "publisher",
        "archive_manifest_sha256": "manifest",
        "training_metadata_sha256": "train",
        "validation_metadata_sha256": "validation",
        "license_sha256": "license",
        "video_size_bytes": 5,
        "video_sha256": sha256(path),
    }
    return selection, receipt, path


def test_verifies_raw_video_receipt_against_plan_and_bytes(tmp_path):
    selection, receipt, path = _fixture(tmp_path)
    assert verified_downloads(selection, [json.dumps(receipt)], tmp_path) == {"source-2-rgb_front"}
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="differs"):
        verified_downloads(selection, [json.dumps(receipt)], tmp_path)


def test_rejects_wrong_archive_member_and_duplicate_receipt(tmp_path):
    selection, receipt, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        verified_downloads(selection, [json.dumps({**receipt, "archive_crc32": 8})], tmp_path)
    with pytest.raises(ValueError, match="differs"):
        verified_downloads(selection, [json.dumps(receipt), json.dumps(receipt)], tmp_path)


def test_source_duration_revision_binds_verified_video_bytes_and_duration(tmp_path):
    selection, receipt, _ = _fixture(tmp_path)
    receipt["video_probe"] = {"duration_sec": 5.0}
    selection["schema"] = "umi-studio-how2sign-raw-training-expansion/2"
    selection["videos"][0]["verified_video_sha256"] = receipt["video_sha256"]
    selection["videos"][0]["verified_video_duration_sec"] = 5.0
    assert verified_downloads(selection, [json.dumps(receipt)], tmp_path) == {"source-2-rgb_front"}
    selection["videos"][0]["verified_video_duration_sec"] = 4.0
    with pytest.raises(ValueError, match="differs"):
        verified_downloads(selection, [json.dumps(receipt)], tmp_path)
