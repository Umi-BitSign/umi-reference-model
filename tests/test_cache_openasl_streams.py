from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from cache_openasl_streams import checked_index, selection_records


def test_checked_index_accepts_empty_missing_index(tmp_path: Path) -> None:
    assert checked_index(tmp_path / "missing.jsonl", tmp_path, "a" * 64, "b" * 64) == set()


def test_checked_index_rejects_wrong_spatial_policy(tmp_path: Path) -> None:
    cache = tmp_path / "openasl-0000-abcdefghijk.npz"
    cache.write_bytes(b"cache")
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": "umi-studio-openasl-stream-cache/1",
                "clip_id": "openasl-0000-abcdefghijk",
                "selection_sha256": "a" * 64,
                "landmark_receipts_sha256": "b" * 64,
                "temporal_stride": 2,
                "source_frames": 5,
                "frames": 3,
                "spatial_policy": "square",
                "encoder_cache_sha256": "ignored",
            }
        )
        + "\n"
    )
    try:
        checked_index(index, tmp_path, "a" * 64, "b" * 64)
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("wrong OpenASL spatial policy was accepted")


def test_selection_records_accepts_scale_tranche() -> None:
    records = [{"example_id": f"clip-{index}"} for index in range(360)]
    selected, total = selection_records(
        {
            "schema": "umi-studio-openasl-scale-selection/1",
            "selected_new_clips": 360,
            "selected_new_sources": 180,
            "records": records,
        }
    )
    assert selected == records
    assert total == 360
