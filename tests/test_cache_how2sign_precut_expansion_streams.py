from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from benchmark_fleurs_dev import sha256
from cache_how2sign_precut_expansion_streams import (
    SCHEMA,
    checked_index,
    checked_receipted_prefix,
    rebind_prior_index,
)


def _selection() -> dict:
    return {
        "schema": "umi-studio-how2sign-precut-expansion/1",
        "training_only": True,
        "clips": 2,
        "records": [
            {
                "sentence_name": name,
                "archive_member": f"raw_videos/{name}.mp4",
                "archive_crc32": position,
                "archive_file_size": 5,
            }
            for position, name in enumerate(("first", "second"), start=1)
        ],
    }


def _receipt(name: str, position: int, selection_hash: str) -> dict:
    return {
        "schema": "umi-studio-how2sign-precut-expansion-clip/1",
        "selection_sha256": selection_hash,
        "sentence_name": name,
        "archive_member": f"raw_videos/{name}.mp4",
        "archive_crc32": position,
        "clip_size_bytes": 5,
        "clip_sha256": "a" * 64,
    }


def test_receipts_must_be_a_deterministic_selection_prefix() -> None:
    selection = _selection()
    selection_hash = "s" * 64
    first = _receipt("first", 1, selection_hash)
    assert checked_receipted_prefix(selection, [first], selection_hash) == [
        selection["records"][0]
    ]
    with pytest.raises(ValueError, match="verified selection prefix"):
        checked_receipted_prefix(
            selection, [_receipt("second", 2, selection_hash)], selection_hash
        )


def test_existing_stream_index_rehashes_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "first.npz"
    np.savez(cache, encoder_hidden_state=np.zeros((2, 1536), dtype=np.float32))
    row = {
        "schema": SCHEMA,
        "sentence_name": "first",
        "selection_sha256": "s" * 64,
        "clip_receipts_sha256": "r" * 64,
        "temporal_stride": 2,
        "source_frames": 4,
        "frames": 2,
        "encoder_cache_sha256": sha256(cache),
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    row["landmark_receipts_sha256"] = "l" * 64
    index.write_text(json.dumps(row) + "\n")
    assert checked_index(index, cache_dir, "s" * 64, "r" * 64, "l" * 64) == {"first"}
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        checked_index(index, cache_dir, "s" * 64, "r" * 64, "l" * 64)


def test_rebinds_validated_pilot_to_complete_landmark_receipts(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "first.npz"
    np.savez(cache, encoder_hidden_state=np.zeros((2, 1536), dtype=np.float32))
    landmark_row = {"sentence_name": "first", "landmarks_sha256": "m" * 64}
    prior_landmarks = tmp_path / "pilot-landmarks.jsonl"
    prior_landmarks.write_text(json.dumps(landmark_row) + "\n")
    prior_row = {
        "schema": SCHEMA,
        "sentence_name": "first",
        "selection_sha256": "s" * 64,
        "clip_receipts_sha256": "r" * 64,
        "landmark_receipts_sha256": sha256(prior_landmarks),
        "landmarks_sha256": "m" * 64,
        "temporal_stride": 2,
        "source_frames": 4,
        "frames": 2,
        "encoder_cache_sha256": sha256(cache),
    }
    prior_index = tmp_path / "pilot-index.jsonl"
    prior_index.write_text(json.dumps(prior_row) + "\n")
    index = tmp_path / "index.jsonl"
    rebind_prior_index(
        prior_index=prior_index,
        prior_landmark_receipts=prior_landmarks,
        index=index,
        cache_dir=cache_dir,
        records=[{"sentence_name": "first"}],
        landmark_rows=[landmark_row, {"sentence_name": "second"}],
        selection_hash="s" * 64,
        receipts_hash="r" * 64,
        landmark_receipts_hash="l" * 64,
    )
    assert checked_index(index, cache_dir, "s" * 64, "r" * 64, "l" * 64) == {
        "first"
    }
