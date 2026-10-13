from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from cache_c5_crop_calibration_pairs import checked_index, shifted_frames


def test_checked_index_binds_pair_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "one.npz"
    np.savez(cache, face_features=np.zeros((2, 384), dtype=np.float32))
    from benchmark_fleurs_dev import sha256

    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": "umi-studio-c5-crop-calibration-pair/1",
                "sentence_name": "one",
                "plan_sha256": "p",
                "visual_crop_size": 168,
                "crop168_cache_sha256": sha256(cache),
            }
        )
        + "\n"
    )
    assert checked_index(index, cache_dir, "p") == {"one"}
    with pytest.raises(ValueError, match="pair differs"):
        checked_index(index, cache_dir, "other")


def test_shifted_frames_is_deterministic_and_bounded() -> None:
    frame = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    first = shifted_frames([frame], "clip")[0]
    second = shifted_frames([frame], "clip")[0]
    assert np.array_equal(first, second)
    assert first.shape == frame.shape
    assert not np.array_equal(first, frame)
