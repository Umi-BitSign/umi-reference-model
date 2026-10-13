"""The C5 ASL STEM continuation keeps every source at effective stride 2."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_c5_stem_continuation import stride2_streams


def test_replay_streams_are_subsampled_to_even_frames():
    widths = {
        "face_features": 384,
        "left_hand_features": 384,
        "right_hand_features": 384,
        "pose_features": 14,
    }
    streams = {
        name: np.arange(5 * width, dtype=np.float32).reshape(5, width)
        for name, width in widths.items()
    }
    row = {"source": "how2sign_original", "frames": 5, "id": "clip"}
    with patch("train_c5_stem_continuation.training_streams", return_value=streams):
        result = stride2_streams(row)
    assert {name: value.shape for name, value in result.items()} == {
        name: (3, width) for name, width in widths.items()
    }
    assert np.array_equal(result["face_features"], streams["face_features"][[0, 2, 4]])


def test_stored_stride2_streams_are_not_subsampled_again(tmp_path: Path):
    widths = {
        "face_features": 384,
        "left_hand_features": 384,
        "right_hand_features": 384,
        "pose_features": 14,
    }
    streams = {
        name: np.ones((3, width), dtype=np.float32) for name, width in widths.items()
    }
    cache = tmp_path / "openasl.npz"
    np.savez(
        cache,
        encoder_hidden_state=np.ones((3, 1536), dtype=np.float32),
        **streams,
    )
    from benchmark_fleurs_dev import sha256

    row = {
        "source": "openasl",
        "frames": 3,
        "id": "clip",
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "stored_temporal_stride": 2,
    }
    result = stride2_streams(row)
    assert {name: value.shape for name, value in result.items()} == {
        name: (3, width) for name, width in widths.items()
    }
