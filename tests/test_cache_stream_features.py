"""Raw stream caches retain finite, frame-aligned visual features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cache_fleurs_encoder import checked_stream_features


def _streams(frames: int = 3) -> dict[str, np.ndarray]:
    return {
        "face_features": np.ones((frames, 384), dtype=np.float64),
        "left_hand_features": np.ones((frames, 384), dtype=np.float64),
        "right_hand_features": np.ones((frames, 384), dtype=np.float64),
        "pose_features": np.ones((frames, 14), dtype=np.float64),
    }


def test_stream_features_are_float32_and_frame_aligned():
    source = _streams()
    checked = checked_stream_features(source, 3)
    assert set(checked) == set(source)
    assert all(value.dtype == np.float32 for value in checked.values())
    assert all(value.shape[0] == 3 for value in checked.values())
    source["face_features"][0, 0] = 9
    assert checked["face_features"][0, 0] == 1


def test_stream_features_reject_missing_or_nonfinite_values():
    source = _streams()
    del source["pose_features"]
    with pytest.raises(ValueError, match="missing"):
        checked_stream_features(source, 3)
    source = _streams()
    source["pose_features"] = np.ones((2, 14))
    with pytest.raises(ValueError, match="shape"):
        checked_stream_features(source, 3)
    source = _streams()
    source["face_features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="value"):
        checked_stream_features(source, 3)
