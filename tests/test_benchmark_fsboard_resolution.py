from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fsboard_resolution import (
    FACE_POINTS,
    LEFT_HAND_START,
    POSE_START,
    RIGHT_HAND_START,
    TOTAL_POINTS,
    baseline_landmark_dict,
)


def test_landmark_conversion_preserves_detected_parts() -> None:
    landmarks = np.full((2, TOTAL_POINTS, 3), np.nan, dtype=np.float32)
    landmarks[0, :FACE_POINTS] = 0.1
    landmarks[0, LEFT_HAND_START:POSE_START] = 0.2
    landmarks[0, POSE_START:RIGHT_HAND_START] = 0.3
    landmarks[0, RIGHT_HAND_START:] = 0.4
    converted = baseline_landmark_dict(landmarks)
    assert len(converted[0]["face_landmarks"][0]) == 468
    assert len(converted[0]["hand_landmarks"]) == 2
    assert len(converted[0]["pose_landmarks"][0]) == 33
    assert converted[1] == {
        "face_landmarks": None,
        "hand_landmarks": None,
        "pose_landmarks": None,
    }


def test_landmark_conversion_rejects_partial_parts() -> None:
    landmarks = np.full((1, TOTAL_POINTS, 3), np.nan, dtype=np.float32)
    landmarks[0, LEFT_HAND_START, :] = 0.1
    with pytest.raises(ValueError, match="partially finite"):
        baseline_landmark_dict(landmarks)
