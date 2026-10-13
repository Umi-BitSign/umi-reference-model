"""The packaged C5 runtime strides only after full-frame MediaPipe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_stride2_runtime import install_post_mediapipe_stride


class Extractor:
    def extract_hand_frames(self, frames, landmarks):
        return frames, landmarks

    def extract_face_frames(self, frames, landmarks):
        return frames, landmarks


class Runtime:
    def __init__(self):
        self._hand_extractor = Extractor()
        self._face_extractor = Extractor()
        self._process_pose_landmarks = lambda landmarks: landmarks


def test_installs_post_mediapipe_stride_two_consistently():
    runtime = Runtime()
    install_post_mediapipe_stride(runtime)
    frames = np.arange(15, dtype=np.uint8).reshape(5, 1, 3)
    landmarks = {index: {"frame": index} for index in range(5)}
    hand_frames, hand_landmarks = runtime._hand_extractor.extract_hand_frames(
        frames, landmarks
    )
    face_frames, face_landmarks = runtime._face_extractor.extract_face_frames(
        frames, landmarks
    )
    pose_landmarks = runtime._process_pose_landmarks(landmarks)
    assert np.array_equal(hand_frames, frames[[0, 2, 4]])
    assert np.array_equal(face_frames, frames[[0, 2, 4]])
    assert hand_landmarks == face_landmarks == pose_landmarks == {
        0: {"frame": 0},
        1: {"frame": 2},
        2: {"frame": 4},
    }


def test_rejects_double_install_or_other_stride():
    runtime = Runtime()
    with pytest.raises(ValueError, match="requires"):
        install_post_mediapipe_stride(runtime, 3)
    install_post_mediapipe_stride(runtime)
    with pytest.raises(RuntimeError, match="already"):
        install_post_mediapipe_stride(runtime)
