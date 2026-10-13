"""Install confirmed post-MediaPipe temporal stride 2 into the C5 runtime."""

from __future__ import annotations

from typing import Any

import numpy as np


def temporal_subsample_landmarks(landmarks: dict[int, Any], stride: int = 2) -> dict[int, Any]:
    if stride < 1 or not landmarks or set(landmarks) != set(range(len(landmarks))):
        raise ValueError("landmark sequence or stride is invalid")
    return {
        output_index: landmarks[source_index]
        for output_index, source_index in enumerate(range(0, len(landmarks), stride))
    }


class StridedHandExtractor:
    def __init__(self, base, stride: int) -> None:
        self.base = base
        self.stride = stride

    def extract_hand_frames(self, frames: np.ndarray, landmarks: dict[int, Any]):
        return self.base.extract_hand_frames(
            np.ascontiguousarray(frames[:: self.stride]),
            temporal_subsample_landmarks(landmarks, self.stride),
        )


class StridedFaceExtractor:
    def __init__(self, base, stride: int) -> None:
        self.base = base
        self.stride = stride

    def extract_face_frames(self, frames: np.ndarray, landmarks: dict[int, Any]):
        return self.base.extract_face_frames(
            np.ascontiguousarray(frames[:: self.stride]),
            temporal_subsample_landmarks(landmarks, self.stride),
        )


def install_post_mediapipe_stride(runtime, stride: int = 2) -> None:
    if stride != 2:
        raise ValueError("the confirmed candidate requires temporal stride 2")
    if getattr(runtime, "_c5_temporal_stride", None) is not None:
        raise RuntimeError("C5 temporal stride is already installed")
    base_pose = runtime._process_pose_landmarks
    runtime._hand_extractor = StridedHandExtractor(runtime._hand_extractor, stride)
    runtime._face_extractor = StridedFaceExtractor(runtime._face_extractor, stride)

    def process_pose(landmarks: dict[int, Any]):
        return base_pose(temporal_subsample_landmarks(landmarks, stride))

    runtime._process_pose_landmarks = process_pose
    runtime._c5_temporal_stride = stride

