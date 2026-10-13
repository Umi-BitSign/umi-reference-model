"""Select compact FSboard landmarks for a separate paper-like experiment.

The 543-point caches omit two iris points used in the published 85-point
selection, so they produce 83 points. A 553-point input with 478 face points
produces all 85. Missing points use -10 after unit-box normalization.
"""

from __future__ import annotations

import numpy as np

FACE_37 = (
    0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285,
    291, 294, 311, 323, 362, 386, 397, 468, 473,
)
POSE_6 = (11, 12, 13, 14, 23, 24)
MISSING_VALUE = -10.0


def selected_point_indices(face_points: int) -> tuple[int, ...]:
    if face_points not in (468, 478):
        raise ValueError("expected 468 or 478 MediaPipe face points")
    face = tuple(index for index in FACE_37 if index < face_points)
    left_hand = tuple(range(face_points, face_points + 21))
    pose_start = face_points + 21
    pose = tuple(pose_start + index for index in POSE_6)
    right_start = pose_start + 33
    right_hand = tuple(range(right_start, right_start + 21))
    return face + left_hand + pose + right_hand


def paper_like_features(
    landmarks: np.ndarray, *, frame_stride: int = 2, max_frames: int = 256,
    normalization: str = "uniform",
) -> np.ndarray:
    """Sample frames and map observed xyz coordinates into a unit bounding box."""
    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] not in ((543, 3), (553, 3))
        or landmarks.dtype != np.float32
        or frame_stride < 1
        or max_frames < 1
        or normalization not in ("uniform", "per_axis")
        or np.isinf(landmarks).any()
    ):
        raise ValueError("unexpected FSboard landmarks or preprocessing parameters")
    face_points = landmarks.shape[1] - 75
    sampled = landmarks[::frame_stride, selected_point_indices(face_points), :]
    sampled = sampled[~np.isnan(sampled).all(axis=(1, 2))]
    if not 0 < len(sampled) <= max_frames:
        raise ValueError("FSboard clip has no selected landmarks or exceeds the frame limit")
    lows = np.zeros(3, dtype=np.float32)
    ranges = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        observed = sampled[:, :, axis][np.isfinite(sampled[:, :, axis])]
        if observed.size:
            lows[axis] = observed.min()
            ranges[axis] = observed.max() - lows[axis]
    if normalization == "uniform":
        scale = max(float(ranges.max()), 1e-6)
        scales = np.full(3, scale, dtype=np.float32)
    else:
        scales = np.maximum(ranges, 1e-6)
    normalized = (sampled - lows) / scales
    normalized[~np.isfinite(sampled)] = MISSING_VALUE
    if not np.isfinite(normalized).all():
        raise ValueError("FSboard unit-box normalization produced nonfinite values")
    return np.ascontiguousarray(normalized.reshape(len(sampled), -1))
