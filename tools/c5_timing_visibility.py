"""Conservative visibility for timing; full-rate gaps never disappear at stride2."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

TIPS = (4, 8, 12, 16, 20)
POLICY = {
    "accepted_detected_hand_count": [1, 2],
    "observed": "all five XY fingertips finite in every detected hand",
    "interval": "every full-rate frame between adjacent sampled frames is observed",
    "out_of_image_coordinates": "retained",
    "native_z_used": False,
    "speaker_identity_verified": False,
    "linguistic_sign_boundaries": False,
}


def visible(row):
    if row is None or row.get("hand_landmarks") is None:
        return False
    hands = np.asarray(row["hand_landmarks"], dtype=np.float64)
    if hands.size == 0:
        return False
    if hands.ndim != 3 or hands.shape[1:] != (21, 3):
        raise ValueError("timing visibility requires complete21-point detector hands")
    return bool(1 <= len(hands) <= 2 and np.isfinite(hands[:, TIPS, :2]).all())


def interval_mask(source_visible, indices):
    if (
        not isinstance(source_visible, np.ndarray)
        or source_visible.dtype != np.bool_
        or source_visible.ndim != 1
        or not len(source_visible)
        or not isinstance(indices, np.ndarray)
        or indices.dtype != np.int64
        or not np.array_equal(indices, np.arange(0, len(source_visible), 2, dtype=np.int64))
    ):
        raise ValueError("timing visibility requires the complete original stride2 grid")
    return np.asarray(
        [source_visible[a : b + 1].all() for a, b in pairwise(indices)],
        dtype=np.bool_,
    )


def visibility(landmarks, source_frames, indices):
    if (
        type(source_frames) is not int
        or source_frames < 1
        or not isinstance(landmarks, dict)
        or set(landmarks) != set(range(source_frames))
    ):
        raise ValueError("timing visibility requires every original detector frame")
    observed = np.asarray([visible(landmarks[i]) for i in range(source_frames)], dtype=np.bool_)
    return observed, interval_mask(observed, indices)
