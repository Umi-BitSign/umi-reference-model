"""Read the existing timing visibility definition from packed landmark arrays.

No face arrays, cached semantic states, neural models or inferred hand identities
are needed. This retains the original one-or-two detected hands policy.
"""

from __future__ import annotations

import numpy as np
from c5_timing_visibility import TIPS, interval_mask


def from_packed(hand_landmarks, hand_counts, frame_valid):
    if (
        not isinstance(hand_landmarks, np.ndarray)
        or hand_landmarks.dtype != np.float64
        or hand_landmarks.ndim != 4
        or hand_landmarks.shape[1:] != (6, 21, 3)
        or hand_landmarks.shape[0] < 1
        or not isinstance(hand_counts, np.ndarray)
        or hand_counts.dtype != np.uint8
        or hand_counts.shape != hand_landmarks.shape[:1]
        or (hand_counts > 6).any()
        or not isinstance(frame_valid, np.ndarray)
        or frame_valid.dtype != np.bool_
        or frame_valid.shape != hand_counts.shape
    ):
        raise ValueError("packed visibility needs complete original float64 hand arrays and masks")
    first_two = hand_landmarks[:, :2, :, :][:, :, TIPS, :2]
    complete_xy = np.isfinite(first_two).all(axis=(2, 3))
    observed = (
        frame_valid
        & ((hand_counts == 1) | (hand_counts == 2))
        & complete_xy[:, 0]
        & ((hand_counts == 1) | complete_xy[:, 1])
    )
    indices = np.arange(0, len(observed), 2, dtype=np.int64)
    valid = interval_mask(observed, indices)
    return {
        "frame_indices": indices,
        "source_hand_tip_visible": observed,
        "valid_intervals": valid,
        "visibility_component": np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(~valid, dtype=np.int64))
        ),
    }
