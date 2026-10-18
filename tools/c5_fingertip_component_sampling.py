"""Compare tip-centroid and centered-shape signals for offline knot selection.

The centroid is computed from the five tips, not from a palm or wrist target.
Centered shape and centroid jointly recover the input exactly. Adaptive weights
use only observed tip motion in the same continuous segment. No fitting occurs.
"""

from __future__ import annotations

import math

import numpy as np
from c5_fingertip_sampling import interpolate, require, segments, select_knots

METHODS = ("raw", "centroid", "centered_shape", "equal_motion")


def selection_view(values, method):
    require(method in METHODS, "unknown component method")
    centroid = values.mean(axis=1, keepdims=True)
    centered = values - centroid
    if method == "raw":
        return values.copy()
    if method == "centroid":
        return np.broadcast_to(centroid, values.shape).copy()
    if method == "centered_shape":
        return centered
    if len(values) == 1:
        return values.copy()

    def normalize_motion(component):
        rms = np.sqrt(np.mean(np.diff(component, axis=0) ** 2))
        return component / rms if rms > 1e-12 else np.zeros_like(component)

    return normalize_motion(centroid) + normalize_motion(centered)


def compress(tips, continuity, seconds, *, fraction, method):
    """Use equal segment budgets; evaluate reconstruction in original XY units."""
    tips = np.asarray(tips, dtype=np.float64)
    continuity, seconds = np.asarray(continuity), np.asarray(seconds)
    require(tips.ndim == 4 and tips.shape[2:] == (5, 2), "expected frame/hand/five-tip/XY")
    frames, hands = tips.shape[:2]
    require(frames > 0 and hands > 0, "empty timeline")
    require(
        continuity.shape == (frames, hands) and continuity.dtype == np.bool_,
        "invalid continuity mask",
    )
    require(
        seconds.shape == (frames,) and np.isfinite(seconds).all() and (np.diff(seconds) > 0).all(),
        "timestamps must be strictly increasing",
    )
    require(np.isfinite(fraction) and 0 < fraction <= 1, "invalid sampling fraction")
    require(method in METHODS, "unknown component method")
    complete = np.isfinite(tips).all(axis=(2, 3))
    sampled = np.zeros((frames, hands), bool)
    restored = np.full_like(tips, np.nan)
    inventory = []
    for hand, start, end in segments(complete, continuity):
        values, times = tips[start:end, hand], seconds[start:end]
        count = min(len(values), max(2, math.ceil(len(values) * fraction)))
        knots = select_knots(selection_view(values, method), times, count, "time_error")
        restored[start:end, hand] = interpolate(values, times, knots)
        sampled[start + knots, hand] = True
        inventory.append(
            {"hand": hand, "start": start, "end": end, "knots": (start + knots).tolist()}
        )
    require(
        np.array_equal(np.isfinite(restored).all(axis=(2, 3)), complete),
        "reconstruction changed support",
    )
    error = restored[complete] - tips[complete]
    centroid_error = error.mean(axis=1, keepdims=True)
    centered_error = error - centroid_error
    return {
        "sampled": sampled,
        "reconstructed": restored,
        "segments": inventory,
        "metrics": {
            "observed_hand_frames": int(complete.sum()),
            "retained_hand_frames": int(sampled.sum()),
            "xy_coordinate_mae": float(np.abs(error).mean()) if complete.any() else None,
            "tip_centroid_xy_mae": float(np.abs(centroid_error).mean()) if complete.any() else None,
            "centered_tip_xy_mae": float(np.abs(centered_error).mean()) if complete.any() else None,
            "p95_tip_euclidean_error": float(np.quantile(np.linalg.norm(error, axis=-1), 0.95))
            if complete.any()
            else None,
        },
    }
