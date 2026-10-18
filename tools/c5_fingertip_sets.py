"""Unordered per-frame fingertip groups from all retained detector hand slots.

Slots follow detector order and carry no persistent hand or speaker identity.
Native depth stays in its original detector convention. No tracking, speaker
selection, spatial normalization or cross-hand world-3D calculation occurs.
"""

from __future__ import annotations

import numpy as np

TIPS = (4, 8, 12, 16, 20)
PALM_TARGETS = (0, 5, 9, 13, 17)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hand_sets(arrays):
    hands = np.asarray(arrays["hand_landmarks"])
    counts = np.asarray(arrays["hand_counts"])
    frame_valid = np.asarray(arrays["frame_valid"])
    require(
        hands.ndim == 4 and hands.shape[1:] == (6, 21, 3) and hands.dtype.kind == "f",
        "expected original six-slot floating hand coordinates",
    )
    frames = len(hands)
    require(frames > 0 and counts.shape == frame_valid.shape == (frames,), "invalid timeline")
    require(
        counts.dtype.kind in "iu" and ((counts >= 0) & (counts <= 6)).all(), "invalid hand count"
    )
    require(frame_valid.dtype == np.bool_, "invalid frame-valid mask")
    present = frame_valid[:, None] & (np.arange(6)[None] < counts[:, None])
    controls = hands.copy()
    controls[~present] = np.nan
    tips = controls[:, :, TIPS].copy()
    targets = controls[:, :, PALM_TARGETS].copy()
    result = {
        "fingertips": tips,
        "wrist_mcp_targets": targets,
        "full_hand_controls": controls,
        "detector_hand_present": present,
        "source_hand_counts": counts.copy(),
        "source_frame_valid": frame_valid.copy(),
    }
    for name in ("fingertips", "wrist_mcp_targets", "full_hand_controls"):
        result[name + "_valid_xy"] = np.isfinite(result[name][..., :2]).all(axis=-1)
        result[name + "_valid_native_xyz"] = np.isfinite(result[name]).all(axis=-1)
    return result


def input_view(features, coordinates="xy"):
    """Expose tips and masks only; target/control arrays never determine inputs."""
    require(coordinates in {"xy", "native_xyz"}, "unknown coordinate convention")
    dimensions = 2 if coordinates == "xy" else 3
    values = features["fingertips"][..., :dimensions]
    valid = features["fingertips_valid_" + coordinates]
    require(np.array_equal(valid, np.isfinite(values).all(axis=-1)), "fingertip mask changed")
    return {"values": np.where(valid[..., None], values, 0), "valid": valid.copy()}
