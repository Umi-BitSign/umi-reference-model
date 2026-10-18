"""Alternative fingertip representations, with original timing and explicit masks.

These are data transforms, not trained predictors. XY distances use the stored
image-normalized axes, not physical units. Native z remains detector-relative;
no cross-hand depth difference or world-3D distance is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from c5_fingertip_probe_data import require

PAIRS = tuple(combinations(range(5), 2))


@dataclass(frozen=True)
class Feature:
    values: np.ndarray
    valid: np.ndarray


def masked(values, valid):
    mask = valid if values.ndim == valid.ndim else valid[..., None]
    return Feature(np.where(mask, values, 0), valid.copy())


def representations(sequence):
    """Read only tips, tip masks, timestamps and tip-derived continuity."""
    tips = sequence.tips
    require(tips.ndim == 4 and tips.shape[1:3] == (2, 5), "invalid tip shape")
    require(tips.shape[-1] in {2, 3}, "unknown coordinate convention")
    seconds = sequence.seconds
    require(
        seconds.shape == (len(tips),)
        and np.isfinite(seconds).all()
        and (np.diff(seconds) > 0).all(),
        "original timestamps must be finite and strictly increasing",
    )
    valid = sequence.tip_valid
    require(np.array_equal(valid, np.isfinite(tips).all(axis=-1)), "tip mask changed")
    require(sequence.continuity.shape == tips.shape[:2], "invalid continuity shape")
    complete = valid.all(axis=2)
    centroid = np.mean(tips, axis=2)
    result = {
        "raw_tips": masked(tips, valid),
        "centroid": masked(centroid, complete),
        "relative_tips": masked(tips - centroid[:, :, None], valid & complete[..., None]),
    }
    left, right = np.array(PAIRS).T
    xy = tips[..., :2]
    xy_valid = np.isfinite(xy).all(axis=-1)
    pairwise = np.linalg.norm(xy[:, :, left] - xy[:, :, right], axis=-1)
    result["pairwise_xy"] = masked(pairwise, xy_valid[:, :, left] & xy_valid[:, :, right])
    xy_complete = xy_valid.all(axis=2)
    xy_centroid = np.mean(xy, axis=2)
    result["cross_hand_xy"] = masked(xy_centroid[:, 1] - xy_centroid[:, 0], xy_complete.all(axis=1))
    dt = np.diff(seconds)[:, None, None, None]
    velocity = np.zeros_like(tips, dtype=np.float64)
    velocity[1:] = np.diff(tips, axis=0) / dt
    velocity_valid = np.zeros_like(valid)
    velocity_valid[1:] = valid[1:] & valid[:-1] & sequence.continuity[1:, :, None]
    result["velocity"] = masked(velocity, velocity_valid)
    speed_valid = np.zeros_like(xy_valid)
    speed_valid[1:] = xy_valid[1:] & xy_valid[:-1] & sequence.continuity[1:, :, None]
    result["speed_xy"] = masked(np.linalg.norm(velocity[..., :2], axis=-1), speed_valid)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = np.diff(result["velocity"].values, axis=0) / dt
    acceleration_valid = np.zeros_like(valid)
    acceleration_valid[1:] = velocity_valid[1:] & velocity_valid[:-1]
    result["acceleration"] = masked(acceleration, acceleration_valid)
    return result
