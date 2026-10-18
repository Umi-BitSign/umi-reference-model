"""Small CPU ridge probes for withheld detector wrist/MCP coordinates.

Features read only XY tips and tip-derived continuity. Geometry reconstruction
does not establish linguistic sufficiency or anatomical ground truth.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from c5_fingertip_probe_data import require

METHODS = (
    "centroid",
    "centered_linear",
    "normalized_linear",
    "normalized_quadratic",
    "temporal_normalized",
    "bimanual_normalized",
)
OFFSETS = (-6, -2, 2, 6)
RIDGE = 0.001
SCALE_FLOOR = 0.0001


def features(tips, continuity, method):
    """Return one row per original hand frame; absent context is zero plus mask.

    Temporal context is offline and never crosses a missing frame or reset.
    Other-hand features describe another unnamed detector track, not a speaker.
    Normalization uses the current hand's five tips only.
    """
    require(method in METHODS, "unknown reconstruction method")
    tips = np.asarray(tips, dtype=np.float64)
    continuity = np.asarray(continuity)
    require(tips.ndim == 4 and tips.shape[1:] == (2, 5, 2), "expected two-hand XY tips")
    frames = len(tips)
    require(frames > 0, "empty timeline")
    require(
        continuity.shape == (frames, 2) and continuity.dtype == np.bool_,
        "invalid continuity",
    )
    complete = np.isfinite(tips).all(axis=(2, 3))
    safe = np.where(complete[..., None, None], tips, 0)
    center = safe.mean(axis=2)
    relative = safe - center[..., None, :]
    scale = np.maximum(np.sqrt(np.mean(relative**2, axis=(2, 3))), SCALE_FLOOR)
    if method in {"centroid", "centered_linear"}:
        scale = np.ones_like(scale)
    shape = (relative / scale[..., None, None]).reshape(frames, 2, 10)
    pieces = [shape]
    if method == "normalized_quadratic":
        left, right = np.triu_indices(10)
        pieces.append(shape[..., left] * shape[..., right])
    if method == "temporal_normalized":
        for offset in OFFSETS:
            indices = np.arange(frames) + offset
            valid = complete.copy() & ((indices >= 0) & (indices < frames))[:, None]
            for frame in range(frames):
                other = indices[frame]
                if 0 <= other < frames:
                    low, high = sorted((frame, other))
                    valid[frame] &= continuity[low + 1 : high + 1].all(axis=0)
                    valid[frame] &= complete[low : high + 1].all(axis=0)
            delta = (safe[np.clip(indices, 0, frames - 1)] - safe) / scale[..., None, None]
            pieces.extend(
                [np.where(valid[..., None], delta.reshape(frames, 2, 10), 0), valid[..., None]]
            )
    if method == "bimanual_normalized":
        valid = complete & complete[:, ::-1]
        other = (safe[:, ::-1] - center[..., None, :]) / scale[..., None, None]
        pieces.extend(
            [np.where(valid[..., None], other.reshape(frames, 2, 10), 0), valid[..., None]]
        )
    values = np.concatenate(pieces, axis=-1)
    values[~complete] = 0
    require(np.isfinite(values).all(), "nonfinite tip features")
    return {
        "x": values.reshape(frames * 2, -1),
        "center": center.reshape(-1, 2),
        "scale": scale.reshape(-1),
        "complete": complete.reshape(-1),
        "layout": (frames, 2, 5, 2),
    }


def balanced_weights(rows, eligible):
    """Equal families, groups per family, clips per group, rows per clip.

    Recomputed for each joint's finite fit targets; no held-out rows enter.
    """
    require(len(rows) == len(eligible) > 0, "invalid fit inventory")
    counts = [int(mask.sum()) for mask in eligible]
    clips = Counter(
        (row["family"], row["group"]) for row, count in zip(rows, counts, strict=True) if count
    )
    groups = Counter(family for family, _ in clips)
    require(len(groups) > 0, "no finite fit targets")
    return [
        np.full(
            count,
            1 / (len(groups) * groups[row["family"]] * clips[row["family"], row["group"]] * count),
        )
        if count
        else np.empty(0)
        for row, count in zip(rows, counts, strict=True)
    ]


def fit(rows, *, ridge=RIDGE):
    """Fit five independent XY ridge regressions; input rows must be fit-only."""
    require(rows and all(row["role"] == "fit" for row in rows), "fit-only rows required")
    require(np.isfinite(ridge) and ridge > 0, "positive ridge required")
    output = {"ridge": np.asarray(ridge)}
    for joint in range(5):
        eligible = [row["features"]["complete"] & row["target_valid"][:, joint] for row in rows]
        weights = np.concatenate(balanced_weights(rows, eligible))
        x = np.concatenate(
            [row["features"]["x"][mask] for row, mask in zip(rows, eligible, strict=True)]
        )
        y = np.concatenate(
            [
                (row["targets"][mask, joint] - row["features"]["center"][mask])
                / row["features"]["scale"][mask, None]
                for row, mask in zip(rows, eligible, strict=True)
            ]
        )
        mean = weights @ x
        std = np.sqrt(weights @ ((x - mean) ** 2))
        std[std < 1e-8] = 1
        design = np.column_stack([np.ones(len(x)), (x - mean) / std])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(
            design.T @ (weights[:, None] * design) + penalty,
            design.T @ (weights[:, None] * y),
        )
        output[f"mean_{joint}"] = mean
        output[f"std_{joint}"] = std
        output[f"coef_{joint}"] = coefficients
        output[f"count_{joint}"] = np.asarray(len(x))
    require(all(np.isfinite(value).all() for value in output.values()), "nonfinite ridge state")
    return output


def predict(view, model=None):
    """Predict every row, masking only incomplete fingertip inputs."""
    predictions = np.broadcast_to(view["center"][:, None, :], (len(view["x"]), 5, 2)).copy()
    if model is not None:
        for joint in range(5):
            design = np.column_stack(
                [
                    np.ones(len(view["x"])),
                    (view["x"] - model[f"mean_{joint}"]) / model[f"std_{joint}"],
                ]
            )
            predictions[:, joint] += (design @ model[f"coef_{joint}"]) * view["scale"][:, None]
    predictions[~view["complete"]] = np.nan
    return predictions.reshape(view["layout"])
