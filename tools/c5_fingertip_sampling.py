"""Offline tip-trajectory compression diagnostics, without model execution.

Knots retain original frame indices/times. Each continuous observed hand segment
has the same budget across methods, including both endpoints. Missing hands and
track resets are never bridged. These knots are not linguistic sign boundaries.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

METHODS = ("uniform", "arc_length", "time_error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def segments(complete, continuity):
    """Yield half-open intervals for each unnamed hand track."""
    for hand in range(complete.shape[1]):
        start = None
        for frame in range(len(complete) + 1):
            observed = frame < len(complete) and complete[frame, hand]
            connected = observed and frame > 0 and continuity[frame, hand]
            if start is not None and not connected:
                yield hand, start, frame
                start = None
            if observed and start is None:
                start = frame


def interpolate(values, seconds, knots):
    flat = values.reshape(len(values), -1)
    result = np.column_stack(
        [np.interp(seconds, seconds[knots], flat[knots, axis]) for axis in range(flat.shape[1])]
    )
    return result.reshape(values.shape)


def select_knots(values, seconds, count, method):
    """Select from a finite single segment; adaptive methods see the full clip.

    Arc length samples cumulative RMS fingertip displacement and fills duplicate
    quantiles deterministically. Time error greedily splits the interval with
    largest RMS XY interpolation error at an original timestamp.
    """
    require(method in METHODS, "unknown sampling method")
    values, seconds = np.asarray(values), np.asarray(seconds)
    n = len(values)
    require(values.shape == (n, 5, 2) and n > 0, "expected nonempty five-tip XY segment")
    require(np.isfinite(values).all(), "segment must be fully observed")
    require(
        seconds.shape == (n,) and np.isfinite(seconds).all() and (np.diff(seconds) > 0).all(),
        "timestamps must be strictly increasing",
    )
    require(type(count) is int and min(2, n) <= count <= n, "invalid knot budget")
    if count == n:
        return np.arange(n)
    uniform = np.rint(np.linspace(0, n - 1, count)).astype(int)
    if method == "uniform":
        return uniform
    if method == "arc_length":
        distance = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=(1, 2)))
        cumulative = np.r_[0.0, np.cumsum(distance)]
        if cumulative[-1] == 0:
            return uniform
        chosen = {0, n - 1}
        for target in np.linspace(0, cumulative[-1], count)[1:-1]:
            chosen.add(int(np.argmin(np.abs(cumulative - target))))
        for index in [*uniform, *range(n)]:
            if len(chosen) == count:
                break
            chosen.add(int(index))
        return np.array(sorted(chosen))
    chosen, intervals = {0, n - 1}, []

    def add_interval(left, right):
        if right - left <= 1:
            return
        alpha = (seconds[left + 1 : right] - seconds[left]) / (seconds[right] - seconds[left])
        estimate = values[left] + alpha[:, None, None] * (values[right] - values[left])
        errors = np.mean((values[left + 1 : right] - estimate) ** 2, axis=(1, 2))
        worst = int(np.argmax(errors))
        heapq.heappush(intervals, (-float(errors[worst]), left, right, left + 1 + worst))

    add_interval(0, n - 1)
    while len(chosen) < count:
        _, left, right, knot = heapq.heappop(intervals)
        chosen.add(knot)
        add_interval(left, knot)
        add_interval(knot, right)
    return np.array(sorted(chosen))


def compress(tips, continuity, seconds, *, fraction, method):
    """Return knots, full-rate reconstruction and coverage/error diagnostics.

    The fraction applies per segment, with an endpoint minimum; actual retained
    fractions can exceed it. Both hands have independent budgets. No wrist,
    target, caption, body, face or native depth field is accepted as input.
    """
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
    require(method in METHODS, "unknown sampling method")
    complete = np.isfinite(tips).all(axis=(2, 3))
    sampled = np.zeros((frames, hands), dtype=bool)
    reconstructed = np.full_like(tips, np.nan)
    inventory, edge_errors = [], []
    path_original, path_reconstructed = 0.0, 0.0
    for hand, start, end in segments(complete, continuity):
        values, times = tips[start:end, hand], seconds[start:end]
        count = min(len(values), max(2, math.ceil(len(values) * fraction)))
        knots = select_knots(values, times, count, method)
        restored = interpolate(values, times, knots)
        sampled[start + knots, hand] = True
        reconstructed[start:end, hand] = restored
        inventory.append(
            {"hand": hand, "start": start, "end": end, "knots": (start + knots).tolist()}
        )
        if len(values) > 1:
            delta, restored_delta = np.diff(values, axis=0), np.diff(restored, axis=0)
            dt = np.diff(times)[:, None, None]
            edge_errors.extend(np.abs((delta - restored_delta) / dt).mean(axis=(1, 2)).tolist())
            path_original += float(np.linalg.norm(delta, axis=-1).sum())
            path_reconstructed += float(np.linalg.norm(restored_delta, axis=-1).sum())
    require(
        np.array_equal(np.isfinite(reconstructed).all(axis=(2, 3)), complete),
        "reconstruction changed observed support",
    )
    residual = reconstructed[complete] - tips[complete]
    point_errors = np.linalg.norm(residual, axis=-1)
    return {
        "sampled": sampled,
        "reconstructed": reconstructed,
        "segments": inventory,
        "metrics": {
            "observed_hand_frames": int(complete.sum()),
            "partial_hand_frames_excluded": int(
                (np.isfinite(tips).any(axis=(2, 3)) & ~complete).sum()
            ),
            "retained_hand_frames": int(sampled.sum()),
            "retained_fraction": float(sampled.sum() / complete.sum()) if complete.any() else None,
            "xy_coordinate_mae": float(np.abs(residual).mean()) if complete.any() else None,
            "p95_tip_euclidean_error": float(np.quantile(point_errors, 0.95))
            if complete.any()
            else None,
            "max_tip_euclidean_error": float(point_errors.max()) if complete.any() else None,
            "xy_velocity_mae_per_second": float(np.mean(edge_errors)) if edge_errors else None,
            "reconstructed_path_ratio": path_reconstructed / path_original
            if path_original > 0
            else None,
            "continuous_segments": len(inventory),
            "original_frames": frames,
            "original_last_minus_first_seconds": float(seconds[-1] - seconds[0]),
        },
    }
