"""Complete-grid feature and token comparisons for authenticated native112 paths."""

from __future__ import annotations

import hashlib

import numpy as np

STREAMS = ("face_features", "left_hand_features", "right_hand_features")
ATOL = 1e-4
RTOL = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_input(values):
    pixels, pose, indices = (values[k] for k in ("pixels", "pose_features", "frame_indices"))
    frames = values["header"]["source_frames"]
    require(
        type(frames) is int and frames > 0 and values["header"]["source_fps"] == 25,
        "parity requires a complete original25fps source timeline",
    )
    expected = np.arange(0, frames, 2, dtype=np.int64)
    require(
        isinstance(pixels, np.ndarray)
        and pixels.dtype == np.uint8
        and pixels.shape == (len(expected), 3, 112, 112, 3)
        and isinstance(pose, np.ndarray)
        and pose.dtype == np.float32
        and pose.shape == (len(expected), 14)
        and np.isfinite(pose).all()
        and isinstance(indices, np.ndarray)
        and indices.dtype == np.int64
        and np.array_equal(indices, expected),
        "parity requires native RGB112 crops, fixed14D pose and the complete stride2 grid",
    )
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in (("pixels", pixels), ("pose_features", pose), ("frame_indices", indices))
    }


def native_streams(embedders, values):
    check_input(values)
    require(set(embedders) == {"hands", "face"}, "both native embedders are required")
    pixels = values["pixels"]
    result = {
        "face_features": embedders["face"].extract_embeddings_from_frames(pixels[:, 0]),
        "left_hand_features": embedders["hands"].extract_embeddings_from_frames(pixels[:, 1]),
        "right_hand_features": embedders["hands"].extract_embeddings_from_frames(pixels[:, 2]),
        "pose_features": values["pose_features"].copy(),
    }
    check_streams(result, len(pixels))
    return result


def check_streams(streams, frames):
    require(set(streams) == {*STREAMS, "pose_features"}, "parity stream inventory changed")
    for name, values in streams.items():
        width = 14 if name == "pose_features" else 384
        require(
            isinstance(values, np.ndarray)
            and values.dtype == np.float32
            and values.shape == (frames, width)
            and np.isfinite(values).all(),
            "parity requires complete finite FP32 streams with unchanged widths",
        )


def compare_streams(reference, actual, frames):
    check_streams(reference, frames)
    check_streams(actual, frames)
    require(
        reference["pose_features"].tobytes() == actual["pose_features"].tobytes(),
        "native parity changed original pose bits",
    )
    report = {}
    for name in STREAMS:
        first, second = reference[name], actual[name]
        delta = second.astype(np.float64) - first
        report[name] = {
            "maximum_absolute_difference": float(np.max(np.abs(delta))),
            "relative_l2_error": float(
                np.linalg.norm(delta) / max(np.linalg.norm(first.astype(np.float64)), 1e-12)
            ),
        }
        require(
            np.allclose(first, second, atol=ATOL, rtol=RTOL),
            "native parity feature tolerance failed: " + name,
        )
    return {
        "streams": report,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "frames": frames,
        "pose_bitwise_equal": True,
    }


def compare_tokens(reference, actual):
    require(
        isinstance(reference, list)
        and isinstance(actual, list)
        and reference
        and all(type(token) is int and token >= 0 for token in [*reference, *actual]),
        "parity requires complete nonempty integer token sequences",
    )
    require(reference == actual, "native parity generated token IDs differ")
    return {"exact_token_match": True, "token_count": len(reference)}
