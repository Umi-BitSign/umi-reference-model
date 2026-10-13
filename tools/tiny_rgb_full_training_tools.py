"""Byte-equivalent RGB streaming and batched gradient checks for the full-data trainer."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def decode_rgb_video_fast(
    video: str, short_side: int, target_fps: float = 12.5, max_frames: int = 512
):
    """Match the reference RGB decoder with an explicit two-thread codec bound."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    import cv2

    path = (ROOT / video).resolve()
    if not path.is_relative_to(ROOT / "data") or short_side < 32 or target_fps <= 0:
        raise ValueError("full RGB decoding requires a private data/ clip and valid scale")
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if capture.isOpened() and int(capture.get(cv2.CAP_PROP_N_THREADS)) != 2:
            raise ValueError("full RGB decoder did not honor its two-thread CPU bound")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps < target_fps or count < 1:
            raise ValueError("full RGB source cannot support the declared temporal policy")
        indices = np.unique(np.floor(np.arange(0, count, fps / target_fps)).astype(np.int64))
        if len(indices) > max_frames:
            raise ValueError("full RGB clip exceeds its frame cap; do not truncate signing")
        pending = set(indices.tolist())
        frames, source_shape = [], None
        index = 0
        while pending:
            success, bgr = capture.read()
            if not success or bgr is None:
                break
            if index in pending:
                height, width = bgr.shape[:2]
                if source_shape is None:
                    source_shape = (height, width)
                if (height, width) != source_shape:
                    raise ValueError("full RGB source dimensions change within a clip")
                scale = min(1.0, short_side / min(height, width))
                size = max(1, round(width * scale)), max(1, round(height * scale))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
                pending.remove(index)
            index += 1
        if pending:
            raise ValueError("full RGB decoder ended before all declared frames")
        return (
            np.stack(frames),
            indices,
            {
                "source_fps": fps,
                "source_frames": count,
                "source_shape": list(source_shape),
                "target_fps": target_fps,
                "short_side_cap": short_side,
            },
        )
    finally:
        capture.release()


def encode_pixel_delta(pixels: np.ndarray) -> np.ndarray:
    """Lossless byte compression transform, not learned features or augmentation."""
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3 or len(pixels) < 1:
        raise ValueError("pixel delta encoding requires nonempty uint8 THWC RGB")
    delta = np.empty_like(pixels)
    delta[0] = pixels[0]
    np.subtract(pixels[1:], pixels[:-1], out=delta[1:])
    return delta


def decode_pixel_delta(delta: np.ndarray) -> np.ndarray:
    if delta.dtype != np.uint8 or delta.ndim != 4 or delta.shape[-1] != 3 or len(delta) < 1:
        raise ValueError("pixel delta decoding requires nonempty uint8 THWC bytes")
    # Unsigned-byte modular accumulation reconstructs the original values exactly.
    return np.cumsum(delta, axis=0, dtype=np.uint8)


def batched_gradient_receipt(model, torch) -> dict:
    """Check every parameter norm with one host transfer, avoiding per-parameter syncs."""
    groups, norms = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"full RGB parameter lacks a gradient: {name}")
        groups.append(
            "vision"
            if name.startswith("vision.")
            else ("language" if name.startswith("language_model.") else "bridge")
        )
        norms.append(parameter.grad.float().norm())
    values = torch.stack(norms).detach().cpu().tolist()
    grouped = {"vision": [], "bridge": [], "language": []}
    for group, value in zip(groups, values, strict=True):
        # A finite L2 norm requires every gradient element to be finite, including NaN/Inf.
        if not math.isfinite(value):
            raise RuntimeError("full RGB parameter gradient norm is nonfinite")
        grouped[group].append(value)
    if any(not numbers or not any(v > 0 for v in numbers) for numbers in grouped.values()):
        raise RuntimeError("full RGB vision, bridge or language gradient signal is absent")
    return {
        group: {
            "parameter_tensors": len(numbers),
            "nonzero_gradient_tensors": sum(v > 0 for v in numbers),
        }
        for group, numbers in grouped.items()
    }
