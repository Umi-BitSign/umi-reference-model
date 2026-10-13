"""Stream original clips through CPU resizing before the scratch RGB network sees pixels."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def decode_rgb_video(video: str, short_side: int, target_fps: float = 12.5, max_frames: int = 512):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    import cv2

    path = (ROOT / video).resolve()
    if not path.is_relative_to(ROOT / "data") or short_side < 32 or target_fps <= 0:
        raise ValueError("scratch RGB decoding requires a private data/ clip and valid scale")
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps < target_fps or count < 1:
            raise ValueError("scratch RGB source cannot support the declared temporal policy")
        indices = np.unique(np.floor(np.arange(0, count, fps / target_fps)).astype(np.int64))
        if len(indices) > max_frames:
            raise ValueError("scratch RGB clip exceeds its frame cap; do not truncate signing")
        pending = set(indices.tolist())
        frames = []
        source_shape = None
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
                    raise ValueError("scratch RGB source dimensions change within a clip")
                scale = min(1.0, short_side / min(height, width))
                size = max(1, round(width * scale)), max(1, round(height * scale))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
                pending.remove(index)
            index += 1
        if pending:
            raise ValueError("scratch RGB decoder ended before all declared frames")
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


def pixel_tensor(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            "scratch network accepts downscaled uint8 RGB pixels, not learned embeddings"
        )
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
