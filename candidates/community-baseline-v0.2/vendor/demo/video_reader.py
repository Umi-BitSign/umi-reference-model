"""Small Decord-compatible reader backed by OpenCV for Apple Silicon.

The compatibility contract is deliberately narrow: zero-based frame indexing,
source ordering, average FPS, frame count, optional fixed-size decode, and uint8
RGB NumPy output. These are the only Decord behaviors used by SHuBERT's
preprocessing and feature-extraction scripts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


class FrameArray:
    def __init__(self, value: np.ndarray):
        self._value = value

    def asnumpy(self) -> np.ndarray:
        return self._value


class VideoReader:
    def __init__(self, filename: str | Path, width: int | None = None, height: int | None = None):
        self.filename = str(filename)
        self.width = width
        self.height = height
        self._capture = cv2.VideoCapture(self.filename)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.filename}")
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self._length <= 0:
            raise RuntimeError(f"video reports no frames: {self.filename}")
        if not np.isfinite(self._fps) or self._fps <= 0:
            raise RuntimeError(f"video reports invalid FPS: {self.filename}")

    def __len__(self) -> int:
        return self._length

    def get_avg_fps(self) -> float:
        return self._fps

    def seek(self, index: int) -> None:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)

    def _read(self, index: int) -> np.ndarray:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"failed to decode frame {index}: {self.filename}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.width is not None or self.height is not None:
            if self.width is None or self.height is None:
                raise ValueError("width and height must be supplied together")
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(frame_rgb, dtype=np.uint8)

    def __getitem__(self, index: int) -> FrameArray:
        return FrameArray(self._read(index))

    def get_batch(self, indices: Iterable[int]) -> FrameArray:
        frames = [self._read(int(index)) for index in indices]
        if not frames:
            shape = (0, self.height or 0, self.width or 0, 3)
            return FrameArray(np.empty(shape, dtype=np.uint8))
        return FrameArray(np.stack(frames, axis=0))

    def __del__(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()

