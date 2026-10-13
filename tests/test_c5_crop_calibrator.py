from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from c5_crop_calibrator import CalibratedEmbedder, LowRankCalibratedEmbedder


class Embedder:
    marker = "kept"

    def extract_embeddings_from_frames(self, frames):
        return np.asarray(frames, dtype=np.float32)


def test_calibrated_embedder_maps_and_delegates() -> None:
    base = Embedder()
    wrapped = CalibratedEmbedder(base, np.eye(384, dtype=np.float32) * 2, np.ones(384))
    source = np.zeros((2, 384), dtype=np.float32)
    assert np.all(wrapped.extract_embeddings_from_frames(source) == 1)
    assert wrapped.marker == "kept"


def test_low_rank_calibrator_applies_residual() -> None:
    base = Embedder()
    left = np.ones((384, 2), dtype=np.float32)
    right = np.ones((2, 384), dtype=np.float32)
    wrapped = LowRankCalibratedEmbedder(base, left, right, np.ones(384))
    source = np.ones((1, 384), dtype=np.float32)
    assert np.all(wrapped.extract_embeddings_from_frames(source) == 770)
