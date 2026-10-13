from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fit_c5_crop_calibrator import capped_arrays, fit_affine, metrics


def test_fit_affine_recovers_shift_and_scale() -> None:
    generator = np.random.default_rng(3)
    source = generator.normal(size=(500, 384)).astype(np.float32)
    target = source * 1.25 + 0.3
    weight, bias = fit_affine([source], [target], 1e-6)
    before = metrics([source], [target])["mse"]
    after = metrics([source], [target], weight, bias)["mse"]
    assert after < before * 1e-6


def test_capped_arrays_samples_endpoints() -> None:
    values = np.arange(10).reshape(10, 1)
    capped = capped_arrays([values], 4)[0]
    assert capped[:, 0].tolist() == [0, 3, 6, 9]
