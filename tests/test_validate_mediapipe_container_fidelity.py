from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from validate_mediapipe_container_fidelity import array_metrics


def test_array_metrics_reports_exact_and_shifted_values() -> None:
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    exact = array_metrics(reference.copy(), reference)
    assert exact["mean_absolute_error"] == 0
    assert exact["cosine_similarity"] == pytest.approx(1)
    assert exact["exact_fraction"] == 1
    shifted = array_metrics(reference + 1, reference)
    assert shifted["mean_absolute_error"] == 1
    assert shifted["root_mean_square_error"] == 1
    assert shifted["max_absolute_error"] == 1
    assert shifted["exact_fraction"] == 0


def test_array_metrics_rejects_mismatched_dtype() -> None:
    with pytest.raises(ValueError, match="topology"):
        array_metrics(np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float64))
