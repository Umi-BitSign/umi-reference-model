from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fsboard_paper_like_features import (
    MISSING_VALUE,
    paper_like_features,
    selected_point_indices,
)


def test_selected_points_include_paper_iris_only_when_present() -> None:
    compact = selected_point_indices(468)
    full = selected_point_indices(478)
    assert len(compact) == 83
    assert len(full) == 85
    assert compact[:3] == (0, 4, 13)
    assert 468 not in compact[:35]
    assert full[:37][-2:] == (468, 473)
    assert compact[-1] == 542
    assert full[-1] == 552


def test_unit_box_features_preserve_missingness_and_uniform_scale() -> None:
    landmarks = np.full((3, 543, 3), np.nan, dtype=np.float32)
    landmarks[0, 0] = [1.0, 10.0, 3.0]
    landmarks[2, 0] = [3.0, 11.0, 3.0]
    values = paper_like_features(landmarks, frame_stride=2).reshape(2, 83, 3)
    assert values.shape == (2, 83, 3)
    np.testing.assert_array_equal(values[:, 0, 0], [0, 1])
    np.testing.assert_array_equal(values[:, 0, 1], [0, 0.5])
    np.testing.assert_array_equal(values[:, 0, 2], [0, 0])
    assert np.all(values[:, 1:] == MISSING_VALUE)
    with pytest.raises(ValueError, match="frame limit"):
        paper_like_features(landmarks, frame_stride=1, max_frames=1)


def test_full_face_features_have_255_columns() -> None:
    landmarks = np.full((1, 553, 3), np.nan, dtype=np.float32)
    landmarks[0, 468] = [0.2, 0.3, 0.4]
    landmarks[0, 473] = [0.5, 0.6, 0.7]
    values = paper_like_features(landmarks, frame_stride=1)
    assert values.shape == (1, 255)
    assert np.isfinite(values).all()
