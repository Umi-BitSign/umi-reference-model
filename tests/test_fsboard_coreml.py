"""Small deterministic checks for the offline fingerspelling feature path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_fsboard_coreml import comparable_units, ctc_decode, model_input
from extract_fsboard_landmarks import assemble


def _part(count: int, x: float):
    return [SimpleNamespace(x=x, y=0.25, z=-0.1) for _ in range(count)]


def test_assemble_uses_canonical_blocks_and_missing_nan():
    result = SimpleNamespace(
        face_landmarks=_part(478, 0.1),
        left_hand_landmarks=[],
        pose_landmarks=_part(33, 0.3),
        right_hand_landmarks=_part(21, 0.4),
    )
    points = assemble(result)
    assert points.shape == (543, 3)
    assert np.all(points[:468, 0] == 0.1)
    assert np.isnan(points[468:489]).all()
    assert np.all(points[489:522, 0] == 0.3)
    assert np.all(points[522:, 0] == 0.4)


def test_assemble_can_retain_478_face_points_for_paper_features():
    result = SimpleNamespace(
        face_landmarks=_part(478, 0.1),
        left_hand_landmarks=_part(21, 0.2),
        pose_landmarks=_part(33, 0.3),
        right_hand_landmarks=_part(21, 0.4),
    )
    points = assemble(result, face_points=478)
    assert points.shape == (553, 3)
    assert np.all(points[468:478, 0] == 0.1)
    assert np.all(points[478:499, 0] == 0.2)
    assert np.all(points[499:532, 0] == 0.3)
    assert np.all(points[532:, 0] == 0.4)
    result.face_landmarks = _part(468, 0.1)
    with pytest.raises(ValueError, match="fewer than 478"):
        assemble(result, face_points=478)


def test_model_input_discards_empty_frames_and_pads():
    frames = np.full((3, 543, 3), np.nan, dtype=np.float32)
    frames[1, 0] = (0.25, 0.5, 1.0)
    frames[2, 0] = (0.75, 0.5, 3.0)
    features, mask, count = model_input(frames)
    assert count == 2
    assert features.shape == (1, 768, 1629)
    assert mask.shape == (1, 768)
    np.testing.assert_array_equal(mask[0, :3], [1, 1, 0])
    np.testing.assert_allclose(features[0, :2, 0], [-1, 1])
    np.testing.assert_allclose(features[0, :2, 2], [-1, 1])
    assert not features[0, 2:].any()
    with pytest.raises(ValueError, match="no usable"):
        model_input(frames[:1])


def test_ctc_collapses_repeats_and_uses_blank():
    logits = np.zeros((1, 384, 62), dtype=np.float32)
    for index, label in enumerate([1, 1, 0, 1, 2]):
        logits[0, index, label] = 10
    vocab = ["<blank>", "a", "b"] + [str(index) for index in range(57)]
    assert ctc_decode(logits, 10, vocab) == "aab"


def test_cer_units_match_umi_normalization_for_ascii():
    assert comparable_units(" A-B  C! ") == ("a", "b", "c")
    assert comparable_units("DON'T") == tuple("don't")
