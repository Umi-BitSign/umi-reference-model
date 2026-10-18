from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_features as features
from c5_fingertip_probe_data import tip_continuity


def sequence():
    tips = np.zeros((7, 2, 5, 3), dtype=np.float64)
    tips[..., :2] = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.2]])
    tips[:, 1, :, 0] += 0.4
    tips += np.arange(7)[:, None, None, None] * 0.001
    return SimpleNamespace(
        tips=tips,
        tip_valid=np.isfinite(tips).all(axis=-1),
        continuity=tip_continuity(tips),
        seconds=np.arange(7) / 25,
    )


def test_shape_is_translation_invariant_but_absolute_position_is_retained_separately():
    original = sequence()
    moved = copy.deepcopy(original)
    moved.tips += [0.3, -0.2, 0.01]
    before = features.representations(original)
    after = features.representations(moved)
    for name in ("relative_tips", "pairwise_xy", "cross_hand_xy", "velocity", "acceleration"):
        np.testing.assert_allclose(before[name].values, after[name].values, atol=1e-12)
    assert not np.array_equal(before["centroid"].values, after["centroid"].values)
    recovered = before["relative_tips"].values + before["centroid"].values[:, :, None]
    np.testing.assert_allclose(recovered, original.tips, atol=1e-12)


def test_pairwise_xy_is_orientation_invariant_while_signed_positions_are_not():
    original = sequence()
    rotated = copy.deepcopy(original)
    rotated.tips[..., :2] = original.tips[..., :2] @ np.array([[0.0, -1.0], [1.0, 0.0]])
    before, after = features.representations(original), features.representations(rotated)
    np.testing.assert_allclose(before["pairwise_xy"].values, after["pairwise_xy"].values)
    assert not np.array_equal(before["relative_tips"].values, after["relative_tips"].values)


def test_actual_time_changes_motion_features_without_changing_hand_shape():
    original = sequence()
    slower = copy.deepcopy(original)
    slower.seconds *= 2
    before, after = features.representations(original), features.representations(slower)
    np.testing.assert_array_equal(before["raw_tips"].values, after["raw_tips"].values)
    np.testing.assert_array_equal(before["pairwise_xy"].values, after["pairwise_xy"].values)
    np.testing.assert_allclose(before["velocity"].values, after["velocity"].values * 2)
    np.testing.assert_allclose(before["acceleration"].values, after["acceleration"].values * 4)


def test_missingness_and_track_resets_do_not_create_velocity_or_acceleration_bridges():
    example = sequence()
    example.tips[2, 0] = np.nan
    example.tips[5:, 0] += 0.8
    example.tip_valid = np.isfinite(example.tips).all(axis=-1)
    example.continuity = tip_continuity(example.tips)
    result = features.representations(example)
    assert not result["velocity"].valid[[0, 2, 3, 5], 0].any()
    assert result["velocity"].valid[[1, 4, 6], 0].all()
    assert not result["acceleration"].valid[:, 0].any()
    assert result["velocity"].valid[1:, 1].all()
    for feature in result.values():
        assert np.isfinite(feature.values).all()


def test_native_depth_never_enters_cross_hand_or_pairwise_xy_features():
    original = sequence()
    changed = copy.deepcopy(original)
    changed.tips[:, 1, :, 2] += 1000
    before, after = features.representations(original), features.representations(changed)
    for name in ("pairwise_xy", "cross_hand_xy", "speed_xy"):
        np.testing.assert_array_equal(before[name].values, after[name].values)
    changed.tips[:, 1, :, 2] = np.nan
    changed.tip_valid = np.isfinite(changed.tips).all(axis=-1)
    absent = features.representations(changed)
    for name in ("pairwise_xy", "cross_hand_xy", "speed_xy"):
        np.testing.assert_array_equal(before[name].values, absent[name].values)
        np.testing.assert_array_equal(before[name].valid, absent[name].valid)


def test_no_target_or_body_fields_are_required_or_read():
    original = sequence()
    before = features.representations(original)
    original.targets = np.full((7, 2, 5, 3), 1e9)
    original.pose = np.full((7, 33, 3), np.nan)
    original.face = "forbidden extra inputs"
    after = features.representations(original)
    for name in before:
        np.testing.assert_array_equal(before[name].values, after[name].values)
        np.testing.assert_array_equal(before[name].valid, after[name].valid)


@pytest.mark.parametrize("seconds", [[0] * 7, [0, 1, 2, 3, 2, 4, 5]])
def test_nonincreasing_timestamps_are_rejected(seconds):
    example = sequence()
    example.seconds = np.array(seconds)
    with pytest.raises(ValueError, match="strictly increasing"):
        features.representations(example)
