from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_sets as sets


def fixture():
    return {
        "hand_landmarks": np.arange(4 * 6 * 21 * 3, dtype=np.float32).reshape(4, 6, 21, 3) / 1000,
        "hand_counts": np.array([0, 2, 4, 6], dtype=np.uint8),
        "frame_valid": np.ones(4, dtype=bool),
    }


def test_all_detected_groups_are_retained_including_three_to_six_hand_frames():
    arrays = fixture()
    before = copy.deepcopy(arrays)
    actual = sets.hand_sets(arrays)
    view = sets.input_view(actual)
    np.testing.assert_array_equal(view["valid"].sum(axis=(1, 2)), [0, 10, 20, 30])
    np.testing.assert_array_equal(
        actual["fingertips"][2, :4], np.take(arrays["hand_landmarks"][2, :4], sets.TIPS, axis=1)
    )
    assert np.isnan(actual["fingertips"][2, 4:]).all()
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])


def test_non_tip_joint_values_and_missing_targets_never_change_tip_inputs():
    arrays = fixture()
    before = sets.input_view(sets.hand_sets(arrays))
    non_tips = [joint for joint in range(21) if joint not in sets.TIPS]
    arrays["hand_landmarks"][:, :, non_tips] = np.nan
    actual = sets.hand_sets(arrays)
    after = sets.input_view(actual)
    np.testing.assert_array_equal(before["values"], after["values"])
    np.testing.assert_array_equal(before["valid"], after["valid"])
    assert not actual["wrist_mcp_targets_valid_xy"].any()


def test_detector_hand_permutations_permute_groups_without_inventing_temporal_identity():
    arrays = fixture()
    expected = sets.hand_sets(arrays)
    permutation = [3, 1, 0, 2]
    arrays["hand_landmarks"][2, :4] = arrays["hand_landmarks"][2, permutation]
    actual = sets.hand_sets(arrays)
    np.testing.assert_array_equal(
        actual["fingertips"][2, :4], expected["fingertips"][2, permutation]
    )
    assert "hand_continuity" not in actual and "speaker_identity" not in actual


def test_invalid_frame_and_missing_points_stay_unknown_without_interpolation():
    arrays = fixture()
    arrays["frame_valid"][1] = False
    arrays["hand_landmarks"][2, 0, 4] = np.nan
    actual = sets.hand_sets(arrays)
    view = sets.input_view(actual)
    assert not view["valid"][1].any()
    assert not view["valid"][2, 0, 0]
    assert view["valid"][2, 0, 1:].all()
    assert np.isfinite(view["values"]).all()
    assert actual["source_hand_counts"][1] == 2


def test_xy_preserves_original_outside_image_values_and_ignores_missing_native_depth():
    arrays = fixture()
    before = sets.input_view(sets.hand_sets(arrays))
    assert before["values"].max() > 1
    arrays["hand_landmarks"][..., 2] = np.nan
    actual = sets.hand_sets(arrays)
    np.testing.assert_array_equal(sets.input_view(actual)["values"], before["values"])
    assert not sets.input_view(actual, "native_xyz")["valid"].any()


@pytest.mark.parametrize("counts", [[0, 1, 2, 7], [0, -1, 2, 3], [0.0, 1.0, 2.0, 3.0]])
def test_invalid_counts_are_rejected_instead_of_selecting_a_target_person(counts):
    arrays = fixture()
    arrays["hand_counts"] = np.array(counts)
    with pytest.raises(ValueError, match="hand count"):
        sets.hand_sets(arrays)
