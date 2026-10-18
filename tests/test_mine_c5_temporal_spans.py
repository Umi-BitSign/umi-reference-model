from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mine_c5_temporal_spans as pilot


def skeleton(frames=8):
    hands = np.full((frames, 6, 21, 3), np.nan)
    for t in range(frames):
        for side in range(2):
            hands[t, side] = np.arange(63).reshape(21, 3) / 1000 + side * 0.5 + t * 0.001
    return {
        "frame_valid": np.ones(frames, dtype=bool),
        "hand_counts": np.full(frames, 2, dtype=np.uint8),
        "hand_landmarks": hands,
        "pose_present": np.ones(frames, dtype=bool),
        "pose_landmarks": np.zeros((frames, 33, 3)),
        "face_counts": np.zeros(frames, dtype=np.uint8),
        "face_landmarks": np.full((frames, 6, 478, 3), np.nan),
    }


def test_detector_order_changes_do_not_create_fingertip_motion():
    arrays = skeleton()
    baseline = pilot.feature_branches(arrays)
    arrays["hand_landmarks"][1::2, :2] = arrays["hand_landmarks"][1::2, :2][:, ::-1]
    actual = pilot.feature_branches(arrays)
    np.testing.assert_array_equal(actual["fingertips"], baseline["fingertips"])
    np.testing.assert_array_equal(actual["hand_continuity"], baseline["hand_continuity"])


def test_strict_fingertips_do_not_use_wrist_palm_body_or_face_for_tracking_or_normalization():
    arrays = skeleton()
    expected = pilot.feature_branches(arrays)
    non_tips = [index for index in range(21) if index not in pilot.TIPS]
    arrays["hand_landmarks"][:, :2, non_tips] += 100
    arrays["pose_landmarks"][:] = 900
    arrays["face_landmarks"][:] = -700
    arrays["face_counts"][:] = 1
    actual = pilot.feature_branches(arrays)
    np.testing.assert_array_equal(actual["fingertips"], expected["fingertips"])
    np.testing.assert_array_equal(actual["hand_continuity"], expected["hand_continuity"])
    assert not np.array_equal(actual["full_hands"], expected["full_hands"])
    assert not np.array_equal(actual["pose"], expected["pose"])


def test_visibility_gaps_preserve_timing_and_never_create_a_motion_bridge():
    arrays = skeleton()
    arrays["hand_counts"][3] = 0
    actual = pilot.feature_branches(arrays)
    assert len(actual["fingertips"]) == 8
    assert np.isnan(actual["fingertips"][3]).all()
    assert not actual["hand_continuity"][4].any()
    continuity = np.repeat(actual["hand_continuity"], 5, axis=1)
    curve = pilot.motion(actual["fingertips"], 25, 640, 480, continuity)
    assert np.isnan(curve[0]) and np.isnan(curve[3]) and np.isnan(curve[4])
    assert np.isfinite(curve[5])


def test_more_than_two_hands_is_retained_as_unknown_instead_of_selecting_with_pose():
    arrays = skeleton()
    arrays["hand_counts"][2] = 3
    arrays["hand_landmarks"][2, 2] = 0.5
    actual = pilot.feature_branches(arrays)
    assert np.isnan(actual["fingertips"][2]).all()
    assert not actual["hand_continuity"][3].any()
    assert np.isfinite(actual["pose"][2]).all()


def test_original_tip_z_is_preserved_but_not_treated_as_a_common_3d_motion_axis():
    arrays = skeleton()
    before = pilot.feature_branches(arrays)
    arrays["hand_landmarks"][:, :2, :, 2] += np.arange(8)[:, None, None] * 10
    after = pilot.feature_branches(arrays)
    assert not np.array_equal(before["fingertips"][..., 2], after["fingertips"][..., 2])
    np.testing.assert_array_equal(
        pilot.motion(before["fingertips"], 25, 640, 480),
        pilot.motion(after["fingertips"], 25, 640, 480),
    )


def test_stationary_or_unobserved_motion_is_not_labelled_as_an_exact_sign_interval():
    for curve in (np.zeros(8), np.full(8, np.nan)):
        spans, _ = pilot.proposals(curve, 25)
        assert spans == []
    spans, threshold = pilot.proposals(np.array([np.nan, 0, 0, 0.1, 0.2, 0.1, 0, 0]), 25)
    assert threshold > 0 and len(spans) == 1
    span = spans[0]
    assert span["label"] is None and span["verified_sign_boundary"] is False
    assert span["start_seconds"] == span["start_frame"] / 25
    assert span["end_seconds"] == span["end_frame_exclusive"] / 25
    assert span["unknown_motion_samples"] == 1


@pytest.mark.parametrize("forbidden", ["development", "P19", "P10"])
def test_training_role_and_signer_checks_happen_without_opening_landmarks(forbidden):
    row = {
        "role": "training",
        "training_allowed": True,
        "participant": "P1",
        "example_id": "fixture",
        "source_group": "fixture",
        "reference": "Word",
    }
    if forbidden == "development":
        row.update(role="development", training_allowed=False)
    else:
        row["participant"] = forbidden
    case = {
        "kind": "word",
        "word": {
            "record": {
                "ROI_lineage_training_allowed": True,
                "original": {"row": row},
                "landmarks": {"path": "does-not-exist"},
                "clip": {"sha256": "fixture"},
                "normalized_probe": {"decoded_frames": 8},
            }
        },
    }
    with pytest.raises(ValueError, match="reject"):
        pilot.identity(copy.deepcopy(case))
