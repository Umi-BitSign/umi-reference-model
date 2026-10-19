"""Packed extraction against the independent existing dictionary reader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_packed_timing_visibility import from_packed
from c5_timing_visibility import visibility


@pytest.mark.parametrize("frames", [1, 2, 3, 17, 198])
def test_matches_original_detector_policy_at_every_frame_and_hidden_interval(frames):
    rng = np.random.default_rng(184 + frames)
    hands = rng.normal(size=(frames, 6, 21, 3))
    counts = rng.integers(0, 7, size=frames, dtype=np.uint8)
    valid = rng.uniform(size=frames) > 0.1
    hands[rng.uniform(size=hands.shape) < 0.02] = np.nan
    landmarks = {
        i: {"hand_landmarks": hands[i, : int(counts[i])].tolist()} if valid[i] else None
        for i in range(frames)
    }
    expected_observed, expected_intervals = visibility(
        landmarks, frames, np.arange(0, frames, 2, dtype=np.int64)
    )
    actual = from_packed(hands, counts, valid)
    assert np.array_equal(actual["source_hand_tip_visible"], expected_observed)
    assert np.array_equal(actual["valid_intervals"], expected_intervals)
    assert actual["visibility_component"].shape == actual["frame_indices"].shape
    assert np.array_equal(np.diff(actual["visibility_component"]), ~expected_intervals)


def test_unobserved_frame_between_two_observed_samples_remains_a_cut():
    hands = np.ones((5, 6, 21, 3), dtype=np.float64)
    hands[..., 2] = np.nan  # Detector-relative z has never defined XY visibility.
    counts = np.array([1, 0, 1, 1, 2], dtype=np.uint8)
    result = from_packed(hands, counts, np.ones(5, dtype=np.bool_))
    assert result["source_hand_tip_visible"].tolist() == [True, False, True, True, True]
    assert result["valid_intervals"].tolist() == [False, True]
    assert result["frame_indices"].tolist() == [0, 2, 4]
    assert result["visibility_component"].tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    "change", ["float32", "count_dtype", "extra_hands", "valid_dtype", "shape"]
)
def test_rejects_changed_topology_dtype_or_mask(change):
    hands = np.ones((5, 6, 21, 3), dtype=np.float64)
    counts, valid = np.ones(5, dtype=np.uint8), np.ones(5, dtype=np.bool_)
    if change == "float32":
        hands = hands.astype(np.float32)
    elif change == "count_dtype":
        counts = counts.astype(np.int64)
    elif change == "extra_hands":
        counts[0] = 7
    elif change == "valid_dtype":
        valid = valid.astype(np.uint8)
    else:
        hands = hands[:, :, :20]
    with pytest.raises(ValueError):
        from_packed(hands, counts, valid)
