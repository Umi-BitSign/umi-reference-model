from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_component_sampling as components
import c5_fingertip_sampling as original


def example():
    rng = np.random.default_rng(17)
    tips = rng.normal(size=(31, 2, 5, 2)) * 0.05
    tips[12, 0] = np.nan
    continuity = np.ones((31, 2), bool)
    continuity[[0, 23]] = False
    return tips, continuity, np.arange(31) / 25


def test_raw_arm_exactly_matches_frozen_timestamp_error_baseline():
    tips, continuity, seconds = example()
    expected = original.compress(tips, continuity, seconds, fraction=0.25, method="time_error")
    result = components.compress(tips, continuity, seconds, fraction=0.25, method="raw")
    assert result["segments"] == expected["segments"]
    np.testing.assert_array_equal(result["reconstructed"], expected["reconstructed"])
    assert result["metrics"]["xy_coordinate_mae"] == expected["metrics"]["xy_coordinate_mae"]


@pytest.mark.parametrize("method", components.METHODS)
def test_identical_support_budgets_and_no_gap_or_reset_bridge(method):
    tips, continuity, seconds = example()
    result = components.compress(tips, continuity, seconds, fraction=0.25, method=method)
    baseline = components.compress(tips, continuity, seconds, fraction=0.25, method="raw")
    np.testing.assert_array_equal(np.isfinite(result["reconstructed"]), np.isfinite(tips))
    assert result["sampled"].sum() == baseline["sampled"].sum()
    assert [(s["hand"], s["start"], s["end"]) for s in result["segments"]] == [
        (s["hand"], s["start"], s["end"]) for s in baseline["segments"]
    ]


def test_centered_signal_is_invariant_to_time_varying_tip_centroid_motion():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(20, 5, 2))
    shifted = values + rng.normal(size=(20, 1, 2)) * 20
    np.testing.assert_allclose(
        components.selection_view(values, "centered_shape"),
        components.selection_view(shifted, "centered_shape"),
        atol=1e-13,
    )


def test_centroid_signal_ignores_zero_mean_finger_changes():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(20, 5, 2))
    shape = rng.normal(size=values.shape)
    shape -= shape.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        components.selection_view(values, "centroid"),
        components.selection_view(values + shape, "centroid"),
        atol=1e-15,
    )


def test_equal_motion_balances_rms_in_orthogonal_centroid_and_shape_subspaces():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(20, 5, 2)) * 0.001 + rng.normal(size=(20, 1, 2))
    normalized = components.selection_view(values, "equal_motion")
    center = normalized.mean(axis=1, keepdims=True)
    centered = normalized - center
    assert np.isclose(np.mean(np.diff(center, axis=0) ** 2), 1)
    assert np.isclose(np.mean(np.diff(centered, axis=0) ** 2), 1)


@pytest.mark.parametrize("method", components.METHODS)
def test_full_budget_restores_raw_coordinates_for_every_selection_view(method):
    tips, continuity, seconds = example()
    result = components.compress(tips, continuity, seconds, fraction=1, method=method)
    np.testing.assert_array_equal(result["reconstructed"], tips)
    assert result["metrics"]["xy_coordinate_mae"] == 0


def test_stationary_singleton_and_empty_observation_are_defined():
    values = np.ones((20, 5, 2))
    assert np.isfinite(components.selection_view(values, "equal_motion")).all()
    np.testing.assert_array_equal(components.selection_view(values[:1], "equal_motion"), values[:1])
    tips, continuity, seconds = example()
    tips[:] = np.nan
    result = components.compress(tips, continuity, seconds, fraction=0.25, method="equal_motion")
    assert result["metrics"]["xy_coordinate_mae"] is None and not result["sampled"].any()
