from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_sampling as sampling


def trajectory(xs):
    tips = np.zeros((len(xs), 2, 5, 2))
    tips[..., 0] = np.asarray(xs)[:, None, None]
    return tips, np.ones((len(xs), 2), bool), np.arange(len(xs)) / 25


@pytest.mark.parametrize("method", sampling.METHODS)
def test_same_exact_budget_endpoints_and_full_timeline(method):
    tips, continuity, seconds = trajectory(np.sin(np.arange(20)))
    result = sampling.compress(tips, continuity, seconds, fraction=0.25, method=method)
    assert result["sampled"].sum() == 10
    assert result["sampled"][[0, -1]].all()
    np.testing.assert_array_equal(
        result["reconstructed"][result["sampled"]], tips[result["sampled"]]
    )
    assert result["metrics"]["original_last_minus_first_seconds"] == seconds[-1]


def test_time_error_keeps_short_excursion_missed_by_uniform():
    tips, continuity, seconds = trajectory([0, 0, 0, 1, 0, 0, 0, 0, 0])
    baseline = sampling.compress(tips, continuity, seconds, fraction=1 / 3, method="uniform")
    adaptive = sampling.compress(tips, continuity, seconds, fraction=1 / 3, method="time_error")
    assert baseline["reconstructed"][3, 0, 0, 0] == 0
    assert adaptive["sampled"][3].all()
    assert (
        adaptive["metrics"]["max_tip_euclidean_error"]
        < baseline["metrics"]["max_tip_euclidean_error"]
    )


@pytest.mark.parametrize("method", sampling.METHODS)
def test_gaps_resets_and_singletons_are_never_bridged(method):
    tips, continuity, seconds = trajectory(np.arange(9))
    tips[2, 0] = np.nan
    continuity[5, 0] = False
    result = sampling.compress(tips, continuity, seconds, fraction=0.25, method=method)
    assert np.isnan(result["reconstructed"][2, 0]).all()
    assert [(s["start"], s["end"]) for s in result["segments"] if s["hand"] == 0] == [
        (0, 2),
        (3, 5),
        (5, 9),
    ]
    tips[1:, 0] = np.nan
    result = sampling.compress(tips, continuity, seconds, fraction=0.25, method=method)
    assert result["segments"][0] == {"hand": 0, "start": 0, "end": 1, "knots": [0]}


@pytest.mark.parametrize("method", sampling.METHODS)
def test_true_elapsed_time_controls_interpolation_and_velocity(method):
    seconds = np.array([0.0, 0.1, 0.7, 1.0])
    tips, continuity, _ = trajectory(seconds)
    result = sampling.compress(tips, continuity, seconds, fraction=0.25, method=method)
    np.testing.assert_allclose(result["reconstructed"], tips, atol=1e-15)
    assert result["metrics"]["xy_velocity_mae_per_second"] < 1e-15


def test_stationary_arc_length_has_deterministic_uniform_fallback():
    values = np.ones((19, 5, 2))
    seconds = np.arange(19) / 25
    np.testing.assert_array_equal(
        sampling.select_knots(values, seconds, 5, "uniform"),
        sampling.select_knots(values, seconds, 5, "arc_length"),
    )


def test_partial_and_absent_hands_keep_explicit_coverage():
    tips, continuity, seconds = trajectory([0, 1, 0])
    tips[:] = np.nan
    tips[0, 0, 0] = 0
    result = sampling.compress(tips, continuity, seconds, fraction=0.5, method="time_error")
    assert result["metrics"]["observed_hand_frames"] == 0
    assert result["metrics"]["partial_hand_frames_excluded"] == 1
    assert result["metrics"]["xy_coordinate_mae"] is None
    assert np.isnan(result["reconstructed"]).all()


def test_short_segment_endpoint_overhead_is_reported():
    result = sampling.compress(*trajectory([0, 1, 0]), fraction=0.25, method="uniform")
    assert result["metrics"]["retained_fraction"] == 2 / 3


@pytest.mark.parametrize("method", sampling.METHODS)
def test_full_budget_is_lossless_and_does_not_clip_out_of_image_coordinates(method):
    tips, continuity, seconds = trajectory([-2, 3, -1, 2])
    result = sampling.compress(tips, continuity, seconds, fraction=1, method=method)
    np.testing.assert_array_equal(result["reconstructed"], tips)
    assert result["metrics"]["xy_coordinate_mae"] == 0


def test_invalid_clock_and_native_depth_are_rejected():
    tips, continuity, seconds = trajectory([0, 1, 0])
    with pytest.raises(ValueError, match="timestamps"):
        sampling.compress(tips, continuity, seconds * 0, fraction=0.5, method="uniform")
    with pytest.raises(ValueError, match="five-tip/XY"):
        sampling.compress(
            np.zeros((3, 2, 5, 3)), continuity, seconds, fraction=0.5, method="uniform"
        )
