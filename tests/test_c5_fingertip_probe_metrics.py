from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_probe_metrics as metrics


def fixture(error=1.0):
    target = np.zeros((2, 1, 5, 2))
    mask = np.ones((2, 1, 5), dtype=bool)
    return np.full_like(target, error), target, mask, mask.copy()


def row(clip, group, error, family="stem"):
    return {"clip_id": clip, "group": group, "family": family, **metrics.clip_xy(*fixture(error))}


def test_primary_metric_is_xy_coordinate_mae_and_keeps_per_joint_counts():
    prediction, target, known, inputs = fixture()
    prediction[..., 1] = 2
    result = metrics.clip_xy(prediction, target, known, inputs)
    assert result["score"] == 1.5 and result["scored_hands"] == 2
    assert result["scored_target_points"] == 10
    assert all(joint == {"count": 2, "mae": 1.5} for joint in result["per_joint"].values())


def test_nonfinite_prediction_cannot_hide_a_difficult_known_target():
    args = fixture()
    args[0][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite prediction"):
        metrics.clip_xy(*args)


def test_a_known_target_cannot_be_silently_removed_by_the_mask():
    prediction, target, known, inputs = fixture()
    known[0, 0, 0] = False
    with pytest.raises(ValueError, match="every known target"):
        metrics.clip_xy(prediction, target, known, inputs)


def test_no_observable_input_produces_no_score_and_preserves_coverage_counts():
    prediction, target, known, inputs = fixture()
    inputs[:] = False
    prediction[:] = np.nan
    result = metrics.clip_xy(prediction, target, known, inputs)
    assert result["score"] is None and result["scored_hands"] == 0
    assert result["known_target_hands"] == 2 and result["input_complete_hands"] == 0


def test_depth_is_excluded_from_the_primary_metric():
    with pytest.raises(ValueError, match="XY arrays"):
        metrics.clip_xy(
            np.zeros((2, 1, 5, 3)),
            np.zeros((2, 1, 5, 3)),
            np.ones((2, 1, 5), bool),
            np.ones((2, 1, 5), bool),
        )


def test_clips_do_not_outvote_groups_and_groups_do_not_outvote_families():
    baseline = [row(str(index), "many-clips", 1) for index in range(10)]
    baseline += [row("single", "one-clip", 0), row("other", "only-group", 1, "other-family")]
    candidate = [{**item, "score": 0} for item in baseline]
    result = metrics.paired_group_comparison(baseline, candidate, bootstraps=128)
    assert result["mean_error_reduction"] == 0.75
    assert result["family_group_counts"] == {"stem": 2, "other-family": 1}
    assert result["families_with_only_one_scored_group"] == ["other-family"]
    assert result["not_a_semantic_quality_gate"] is True


def test_changed_support_or_missing_clips_are_rejected():
    baseline = [row("one", "group", 1)]
    candidate = copy.deepcopy(baseline)
    candidate[0]["support_sha256"] = "different"
    with pytest.raises(ValueError, match="support differs"):
        metrics.paired_group_comparison(baseline, candidate)
    with pytest.raises(ValueError, match="clip inventory differs"):
        metrics.paired_group_comparison(baseline, [row("two", "group", 1)])


def test_bootstrap_is_permutation_stable_and_constant_gains_stay_constant():
    baseline = [row(str(index), f"group{index}", 1) for index in range(4)]
    candidate = [{**item, "score": 0.5} for item in baseline]
    result = metrics.paired_group_comparison(baseline, candidate, bootstraps=128)
    reversed_result = metrics.paired_group_comparison(
        baseline[::-1], candidate[::-1], bootstraps=128
    )
    assert result == reversed_result
    assert result["paired_group_bootstrap95"] == [0.5, 0.5]


def test_one_group_cannot_be_resampled_independently_in_two_families():
    baseline = [row("one", "same-cluster", 1, "first"), row("two", "same-cluster", 1, "second")]
    with pytest.raises(ValueError, match="cross-family group"):
        metrics.paired_group_comparison(baseline, baseline, bootstraps=128)


def test_an_entire_unscorable_family_cannot_disappear_from_the_summary():
    prediction, target, known, inputs = fixture()
    inputs[:] = False
    empty = {
        "clip_id": "empty",
        "family": "unknown",
        "group": "none",
        **metrics.clip_xy(prediction, target, known, inputs),
    }
    baseline = [row("one", "group", 1), empty]
    with pytest.raises(ValueError, match="every family"):
        metrics.paired_group_comparison(baseline, baseline, bootstraps=128)
