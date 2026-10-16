from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from terra_semantic_selection import decision


def audit(**changes):
    return {
        "complete": True,
        "calibration_controls_passed_both_passes": True,
        "C5_mean_quality": 4.0,
        "native4096_mean_quality": 4.5,
        "source_group_bootstrap95_interval": [0.2, 0.8],
        "clips": 124,
        **changes,
    }


def test_better_semantics_can_pass_even_when_WER_worsens():
    result = decision(audit(), {"C5_mean_WER": 0.8, "candidate_mean_WER": 0.9})
    assert result["passes_semantic_research_gate"] and not result["WER_veto_used"]
    assert not result["promotion_authorized"]


def test_better_WER_cannot_rescue_semantic_regression():
    result = decision(
        audit(native4096_mean_quality=3.8, source_group_bootstrap95_interval=[-0.4, 0]),
        {"C5_mean_WER": 0.8, "candidate_mean_WER": 0.6},
    )
    assert not result["passes_semantic_research_gate"]


@pytest.mark.parametrize(
    "change",
    [
        {"complete": False},
        {"calibration_controls_passed_both_passes": False},
        {"source_group_bootstrap95_interval": [-0.1, 0.8]},
        {"native4096_mean_quality": 4.1},
    ],
)
def test_incomplete_uncalibrated_or_uncertain_scoring_cannot_win(change):
    assert not decision(audit(**change))["passes_semantic_research_gate"]


@pytest.mark.parametrize(
    "change",
    [
        {"native4096_mean_quality": float("nan")},
        {"native4096_mean_quality": True},
        {"C5_mean_quality": -1},
        {"source_group_bootstrap95_interval": [1, -1]},
    ],
)
def test_invalid_or_nonfinite_scores_are_rejected(change):
    with pytest.raises(ValueError):
        decision(audit(**change))


def test_zero_baseline_uses_predeclared_absolute_floor():
    result = decision(
        audit(
            C5_mean_quality=0,
            native4096_mean_quality=0.5,
            source_group_bootstrap95_interval=[0.2, 0.7],
        )
    )
    assert result["passes_semantic_research_gate"] and result["relative_gain"] is None
