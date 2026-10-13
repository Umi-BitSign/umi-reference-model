"""The C5-to-STEM interpolation screen enforces aggregate and signer gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_c5_stem_interpolation_screen import select_candidate


def _summary(weight: float, h_delta: float, f_delta: float, improved: int) -> dict:
    signer_means = {}
    for index in range(5):
        signer_means[str(index)] = {
            "baseline_mean_wer": 1.0,
            "candidate_mean_wer": 0.9 if index < improved else 1.0,
        }
    return {
        "second_weight": weight,
        "how2sign": {
            "candidate_minus_baseline_mean_wer": h_delta,
            "signer_means": signer_means,
        },
        "fleurs": {"candidate_minus_baseline_mean_wer": f_delta},
    }


SCREEN = {
    "how2sign_max_mean_wer_change": 0.01,
    "fleurs_max_mean_wer_change": 0.01,
    "equal_domain_max_mean_wer_change": -0.005,
    "minimum_improved_how2sign_signers": 3,
}


def test_selects_best_passing_interpolation():
    summaries = {
        "light": _summary(0.25, -0.02, 0.0, 3),
        "heavy": _summary(0.75, -0.03, 0.0, 4),
    }
    assert select_candidate(summaries, SCREEN) == "heavy"
    assert summaries["light"]["passes_screen"] is True


def test_rejects_aggregate_gain_concentrated_in_two_signers():
    summaries = {"candidate": _summary(0.5, -0.04, -0.02, 2)}
    assert select_candidate(summaries, SCREEN) is None
    assert summaries["candidate"]["passes_screen"] is False
