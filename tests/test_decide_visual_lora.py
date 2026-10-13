"""Visual checkpoints are selected only after the frozen two-domain gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_visual_lora import select_checkpoint
from freeze_visual_lora_plan import RECIPE


def _decision(passed: bool, change: float, name: str) -> dict:
    return {
        "decision": "proceed_to_confirmation" if passed else "retain_frozen_baseline",
        "equal_domain_mean_wer_change": change,
        "candidate_adapter_sha256": name,
        "fleurs": {},
        "how2sign": {},
        "how2sign_signers_improved": 3,
        "fleurs_candidate_results_sha256": f"f-{name}",
        "how2sign_candidate_results_sha256": f"h-{name}",
    }


def test_selects_best_passing_visual_checkpoint_and_keeps_confirmation_sealed():
    steps = RECIPE["checkpoint_after_examples"]
    decisions = [
        (steps[0], _decision(False, -0.1, "first")),
        (steps[1], _decision(True, -0.04, "second")),
        (steps[2], _decision(True, -0.05, "third")),
        (steps[3], _decision(True, -0.05, "fourth")),
    ]
    selected = select_checkpoint(decisions)
    assert selected["selected_example_step"] == steps[2]
    assert selected["selected_adapter_sha256"] == "third"
    assert selected["decision"] == "proceed_to_confirmation"
    assert selected["confirmation_evaluated"] is False
    rejected = select_checkpoint(
        [(step, _decision(False, 0.1, str(step))) for step in steps]
    )
    assert rejected["decision"] == "retain_frozen_baseline"
    assert rejected["selected_adapter_sha256"] is None
    with pytest.raises(ValueError, match="scheduled checkpoint"):
        select_checkpoint(decisions[:-1])
