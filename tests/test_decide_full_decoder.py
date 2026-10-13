"""Select only checkpoints that pass the two-domain gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_full_decoder import select_checkpoint
from freeze_full_decoder_plan import RECIPE


def _decisions():
    return [
        (
            step,
            {
                "decision": "retain_frozen_baseline",
                "candidate_adapter_sha256": str(step),
                "equal_domain_mean_wer_change": 0.1,
                "fleurs": {},
                "how2sign": {},
                "how2sign_signers_improved": 0,
                "fleurs_candidate_results_sha256": str(step),
                "how2sign_candidate_results_sha256": str(step),
            },
        )
        for step in RECIPE["checkpoint_after_examples"]
    ]


def test_rejects_every_checkpoint_when_none_passes():
    selected = select_checkpoint(_decisions())
    assert selected["decision"] == "retain_frozen_baseline"
    assert selected["selected_example_step"] is None


def test_chooses_best_passing_checkpoint_with_earliest_tie():
    decisions = _decisions()
    for index in (1, 2):
        decisions[index][1]["decision"] = "proceed_to_confirmation"
        decisions[index][1]["equal_domain_mean_wer_change"] = -0.06
    selected = select_checkpoint(decisions)
    assert selected["selected_example_step"] == RECIPE["checkpoint_after_examples"][1]
    assert selected["selected_adapter_sha256"] == str(selected["selected_example_step"])


def test_requires_all_pre_registered_checkpoints():
    with pytest.raises(ValueError, match="scheduled"):
        select_checkpoint(_decisions()[:-1])
