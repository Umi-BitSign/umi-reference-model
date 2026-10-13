"""A full-decoder follow-up must inherit the rejected run's exact data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_full_decoder_plan import RECIPE, full_decoder_plan


def _inputs():
    previous = {
        "schema": "umi-studio-full-cross-attention-plan/1",
        "training_examples": {"fleurs": 313, "how2sign": 1024, "flores2m": 0},
        "seed": 17,
        "fleurs_repeat": 1,
        "how2sign_repeat": 1,
        "fleurs_training_index_sha256": "fleurs",
        "how2sign_training_index_sha256": "how2sign",
    }
    decision = {
        "schema": "umi-studio-full-cross-attention-tuning-decision/1",
        "decision": "retain_frozen_baseline",
        "plan_sha256": "previous",
        "confirmation_evaluated": False,
        "live_miner_changed": False,
    }
    return previous, decision


def test_full_decoder_plan_pins_both_prior_artifacts():
    previous, decision = _inputs()
    plan = full_decoder_plan(previous, decision, "previous", "decision")
    assert plan["schema"] == "umi-studio-full-decoder-plan/1"
    assert plan["recipes"] == [RECIPE]
    assert plan["previous_decision_sha256"] == "decision"
    assert plan["how2sign_training_index_sha256"] == "how2sign"


def test_full_decoder_plan_rejects_unscored_or_deployed_prior_run():
    previous, decision = _inputs()
    with pytest.raises(ValueError, match="rejected"):
        full_decoder_plan(previous, {**decision, "decision": "promote"}, "previous", "decision")
    with pytest.raises(ValueError, match="rejected"):
        full_decoder_plan(
            previous, {**decision, "confirmation_evaluated": True}, "previous", "decision"
        )
