"""Projection-frozen tuning applies both aggregate and signer gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_c5_openasl_pilot_plan import RECIPE as OPENASL_RECIPE
from freeze_c5_how2sign_expansion_plan import RECIPE as EXPANSION_RECIPE
from freeze_c5_signer_recovery_plan import RECIPE as SIGNER_RECOVERY_RECIPE
from freeze_c5_stem_scale_plan import RECIPE as SCALE_RECIPE
from run_c5_stem_projection_frozen_tuning import select_step, tuning_config


def _row(h_delta: float, f_delta: float, improved: int) -> dict:
    return {
        "how2sign": {
            "candidate_minus_baseline_mean_wer": h_delta,
            "signer_means": {
                str(index): {
                    "baseline_mean_wer": 1.0,
                    "candidate_mean_wer": 0.9 if index < improved else 1.0,
                }
                for index in range(5)
            },
        },
        "fleurs": {"candidate_minus_baseline_mean_wer": f_delta},
    }


SCREEN = {
    "how2sign_max_mean_wer_change": 0.01,
    "fleurs_max_mean_wer_change": 0.01,
    "equal_domain_max_mean_wer_change": -0.005,
    "minimum_improved_how2sign_signers": 3,
}


def test_selects_lowest_equal_domain_change():
    summaries = {"128": _row(-0.02, 0.0, 3), "256": _row(-0.03, 0.0, 4)}
    assert select_step(summaries, SCREEN) == 256


def test_rejects_two_signer_improvement():
    summaries = {"128": _row(-0.04, -0.02, 2)}
    assert select_step(summaries, SCREEN) is None


def test_scale_plan_uses_scale_recipe_and_output_schema():
    recipe, schema = tuning_config(
        {"schema": "umi-studio-c5-stem-scale-plan/1", "recipe": SCALE_RECIPE}
    )
    assert recipe == SCALE_RECIPE
    assert schema == "umi-studio-c5-stem-scale-tuning-decision/1"


def test_signer_recovery_plan_uses_recovery_recipe():
    recipe, schema = tuning_config(
        {
            "schema": "umi-studio-c5-frozen-replay-plan/1",
            "recipe": SIGNER_RECOVERY_RECIPE,
        }
    )
    assert recipe == SIGNER_RECOVERY_RECIPE
    assert schema == "umi-studio-c5-signer-recovery-tuning-decision/1"


def test_openasl_plan_uses_openasl_output_schema():
    recipe, schema = tuning_config(
        {
            "schema": "umi-studio-c5-frozen-replay-plan/1",
            "recipe": OPENASL_RECIPE,
        }
    )
    assert recipe == OPENASL_RECIPE
    assert schema == "umi-studio-c5-openasl-pilot-tuning-decision/1"


def test_how2sign_expansion_plan_uses_distinct_output_schema():
    recipe, schema = tuning_config(
        {
            "schema": "umi-studio-c5-frozen-replay-plan/1",
            "recipe": EXPANSION_RECIPE,
        }
    )
    assert recipe == EXPANSION_RECIPE
    assert schema == "umi-studio-c5-how2sign-expansion-tuning-decision/1"
