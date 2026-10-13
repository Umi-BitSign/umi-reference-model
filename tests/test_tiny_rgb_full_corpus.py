from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_full_corpus import FULL_RECIPE, fit_passes, full_schedule, validate_split


def rows():
    return [
        {
            "source": source,
            "example_id": str(i),
            "signer": signer,
            "role": "training",
            "training_allowed": True,
            "source_group": f"group{i}",
            "reference": f"caption {i}",
        }
        for i, (source, signer) in enumerate(
            (
                ("how2sign_raw", 1),
                ("how2sign_original", 2),
                ("fleurs", 3),
                ("asl_stem_original", "P1"),
                ("openasl_scale", "source"),
            )
        )
    ]


def test_full_raw_replay_preserves_all_sources_and_unit_exposure_weights():
    data = rows()
    schedule = full_schedule(data)
    assert {(r["source"], r["id"]) for r in schedule} == {
        (r["source"], r["example_id"]) for r in data
    }
    assert all(r["loss_weight"] == 1 for r in schedule)
    assert FULL_RECIPE["domain_mass"] == {
        "how2sign": 0.60,
        "fleurs": 0.20,
        "asl_stem": 0.15,
        "openasl": 0.05,
    }
    data[0]["training_allowed"] = False
    with pytest.raises(ValueError, match="evaluation or protected"):
        full_schedule(data)


@pytest.mark.parametrize("bad", (None, "reference", "source", "training_role", "eval_role"))
def test_full_rgb_holdout_source_and_reference_isolation(bad):
    training = rows()
    evaluation = [
        {
            "source": "new_open",
            "example_id": "new",
            "source_group": "new",
            "reference": "new reference",
            "role": "open_development",
            "training_allowed": False,
        }
    ]
    if bad == "reference":
        evaluation[0]["reference"] = training[0]["reference"].upper() + "."
    elif bad == "source":
        evaluation[0]["source_group"] = training[0]["source_group"]
    elif bad == "training_role":
        training[0]["role"] = "holdout"
    elif bad == "eval_role":
        evaluation[0]["training_allowed"] = True
    if bad is None:
        validate_split(training, evaluation)
    else:
        with pytest.raises(ValueError, match="full RGB"):
            validate_split(training, evaluation)


def test_full_rgb_requires_completed_passing_fit_at_both_resolutions_and_never_quality_claim():
    decision = {
        "plan_sha256": "plan",
        "complete": True,
        "training_examples_per_arm": 24,
        "optimizer_updates_per_arm": 6144,
        "held_out_quality_verified": False,
        "selected_model_changed": False,
        "checks": [
            {
                "short_side": side,
                "epoch": 256,
                "passes_training_fit_gate": True,
                "training_mean_wer": 0.1,
                "training_exact_translations": 20,
                "wrong_video_wer_increase": 1.0,
                "held_out_quality_verified": False,
                "promotion_authorized": False,
            }
            for side in (128, 192)
        ],
    }
    assert set(fit_passes(decision, "plan")) == {128, 192}
    decision["checks"][1]["passes_training_fit_gate"] = False
    with pytest.raises(ValueError, match="both full RGB"):
        fit_passes(decision, "plan")
    decision["checks"][1]["passes_training_fit_gate"] = True
    decision["held_out_quality_verified"] = True
    with pytest.raises(ValueError, match="completed matched256"):
        fit_passes(decision, "plan")
