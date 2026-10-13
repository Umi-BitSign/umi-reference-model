"""Full cross-attention decision uses the same frozen two-domain gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_full_cross_attention import build_full_decision


def test_full_decision_keeps_the_two_domain_gate_and_lineage():
    def paired(source, count, before, after):
        schema = (
            "umi-studio-fleurs-cached-evaluation-example/1"
            if source == "fleurs"
            else "umi-studio-how2sign-cached-evaluation-example/1"
        )
        signers = [1, 2, 3, 5, 8] * 3 + [1]
        baseline = []
        candidate = []
        for number in range(count):
            row = {
                "schema": schema,
                "example_id": f"{source}-{number}",
                "reference": f"Reference {number}.",
                "split": "tuning" if source == "fleurs" else "validation",
                "index_sha256": f"{source}-index",
                "checkpoint_sha256": "checkpoint",
                "adapter_sha256": None,
                "single_reference_wer": before,
            }
            if source == "how2sign":
                row["signer"] = signers[number]
            baseline.append(row)
            candidate.append(
                {
                    **row,
                    "signer": 4 if source == "fleurs" else row["signer"],
                    "adapter_sha256": "adapter",
                    "single_reference_wer": after,
                }
            )
        return baseline, candidate

    fleurs_baseline, fleurs_candidate = paired("fleurs", 7, 1.0, 1.01)
    how2sign_baseline, how2sign_candidate = paired("how2sign", 16, 0.9, 0.8)
    plan = {
        "schema": "umi-studio-full-cross-attention-plan/1",
        "training_examples": {"fleurs": 313, "how2sign": 1024, "flores2m": 0},
        "prior_rejected_pilot_decision_sha256": "prior",
        "fleurs_tuning_baseline_sha256": "fleurs-baseline",
        "how2sign_tuning_baseline_sha256": "how2sign-baseline",
        "baseline_mean_single_reference_wer": {"fleurs": 1.0, "how2sign": 0.9},
    }
    hashes = {
        "plan": "plan",
        "adapter": "adapter",
        "fleurs_baseline": "fleurs-baseline",
        "fleurs_candidate": "fleurs-candidate",
        "how2sign_baseline": "how2sign-baseline",
        "how2sign_candidate": "how2sign-candidate",
    }
    decision = build_full_decision(
        plan,
        fleurs_baseline,
        fleurs_candidate,
        how2sign_baseline,
        how2sign_candidate,
        hashes,
    )
    assert decision["schema"] == "umi-studio-full-cross-attention-tuning-decision/1"
    assert decision["decision"] == "proceed_to_confirmation"
    assert decision["prior_rejected_pilot_decision_sha256"] == "prior"
    assert decision["confirmation_evaluated"] is False


def test_full_decision_rejects_partial_training_plan():
    with pytest.raises(ValueError, match="full cross-attention"):
        build_full_decision(
            {
                "schema": "umi-studio-full-cross-attention-plan/1",
                "training_examples": {"fleurs": 313, "how2sign": 168, "flores2m": 0},
            },
            [],
            [],
            [],
            [],
            {},
        )
