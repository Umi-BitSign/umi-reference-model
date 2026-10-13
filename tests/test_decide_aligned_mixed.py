"""The aligned LoRA gate compares paired tuning results and keeps confirmation sealed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_aligned_mixed import build_decision


def _results(source: str, count: int, baseline_wer: float, candidate_wer: float):
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
            "single_reference_wer": baseline_wer,
        }
        if source == "how2sign":
            row["signer"] = signers[number]
        baseline.append(row)
        candidate.append(
            {
                **row,
                "signer": 4 if source == "fleurs" else row["signer"],
                "adapter_sha256": "adapter",
                "single_reference_wer": candidate_wer,
            }
        )
    return baseline, candidate


def test_aligned_tuning_gate_accepts_broad_improvement():
    fleurs_baseline, fleurs_candidate = _results("fleurs", 7, 1.0, 1.01)
    how2sign_baseline, how2sign_candidate = _results("how2sign", 16, 0.9, 0.8)
    plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
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
    decision = build_decision(
        plan,
        fleurs_baseline,
        fleurs_candidate,
        how2sign_baseline,
        how2sign_candidate,
        hashes,
    )
    assert decision["decision"] == "proceed_to_confirmation"
    assert decision["how2sign_signers_improved"] == 5
    assert decision["confirmation_evaluated"] is False
    how2sign_candidate = [{**row, "single_reference_wer": 0.88} for row in how2sign_candidate]
    decision = build_decision(
        plan,
        fleurs_baseline,
        fleurs_candidate,
        how2sign_baseline,
        how2sign_candidate,
        hashes,
    )
    assert decision["decision"] == "retain_frozen_baseline"
    how2sign_candidate[0]["reference"] = "Different reference."
    with pytest.raises(ValueError, match="different inputs"):
        build_decision(
            plan,
            fleurs_baseline,
            fleurs_candidate,
            how2sign_baseline,
            how2sign_candidate,
            hashes,
        )
