"""The fixed confirmation gate compares one stride-2 candidate to stride 1."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_temporal_stride_confirmation import build_decision


def _domain(prefix: str, count: int, signers: list[int], baseline: float, candidate: float):
    first = []
    second = []
    for index in range(count):
        frames = 5 + index
        row = {
            "example_id": f"{prefix}-{index}",
            "reference": "one two",
            "signer": signers[index % len(signers)],
            "index_sha256": f"{prefix}-index",
            "checkpoint_sha256": "checkpoint",
            "adapter_sha256": None,
            "temporal_stride": 1,
            "source_frame_count": frames,
            "evaluated_frame_count": frames,
            "single_reference_wer": baseline,
            "single_reference_word_edits": round(baseline * 100),
            "reference_word_count": 100,
        }
        first.append(row)
        second.append(
            {
                **row,
                "adapter_sha256": "adapter",
                "temporal_stride": 2,
                "evaluated_frame_count": (frames + 1) // 2,
                "single_reference_wer": candidate,
                "single_reference_word_edits": round(candidate * 100),
            }
        )
    return first, second


def test_confirmation_gate_packages_broad_improvement():
    hb, hc = _domain("h", 42, [1, 3, 5, 8], 0.9, 0.82)
    fb, fc = _domain("f", 46, [4], 1.0, 0.9)
    plan = {
        "schema": "umi-studio-temporal-stride-confirmation-plan/1",
        "candidate_temporal_stride": 2,
        "candidate_adapter_sha256": "adapter",
        "checkpoint_sha256": "checkpoint",
        "confirmation_gate": {
            "how2sign_max_mean_wer_change": -0.025,
            "how2sign_min_improved_signers": 3,
            "fleurs_max_mean_wer_change": 0.0,
            "equal_domain_max_mean_wer_change": -0.025,
        },
    }
    result = build_decision(plan, hb, hc, fb, fc)
    assert result["decision"] == "package_candidate"
    assert result["how2sign_signers_improved"] == 4
    assert result["confirmation_evaluated"] is True


def test_confirmation_gate_rejects_how2sign_regression():
    hb, hc = _domain("h", 42, [1, 3, 5, 8], 0.9, 0.91)
    fb, fc = _domain("f", 46, [4], 1.0, 0.8)
    plan = {
        "schema": "umi-studio-temporal-stride-confirmation-plan/1",
        "candidate_temporal_stride": 2,
        "candidate_adapter_sha256": "adapter",
        "checkpoint_sha256": "checkpoint",
        "confirmation_gate": {
            "how2sign_max_mean_wer_change": -0.025,
            "how2sign_min_improved_signers": 3,
            "fleurs_max_mean_wer_change": 0.0,
            "equal_domain_max_mean_wer_change": -0.025,
        },
    }
    result = build_decision(plan, hb, hc, fb, fc)
    assert result["decision"] == "retain_frozen_baseline"
