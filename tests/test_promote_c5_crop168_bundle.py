from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from promote_c5_crop168_bundle import validate_confirmation


def passing() -> dict:
    return {
        "schema": "umi-studio-c5-crop-openasl-confirmation-decision/1",
        "decision": "promote_crop_candidate",
        "quality_gate_passed": True,
        "fresh_external_confirmation": True,
        "source_videos": 60,
        "miner_installed": False,
        "mean_wer_delta": -0.01,
        "word_weighted_wer_delta": -0.001,
        "paired": {"improved": 15, "regressed": 10},
    }


def test_confirmation_validation_requires_every_promotion_gate() -> None:
    validate_confirmation(passing())
    for field, value in (
        ("quality_gate_passed", False),
        ("fresh_external_confirmation", False),
        ("source_videos", 59),
        ("mean_wer_delta", -0.001),
        ("word_weighted_wer_delta", 0.001),
    ):
        failed = {**passing(), field: value}
        with pytest.raises(ValueError, match="lacks passing"):
            validate_confirmation(failed)
