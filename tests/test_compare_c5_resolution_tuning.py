from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_c5_resolution_tuning import paired_summary


def row(source: str, example: str, wer: float, seconds: float, signer: int = 1) -> dict:
    return {
        "source": source,
        "example_id": example,
        "signer": signer,
        "hypothesis": example,
        "single_reference_wer": wer,
        "single_reference_word_edits": round(wer * 10),
        "reference_word_count": 10,
        "active_seconds": seconds,
        "source_resize_policy": "native",
        "source_short_side": None,
        "tuning_inventory_sha256": "i",
        "bundle_identity_sha256": "b",
        "bundle_runtime_sha256": "r",
        "stride_runtime_sha256": "s",
        "model_sha256": "m",
        "landmark_receipts_sha256": "l",
        "landmarks_sha256": example,
        "temporal_stride": 2,
        "visual_crop_size": 224,
    }


def test_paired_summary_rejects_slow_quality_regression() -> None:
    control = {
        ("how2sign", "h"): row("how2sign", "h", 0.5, 2),
        ("fleurs", "f"): row("fleurs", "f", 0.5, 2, 4),
    }
    candidate = {key: dict(value) for key, value in control.items()}
    for value in candidate.values():
        value["source_resize_policy"] = "aspect_preserving_downscale_only"
        value["source_short_side"] = 256
        value["single_reference_wer"] = 0.6
        value["single_reference_word_edits"] = 6
        value["active_seconds"] = 3
    result = paired_summary(control, candidate)
    assert result["decision"] == "reject_resolution_candidate"
    assert result["quality_gate_passed"] is False
    assert result["speed_gate_passed"] is False
    assert result["paired_by_source"]["how2sign"]["regressed"] == 1
