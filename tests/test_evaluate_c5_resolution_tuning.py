from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from evaluate_c5_resolution_tuning import checked_completed, resolution_policy, summarize


def test_resolution_policy_preserves_crop_and_stride() -> None:
    assert resolution_policy(None) == {
        "source_short_side": None,
        "source_resize_policy": "native",
        "visual_crop_size": 224,
        "dino_patch_size": 14,
        "patch_tokens_per_stream_frame": 256,
        "temporal_stride": 2,
    }
    assert resolution_policy(256)["source_resize_policy"] == (
        "aspect_preserving_downscale_only"
    )
    assert resolution_policy(None, 168)["patch_tokens_per_stream_frame"] == 144
    with pytest.raises(ValueError, match="between 64 and 1024"):
        resolution_policy(32)


def test_checked_completed_rejects_policy_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    record = {
        "schema": "umi-studio-c5-resolution-tuning-example/1",
        "source": "how2sign",
        "example_id": "one",
        "tuning_inventory_sha256": "i",
        "bundle_identity_sha256": "b",
        "bundle_runtime_sha256": "r",
        "stride_runtime_sha256": "s",
        "landmark_receipts_sha256": "l",
        **resolution_policy(None),
    }
    path.write_text(json.dumps(record) + "\n")
    completed = checked_completed(
        path,
        inventory_hash="i",
        identity_hash="b",
        runtime_hash="r",
        stride_runtime_hash="s",
        landmark_receipts_hash="l",
        calibrator_hash=None,
        policy=resolution_policy(None),
    )
    assert set(completed) == {("how2sign", "one")}
    with pytest.raises(ValueError, match="another frozen input"):
        checked_completed(
            path,
            inventory_hash="i",
            identity_hash="b",
            runtime_hash="r",
            stride_runtime_hash="s",
            landmark_receipts_hash="l",
            calibrator_hash=None,
            policy=resolution_policy(256),
        )


def test_summarize_reports_mean_and_word_weighted_wer() -> None:
    records = [
        {
            "source": "fleurs",
            "hypothesis": "a",
            "single_reference_wer": 0.5,
            "single_reference_word_edits": 1,
            "reference_word_count": 2,
        },
        {
            "source": "fleurs",
            "hypothesis": "b",
            "single_reference_wer": 1.0,
            "single_reference_word_edits": 3,
            "reference_word_count": 3,
        },
    ]
    summary = summarize(records)["fleurs"]
    assert summary["mean_single_reference_wer"] == 0.75
    assert summary["word_weighted_single_reference_wer"] == 0.8
    assert summary["distinct_hypotheses"] == 2
