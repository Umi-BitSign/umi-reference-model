"""Visual-stream backfill accepts training-only immutable cache records."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from backfill_visual_streams import check_training_rows, checked_completed, embedded_streams
from benchmark_fleurs_dev import sha256


def _row(baseline):
    return {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "sentence_name": "clip",
        "split": "training",
        "signer": 1,
        "clip_fps": 25,
        "video_decode_strategy": "opencv_sequential_full_range",
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
        "encoder_dtype": "float32",
        "encoder_shape": [2, 1536],
        "selection_sha256": "selection",
        "encoder_cache_sha256": "cache",
        "resampled_clip_sha256": "clip-hash",
    }


def test_requires_training_only_source_and_valid_embedded_streams(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "MODEL_IDENTITY.json").write_text("identity")
    (baseline / "runtime.py").write_text("runtime")
    row = _row(baseline)
    check_training_rows([row], baseline)
    row["split"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        check_training_rows([row], baseline)
    row["split"] = "training"
    row["stream_feature_dtype"] = "float32"
    with pytest.raises(ValueError, match="embedded visual streams"):
        check_training_rows([row], baseline)
    row["stream_feature_shapes"] = {
        "face_features": [2, 384],
        "left_hand_features": [2, 384],
        "right_hand_features": [2, 384],
        "pose_features": [2, 14],
    }
    check_training_rows([row], baseline)
    assert embedded_streams(row)


def test_completed_sidecars_must_match_frozen_source(tmp_path):
    import json

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "MODEL_IDENTITY.json").write_text("identity")
    (baseline / "runtime.py").write_text("runtime")
    row = _row(baseline)
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    (sidecar_dir / "clip.npz").write_bytes(b"streams")
    record = {
        "schema": "umi-studio-how2sign-visual-stream-sidecar/1",
        "sentence_name": "clip",
        "selection_sha256": "selection",
        "original_index_sha256": "index",
        "encoder_cache_sha256": "cache",
        "resampled_clip_sha256": "clip-hash",
        "stream_feature_dtype": "float32",
        "stream_sha256": sha256(sidecar_dir / "clip.npz"),
    }
    ledger = tmp_path / "sidecars.jsonl"
    ledger.write_text(json.dumps(record) + "\n")
    assert set(checked_completed(ledger, [row], "index", sidecar_dir)) == {"clip"}
    record["encoder_cache_sha256"] = "other"
    ledger.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="differs"):
        checked_completed(ledger, [row], "index", sidecar_dir)
    row["stream_feature_dtype"] = "float32"
    row["stream_feature_shapes"] = {
        "face_features": [2, 384],
        "left_hand_features": [2, 384],
        "right_hand_features": [2, 384],
        "pose_features": [2, 14],
    }
    with pytest.raises(ValueError, match="differs"):
        checked_completed(ledger, [row], "index", sidecar_dir)
