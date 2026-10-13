"""FLEURS training streams retain the frozen encoder state exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from verify_fleurs_training_stream_fidelity import verify_training_rows


def _streams():
    return {
        "face_features": np.ones((2, 384), dtype=np.float32),
        "left_hand_features": np.ones((2, 384), dtype=np.float32),
        "right_hand_features": np.ones((2, 384), dtype=np.float32),
        "pose_features": np.ones((2, 14), dtype=np.float32),
    }


def test_partial_training_fidelity_checks_exact_state_and_metadata(tmp_path):
    original_dir = tmp_path / "old"
    stream_dir = tmp_path / "new"
    original_dir.mkdir()
    stream_dir.mkdir()
    selection = {
        "schema": "umi-studio-fleurs-dev-split/1",
        "training": [{"example_id": f"clip-{number}"} for number in range(313)],
    }
    old_rows = []
    new_rows = []
    for number in range(313):
        name = f"clip-{number}"
        row = {
            "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
            "split": "training",
            "selection_sha256": "s" * 64,
            "reference": "a sentence",
            "signer": 3,
            "baseline_identity_sha256": "i" * 64,
            "baseline_runtime_sha256": "r" * 64,
            "encoder_shape": [2, 1536],
            "encoder_dtype": "float32",
            "example_id": name,
            "document_id": "document",
            "source_receipt_sha256": "p" * 64,
            "source_video": "source.mp4",
            "source_video_sha256": "v" * 64,
            "clip_sha256": "c" * 64,
        }
        if number < 2:
            if number == 1:
                row["video_decode_strategy"] = "opencv_sequential_full_range"
            old_path = original_dir / f"{name}.npz"
            new_path = stream_dir / f"{name}.npz"
            hidden = np.ones((2, 1536), dtype=np.float32)
            streams = _streams()
            np.savez(old_path, encoder_hidden_state=hidden)
            np.savez(new_path, encoder_hidden_state=hidden, **streams)
            row["encoder_cache_sha256"] = sha256(old_path)
            new_rows.append(
                {
                    **row,
                    "video_decode_strategy": "baseline_seek_per_frame",
                    "encoder_cache_sha256": sha256(new_path),
                    "stream_feature_dtype": "float32",
                    "stream_feature_shapes": {
                        key: list(value.shape) for key, value in streams.items()
                    },
                }
            )
        old_rows.append(row)
    old_rows.reverse()
    result = verify_training_rows(old_rows, new_rows, original_dir, stream_dir, selection)
    assert result["examples_verified"] == 2
    assert result["complete"] is False
    assert result["prior_decode_strategies"] == {
        "opencv_sequential_full_range": 1,
        "unspecified": 1,
    }
    corrupted = np.zeros((2, 1536), dtype=np.float32)
    np.savez(stream_dir / "clip-0.npz", encoder_hidden_state=corrupted, **_streams())
    new_rows[0]["encoder_cache_sha256"] = sha256(stream_dir / "clip-0.npz")
    with pytest.raises(ValueError, match="encoder state changed"):
        verify_training_rows(old_rows, new_rows, original_dir, stream_dir, selection)
