"""Stream caches must preserve both source identity and frozen encoder states."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from verify_stream_cache_fidelity import verify_rows


def _case(tmp_path: Path):
    original_dir = tmp_path / "old"
    stream_dir = tmp_path / "new"
    original_dir.mkdir()
    stream_dir.mkdir()
    state = np.ones((2, 1536), dtype=np.float32)
    old = original_dir / "clip.npz"
    new = stream_dir / "clip.npz"
    np.savez(old, encoder_hidden_state=state)
    streams = {
        "face_features": np.ones((2, 384), dtype=np.float32),
        "left_hand_features": np.ones((2, 384), dtype=np.float32),
        "right_hand_features": np.ones((2, 384), dtype=np.float32),
        "pose_features": np.ones((2, 14), dtype=np.float32),
    }
    np.savez(new, encoder_hidden_state=state, **streams)
    shared = {
        "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
        "example_id": "clip",
        "split": "tuning",
        "selection_sha256": "selection",
        "reference": "a sentence",
        "signer": 4,
        "baseline_identity_sha256": "identity",
        "baseline_runtime_sha256": "runtime",
        "encoder_shape": [2, 1536],
        "encoder_dtype": "float32",
        "video_decode_strategy": "baseline_seek_per_frame",
        "source_receipt_sha256": "receipt",
        "source_video": "source.mp4",
        "source_video_sha256": "video",
        "document_id": "doc",
        "clip_sha256": "clip-hash",
    }
    original = {**shared, "encoder_cache_sha256": sha256(old)}
    streamed = {
        **shared,
        "encoder_cache_sha256": sha256(new),
        "stream_feature_dtype": "float32",
        "stream_feature_shapes": {key: list(value.shape) for key, value in streams.items()},
    }
    partition = {
        "schema": "umi-studio-fleurs-dev-evaluation-partition/1",
        "selection_sha256": "selection",
        "tuning_examples": 1,
        "tuning_documents": ["doc"],
    }
    return original, streamed, original_dir, stream_dir, partition


def test_accepts_identical_states_and_finite_streams(tmp_path):
    original, streamed, original_dir, stream_dir, partition = _case(tmp_path)
    assert verify_rows([original], [streamed], original_dir, stream_dir, partition)["examples"] == 1


def test_rejects_changed_source_or_encoder_state(tmp_path):
    original, streamed, original_dir, stream_dir, partition = _case(tmp_path)
    streamed["reference"] = "different sentence"
    with pytest.raises(ValueError, match="source differs"):
        verify_rows([original], [streamed], original_dir, stream_dir, partition)
    streamed["reference"] = original["reference"]
    target = stream_dir / "clip.npz"
    with np.load(target) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["encoder_hidden_state"][0, 0] = 2
    np.savez(target, **arrays)
    streamed["encoder_cache_sha256"] = sha256(target)
    with pytest.raises(ValueError, match="state changed"):
        verify_rows([original], [streamed], original_dir, stream_dir, partition)


def test_rejects_confirmation_or_duplicate_ids(tmp_path):
    original, streamed, original_dir, stream_dir, partition = _case(tmp_path)
    original["split"] = "confirmation"
    streamed["split"] = "confirmation"
    with pytest.raises(ValueError, match="tuning-only"):
        verify_rows([original], [streamed], original_dir, stream_dir, partition)
    original["split"] = "tuning"
    streamed["split"] = "tuning"
    with pytest.raises(ValueError, match="repeats"):
        verify_rows([original, original], [streamed, streamed], original_dir, stream_dir, partition)


def test_rejects_wrong_frozen_tuning_partition(tmp_path):
    original, streamed, original_dir, stream_dir, partition = _case(tmp_path)
    partition["tuning_documents"] = ["other-doc"]
    with pytest.raises(ValueError, match="frozen tuning"):
        verify_rows([original], [streamed], original_dir, stream_dir, partition)
