"""Training streams and signer balancing keep frozen source identities."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from visual_training_data import (
    FLEURS_REPEAT,
    SIGNER_REPEATS,
    balanced_schedule,
    load_visual_examples,
)


def _baseline(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "MODEL_IDENTITY.json").write_text("identity")
    (baseline / "runtime.py").write_text("runtime")
    return baseline


def _streams():
    return {
        "face_features": np.ones((2, 384), dtype=np.float32),
        "left_hand_features": np.ones((2, 384), dtype=np.float32),
        "right_hand_features": np.ones((2, 384), dtype=np.float32),
        "pose_features": np.ones((2, 14), dtype=np.float32),
    }


def test_loads_verified_embedded_visual_features(tmp_path):
    baseline = _baseline(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "clip.npz"
    streams = _streams()
    np.savez(cache, encoder_hidden_state=np.ones((2, 1536), dtype=np.float32), **streams)
    row = {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "sentence_name": "clip",
        "split": "training",
        "signer": 1,
        "reference": "a sentence",
        "clip_fps": 25,
        "video_decode_strategy": "opencv_sequential_full_range",
        "selection_sha256": "a" * 64,
        "resampled_clip_sha256": "b" * 64,
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
        "encoder_dtype": "float32",
        "encoder_shape": [2, 1536],
        "encoder_cache_sha256": sha256(cache),
        "stream_feature_dtype": "float32",
        "stream_feature_shapes": {name: list(value.shape) for name, value in streams.items()},
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    loaded = load_visual_examples(index, cache_dir, baseline, "how2sign_new")
    assert loaded[0]["id"] == "clip"
    assert loaded[0]["sidecar"] is None
    row["split"] = "validation"
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="non-training"):
        load_visual_examples(index, cache_dir, baseline, "how2sign_new")


def test_loads_verified_sidecar_without_changing_original_cache(tmp_path):
    baseline = _baseline(tmp_path)
    cache_dir = tmp_path / "cache"
    sidecar_dir = tmp_path / "sidecars"
    cache_dir.mkdir()
    sidecar_dir.mkdir()
    cache = cache_dir / "clip.npz"
    sidecar = sidecar_dir / "clip.npz"
    streams = _streams()
    np.savez(cache, encoder_hidden_state=np.ones((2, 1536), dtype=np.float32))
    np.savez(sidecar, **streams)
    row = {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "sentence_name": "clip",
        "split": "training",
        "signer": 1,
        "reference": "a sentence",
        "clip_fps": 25,
        "video_decode_strategy": "opencv_sequential_full_range",
        "selection_sha256": "a" * 64,
        "resampled_clip_sha256": "b" * 64,
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
        "encoder_dtype": "float32",
        "encoder_shape": [2, 1536],
        "encoder_cache_sha256": sha256(cache),
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    sidecar_row = {
        "schema": "umi-studio-how2sign-visual-stream-sidecar/1",
        "sentence_name": "clip",
        "original_index_sha256": sha256(index),
        "selection_sha256": row["selection_sha256"],
        "encoder_cache_sha256": row["encoder_cache_sha256"],
        "resampled_clip_sha256": row["resampled_clip_sha256"],
        "baseline_identity_sha256": row["baseline_identity_sha256"],
        "baseline_runtime_sha256": row["baseline_runtime_sha256"],
        "stream_sha256": sha256(sidecar),
        "stream_feature_dtype": "float32",
        "stream_feature_shapes": {
            name: list(value.shape) for name, value in streams.items()
        },
    }
    sidecar_index = tmp_path / "sidecar-index.jsonl"
    sidecar_index.write_text(json.dumps(sidecar_row) + "\n")
    loaded = load_visual_examples(
        index,
        cache_dir,
        baseline,
        "how2sign_original",
        sidecar_index=sidecar_index,
        sidecar_dir=sidecar_dir,
    )
    assert loaded[0]["sidecar_sha256"] == sha256(sidecar)
    assert loaded[0]["cache_sha256"] == sha256(cache)
    sidecar_row["original_index_sha256"] = "0" * 64
    sidecar_index.write_text(json.dumps(sidecar_row) + "\n")
    with pytest.raises(ValueError, match="sidecar source differs"):
        load_visual_examples(
            index,
            cache_dir,
            baseline,
            "how2sign_original",
            sidecar_index=sidecar_index,
            sidecar_dir=sidecar_dir,
        )


def test_schedule_covers_every_source_and_balances_signers():
    examples = [{"id": "f", "source": "fleurs", "signer": 3}]
    examples.extend(
        {"id": str(signer), "source": "how2sign_raw", "signer": signer}
        for signer in SIGNER_REPEATS
    )
    scheduled = balanced_schedule(examples, seed=17)
    counts = Counter((row["source"], row["id"]) for row in scheduled)
    assert counts[("fleurs", "f")] == FLEURS_REPEAT
    assert all(counts[("how2sign_raw", str(signer))] == repeat for signer, repeat in SIGNER_REPEATS.items())
    assert scheduled == balanced_schedule(examples, seed=17)
    with pytest.raises(ValueError, match="signer"):
        balanced_schedule(examples[:-1], seed=17)
    duplicated = [*examples, {"id": "1", "source": "how2sign_new", "signer": 1}]
    with pytest.raises(ValueError, match="across sources"):
        balanced_schedule(duplicated, seed=17)
