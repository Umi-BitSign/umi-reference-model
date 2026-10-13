"""Visual training reads only checksum-matched frame-aligned streams."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_visual_lora_plan import RECIPE, SEED
from train_visual_lora import checked_plan, checked_resume_record, training_streams
import train_visual_lora


def _streams():
    return {
        "face_features": np.ones((2, 384), dtype=np.float32),
        "left_hand_features": np.ones((2, 384), dtype=np.float32),
        "right_hand_features": np.ones((2, 384), dtype=np.float32),
        "pose_features": np.ones((2, 14), dtype=np.float32),
    }


def test_plan_comparison_normalizes_integer_signer_keys():
    reproduced = {
        "recipe": RECIPE,
        "seed": SEED,
        "signer_counts": {1: 5, 8: 9},
    }
    checked_plan(json.loads(json.dumps(reproduced)), reproduced)
    with pytest.raises(ValueError, match="frozen plan"):
        checked_plan({**json.loads(json.dumps(reproduced)), "seed": 19}, reproduced)


def test_reads_embedded_or_sidecar_streams_with_frozen_hashes(tmp_path):
    cache = tmp_path / "cache.npz"
    sidecar = tmp_path / "sidecar.npz"
    streams = _streams()
    np.savez(cache, encoder_hidden_state=np.ones((2, 1536), dtype=np.float32), **streams)
    row = {
        "id": "clip",
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "sidecar": None,
        "sidecar_sha256": None,
        "frames": 2,
    }
    assert set(training_streams(row)) == set(streams)
    np.savez(sidecar, **streams)
    np.savez(cache, encoder_hidden_state=np.ones((2, 1536), dtype=np.float32))
    row.update(
        cache_sha256=sha256(cache),
        sidecar=str(sidecar),
        sidecar_sha256=sha256(sidecar),
    )
    assert set(training_streams(row)) == set(streams)
    row["sidecar_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sidecar changed"):
        training_streams(row)


def test_resume_requires_committed_matching_adapter_and_optimizer(tmp_path):
    import torch
    from safetensors.torch import save_file

    directory = tmp_path / "checkpoints" / "first-segment"
    directory.mkdir(parents=True)
    checkpoint = directory / "visual-lora-after-000128.safetensors"
    optimizer = checkpoint.with_suffix(".optimizer.pt")
    plan_hash = "a" * 64
    plan = {
        "baseline_checkpoint_sha256": "b" * 64,
        "visual_adapter_source_sha256": "c" * 64,
    }
    metadata = {
        "schema": "umi-studio-shubert-visual-lora/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "rank": str(RECIPE["rank"]),
        "alpha": str(RECIPE["alpha"]),
        "visual_last_layers": str(RECIPE["visual_last_layers"]),
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "trainer_source_sha256": sha256(Path(train_visual_lora.__file__)),
        "adapter_source_sha256": plan["visual_adapter_source_sha256"],
        "example_step": "128",
    }
    save_file({"weight": torch.ones(1)}, str(checkpoint), metadata=metadata)
    optimizer.write_bytes(b"optimizer state")
    record = {
        "schema": "umi-studio-visual-lora-recovery/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "example_step": 128,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "optimizer_name": optimizer.name,
        "optimizer_sha256": sha256(optimizer),
        "selected_names": ["weight"],
        "tuning_checkpoint": False,
    }
    index = directory / "checkpoint-index.jsonl"
    index.write_text(json.dumps(record) + "\n")
    verified = checked_resume_record(
        checkpoint, plan, plan_hash, ["weight"], tmp_path / "checkpoints"
    )
    assert verified["example_step"] == 128
    optimizer.write_bytes(b"different state")
    with pytest.raises(ValueError, match="recovery record"):
        checked_resume_record(checkpoint, plan, plan_hash, ["weight"], tmp_path / "checkpoints")
