"""The projection-frozen run keeps full adapter topology and sealed checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_projection_frozen_lora_plan import RECIPE
from run_projection_frozen_lora_tuning import checked_checkpoints
from train_projection_frozen_lora import complete_adapter_state


def test_adapter_state_includes_frozen_tensors_without_aliasing():
    direct = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    lora = torch.nn.Parameter(torch.zeros(2), requires_grad=True)
    state = complete_adapter_state({"direct": direct, "lora": lora})
    assert set(state) == {"direct", "lora"}
    direct.data.add_(1)
    lora.data.add_(1)
    assert state["direct"].tolist() == [1, 1]
    assert state["lora"].tolist() == [0, 0]


def test_checkpoint_ledger_requires_all_recovery_steps(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    plan_hash = "a" * 64
    plan = {
        "starting_adapter_sha256": "b" * 64,
        "trainer_source_sha256": "c" * 64,
        "baseline_checkpoint_sha256": "d" * 64,
        "adapter_source_sha256": "e" * 64,
    }
    run = {
        "schema": "umi-studio-visual-lora-projection-frozen-training/1",
        "plan_sha256": plan_hash,
        "recipe_id": RECIPE["id"],
        "starting_adapter_sha256": plan["starting_adapter_sha256"],
        "trainer_source_sha256": plan["trainer_source_sha256"],
    }
    (directory / "run.json").write_text(json.dumps(run))
    rows = []
    for step in range(
        RECIPE["recovery_after_examples"],
        RECIPE["train_examples"] + 1,
        RECIPE["recovery_after_examples"],
    ):
        checkpoint = directory / f"visual-lora-after-{step:06d}.safetensors"
        optimizer = checkpoint.with_suffix(".optimizer.pt")
        metadata = {
            "schema": "umi-studio-shubert-visual-lora/1",
            "plan_sha256": plan_hash,
            "recipe_id": RECIPE["id"],
            "example_step": str(step),
            "rank": str(RECIPE["rank"]),
            "alpha": str(RECIPE["alpha"]),
            "visual_last_layers": str(RECIPE["visual_last_layers"]),
            "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
            "trainer_source_sha256": plan["trainer_source_sha256"],
            "adapter_source_sha256": plan["adapter_source_sha256"],
            "starting_adapter_sha256": plan["starting_adapter_sha256"],
        }
        save_file({"weight": torch.tensor([float(step)])}, str(checkpoint), metadata=metadata)
        optimizer.write_bytes(b"optimizer")
        rows.append({
            "schema": "umi-studio-visual-lora-projection-frozen-recovery/1",
            "plan_sha256": plan_hash,
            "recipe_id": RECIPE["id"],
            "example_step": step,
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "optimizer_name": optimizer.name,
            "optimizer_sha256": sha256(optimizer),
            "tuning_checkpoint": step in RECIPE["checkpoint_after_examples"],
        })
    ledger = directory / "checkpoint-index.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selected = checked_checkpoints(directory, plan, plan_hash)
    assert list(selected) == RECIPE["checkpoint_after_examples"]
    ledger.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        checked_checkpoints(directory, plan, plan_hash)
