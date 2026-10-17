"""Read-only audit of word epoch recovery commits against an authenticated live prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_word_training import RECIPE
from launch_terra_visual_job import write
from safetensors import safe_open
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
TENSORS = 350
PARAMETERS = 43260672


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optimizer_inventory(path, shapes, updates, recipe):
    expected = {
        f"{name}.{field}" for name in shapes for field in ("step", "exp_avg", "exp_avg_sq")
    } | {"rng.cpu", "rng.mps"}
    require(type(updates) is int and updates > 0, "epoch Adam audit needs committed updates")
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        metadata = saved.metadata()
        order = json.loads(metadata["parameter_order"])
        groups = json.loads(metadata["groups_without_parameters"])
        require(set(saved.keys()) == expected, "epoch Adam/RNG tensor inventory changed")
        require(
            len(order) == len(set(order)) == TENSORS and set(order) == set(shapes),
            "epoch Adam original350-name order is incomplete",
        )
        require(len(groups) == 1, "epoch Adam parameter group count changed")
        group = groups[0]
        expected_group = {
            "lr": recipe["learning_rate"],
            "weight_decay": recipe["weight_decay"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "foreach": False,
        }
        require(
            all(group.get(key) == value for key, value in expected_group.items())
            and "fused" in group
            and group["fused"] in (False, None),
            "epoch Adam hyperparameters changed",
        )
        for name, shape in shapes.items():
            step = saved.get_tensor(f"{name}.step")
            require(
                step.shape == () and step.dtype == torch.float32 and step.item() == updates,
                "epoch Adam tensor step differs from ledger",
            )
            for field in ("exp_avg", "exp_avg_sq"):
                value = saved.get_tensor(f"{name}.{field}")
                require(
                    list(value.shape) == shape
                    and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()),
                    "epoch Adam moment has invalid shape, dtype or values",
                )
                if field == "exp_avg_sq":
                    require(not bool((value < 0).any()), "epoch Adam second moment is negative")
        rng = {}
        for name in ("cpu", "mps"):
            value = saved.get_tensor(f"rng.{name}")
            require(
                value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                "epoch RNG state is not a complete byte vector",
            )
            if name == "cpu":
                # A separate generator checks validity without changing the process RNG.
                torch.Generator(device="cpu").set_state(value)
            rng[name] = {
                "bytes": value.numel(),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
    return {
        "all_Adam_steps": updates,
        "parameter_tensors": len(shapes),
        "RNG": rng,
        "groups": groups,
    }


def audit_epoch(plan_path, observation_path, commit_path):
    plan = json.loads(plan_path.read_text())
    observation = json.loads(observation_path.read_text())
    require(plan["recipe"] == RECIPE, "epoch audit requires unchanged word curriculum recipe")
    verify_snapshot(plan_path, plan)
    for name, digest in plan["source_sha256"].items():
        require(sha256(ROOT / name) == digest, "epoch audit found changed pinned training source")
    require(
        observation.get("schema") == "umi-studio-word-training-live-observation/1"
        and observation.get("same_original_worker_confirmed_live") is True
        and observation.get("previous_committed_prefix_and_evidence_unchanged") is True
        and observation.get("new_Terra_requests_during_observation") == 0
        and observation.get("optimizer_updates_during_observation") == 0,
        "epoch audit needs an authenticated live-prefix observation",
    )
    # Verify every immutable evidence file recorded by the prefix authenticator.
    descriptors = [
        observation["original_worker"],
        observation["original_run"],
        observation["observer_source"],
        observation["completed_steps_prefix"],
        *observation["actual_raw_training_evidence"],
    ]
    for descriptor in descriptors:
        require(
            artifact(ROOT / descriptor["path"]) == descriptor, "epoch observation evidence changed"
        )
    run_path = ROOT / observation["original_run"]["path"]
    directory = run_path.parent
    run = json.loads(run_path.read_text())
    require(
        run["plan_sha256"] == sha256(plan_path)
        and run["recipe"] == plan["recipe"]
        and run["initial_visual_sha256"] == plan["vision_initialization_sha256"]
        and run["fixed_backbones"] == plan["model_plan"]["initial_reference_hashes"],
        "epoch observation belongs to another training initialization or plan",
    )
    prefix = ROOT / observation["completed_steps_prefix"]["path"]
    raw = prefix.read_bytes()
    require(raw.endswith(b"\n"), "epoch observed prefix ends in an incomplete batch")
    steps = [json.loads(line) for line in raw.splitlines()]
    require(
        len(steps) == observation["complete_batch_records"]
        and len(steps) <= len(plan["batches"])
        and len(plan["records"]) == 214
        and plan["planned_training_examples"] == 856,
        "epoch observation changed complete word curriculum coverage",
    )
    commit_path = commit_path.resolve()
    require(
        commit_path.parent == directory.resolve()
        and artifact(commit_path) in observation["epoch_checkpoint_commits"],
        "epoch commit is absent from this authenticated snapshot",
    )
    commit = json.loads(commit_path.read_text())
    examples, updates = 0, 0
    boundary = None
    independent_batches = []
    for epoch in range(RECIPE["epochs"]):
        order = list(range(214))
        random.Random(RECIPE["seed"] + epoch).shuffle(order)
        independent_batches.extend(
            {"epoch": epoch, "training_record_indices": order[offset : offset + 4]}
            for offset in range(0, 214, 4)
        )
    require(plan["batches"] == independent_batches, "epoch independent deterministic order changed")
    for number, step in enumerate(steps):
        batch = independent_batches[number]
        require(
            all(step.get(key) == value for key, value in batch.items()) and step["batch"] == number,
            "epoch observed batch order changed",
        )
        examples += len(batch["training_record_indices"])
        contrast = sum(len(set(values)) > 1 for values in step["Terra_qualities"])
        updates += bool(contrast)
        require(
            step["training_examples"] == examples
            and step["optimizer_updates"] == updates
            and step["contrasting_clips"] == contrast,
            "epoch observed exposure/update counters changed",
        )
        if examples == commit.get("training_examples"):
            require((number + 1) % 54 == 0, "epoch commit is not a complete214-clip boundary")
            boundary = {
                "training_examples": examples,
                "optimizer_updates": updates,
                "epoch": batch["epoch"],
                "tail_clips": len(batch["training_record_indices"]),
            }
    require(boundary is not None, "epoch commit exceeds the authenticated ledger prefix")
    require(
        observation["training_examples"] == examples
        and observation["optimizer_updates"] == updates,
        "epoch observation counters differ from its saved prefix",
    )
    visual = directory / f"visual-after-{boundary['training_examples']:06d}.safetensors"
    optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
    require(
        commit_path == visual.with_suffix(".json")
        and commit.get("complete") is True
        and commit.get("visual") == artifact(visual)
        and commit.get("optimizer") == artifact(optimizer)
        and commit.get("optimizer_updates") == boundary["optimizer_updates"]
        and commit.get("committed_batch_RNG") is True,
        "epoch lacks a complete visual/Adam/committed-RNG transaction",
    )
    scope_descriptor = plan["model_plan"]["visual_scope"]
    scope_path = ROOT / scope_descriptor["path"]
    require(artifact(scope_path) == scope_descriptor, "epoch visual-shape scope changed")
    shapes = {
        "vision." + row["parameter"]: row["shape"]
        for row in json.loads(scope_path.read_text())["parameters"]
    }
    require(
        len(shapes) == TENSORS
        and set(shapes) == set(plan["model_plan"]["expected_visual_gradient_names"])
        and sum(math.prod(shape) for shape in shapes.values()) == PARAMETERS,
        "epoch visual tensors or parameter count changed",
    )
    for path in (visual, optimizer):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            require(
                metadata.get("plan_sha256") == sha256(plan_path)
                and metadata.get("update") == str(boundary["training_examples"]),
                "epoch tensor artifact belongs to another plan or exposure count",
            )
            if path == visual:
                require(
                    metadata.get("optimizer_updates") == str(boundary["optimizer_updates"])
                    and metadata.get("fixed_winner_checkpoint")
                    == plan["model_plan"]["reference"]["translator_checkpoint_sha256"]
                    and set(saved.keys()) == set(shapes),
                    "epoch visual names, update count or backbone provenance changed",
                )
                for name, shape in shapes.items():
                    value = saved.get_tensor(name)
                    require(
                        list(value.shape) == shape
                        and value.dtype == torch.float32
                        and bool(torch.isfinite(value).all()),
                        "epoch visual tensor is invalid",
                    )
    inventory = optimizer_inventory(
        optimizer, shapes, boundary["optimizer_updates"], plan["recipe"]
    )
    # Detect a changed artifact during inspection, including its commit descriptor.
    require(
        commit["visual"] == artifact(visual)
        and commit["optimizer"] == artifact(optimizer)
        and artifact(commit_path) in observation["epoch_checkpoint_commits"],
        "epoch committed artifact changed during inspection",
    )
    return {
        "schema": "umi-studio-word-epoch-recovery-artifact-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "observation": artifact(observation_path),
        "audit_source_sha256": sha256(Path(__file__)),
        "commit": artifact(commit_path),
        "visual": artifact(visual),
        "optimizer": artifact(optimizer),
        **boundary,
        **inventory,
        "parameter_values": PARAMETERS,
        "optimizer_or_RNG_restored": False,
        "MPS_RNG_device_restoration_tested": False,
        "recovery_artifacts_validated": True,
        "automatic_recovery_authorized": False,
        "fixed_final_evaluation_eligible": False,
        "held_out_quality_evaluated": False,
        "new_Terra_requests": 0,
        "optimizer_updates_during_audit": 0,
        "promotion_authorized": False,
    }


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "observation", "commit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        parser.error("epoch audit needs a fresh owned receipt in runs/")
    result = audit_epoch(args.plan.resolve(), args.observation.resolve(), args.commit.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(
        json.dumps(
            {
                "audit": artifact(output),
                "training_examples": result["training_examples"],
                "optimizer_updates": result["optimizer_updates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
