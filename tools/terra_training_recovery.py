"""Hash-bound Terra continuation without changing the qualified learning recipe."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_SOURCES = {"tools/train_terra_visual.py", "tools/terra_training_recovery.py"}


def artifact(path, *, root=ROOT):
    from benchmark_fleurs_dev import sha256

    path = path.resolve()
    if not path.is_relative_to(root / "runs") and not path.is_relative_to(root / "checkpoints"):
        raise ValueError("recovery artifacts must be in runs/ or checkpoints/")
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def checked_artifact(value, *, root=ROOT):
    path = (root / value["path"]).resolve()
    if artifact(path, root=root) != value:
        raise ValueError("Terra recovery artifact path or hash changed")
    return path


def compatible_plans(previous, current):
    """Only orchestration may differ when reusing the actual gradient proof."""
    excluded = {"source_sha256", "resume"}
    if {k: v for k, v in previous.items() if k not in excluded} != {
        k: v for k, v in current.items() if k not in excluded
    }:
        raise ValueError("Terra recovery cannot change the qualified data/model/recipe/gates")
    before, after = previous["source_sha256"], current["source_sha256"]
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed <= CONTROL_SOURCES:
        raise ValueError("Terra recovery changed a numerically qualified implementation")


def verify_snapshot(path, plan):
    from benchmark_fleurs_dev import sha256

    snapshot = path.with_name("source-snapshot")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["plan_sha256"] != sha256(path):
        raise ValueError("Terra recovery source snapshot has a different plan")
    for name, digest in plan["source_sha256"].items():
        if manifest["source_sha256"].get(name) != digest or sha256(snapshot / name) != digest:
            raise ValueError("Terra recovery source snapshot changed")


def make_resume(previous_path, checkpoint, qualification, current, *, root=ROOT):
    from safetensors import safe_open

    previous_path, checkpoint, qualification = (
        value.resolve() for value in (previous_path, checkpoint, qualification)
    )
    previous = json.loads(previous_path.read_text())
    compatible_plans(previous, current)
    verify_snapshot(previous_path, previous)
    with safe_open(str(checkpoint), framework="pt") as file:
        metadata = file.metadata()
    source = artifact(previous_path, root=root)
    if metadata["plan_sha256"] != source["sha256"]:
        raise ValueError("Terra recovery weights belong to a different source plan")
    examples, updates = int(metadata["update"]), int(metadata["optimizer_updates"])
    run_path = checkpoint.parent / "run.json"
    run = json.loads(run_path.read_text())
    if run["plan_sha256"] != source["sha256"]:
        raise ValueError("Terra recovery run belongs to a different source plan")
    base = run.get("resume") or {}
    base_updates = base.get("optimizer_updates", 0)
    base_examples = base.get("training_examples", 0)
    steps_path = checkpoint.parent / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    saved_step = next((row for row in steps if row["optimizer_update"] == updates), None)
    if (
        not 0 < examples <= current["planned_training_examples"]
        or not base_updates <= updates <= base_updates + len(steps)
        or (saved_step is None and (updates != base_updates or examples != base_examples))
        or (saved_step is not None and saved_step["training_examples"] != examples)
        or any(
            row["optimizer_update"] != index for index, row in enumerate(steps, base_updates + 1)
        )
    ):
        raise ValueError("Terra recovery counters do not match committed training steps")
    proof_plan = previous.get("resume", {}).get("qualification_plan", source)
    proof_path = checked_artifact(proof_plan, root=root)
    proof = json.loads(proof_path.read_text())
    compatible_plans(proof, current)
    verify_snapshot(proof_path, proof)
    if json.loads(qualification.read_text())["plan_sha256"] != proof_plan["sha256"]:
        raise ValueError("Terra recovery qualification belongs to a different plan")
    result = {
        "source_plan": source,
        "qualification_plan": proof_plan,
        "qualification": artifact(qualification, root=root),
        "visual": artifact(checkpoint, root=root),
        "optimizer": artifact(
            checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors"), root=root
        ),
        "source_run": artifact(run_path, root=root),
        "source_steps": artifact(steps_path, root=root),
        "training_examples": examples,
        "optimizer_updates": updates,
        "prior_observed_training_examples": steps[-1]["training_examples"]
        if steps
        else base_examples,
        "unsaved_examples_to_recompute": (
            steps[-1]["training_examples"] if steps else base_examples
        )
        - examples,
    }
    decisions = checkpoint.parent / "open-decisions.json"
    if decisions.exists():
        result["source_decisions"] = artifact(decisions, root=root)
    if "tools/terra_training_recovery.py" in previous["source_sha256"]:
        commit_path = checkpoint.with_suffix(".json")
        commit = json.loads(commit_path.read_text())
        if (
            commit.get("complete") is not True
            or commit.get("visual") != result["visual"]
            or commit.get("optimizer") != result["optimizer"]
            or commit.get("training_examples") != examples
            or commit.get("optimizer_updates") != updates
        ):
            raise ValueError("Terra recovery checkpoint pair lacks a complete hash receipt")
        result["checkpoint_commit"] = artifact(commit_path, root=root)
    return result


def checked_resume(plan, *, root=ROOT):
    resume = plan["resume"]
    paths = {
        key: checked_artifact(value, root=root)
        for key, value in resume.items()
        if isinstance(value, dict)
    }
    for key in ("source_plan", "qualification_plan"):
        previous = json.loads(paths[key].read_text())
        compatible_plans(previous, plan)
        verify_snapshot(paths[key], previous)
    return paths


def skip_epoch(remaining, length):
    if remaining < 0 or length <= 0:
        raise ValueError("invalid Terra recovery corpus offset")
    offset = min(remaining, length)
    return offset, remaining - offset


def restore_state(parameters, optimizer, values, metadata, *, updates, restore_rng):
    """Validate completely before mutation; AdamW step stays on CPU, moments on device."""
    import torch

    order = json.loads(metadata["parameter_order"])
    groups = json.loads(metadata["groups_without_parameters"])
    expected = {
        f"{name}.{field}" for name in parameters for field in ("step", "exp_avg", "exp_avg_sq")
    }
    expected |= {"rng.cpu", "rng.mps"}
    live_order = [p for group in optimizer.param_groups for p in group["params"]]
    if (
        order != list(parameters)
        or len(groups) != len(optimizer.param_groups)
        or len(live_order) != len(parameters)
        or any(p is not q for p, q in zip(live_order, parameters.values(), strict=True))
        or set(values) != expected
        or any(group.get("capturable") or group.get("fused") for group in groups)
    ):
        raise ValueError("Terra optimizer topology or execution mode changed")
    for saved, live in zip(groups, optimizer.param_groups, strict=True):
        comparable = {key: value for key, value in live.items() if key not in ("params", "lr")}
        if json.loads(json.dumps(comparable)) != {k: v for k, v in saved.items() if k != "lr"}:
            raise ValueError("Terra optimizer hyperparameters changed")
    for name, parameter in parameters.items():
        step = values[f"{name}.step"]
        if step.shape != () or step.dtype != torch.float32 or step.item() != updates:
            raise ValueError("Terra optimizer step counter differs from the checkpoint")
        for field in ("exp_avg", "exp_avg_sq"):
            value = values[f"{name}.{field}"]
            if (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
                or (field == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError("Terra optimizer moment shape/dtype/finite values differ")
    for key in ("rng.cpu", "rng.mps"):
        if values[key].dtype != torch.uint8 or values[key].ndim != 1 or not values[key].numel():
            raise ValueError("Terra RNG state must be a nonempty byte vector")
    # Validate the CPU RNG bytes without changing the process-global generator.
    torch.Generator(device="cpu").set_state(values["rng.cpu"].cpu())
    prepared = {
        parameter: {
            "step": values[f"{name}.step"].cpu().clone(),
            "exp_avg": values[f"{name}.exp_avg"].to(parameter.device).clone(),
            "exp_avg_sq": values[f"{name}.exp_avg_sq"].to(parameter.device).clone(),
        }
        for name, parameter in parameters.items()
    }
    restore_rng(values["rng.cpu"].cpu(), values["rng.mps"].cpu())
    optimizer.state.clear()
    optimizer.state.update(prepared)
    for live, saved in zip(optimizer.param_groups, groups, strict=True):
        live["lr"] = saved["lr"]


def restore_checkpoint(plan, paths, parameters, optimizer, *, initial_visual, fixed, restore_rng):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    resume = plan["resume"]
    run = json.loads(paths["source_run"].read_text())
    if run["initial_visual_sha256"] != initial_visual or run["fixed_backbones"] != fixed:
        raise ValueError("Terra resume model initialization or fixed backbones differ")
    with safe_open(str(paths["visual"]), framework="pt") as file:
        visual_metadata = file.metadata()
    with safe_open(str(paths["optimizer"]), framework="pt") as file:
        optimizer_metadata = file.metadata()
    for metadata in (visual_metadata, optimizer_metadata):
        if metadata.get("plan_sha256") != resume["source_plan"]["sha256"] or metadata.get(
            "update"
        ) != str(resume["training_examples"]):
            raise ValueError("Terra recovery visual/optimizer provenance differs")
    if (
        visual_metadata["optimizer_updates"] != str(resume["optimizer_updates"])
        or visual_metadata["fixed_winner_checkpoint"]
        != plan["reference"]["translator_checkpoint_sha256"]
    ):
        raise ValueError("Terra recovery counters or fixed checkpoint differ")
    visual = load_file(str(paths["visual"]), device="cpu")
    if set(visual) != set(parameters) or any(
        value.shape != parameters[name].shape
        or value.dtype != parameters[name].dtype
        or not torch.isfinite(value).all()
        for name, value in visual.items()
    ):
        raise ValueError("Terra recovery visual topology or finite values differ")
    restore_state(
        parameters,
        optimizer,
        load_file(str(paths["optimizer"]), device="cpu"),
        optimizer_metadata,
        updates=resume["optimizer_updates"],
        restore_rng=restore_rng,
    )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(visual[name])
    return resume["training_examples"], resume["optimizer_updates"]


def save_checkpoint(
    model,
    optimizer,
    output,
    plan_path,
    reference_sha,
    examples,
    updates,
    *,
    recovery=False,
    root=ROOT,
):
    import torch
    from benchmark_fleurs_dev import sha256
    from safetensors.torch import save_file
    from terra_training_storage import storage
    from train_tiny_rgb_feature_winner_full import optimizer_payload

    prefix = "visual-recovery-after" if recovery else "visual-after"
    checkpoint = output / f"{prefix}-{examples:06d}.safetensors"
    optimizer_path = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    receipt = checkpoint.with_suffix(".json")
    if any(path.exists() for path in (checkpoint, optimizer_path, receipt)):
        raise ValueError("Terra recovery never overwrites checkpoint evidence")
    storage(215000000, output=output, root=root)
    values, metadata = optimizer_payload(
        optimizer,
        model.trainable_parameters(),
        {"cpu": torch.get_rng_state(), "mps": torch.mps.get_rng_state()},
    )
    provenance = {"plan_sha256": sha256(plan_path), "update": str(examples)}
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.trainable_parameters().items()
        },
        str(checkpoint),
        metadata={
            **provenance,
            "optimizer_updates": str(updates),
            "fixed_winner_checkpoint": reference_sha,
        },
    )
    save_file(values, str(optimizer_path), metadata={**provenance, **metadata})
    receipt.write_text(
        json.dumps(
            {
                "training_examples": examples,
                "optimizer_updates": updates,
                "visual": artifact(checkpoint, root=root),
                "optimizer": artifact(optimizer_path, root=root),
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    return checkpoint
