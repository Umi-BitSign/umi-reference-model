"""Explicit one-shot recovery after a calibrated ROI training batch was rejected."""

from __future__ import annotations

import json

from benchmark_fleurs_dev import sha256
from roi_appearance_protocol import ROOT
from terra_training_recovery import artifact, verify_snapshot

CONTROL_SOURCES = {"tools/train_roi_semantic.py", "tools/roi_semantic_resume.py"}


def compatible(previous, current):
    ignored = {"source_sha256", "resume"}
    if {k: v for k, v in previous.items() if k not in ignored} != {
        k: v for k, v in current.items() if k not in ignored
    }:
        raise ValueError("ROI recovery cannot change data, objective, sampler, model or proof")
    before, after = previous["source_sha256"], current["source_sha256"]
    if (
        not {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        <= CONTROL_SOURCES
    ):
        raise ValueError("ROI recovery changed a numerically qualified scientific component")


def make_resume(previous_path, checkpoint, current):
    from safetensors import safe_open

    previous_path, checkpoint = previous_path.resolve(), checkpoint.resolve()
    previous = json.loads(previous_path.read_text())
    compatible(previous, current)
    if "resume" in previous:
        raise ValueError("this recovery authorizes one manual continuation, not a retry chain")
    verify_snapshot(previous_path, previous)
    if not checkpoint.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI recovery checkpoint must remain in owned checkpoints/")
    output = checkpoint.parent
    commit = json.loads(checkpoint.with_suffix(".json").read_text())
    stopped = json.loads((output / "stopped.json").read_text())
    terminal = json.loads((output / "exit.json").read_text())
    worker = json.loads((output / "worker.json").read_text())
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    examples, updates = commit["training_examples"], commit["optimizer_updates"]
    optimizer = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    if (
        terminal.get("exit_code") != 1
        or terminal.get("pid") != worker.get("pid")
        or stopped.get("error") != "ROI training judge failed original calibration controls"
        or stopped.get("automatic_retries") != 0
        or stopped.get("recovery") != artifact(checkpoint)
        or stopped.get("training_examples") != examples
        or stopped.get("optimizer_updates") != updates
        or commit.get("complete") is not True
        or commit.get("visual") != artifact(checkpoint)
        or commit.get("optimizer") != artifact(optimizer)
        or len(steps) != updates
        or not 0 < examples < len(previous["order"])
        or examples != updates * previous["recipe"]["batch_clips"]
    ):
        raise ValueError(
            "ROI manual recovery requires an exact terminal preupdate calibration stop"
        )
    for index, step in enumerate(steps):
        if (
            step["batch"] != index
            or step["optimizer_updates"] != index + 1
            or step["training_examples"] != (index + 1) * 4
            or step["training_record_indices"] != previous["order"][index * 4 : (index + 1) * 4]
        ):
            raise ValueError("ROI recovery lost the complete original optimizer/sampler prefix")
    for path in (checkpoint, optimizer):
        with safe_open(path, framework="pt") as saved:
            metadata = saved.metadata()
        if metadata.get("plan_sha256") != sha256(previous_path) or metadata.get("update") != str(
            examples
        ):
            raise ValueError("ROI recovery weights/optimizer belong to a different plan or offset")
    with safe_open(checkpoint, framework="pt") as saved:
        metadata = saved.metadata()
    if (
        metadata.get("optimizer_updates") != str(updates)
        or metadata.get("fixed_winner_checkpoint")
        != current["model_plan"]["reference"]["translator_checkpoint_sha256"]
    ):
        raise ValueError("ROI recovery policy counter or fixed C5 reference differs")
    return {
        "source_plan": artifact(previous_path),
        "visual": artifact(checkpoint),
        "optimizer": artifact(optimizer),
        "source_steps": artifact(output / "steps.jsonl"),
        "source_stop": artifact(output / "stopped.json"),
        "source_terminal": artifact(output / "exit.json"),
        "source_commit": artifact(checkpoint.with_suffix(".json")),
        "training_examples": examples,
        "optimizer_updates": updates,
        "failed_batch_grades_discarded": True,
        "manual_continuation_limit": 1,
    }


def restore(plan, model, optimizer):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from terra_training_recovery import restore_state

    resume = plan["resume"]
    for value in resume.values():
        if isinstance(value, dict) and artifact(ROOT / value["path"]) != value:
            raise ValueError("ROI recovery artifact bytes changed")
    parameters = model.trainable_parameters()
    visual = load_file(ROOT / resume["visual"]["path"])
    if set(visual) != parameters.keys() or any(
        visual[n].shape != p.shape
        or visual[n].dtype != p.dtype
        or not torch.isfinite(visual[n]).all()
        for n, p in parameters.items()
    ):
        raise ValueError("ROI recovery visual tensor topology or finite values differ")
    path = ROOT / resume["optimizer"]["path"]
    with safe_open(path, framework="pt") as saved:
        metadata = saved.metadata()

    def restore_rng(cpu, mps):
        torch.set_rng_state(cpu)
        torch.mps.set_rng_state(mps)

    restore_state(
        parameters,
        optimizer,
        load_file(path),
        metadata,
        updates=resume["optimizer_updates"],
        restore_rng=restore_rng,
    )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(visual[name].to(parameter.device))
    return resume["training_examples"], resume["optimizer_updates"]
