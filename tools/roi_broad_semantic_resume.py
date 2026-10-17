"""One explicit state-exact continuation of the broad curriculum after calibration stop."""

from __future__ import annotations

import json
import math

from benchmark_fleurs_dev import sha256
from roi_appearance_protocol import ROOT
from roi_semantic_resume import restore  # noqa: F401, audited shared weight/Adam/RNG restoration
from terra_training_recovery import artifact, verify_snapshot

CONTROL_SOURCES = {"tools/train_roi_broad_semantic.py", "tools/roi_broad_semantic_resume.py"}


def compatible(previous, current):
    ignored = {"source_sha256", "resume"}
    if {k: v for k, v in previous.items() if k not in ignored} != {
        k: v for k, v in current.items() if k not in ignored
    }:
        raise ValueError(
            "broad ROI recovery cannot change data, curriculum, objective, model or proof"
        )
    before, after = previous["source_sha256"], current["source_sha256"]
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    if not changed <= CONTROL_SOURCES:
        raise ValueError("broad ROI recovery changed a qualified scientific component")


def prefix_state(plan, steps):
    if not steps or len(steps) >= len(plan["batches"]):
        raise ValueError("broad ROI recovery requires a nonempty unfinished sampler prefix")
    examples = updates = 0
    for number, step in enumerate(steps):
        batch = plan["batches"][number]
        quality = step["Terra_qualities"]
        if (
            any(step.get(k) != v for k, v in batch.items())
            or step.get("batch") != number
            or len(quality) != len(batch["training_record_indices"])
            or any(
                not q or q[0] != 10 or any(type(v) is not int or not 0 <= v <= 10 for v in q)
                for q in quality
            )
        ):
            raise ValueError("broad ROI recovery lost the complete original batch prefix")
        contrasting = sum(len(set(q)) > 1 for q in quality)
        norm = step.get("gradient_norm_before_clip")
        if contrasting:
            updates += 1
            if type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0:
                raise ValueError("broad ROI recovery applied-step gradient is invalid")
        elif norm is not None:
            raise ValueError("broad ROI tied rewards cannot claim an update")
        examples += len(batch["training_record_indices"])
        if (
            step.get("contrasting_clips") != contrasting
            or step.get("training_examples") != examples
            or step.get("optimizer_updates") != updates
        ):
            raise ValueError("broad ROI recovery prefix counters changed")
    return examples, updates


def make_resume(previous_path, checkpoint, current):
    import torch
    from probe_tiny_rgb_capacity import state_digest
    from safetensors import safe_open
    from safetensors.torch import load_file

    previous_path, checkpoint = previous_path.resolve(), checkpoint.resolve()
    if not previous_path.is_relative_to(ROOT / "runs") or not checkpoint.is_relative_to(
        ROOT / "checkpoints"
    ):
        raise ValueError("broad ROI recovery needs owned runs/ and checkpoints/ artifacts")
    previous = json.loads(previous_path.read_text())
    compatible(previous, current)
    if "resume" in previous:
        raise ValueError("broad ROI permits one manual continuation, not a retry chain")
    verify_snapshot(previous_path, previous)
    output = checkpoint.parent
    commit_path = checkpoint.with_suffix(".json")
    commit = json.loads(commit_path.read_text())
    stopped = json.loads((output / "stopped.json").read_text())
    terminal = json.loads((output / "exit.json").read_text())
    worker = json.loads((output / "worker.json").read_text())
    run = json.loads((output / "run.json").read_text())
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    examples, updates = prefix_state(previous, steps)
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
        or commit.get("training_examples") != examples
        or commit.get("optimizer_updates") != updates
        or not 0 < updates <= len(steps)
        or run.get("plan_sha256") != sha256(previous_path)
        or run.get("initial_visual_sha256") != current["vision_initialization_sha256"]
        or run.get("fixed_backbones") != current["model_plan"]["initial_reference_hashes"]
        or run.get("recipe") != current["recipe"]
    ):
        raise ValueError("broad ROI needs an exact terminal preupdate calibration-stop recovery")
    for path in (checkpoint, optimizer):
        with safe_open(path, framework="pt") as saved:
            metadata = saved.metadata()
        if metadata.get("plan_sha256") != sha256(previous_path) or metadata.get("update") != str(
            examples
        ):
            raise ValueError("broad ROI recovery weights/optimizer have a different plan or offset")
    with safe_open(checkpoint, framework="pt") as saved:
        metadata = saved.metadata()
    if (
        metadata.get("optimizer_updates") != str(updates)
        or metadata.get("fixed_winner_checkpoint")
        != current["model_plan"]["reference"]["translator_checkpoint_sha256"]
    ):
        raise ValueError("broad ROI recovery counter or fixed C5 reference changed")
    values = load_file(checkpoint)
    if len(values) != current["model_plan"]["policy"]["trainable_tensors"] or any(
        not name.startswith("vision.")
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for name, value in values.items()
    ):
        raise ValueError("broad ROI recovery requires the exact finite visual-only tensors")
    return {
        "source_plan": artifact(previous_path),
        "visual": artifact(checkpoint),
        "optimizer": artifact(optimizer),
        "source_steps": artifact(output / "steps.jsonl"),
        "source_run": artifact(output / "run.json"),
        "source_stop": artifact(output / "stopped.json"),
        "source_terminal": artifact(output / "exit.json"),
        "source_commit": artifact(commit_path),
        "training_examples": examples,
        "optimizer_updates": updates,
        "completed_batches": len(steps),
        "starting_visual_state_sha256": state_digest(
            {n.removeprefix("vision."): v for n, v in values.items()}
        ),
        "failed_batch_grades_discarded": True,
        "manual_continuation_limit": 1,
    }
