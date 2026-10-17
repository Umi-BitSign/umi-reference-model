"""Authenticate a completed word curriculum before any fixed-final evaluation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_word_training import paired_inputs, verified_rewards, verify_coverage
from evaluate_roi_short_translation import terminal
from probe_c5_word_semantic import require_resources
from terra_training_recovery import artifact
from train_c5_word_semantic import GATES
from train_c5_word_semantic import checked as checked_training

ROOT = Path(__file__).resolve().parents[1]


def require_final(result, plan, plan_sha):
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != plan_sha
        or result.get("training_examples") != plan["planned_training_examples"]
        or type(result.get("optimizer_updates")) is not int
        or not 1 <= result["optimizer_updates"] <= plan["maximum_optimizer_updates"]
        or result.get("all_fixed_backbones_and_unused_masks_unchanged") is not True
        or result.get("semantic_translation_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or result.get("evaluation_gates") != GATES
        or result.get("visual_initialization_sha256") != plan["vision_initialization_sha256"]
        or not result.get("visual")
        or not result.get("checkpoints")
        or result["checkpoints"][-1] != result["visual"]
        or re.fullmatch(r"[0-9a-f]{64}", result.get("visual_final_sha256", "")) is None
    ):
        raise ValueError("word evaluation requires the completed fixed-final curriculum")


def verify_steps(plan, steps, result):
    if len(steps) != len(plan["batches"]):
        raise ValueError("word evaluation cannot omit unfinished curriculum batches")
    examples = updates = 0
    observed = []
    previous_active = previous_idle = 0.0
    for number, (step, batch) in enumerate(zip(steps, plan["batches"], strict=True)):
        indices = batch["training_record_indices"]
        qualities = step["Terra_qualities"]
        if (
            any(step.get(key) != value for key, value in batch.items())
            or step.get("batch") != number
            or len(qualities) != len(indices)
            or any(
                not quality
                or quality[0] != 10
                or any(type(value) is not int or not 0 <= value <= 10 for value in quality)
                for quality in qualities
            )
            or len(step.get("before_update_risk", [])) != len(indices)
        ):
            raise ValueError("word final model lost actual order, complete rewards or risks")
        contrasting = sum(len(set(quality)) > 1 for quality in qualities)
        norm = step.get("gradient_norm_before_clip")
        if contrasting:
            updates += 1
            if type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0:
                raise ValueError("word applied update lacks finite nonzero visual gradients")
        elif norm is not None:
            raise ValueError("word tied batch cannot claim an optimizer update")
        for quality, measured in zip(qualities, step["before_update_risk"], strict=True):
            risk = measured.get("expected_terra_error")
            if type(risk) not in (int, float) or not math.isfinite(risk) or not 0 <= risk <= 10:
                raise ValueError("word training logged an invalid actual expected error")
            if len(set(quality)) == 1:
                if measured.get("zero_reward_contrast") is not True or risk != 10 - quality[0]:
                    raise ValueError("word tied clip changed its true expected error")
            elif measured.get("zero_reward_contrast") is True:
                raise ValueError("word contrasting rewards were silently skipped")
        examples += len(indices)
        observed.extend(indices)
        if (
            step.get("contrasting_clips") != contrasting
            or step.get("optimizer_updates") != updates
            or step.get("training_examples") != examples
        ):
            raise ValueError("word final update/exposure counters differ from actual steps")
        resources = step["resources"]
        require_resources(resources)
        if (
            resources["mps_active_seconds"] < previous_active
            or resources["mps_idle_seconds"] < previous_idle
        ):
            raise ValueError("word resource counters moved backwards")
        previous_active = resources["mps_active_seconds"]
        previous_idle = resources["mps_idle_seconds"]
    verify_coverage(plan, observed, examples, result["checkpoints"])
    if updates != result["optimizer_updates"]:
        raise ValueError("word final optimizer count differs from the complete actual ledger")
    require_resources(result["resources"])
    if (
        result["resources"]["mps_active_seconds"] < previous_active
        or result["resources"]["mps_idle_seconds"] < previous_idle
    ):
        raise ValueError("word final resources omit completed numerical work")


def verified_batch_evidence(plan, directory, steps, tokenizer):
    expected = {f"batch-{index:04d}" for index in range(len(plan["batches"]))}
    if {path.name for path in directory.glob("batch-*")} != expected:
        raise ValueError("word final model changed complete actual batch-directory coverage")
    assets = []
    for number, (batch, step) in enumerate(zip(plan["batches"], steps, strict=True)):
        path = directory / f"batch-{number:04d}"
        indices = batch["training_record_indices"]
        records = [plan["records"][i] for i in indices]
        donors = [plan["records"][plan["donor_indices"][i]] for i in indices]
        pools = []
        expected_pools = {f"pool-{position:02d}.json" for position in range(len(indices))}
        if {item.name for item in path.glob("pool-*.json")} != expected_pools:
            raise ValueError("word final batch lost exact rollout-file coverage")
        for position, (case, donor) in enumerate(zip(records, donors, strict=True)):
            filename = path / f"pool-{position:02d}.json"
            saved = json.loads(filename.read_text())
            seed = plan["recipe"]["seed"] + number * plan["recipe"]["batch_clips"] + position
            if (
                saved.get("record") != case
                or saved.get("donor") != donor
                or saved.get("seed") != seed
            ):
                raise ValueError("word final pool changed input, donor or rollout seed")
            pools.append(saved["candidates"])
            assets.append(artifact(filename))
        paired = paired_inputs(records, donors, pools, tokenizer)
        rewards = verified_rewards(paired, path, plan["recipe"]["seed"] + number)
        saved = path / "rewards.json"
        if (
            json.loads(saved.read_text()) != rewards
            or step["Terra_qualities"] != rewards["qualities"]
        ):
            raise ValueError("word final ledger differs from actual calibrated Terra rewards")
        assets.extend(artifact(path / name) for name in ("blinded-inputs.json", "rewards.json"))
        for request in sorted(path.glob("request-*")):
            assets.extend(
                artifact(request / name)
                for name in ("request.json", "started.json", "response.json", "scores.json")
            )
    return assets


def checked_checkpoints(plan, result, directory, plan_sha, steps):
    from safetensors import safe_open

    scope_descriptor = plan["model_plan"]["visual_scope"]
    scope_path = ROOT / scope_descriptor["path"]
    if artifact(scope_path) != scope_descriptor:
        raise ValueError("word final model lost original public visual-shape scope")
    scope = json.loads(scope_path.read_text())
    shapes = {"vision." + row["parameter"]: row["shape"] for row in scope["parameters"]}
    if set(shapes) != set(plan["model_plan"]["expected_visual_gradient_names"]):
        raise ValueError("word final checkpoint shape scope changed")
    boundaries = {
        step["training_examples"]: step["optimizer_updates"]
        for index, step in enumerate(steps)
        if index + 1 == len(steps) or steps[index + 1]["epoch"] != step["epoch"]
    }
    expected_examples = [examples for examples, updates in boundaries.items() if updates > 0]
    if len(result["checkpoints"]) != len(expected_examples):
        raise ValueError("word final result changed planned epoch-checkpoint coverage")
    assets = []
    for descriptor, examples in zip(result["checkpoints"], expected_examples, strict=True):
        path = directory / f"visual-after-{examples:06d}.safetensors"
        if artifact(path) != descriptor:
            raise ValueError("word epoch checkpoint differs from actual planned weights")
        commit_path = path.with_suffix(".json")
        optimizer_path = path.with_name(path.stem + "-optimizer.safetensors")
        commit = json.loads(commit_path.read_text())
        if (
            commit.get("complete") is not True
            or commit.get("visual") != descriptor
            or commit.get("optimizer") != artifact(optimizer_path)
            or commit.get("training_examples") != examples
            or commit.get("optimizer_updates") != boundaries[examples]
            or commit.get("committed_batch_RNG") is not True
        ):
            raise ValueError("word checkpoint lacks complete weight/Adam/RNG provenance")
        for saved_path in (path, optimizer_path):
            with safe_open(str(saved_path), framework="pt", device="cpu") as saved:
                metadata = saved.metadata()
                if metadata.get("plan_sha256") != plan_sha or metadata.get("update") != str(
                    examples
                ):
                    raise ValueError("word checkpoint belongs to another plan or exposure count")
                if saved_path == path:
                    if (
                        metadata.get("optimizer_updates") != str(boundaries[examples])
                        or metadata.get("fixed_winner_checkpoint")
                        != plan["model_plan"]["reference"]["translator_checkpoint_sha256"]
                        or set(saved.keys()) != set(shapes)
                    ):
                        raise ValueError("word checkpoint changed original350-tensor topology")
                    for name, shape in shapes.items():
                        value = saved.get_tensor(name)
                        if (
                            list(value.shape) != shape
                            or value.dtype != torch.float32
                            or not torch.isfinite(value).all()
                        ):
                            raise ValueError(
                                "word checkpoint contains altered shapes or invalid values"
                            )
        assets.extend((artifact(path), artifact(commit_path), artifact(optimizer_path)))
    final = directory / f"visual-after-{plan['planned_training_examples']:06d}.safetensors"
    if artifact(final) != result["visual"]:
        raise ValueError("word evaluation cannot substitute an earlier epoch or recovery snapshot")
    return assets


def checked_final(training_path, result_path):
    from transformers import ByT5Tokenizer

    training_path, result_path = training_path.resolve(), result_path.resolve()
    if not result_path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word final result must remain in owned checkpoints/")
    plan = checked_training(training_path)
    terminal(result_path)
    result = json.loads(result_path.read_text())
    require_final(result, plan, sha256(training_path))
    directory = result_path.parent
    if (directory / "failure.json").exists() or (directory / "stopped.json").exists():
        raise ValueError("word evaluation cannot consume a stopped training lineage")
    run_path, steps_path = directory / "run.json", directory / "steps.jsonl"
    run = json.loads(run_path.read_text())
    if (
        run.get("plan_sha256") != sha256(training_path)
        or run.get("initial_visual_sha256") != plan["vision_initialization_sha256"]
        or run.get("fixed_backbones") != plan["model_plan"]["initial_reference_hashes"]
        or run.get("recipe") != plan["recipe"]
        or run.get("unique_training_clips") != len(plan["records"])
        or run.get("training_examples") != 0
        or run.get("optimizer_updates") != 0
        or run.get("trainable_parameters") != 43260672
    ):
        raise ValueError("word final result has a different original-value training run")
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    verify_steps(plan, steps, result)
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    assets = [artifact(run_path), artifact(steps_path)]
    assets.extend(verified_batch_evidence(plan, directory, steps, tokenizer))
    assets.extend(checked_checkpoints(plan, result, directory, sha256(training_path), steps))
    return plan, result, assets
