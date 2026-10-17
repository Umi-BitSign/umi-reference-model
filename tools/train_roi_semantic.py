"""Bounded on-policy Terra-only updates of the numerically qualified64px ROI vision."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import normalize, sha256
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import identity, now, write
from probe_roi_semantic_learning import (
    POLICY as PROOF_POLICY,
)
from probe_roi_semantic_learning import (
    checked as checked_proof_plan,
)
from probe_roi_semantic_learning import (
    checked_candidates,
    comparison,
    fixed_hashes,
    load_clip,
    load_student,
    training_negative,
)
from probe_tiny_rgb_capacity import state_digest
from roi_appearance_protocol import ROOT
from roi_semantic_student import visual_gradients
from terra_short_resume import RecoveryPacer
from terra_training_recovery import CheckpointRequest, artifact, save_checkpoint, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import verified_comparison

PROOF_PLAN = ROOT / "runs/roi-semantic-gradient-v2/plan.json"
PROOF = ROOT / "checkpoints/roi-semantic-gradient-v2/qualify/qualification.json"
RECIPE = {
    "epochs": 1,
    "batch_clips": 4,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "seed": 20261017,
    "augmentation": "none; isolate semantic-through-vision updates",
    "checkpoint_after_training_examples": [32, 64, 88],
    "checkpoint_selection": "fixed final one-pass checkpoint, no intermediate quality selection",
    "candidate_policy": PROOF_POLICY["candidate_generation"],
    "additional_candidate": "next different-text training identity in the fixed full inventory",
    "reward": "actual calibrated Terra medium integer0-10, expected error10-quality",
    "zero_contrast": "valid zero gradient; record it, skip optimizer only if whole batch is tied",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "automatic_retries": 0,
    "WER_is_diagnostic_only": True,
    "new_downloads": 0,
    "promotion_authorized": False,
}


def qualified():
    from transformers import ByT5Tokenizer

    plan = checked_proof_plan(PROOF_PLAN)
    terminal(PROOF)
    proof = json.loads(PROOF.read_text())
    if (
        proof.get("complete") is not True
        or proof.get("numerics_passed") is not True
        or proof.get("plan_sha256") != sha256(PROOF_PLAN)
        or proof.get("optimizer_updates") != 0
        or proof.get("new_Terra_requests") != 0
        or proof.get("pure_Terra_objective") is not True
        or proof.get("all_fixed_weights_unchanged") is not True
        or proof.get("trainable_tensors") != 116
        or len(proof.get("comparisons", [])) != 3
        or plan["policy"] != PROOF_POLICY
    ):
        raise ValueError("ROI optimization requires the complete actual Terra-only gradient proof")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    candidates = checked_candidates(ROOT / proof["candidates"]["path"], PROOF_PLAN, plan, tokenizer)
    if artifact(ROOT / proof["candidates"]["path"]) != proof["candidates"]:
        raise ValueError("ROI proof's actual scored candidate bytes changed")
    for index, row in enumerate(proof["comparisons"]):
        measured = json.loads(PROOF.with_name(f"comparison-{index:02d}.json").read_text())
        worst = verified_comparison(measured["comparison"], 116)
        if (
            measured["cache"] != plan["records"][index]["cache"]
            or measured["qualities"] != candidates["qualities"][index]
            or row["record"] != measured["cache"]
            or row["actual_Terra_qualities"] != measured["qualities"]
            or row["worst_relative_gradient_error"] != worst
            or any(
                g["cpu_norm"] <= 0 or g["actual_norm"] <= 0
                for g in measured["comparison"]["gradient_differences"]
            )
        ):
            raise ValueError("ROI proof lost actual input/reward/all-gradient coverage")
    return plan, proof


def records_for_training(data, reserved):
    rows = [row for row in data["records"] if row["roi_training_allowed"]]
    if len(rows) != 88 or sum(row["frames"] for row in rows) != 7865:
        raise ValueError("ROI semantic pilot cannot change the complete88-clip training inventory")
    identities = set()
    for row in rows:
        source = row["original_row"]
        key = (source["source"], source["example_id"])
        if (
            row["roi_role"] != "training"
            or row["roi_training_allowed"] is not True
            or source["role"] != "training"
            or source["training_allowed"] is not True
            or source["signer"] in reserved
            or key in identities
        ):
            raise ValueError(
                "ROI semantic learning cannot use reserved, duplicate or diagnostic clips"
            )
        identities.add(key)
    return rows


def donor_for(records, index):
    reference = normalize(records[index]["original_row"]["reference"])
    for offset in range(1, len(records)):
        candidate = records[(index + offset) % len(records)]
        if normalize(candidate["original_row"]["reference"]) != reference:
            return candidate
    raise ValueError("ROI training inventory has no distinct-caption negative")


def training_rewards(paired, scores):
    if len(scores) != len(paired["items"]) or any(
        type(v) is not int or not 0 <= v <= 10 for v in scores
    ):
        raise ValueError("ROI training needs complete integer rewards from Terra")
    if any(not c["range"][0] <= scores[c["index"]] <= c["range"][1] for c in paired["controls"]):
        raise ValueError("ROI training judge failed original calibration controls")
    result = [[scores[i] for i in case["indices"]] for case in paired["cases"]]
    if any(q[0] != 10 for q in result):
        raise ValueError("ROI training judge rejected an exact human anchor")
    return result


def specification():
    proof_plan, proof = qualified()
    fit = json.loads((ROOT / proof_plan["ROI_initializer"]["plan"]["path"]).read_text())
    data = json.loads((ROOT / proof_plan["data"]["path"]).read_text())
    rows = records_for_training(
        data, fit["reserved_STEM_participants"] + fit["ROI_diagnostic_participants"]
    )
    order = list(range(len(rows)))
    random.Random(RECIPE["seed"]).shuffle(order)
    source = dict(proof_plan["source_sha256"])
    source["tools/train_roi_semantic.py"] = sha256(Path(__file__))
    return {
        "schema": "umi-studio-roi-semantic-training/1",
        "recipe": RECIPE,
        "qualified_plan": artifact(PROOF_PLAN),
        "qualified_proof": artifact(PROOF),
        "proof_comparisons": [
            artifact(PROOF.with_name(f"comparison-{i:02d}.json")) for i in range(3)
        ],
        "records": rows,
        "order": order,
        "planned_training_examples": len(rows),
        "maximum_optimizer_updates": math.ceil(len(rows) / 4),
        "model_plan": proof_plan,
        "vision_initialization_sha256": proof["vision_initialization_sha256"],
        "source_sha256": source,
        "primary_quality_gate": "unchanged34-clip Terra gate, then broader confirmation",
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("ROI semantic optimizer/data/proof/source recipe changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("ROI semantic training requires a fresh owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage(PROOF_POLICY["trainable_parameters"] * 12 + 1024**2)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"stage": "ROI_semantic_training_plan", "plan": artifact(path), "examples": 88}),
        flush=True,
    )


def append(output, name, value):
    with output.joinpath(name).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def score_batch(records, pools, output, pacer, seed):
    paired = comparison(records, pools)
    write(output / "blinded-inputs.json", paired)
    indices = list(range(len(paired["items"])))
    random.Random(seed).shuffle(indices)
    scores = [None] * len(indices)
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        result = pacer.score(
            [paired["items"][i] for i in chosen], output / f"request-{offset // 16:03d}"
        )
        for i, score in zip(chosen, result["scores"], strict=True):
            scores[i] = score
    rewards = training_rewards(paired, scores)
    write(output / "rewards.json", {"scores": scores, "qualities": rewards})
    return rewards


@torch.no_grad()
def synchronize_visual(gpu, cpu):
    source, destination = dict(gpu.vision.named_parameters()), dict(cpu.vision.named_parameters())
    if source.keys() != destination.keys() or len(source) != 116:
        raise ValueError("ROI rollout/trainer visual topology differs")
    for name, value in source.items():
        if value.shape != destination[name].shape or value.dtype != destination[name].dtype:
            raise ValueError("ROI rollout tensor shape/dtype differs")
        destination[name].copy_(value.detach().cpu())


def fit(path, output, plan):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import candidate_labels
    from evaluate_tiny_rgb_feature_winner import peak_bytes
    from terra_teacher_candidates import TeacherTextCache, teacher_augmented_pool
    from terra_visual_learning import backward_candidate_risk

    budget = CertificationComputeWindow(ROOT / plan["model_plan"]["resource_policy"])
    if (
        budget.cpu_threads() != 2
        or budget.duty_fraction() != 0.5
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("ROI semantic training requires CPU2/50% Metal")
    duty = WinnerDutyCycle(torch, budget)
    gpu, _ = load_student(plan["model_plan"])
    cpu, tokenizer = load_student(plan["model_plan"])
    cpu.requires_grad_(False).eval()
    if any(
        state_digest(m.vision.state_dict()) != plan["vision_initialization_sha256"]
        for m in (gpu, cpu)
    ):
        raise ValueError("ROI semantic training did not start from qualified values")
    torch.manual_seed(RECIPE["seed"])
    torch.mps.manual_seed(RECIPE["seed"])
    duty.run(lambda: gpu.to("mps"))
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    cache = TeacherTextCache(path.parent / "teacher-text-cache")
    pacer, stop = RecoveryPacer(), CheckpointRequest()
    examples = updates = 0
    saved = []
    fixed = plan["model_plan"]["initial_reference_hashes"]
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "initial_visual_sha256": plan["vision_initialization_sha256"],
            "fixed_backbones": fixed,
            "recipe": RECIPE,
            "training_examples": 0,
            "optimizer_updates": 0,
        },
    )

    def save(*, recovery=False):
        storage(PROOF_POLICY["trainable_parameters"] * 12 + 1024**2, output=output)
        return duty.run(
            lambda: save_checkpoint(
                gpu,
                optimizer,
                output,
                path,
                plan["model_plan"]["reference"]["translator_checkpoint_sha256"],
                examples,
                updates,
                recovery=recovery,
            )
        )

    try:
        for batch_start in range(0, len(plan["order"]), RECIPE["batch_clips"]):
            if stop.requested:
                raise RuntimeError("owner-requested batch-boundary checkpoint stop")
            indices = plan["order"][batch_start : batch_start + RECIPE["batch_clips"]]
            count = len(indices)
            rows = [plan["records"][i] for i in indices]
            directory = output / f"batch-{batch_start // 4:03d}"
            storage(output=output)
            directory.mkdir()
            duty.run(lambda: synchronize_visual(gpu, cpu))
            pools = []
            for position, (i, row) in enumerate(zip(indices, rows, strict=True)):
                clip, mask = load_clip(row)
                seed = RECIPE["seed"] + batch_start + position
                pool = teacher_augmented_pool(
                    cpu,
                    tokenizer,
                    clip,
                    mask,
                    row["original_row"]["reference"],
                    pair=row["original_row"]["original_pair"],
                    seed=seed,
                    checkpoint_sha=plan["model_plan"]["reference"]["translator_checkpoint_sha256"],
                    cache=cache,
                )
                negative = training_negative(row, donor_for(plan["records"], i), tokenizer)
                if negative["tokens"] not in [c["tokens"] for c in pool]:
                    pool.append(negative)
                pools.append(pool)
                write(
                    directory / f"pool-{position:02d}.json",
                    {"record": row, "seed": seed, "candidates": pool},
                )
                print(
                    json.dumps(
                        {
                            "stage": "ROI_semantic_rollout",
                            "training_examples_before_batch": examples,
                            "position": position,
                            "candidates": len(pool),
                        }
                    ),
                    flush=True,
                )
            rewards = score_batch(rows, pools, directory, pacer, RECIPE["seed"] + batch_start)
            gpu.zero_grad(set_to_none=True)
            measured = []
            contrasting = 0
            for row, pool, quality in zip(rows, pools, rewards, strict=True):
                if len(set(quality)) < 2:
                    measured.append({"expected_terra_error": 0.0, "zero_reward_contrast": True})
                    continue
                contrasting += 1
                clip, mask = load_clip(row)
                labels = candidate_labels(pool)

                def backward(clip=clip, mask=mask, labels=labels, quality=quality, count=count):
                    result = backward_candidate_risk(
                        gpu,
                        clip,
                        mask,
                        labels,
                        quality,
                        alpha=RECIPE["risk_alpha"],
                        caption_weight=0,
                        divisor=count,
                    )
                    del result["gold_logits"]
                    return result

                measured.append(duty.run(backward))
            gradient_norm = None
            if contrasting:

                def update():
                    nonlocal updates
                    gradients = visual_gradients(gpu)
                    del gradients
                    norm = torch.nn.utils.clip_grad_norm_(
                        gpu.trainable_parameters().values(),
                        RECIPE["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    gpu.zero_grad(set_to_none=True)
                    updates += 1
                    return float(norm)

                gradient_norm = duty.run(update)
            examples += len(rows)
            step = {
                "batch": batch_start // 4,
                "training_examples": examples,
                "optimizer_updates": updates,
                "training_record_indices": indices,
                "contrasting_clips": contrasting,
                "Terra_qualities": rewards,
                "before_update_risk": measured,
                "gradient_norm_before_clip": gradient_norm,
                "process_peak_bytes": peak_bytes(),
                "resources": duty.receipt(),
            }
            append(output, "steps.jsonl", step)
            print(
                json.dumps(
                    {
                        "stage": "ROI_semantic_batch_complete",
                        "training_examples": examples,
                        "optimizer_updates": updates,
                        "contrasting_clips": contrasting,
                    }
                ),
                flush=True,
            )
            if examples in RECIPE["checkpoint_after_training_examples"] and updates:
                saved.append(artifact(save()))
        if examples != 88 or len(plan["order"]) != len(set(plan["order"])):
            raise ValueError("ROI semantic training coverage changed")
        if any(fixed_hashes(m) != fixed for m in (gpu, cpu)):
            raise ValueError("ROI semantic training changed a frozen C5 backbone")
        checked(path)
        write(
            output / "result.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "training_examples": examples,
                "optimizer_updates": updates,
                "visual": saved[-1] if saved else None,
                "checkpoints": saved,
                "visual_initialization_sha256": plan["vision_initialization_sha256"],
                "visual_final_sha256": state_digest(gpu.vision.state_dict()),
                "all_fixed_backbones_unchanged": True,
                "semantic_translation_evaluated": False,
                "primary_metric": "Terra semantic0-10; WER diagnostic only",
                "promotion_authorized": False,
                "resources": duty.receipt(),
            },
        )
    except Exception as error:
        recovery = artifact(save(recovery=True)) if updates else None
        write(
            output / "stopped.json",
            {
                "training_examples": examples,
                "optimizer_updates": updates,
                "error": str(error),
                "recovery": recovery,
                "automatic_retries": 0,
            },
        )
        raise


def run(path, output):
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("ROI semantic training output must be fresh and private")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            fit(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("choose --freeze, optionally --output, or --plan --output")


if __name__ == "__main__":
    main()
