"""Terra-only cross-source ROI curriculum, with complete clips and a fixed final model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import normalize, sha256
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import identity, now, write
from probe_roi_broad_semantic import ROOT, checked_data, training_rows
from probe_roi_broad_semantic import checked as checked_proof_plan
from probe_roi_semantic_learning import (
    POLICY as NUMERICAL_POLICY,
)
from probe_roi_semantic_learning import (
    checked_candidates,
    fixed_hashes,
    load_clip,
    load_student,
    training_negative,
)
from probe_tiny_rgb_capacity import state_digest
from roi_semantic_student import visual_gradients
from terra_short_resume import RecoveryPacer
from terra_training_recovery import CheckpointRequest, artifact, save_checkpoint, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import verified_comparison
from train_roi_semantic import append, donor_for, score_batch, synchronize_visual

PROOF_PLAN = ROOT / "runs/roi-broad-semantic-gradient-v1/plan.json"
PROOF = ROOT / "checkpoints/roi-broad-semantic-gradient-v1/qualify/qualification.json"
RECIPE = {
    "stages": [
        {"name": "two_to_three_words", "maximum_words": 3, "epochs": 4},
        {"name": "two_to_six_words_with_short_replay", "maximum_words": 6, "epochs": 4},
    ],
    "batch_clips": 4,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "caption_weight": 0,
    "seed": 20261017,
    "augmentation": "none; test cross-source breadth and short-to-long exposure first",
    "checkpoint_selection": "fixed final; stage-boundary checkpoints are recovery only",
    "additional_candidate": "next different-caption training identity from complete318 inventory",
    "reward": "actual calibrated Terra-medium0-10; minimize expected10-quality",
    "zero_contrast": "valid zero gradient; skip optimizer only if whole batch is tied",
    "maximum_pairs_per_request": 16,
    "minimum_request_start_gap_seconds": 60,
    "automatic_retries": 0,
    "WER_is_diagnostic_only": True,
    "new_downloads": 0,
    "promotion_authorized": False,
}


def batches_for(rows, recipe=RECIPE):
    if (
        type(recipe["batch_clips"]) is not int
        or recipe["batch_clips"] < 1
        or not recipe["stages"]
        or any(
            type(stage["epochs"]) is not int or stage["epochs"] < 1 for stage in recipe["stages"]
        )
        or any(
            row["roi_training_allowed"] is not True
            or row["roi_role"] != "training"
            or row["original_row"]["training_allowed"] is not True
            or row["original_row"]["role"] != "training"
            for row in rows
        )
    ):
        raise ValueError("broad ROI curriculum requires real training rows and positive work")
    batches = []
    for stage_index, stage in enumerate(recipe["stages"]):
        eligible = [
            i
            for i, row in enumerate(rows)
            if 2 <= len(normalize(row["original_row"]["reference"])) <= stage["maximum_words"]
        ]
        if not eligible:
            raise ValueError("broad ROI curriculum has an empty training stage")
        for epoch in range(stage["epochs"]):
            order = list(eligible)
            random.Random(recipe["seed"] + stage_index * 10000 + epoch).shuffle(order)
            for offset in range(0, len(order), recipe["batch_clips"]):
                batches.append(
                    {
                        "stage": stage["name"],
                        "epoch": epoch,
                        "training_record_indices": order[offset : offset + recipe["batch_clips"]],
                    }
                )
    return batches


def verify_coverage(plan, examples, observed_order, saved):
    expected = [i for batch in plan["batches"] for i in batch["training_record_indices"]]
    if examples != plan["planned_training_examples"] or observed_order != expected:
        raise ValueError("broad ROI complete curriculum order/exposures changed")
    if set(observed_order) != set(range(len(plan["records"]))) or not saved:
        raise ValueError("broad ROI final model does not cover all training clips")


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
        or len(proof.get("comparisons", [])) != len(plan["records"])
        or proof.get("vision_initialization_sha256") != plan["original_initialization_sha256"]
    ):
        raise ValueError(
            "broad ROI optimization requires actual complete cross-source gradient proof"
        )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    candidates_path = ROOT / proof["candidates"]["path"]
    if artifact(candidates_path) != proof["candidates"]:
        raise ValueError("broad ROI proof candidate bytes changed")
    candidates = checked_candidates(candidates_path, PROOF_PLAN, plan, tokenizer)
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
            raise ValueError("broad ROI proof lost real-input/reward/all116-gradient coverage")
    return plan, proof, tokenizer


def specification():
    proof_plan, proof, tokenizer = qualified()
    data_plan, data, rows = checked_data()
    if training_rows(data_plan, data) != rows:
        raise ValueError("broad ROI training data roles changed")
    for row in rows:
        tokens = tokenizer(row["original_row"]["reference"], truncation=False).input_ids
        if not tokens or len(tokens) > 512 or tokens[-1] != 1:
            raise ValueError("broad ROI cannot truncate any complete training caption")
    batches = batches_for(rows)
    exposures = sum(len(batch["training_record_indices"]) for batch in batches)
    if exposures != 1668 or len(batches) != 420:
        raise ValueError("broad ROI fixed curriculum inventory/exposure counts changed")
    source = dict(proof_plan["source_sha256"])
    source["tools/train_roi_broad_semantic.py"] = sha256(Path(__file__))
    return {
        "schema": "umi-studio-roi-broad-semantic-training/1",
        "recipe": RECIPE,
        "qualified_plan": artifact(PROOF_PLAN),
        "qualified_proof": artifact(PROOF),
        "proof_comparisons": [
            artifact(PROOF.with_name(f"comparison-{i:02d}.json"))
            for i in range(len(proof_plan["records"]))
        ],
        "data_plan": artifact(ROOT / proof_plan["data_plan"]["path"]),
        "data": artifact(ROOT / proof_plan["data"]["path"]),
        "records": rows,
        "batches": batches,
        "unique_training_clips": len(rows),
        "planned_training_examples": exposures,
        "maximum_optimizer_updates": len(batches),
        "short_stage_clips": sum(len(normalize(r["original_row"]["reference"])) <= 3 for r in rows),
        "model_plan": proof_plan,
        "vision_initialization_sha256": proof["vision_initialization_sha256"],
        "source_sha256": source,
        "primary_quality_gate": "unchanged reserved34 paired Terra gate, then broader confirmation",
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("broad ROI optimizer/data/qualified implementation or recipe changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("broad ROI training requires a fresh owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage(NUMERICAL_POLICY["trainable_parameters"] * 12 + 1024**2)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "stage": "broad_ROI_training_plan",
                "plan": artifact(path),
                "unique_clips": len(plan["records"]),
                "exposures": plan["planned_training_examples"],
                "maximum_updates": plan["maximum_optimizer_updates"],
            }
        ),
        flush=True,
    )


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
        raise ValueError("broad ROI training requires CPU2/50% Metal")
    duty = WinnerDutyCycle(torch, budget)
    gpu, _ = load_student(plan["model_plan"])
    cpu, tokenizer = load_student(plan["model_plan"])
    cpu.requires_grad_(False).eval()
    if any(
        state_digest(m.vision.state_dict()) != plan["vision_initialization_sha256"]
        for m in (gpu, cpu)
    ):
        raise ValueError("broad ROI training must start from the qualified visual initialization")
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
    fixed = plan["model_plan"]["initial_reference_hashes"]
    examples = updates = 0
    saved, observed_order = [], []
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "initial_visual_sha256": plan["vision_initialization_sha256"],
            "fixed_backbones": fixed,
            "recipe": RECIPE,
            "training_examples": 0,
            "optimizer_updates": 0,
            "unique_training_clips": len(plan["records"]),
        },
    )

    def save(*, recovery=False):
        storage(NUMERICAL_POLICY["trainable_parameters"] * 12 + 1024**2, output=output)
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
        for batch_number, batch in enumerate(plan["batches"]):
            if stop.requested:
                raise RuntimeError("owner-requested batch-boundary checkpoint stop")
            indices = batch["training_record_indices"]
            rows = [plan["records"][i] for i in indices]
            count = len(rows)
            directory = output / f"batch-{batch_number:04d}"
            storage(output=output)
            directory.mkdir()
            duty.run(lambda: synchronize_visual(gpu, cpu))
            pools = []
            for position, (i, row) in enumerate(zip(indices, rows, strict=True)):
                clip, mask = load_clip(row)
                seed = RECIPE["seed"] + batch_number * RECIPE["batch_clips"] + position
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
                del clip, mask
                negative = training_negative(row, donor_for(plan["records"], i), tokenizer)
                if negative["tokens"] not in [c["tokens"] for c in pool]:
                    pool.append(negative)
                pools.append(pool)
                write(
                    directory / f"pool-{position:02d}.json",
                    {
                        "record": row,
                        "seed": seed,
                        "candidates": pool,
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": "broad_ROI_rollout",
                            "batch": batch_number,
                            "position": position,
                            "training_examples": examples,
                        }
                    ),
                    flush=True,
                )
            rewards = score_batch(rows, pools, directory, pacer, RECIPE["seed"] + batch_number)
            gpu.zero_grad(set_to_none=True)
            measured, contrasting = [], 0
            for row, pool, quality in zip(rows, pools, rewards, strict=True):
                if len(set(quality)) < 2:
                    measured.append(
                        {
                            "expected_terra_error": float(10 - quality[0]),
                            "zero_reward_contrast": True,
                        }
                    )
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
                del backward, clip, mask, labels
            norm = None
            if contrasting:

                def update():
                    gradients = visual_gradients(gpu)
                    del gradients
                    result = torch.nn.utils.clip_grad_norm_(
                        gpu.trainable_parameters().values(),
                        RECIPE["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    gpu.zero_grad(set_to_none=True)
                    return float(result)

                norm = duty.run(update)
                updates += 1
            examples += len(rows)
            observed_order.extend(indices)
            append(
                output,
                "steps.jsonl",
                {
                    **batch,
                    "batch": batch_number,
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    "contrasting_clips": contrasting,
                    "Terra_qualities": rewards,
                    "before_update_risk": measured,
                    "gradient_norm_before_clip": norm,
                    "process_peak_bytes": peak_bytes(),
                    "resources": duty.receipt(),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "broad_ROI_batch_complete",
                        "batch": batch_number,
                        "training_examples": examples,
                        "optimizer_updates": updates,
                    }
                ),
                flush=True,
            )
            if updates and (
                batch_number + 1 == len(plan["batches"])
                or plan["batches"][batch_number + 1]["stage"] != batch["stage"]
            ):
                saved.append(artifact(save()))
        verify_coverage(plan, examples, observed_order, saved)
        if any(fixed_hashes(m) != fixed for m in (gpu, cpu)):
            raise ValueError("broad ROI training changed a frozen C5 backbone")
        checked(path)
        write(
            output / "result.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "training_examples": examples,
                "unique_training_clips": len(set(observed_order)),
                "optimizer_updates": updates,
                "maximum_optimizer_updates": plan["maximum_optimizer_updates"],
                "visual": saved[-1],
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
        raise ValueError("broad ROI training output must be fresh and private")
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
