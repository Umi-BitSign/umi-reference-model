"""Fixed complete-word Terra curriculum, gated on actual full C5 risk qualification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_word_semantic_model import PARAMETERS, load_clip, load_student
from c5_word_training import (
    RECIPE,
    STORAGE,
    batches_for,
    donor_indices,
    save_checkpoint,
    score_batch,
    storage,
    synchronize_visual,
    training_records,
    update_visual,
    verify_coverage,
)
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import identity, now, write
from probe_c5_word_semantic import checked as checked_proof_plan
from probe_c5_word_semantic import checked_qualification, input_evidence
from probe_roi_semantic_learning import fixed_hashes
from probe_tiny_rgb_capacity import state_digest
from terra_short_resume import RecoveryPacer
from terra_training_recovery import CheckpointRequest, artifact, verify_snapshot
from terra_word_semantic_protocol import training_case, training_negative

ROOT = Path(__file__).resolve().parents[1]
PROOF_PLAN = ROOT / "runs/c5-word-semantic-gradient-v1/plan.json"
PROOF = ROOT / "checkpoints/c5-word-semantic-gradient-v1/qualify/qualification.json"
GATES = {
    "checkpoint": "fixed final after all856 word exposures; no partial-checkpoint selection",
    "word": "all96 reserved signer-disjoint inputs, reproduced original224px C5 controls first",
    "word_metric": "unchanged two-pass calibrated Terra semantic gate; WER diagnostic only",
    "sentence": "unchanged34 reserved paired C5 comparison; word gain alone is insufficient",
    "confirmation": (
        "broader source/signer-disjoint semantic evidence and native serving compatibility"
    ),
    "later_phrase_training": "requires original-video native112 input and full-risk qualification",
    "promotion_authorized": False,
}


def qualified():
    plan = checked_proof_plan(PROOF_PLAN)
    parent = PROOF.parent.parent / "complete.json"
    terminal(parent)
    proof = checked_qualification(PROOF, PROOF_PLAN, plan)
    if json.loads(parent.read_text()) != {
        "complete": True,
        "plan": artifact(PROOF_PLAN),
        "qualification": artifact(PROOF),
        "optimizer_updates": 0,
        "automatic_retries": 0,
        "promotion_authorized": False,
    }:
        raise ValueError("word optimization requires complete actual parent/child qualification")
    return plan, proof


def specification():
    from transformers import ByT5Tokenizer

    proof_plan, proof = qualified()
    binding, _ = input_evidence()
    records = training_records(binding)
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / proof_plan["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    for case in records:
        row = training_case(case)
        tokens = tokenizer(row["reference"], truncation=False).input_ids
        if (
            not tokens
            or tokens[-1] != 1
            or len(tokens) > RECIPE["candidate_horizon"]
            or tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            != row["reference"]
        ):
            raise ValueError("word curriculum cannot truncate or alter any complete caption")
    batches = batches_for(records)
    exposures = sum(len(batch["training_record_indices"]) for batch in batches)
    if exposures != 856 or len(batches) != 216:
        raise ValueError("word four-pass inventory/exposure counts changed")
    source = dict(proof_plan["source_sha256"])
    for name in (
        "train_c5_word_semantic.py",
        "c5_word_training.py",
        "train_tiny_rgb_feature_winner_full.py",
        "c5_experiment_sources.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-C5-word-semantic-training/1",
        "recipe": RECIPE,
        "storage_policy": STORAGE,
        "evaluation_gates": GATES,
        "qualified_plan": artifact(PROOF_PLAN),
        "qualified_proof": artifact(PROOF),
        "qualified_parent": artifact(PROOF.parent.parent / "complete.json"),
        "proof_comparisons": proof["comparisons"],
        "verified_inputs": proof_plan["verified_inputs"],
        "records": records,
        "donor_indices": donor_indices(records),
        "batches": batches,
        "unique_training_clips": len(records),
        "planned_training_examples": exposures,
        "maximum_optimizer_updates": len(batches),
        "model_plan": proof_plan,
        "vision_initialization_sha256": proof["vision_initialization_sha256"],
        "source_sha256": source,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("word training changed qualified values, complete data or frozen recipe")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word training requires a fresh owned plan")
    plan = specification()
    storage(STORAGE["checkpoint_pair_reserve_bytes"])
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "training_clips": len(plan["records"]),
                "exposures": plan["planned_training_examples"],
                "maximum_updates": plan["maximum_optimizer_updates"],
            }
        ),
        flush=True,
    )


def append_step(output, value):
    with output.joinpath("steps.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rng_states():
    return {"cpu": torch.get_rng_state().clone(), "mps": torch.mps.get_rng_state().clone()}


def mask_digest(model):
    return state_digest(
        {name: getattr(model.vision, name).mask_token for name in ("hands", "face")}
    )


def fit(path, output, plan):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import candidate_labels
    from evaluate_tiny_rgb_feature_winner import peak_bytes
    from terra_teacher_candidates import TrainingRolloutBudget
    from terra_visual_learning import backward_candidate_risk, candidate_pool

    budget = CertificationComputeWindow(ROOT / plan["model_plan"]["resource_policy"])
    if (
        budget.cpu_threads() != 2
        or budget.duty_fraction() != 0.5
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("word optimization requires CPU2 and Metal50%")
    duty = WinnerDutyCycle(torch, budget)
    gpu, _ = load_student(plan["model_plan"])
    cpu, tokenizer = load_student(plan["model_plan"])
    cpu.requires_grad_(False).eval()
    initial = plan["vision_initialization_sha256"]
    if any(state_digest(model.vision.state_dict()) != initial for model in (gpu, cpu)):
        raise ValueError("word optimizer must start from original qualified public visual values")
    frozen_masks = mask_digest(gpu)
    fixed = plan["model_plan"]["initial_reference_hashes"]
    torch.manual_seed(RECIPE["seed"])
    torch.mps.manual_seed(RECIPE["seed"])
    duty.run(lambda: gpu.to("mps"))
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    pacer, stop = RecoveryPacer(), CheckpointRequest()
    examples = updates = 0
    saved, observed_order = [], []
    committed_rng = duty.run(rng_states)
    state_is_committed = True
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "initial_visual_sha256": initial,
            "fixed_backbones": fixed,
            "frozen_mask_sha256": frozen_masks,
            "recipe": RECIPE,
            "unique_training_clips": len(plan["records"]),
            "training_examples": 0,
            "optimizer_updates": 0,
            "trainable_parameters": PARAMETERS,
        },
    )

    def save(*, recovery=False):
        if not state_is_committed:
            raise ValueError("word checkpoint cannot claim an uncommitted optimizer mutation")
        return duty.run(
            lambda: save_checkpoint(
                gpu,
                optimizer,
                output,
                path,
                plan["model_plan"]["reference"]["translator_checkpoint_sha256"],
                examples,
                updates,
                committed_rng,
                recovery=recovery,
            )
        )

    try:
        for batch_number, batch in enumerate(plan["batches"]):
            if stop.requested:
                raise RuntimeError("owner-requested batch-boundary checkpoint stop")
            indices = batch["training_record_indices"]
            records = [plan["records"][i] for i in indices]
            donors = [plan["records"][plan["donor_indices"][i]] for i in indices]
            directory = output / f"batch-{batch_number:04d}"
            storage(output=output)
            directory.mkdir()
            duty.run(lambda: synchronize_visual(gpu, cpu))
            pools = []
            for position, (case, donor) in enumerate(zip(records, donors, strict=True)):
                clip, mask = load_clip(case)
                seed = RECIPE["seed"] + batch_number * RECIPE["batch_clips"] + position
                pool = candidate_pool(
                    cpu,
                    tokenizer,
                    clip,
                    mask,
                    training_case(case)["reference"],
                    seed=seed,
                    maximum_tokens=RECIPE["candidate_horizon"],
                    budget_factory=TrainingRolloutBudget,
                )
                del clip, mask
                negative = training_negative(case, donor, tokenizer)
                if negative["tokens"] not in [candidate["tokens"] for candidate in pool]:
                    pool.append(negative)
                pools.append(pool)
                write(
                    directory / f"pool-{position:02d}.json",
                    {"record": case, "donor": donor, "seed": seed, "candidates": pool},
                )
                print(
                    json.dumps(
                        {
                            "stage": "word_rollout",
                            "batch": batch_number,
                            "position": position,
                            "training_examples": examples,
                        }
                    ),
                    flush=True,
                )
            rewards = score_batch(
                records, donors, pools, tokenizer, directory, pacer, RECIPE["seed"] + batch_number
            )
            gpu.zero_grad(set_to_none=True)
            measured, contrasting = [], 0
            count = len(records)
            for case, pool, quality in zip(records, pools, rewards, strict=True):
                if len(set(quality)) < 2:
                    measured.append(
                        {
                            "expected_terra_error": float(10 - quality[0]),
                            "zero_reward_contrast": True,
                        }
                    )
                    continue
                contrasting += 1
                clip, mask = load_clip(case)
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

                def mark_uncommitted():
                    nonlocal state_is_committed
                    state_is_committed = False

                def update():
                    if mask_digest(gpu) != frozen_masks:
                        raise ValueError("word training changed a frozen unused mask token")
                    return update_visual(
                        gpu,
                        optimizer,
                        plan["model_plan"]["expected_visual_gradient_names"],
                        before_mutation=mark_uncommitted,
                    )

                norm = duty.run(update)
                updates += 1
            state_is_committed = False
            examples += len(records)
            observed_order.extend(indices)
            append_step(
                output,
                {
                    **batch,
                    "batch": batch_number,
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    "training_record_indices": indices,
                    "contrasting_clips": contrasting,
                    "Terra_qualities": rewards,
                    "before_update_risk": measured,
                    "gradient_norm_before_clip": norm,
                    "process_peak_bytes": peak_bytes(),
                    "resources": duty.receipt(),
                },
            )
            committed_rng = duty.run(rng_states)
            state_is_committed = True
            print(
                json.dumps(
                    {
                        "stage": "word_batch_complete",
                        "batch": batch_number,
                        "training_examples": examples,
                        "optimizer_updates": updates,
                    }
                ),
                flush=True,
            )
            if updates and (
                batch_number + 1 == len(plan["batches"])
                or plan["batches"][batch_number + 1]["epoch"] != batch["epoch"]
            ):
                saved.append(artifact(save()))
        verify_coverage(plan, observed_order, examples, saved)
        if (
            duty.run(lambda: fixed_hashes(gpu)) != fixed
            or fixed_hashes(cpu) != fixed
            or duty.run(lambda: mask_digest(gpu)) != frozen_masks
        ):
            raise ValueError("word training changed original frozen C5 values")
        checked(path)
        write(
            output / "result.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "training_examples": examples,
                "optimizer_updates": updates,
                "visual": saved[-1],
                "checkpoints": saved,
                "visual_initialization_sha256": initial,
                "visual_final_sha256": duty.run(lambda: state_digest(gpu.vision.state_dict())),
                "all_fixed_backbones_and_unused_masks_unchanged": True,
                "semantic_translation_evaluated": False,
                "evaluation_gates": GATES,
                "promotion_authorized": False,
                "resources": duty.receipt(),
            },
        )
    except Exception as error:
        gpu.zero_grad(set_to_none=True)
        recovery, recovery_error = None, None
        if updates and state_is_committed:
            try:
                recovery = artifact(save(recovery=True))
            except Exception as checkpoint_error:
                recovery_error = str(checkpoint_error)
        write(
            output / "stopped.json",
            {
                "training_examples": examples,
                "optimizer_updates": updates,
                "error": str(error),
                "recovery": recovery,
                "recovery_error": recovery_error,
                "recovery_is_committed_prefix": bool(recovery),
                "uncommitted_optimizer_mutation_possible": not state_is_committed,
                "automatic_retries": 0,
                "automatic_continuations": 0,
            },
        )
        raise


def run(path, output):
    import cv2
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("word training requires its bound two-thread decoder")
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word training requires a fresh owned output")
    storage(STORAGE["checkpoint_pair_reserve_bytes"])
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            fit(path, output, plan)
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.freeze and not args.plan:
        freeze(args.freeze)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("choose --freeze [--output] or --plan with --output")


if __name__ == "__main__":
    main()
