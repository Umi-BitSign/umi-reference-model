"""Fresh small-input vision trained against Terra medium's semantic error."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/tiny-rgb-feature-winner-full-v1/plan.json"
RECIPE = {
    "seed": 317,
    "capacity": "large",
    "short_side": 192,
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 512,
    "epochs": 4,
    "batch_clips": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "risk_weight": 1.0,
    "caption_weight": 0.1,
    "checks_after_updates": [128, 512, 2048, 8192, 16384, 30092],
    "checkpoint_counter_unit": "training_examples",
    "generated_candidates": 2,
    "gold_in_candidate_pool": True,
    "old_visual_weights_reused": False,
    "latent_distillation": False,
    "feature_distillation": False,
    "judge_model": "gpt-5.6-terra",
    "judge_reasoning_effort": "medium",
    "error": "10 minus Terra semantic quality",
    "augmentation": "none",
    "storage_policy": "60GiB physical floor plus128MiB headroom and remaining2GiB output reserve",
    "training_rollouts": "EOS or explicit512-token horizon; no invented termination token",
}


def sources():
    from benchmark_fleurs_dev import sha256
    from train_tiny_rgb_feature_winner_full import sources as parent_sources

    result = parent_sources()
    for name in (
        "train_terra_visual.py",
        "terra_visual_learning.py",
        "terra_quality_training.py",
        "semantic_quality_judge.py",
        "probe_jeff_terra_quality.py",
        "evaluate_terra_visual.py",
        "terra_training_storage.py",
        "terra_training_recovery.py",
    ):
        result[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_feature_winner_protocol import select_cases
    from train_tiny_rgb_feature_winner_full import checked_plan as parent_plan

    plan = json.loads(path.read_text())
    parent, pairs = parent_plan(PARENT)
    if (
        plan["schema"] != "umi-studio-terra-visual-training/1"
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["parent_plan_sha256"] != sha256(PARENT)
        or plan["proof_cases"] != select_cases(pairs)
        or any(
            plan[key] != parent[key]
            for key in (
                "reference",
                "evaluation",
                "winner_control",
                "historical_controls",
                "gates",
                "resource_policy",
                "resource_policy_sha256",
                "runtime_binding",
                "native_pairs_sha256",
            )
        )
        or plan["planned_training_examples"] != len(pairs) * RECIPE["epochs"]
        or json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())[
            "plan_sha256"
        ]
        != sha256(path)
    ):
        raise ValueError("Terra training plan/source/data/benchmark bindings changed")
    if "resume" in plan:
        from terra_training_recovery import checked_resume

        checked_resume(plan)
    return plan, pairs


def freeze(path, *, resume_plan=None, resume_checkpoint=None, qualification=None):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage
    from tiny_rgb_feature_winner_protocol import select_cases
    from train_tiny_rgb_feature_winner_full import checked_plan as parent_plan

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Terra training requires a fresh runs/ plan")
    parent, pairs = parent_plan(PARENT)
    plan = {
        "schema": "umi-studio-terra-visual-training/1",
        "recipe": RECIPE,
        "source_sha256": sources(),
        "parent_plan_sha256": sha256(PARENT),
        "proof_cases": select_cases(pairs),
        "training_examples": len(pairs),
        "planned_training_examples": len(pairs) * RECIPE["epochs"],
        "owner_approved_terra_training_cost": True,
        "training_only_judge_requests": True,
        "miner_modified": False,
        "promotion_authorized": False,
        "confirmation_decoded": False,
        **{
            key: parent[key]
            for key in (
                "reference",
                "evaluation",
                "winner_control",
                "historical_controls",
                "gates",
                "resource_policy",
                "resource_policy_sha256",
                "runtime_binding",
                "native_pairs_sha256",
            )
        },
    }
    if resume_plan is not None or resume_checkpoint is not None:
        from terra_training_recovery import make_resume

        if any(value is None for value in (resume_plan, resume_checkpoint, qualification)):
            raise ValueError("recovery freeze needs source plan, checkpoint and qualification")
        plan["resume"] = make_resume(resume_plan, resume_checkpoint, qualification, plan)
    storage(len(json.dumps(plan).encode()) + 1024**2)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_examples": len(pairs),
                "proof_cases": len(plan["proof_cases"]),
            }
        ),
        flush=True,
    )


def append(output, name, row):
    with output.joinpath(name).open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def score_pools(pairs, pools, output):
    from terra_quality_training import score_with_terra

    rows = [
        {"reference": pair["raw"]["reference"], "candidate": candidate["text"]}
        for pair, pool in zip(pairs, pools, strict=True)
        for candidate in pool
    ]
    result = score_with_terra(rows, output)
    qualities, offset = [], 0
    for pool in pools:
        values = result["scores"][offset : offset + len(pool)]
        if values[0] != 10:
            raise RuntimeError("Terra did not give the exact human caption ten; no update allowed")
        qualities.append(values)
        offset += len(pool)
    return qualities, result


def checked_qualification(receipt, path, initial_visual, fixed, proof_cases):
    from benchmark_fleurs_dev import sha256
    from probe_jeff_terra_quality import terra_payload
    from semantic_quality_judge import parse_terra_scores
    from tiny_rgb_capacity_protocol import verified_comparison

    result = json.loads(receipt.read_text())
    if (
        result.get("complete") is not True
        or result.get("numerics_passed") is not True
        or result.get("plan_sha256") != sha256(path)
        or result.get("initial_visual_sha256") != initial_visual
        or result.get("fixed_backbones") != fixed
        or result.get("optimizer_updates") != 0
        or result.get("caption_weight_during_proof") != 0
    ):
        raise ValueError("Terra qualification is incomplete or bound to different initial tensors")
    comparisons_path = receipt.with_name("qualification-comparisons.jsonl")
    inputs_path = receipt.with_name("qualification-inputs.jsonl")
    response_path = receipt.parent / "qualification-terra/response.json"
    if any(
        sha256(file) != result[field]
        for file, field in (
            (comparisons_path, "comparisons_sha256"),
            (inputs_path, "inputs_sha256"),
            (response_path, "judge_response_sha256"),
        )
    ):
        raise ValueError("Terra qualification evidence changed")
    inputs = json.loads(inputs_path.read_text())
    if inputs["pairs"] != proof_cases:
        raise ValueError("Terra qualification used different training identities")
    rows = [
        {"reference": pair["raw"]["reference"], "candidate": candidate["text"]}
        for pair, pool in zip(proof_cases, inputs["pools"], strict=True)
        for candidate in pool
    ]
    if json.loads(
        (receipt.parent / "qualification-terra/request.json").read_text()
    ) != terra_payload(rows):
        raise ValueError("Terra qualification request differs from its frozen training texts")
    scores = parse_terra_scores(
        json.loads(response_path.read_text()), [f"pair_{index:02d}" for index in range(len(rows))]
    )
    flattened = [score for group in inputs["quality"] for score in group]
    if flattened != [scores[f"pair_{index:02d}"] for index in range(len(rows))]:
        raise ValueError("Terra qualification gradients used different judge scores")
    comparisons = [json.loads(line) for line in comparisons_path.read_text().splitlines()]
    identities = [(pair["raw"]["source"], pair["raw"]["example_id"]) for pair in proof_cases]
    if [(row["source"], row["example_id"]) for row in comparisons] != identities:
        raise ValueError("Terra qualification gradient coverage differs")
    for row, quality in zip(comparisons, inputs["quality"], strict=True):
        if (
            row["caption_weight"] != 0
            or row["terra_gradient_coefficient_l1"] <= 0
            or row["terra_quality"] != quality
        ):
            raise ValueError("Terra qualification lacks actual Terra-only gradients")
        verified_comparison(row["comparison"], 90)
    return result


def run(path, output, *, qualification_only=False, qualification=None, restore_only=False):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import compare, state_digest
    from probe_tiny_rgb_feature_winner import gradients
    from terra_training_recovery import (
        checked_resume,
        restore_checkpoint,
        save_checkpoint,
        skip_epoch,
    )
    from terra_training_storage import storage
    from terra_visual_learning import (
        backward_candidate_risk,
        candidate_labels,
        candidate_pool,
        synchronize_visual_policy,
        training_pixels,
    )
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_tiny_rgb_feature_winner_full import lr_factor, order_key
    from transformers import ByT5Tokenizer

    plan, pairs = checked_plan(path)
    resume_paths = checked_resume(plan) if "resume" in plan else None
    if restore_only and resume_paths is None:
        raise ValueError("local restore-only mode requires a frozen recovery plan")
    if resume_paths is not None:
        if qualification is not None or qualification_only:
            raise ValueError("recovery uses only its frozen zero-update qualification")
        qualification = resume_paths["qualification"]
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("Terra training needs a new checkpoint directory")
    storage(215000000)
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Terra visual training requires qualified Metal")
    duty = WinnerDutyCycle(torch, budget)
    language, reference = load_winner_language()
    encoder = load_frozen_winner_encoder(reference)
    cpu = RawFeatureWinnerStudent(encoder, language).eval()
    initial_visual = state_digest(cpu.trainable_parameters())
    fixed = {
        "language": state_digest(language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    # Legacy weight_norm caches a nonleaf tensor until its first forward.
    # Construct the second policy from pinned tensors; never deepcopy that cache.
    gpu_language, gpu_reference = load_winner_language()
    if gpu_reference != reference:
        raise RuntimeError("CPU and Metal policies loaded different winning references")
    gpu = RawFeatureWinnerStudent(load_frozen_winner_encoder(reference), gpu_language).eval()
    if state_digest(gpu.trainable_parameters()) != initial_visual or fixed != {
        "language": state_digest(gpu.language.state_dict()),
        "encoder": state_digest(gpu.semantic_encoder.state_dict()),
    }:
        raise RuntimeError("independently loaded CPU and Metal policy tensors differ")
    gpu = duty.run(lambda: gpu.to("mps"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "initial_visual_sha256": initial_visual,
                "fixed_backbones": fixed,
                "fresh_visual_weights": resume_paths is None,
                "old_visual_weights_loaded": False,
                "terra_checkpoint_resumed": resume_paths is not None,
                "resume": plan.get("resume"),
                "loss": "expected Terra semantic error plus 0.1 human caption CE",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"stage": "fresh_model_loaded", "initial_visual_sha256": initial_visual}),
        flush=True,
    )

    def check_fixed(model):
        for module in (model.language, model.semantic_encoder):
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("Terra training changed the fixed-backbone gradient boundary")

    def check_fixed_hashes(model):
        if fixed != {
            "language": state_digest(model.language.state_dict()),
            "encoder": state_digest(model.semantic_encoder.state_dict()),
        }:
            raise RuntimeError("Terra training changed fixed backbone values")

    if qualification is not None:
        proof_path = resume_paths["qualification_plan"] if resume_paths is not None else path
        checked_qualification(qualification, proof_path, initial_visual, fixed, plan["proof_cases"])
        (output / "qualification-reuse.json").write_text(
            json.dumps(
                {
                    "receipt": str(qualification),
                    "receipt_sha256": sha256(qualification),
                    "fresh_initial_visual_sha256": initial_visual,
                    "optimizer_updates": 0,
                    "qualification_plan": str(proof_path),
                    "qualification_plan_sha256": sha256(proof_path),
                    "current_plan_sha256": sha256(path),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps({"stage": "verified_qualification_reused", "optimizer_updates": 0}),
            flush=True,
        )
    else:
        # Qualify the actual Terra-only gradient before admitting any optimizer update.
        proof_pairs = plan["proof_cases"]
        proof_data = [training_pixels(pair, RECIPE) for pair in proof_pairs]
        pools = [
            candidate_pool(
                cpu, tokenizer, video, mask, pair["raw"]["reference"], seed=RECIPE["seed"] + index
            )
            for index, (pair, (video, mask)) in enumerate(zip(proof_pairs, proof_data, strict=True))
        ]
        qualities, judge = score_pools(proof_pairs, pools, output / "qualification-terra")
        append(
            output,
            "qualification-inputs.jsonl",
            {"pairs": proof_pairs, "pools": pools, "quality": qualities, "judge": judge},
        )
        comparisons = []
        for pair, (video, mask), pool, quality in zip(
            proof_pairs, proof_data, pools, qualities, strict=True
        ):
            labels = candidate_labels(video, pool)
            cpu.zero_grad(set_to_none=True)
            gpu.zero_grad(set_to_none=True)
            before = backward_candidate_risk(
                cpu, video, mask, labels, quality, alpha=RECIPE["risk_alpha"], caption_weight=0
            )
            cpu_gradients = gradients(cpu)
            after = duty.run(
                lambda video=video,
                mask=mask,
                labels=labels,
                quality=quality: backward_candidate_risk(
                    gpu, video, mask, labels, quality, alpha=RECIPE["risk_alpha"], caption_weight=0
                )
            )
            metal_gradients = gradients(gpu)
            measured = compare(
                cpu_gradients,
                metal_gradients,
                before["loss"],
                after["loss"],
                before["gold_logits"],
                after["gold_logits"],
            )
            record = {
                "source": pair["raw"]["source"],
                "example_id": pair["raw"]["example_id"],
                "terra_quality": quality,
                "caption_weight": 0,
                "terra_gradient_coefficient_l1": before["terra_gradient_coefficient_l1"],
                "comparison": measured,
            }
            append(output, "qualification-comparisons.jsonl", record)
            verified_comparison(measured, 90)
            if before["terra_gradient_coefficient_l1"] <= 0:
                raise RuntimeError("Terra-only qualification has no actual semantic gradient")
            comparisons.append({key: value for key, value in record.items() if key != "comparison"})
            print(
                json.dumps(
                    {
                        "stage": "terra_gradient_qualified",
                        "example_id": record["example_id"],
                        "worst_relative_gradient_error": max(
                            row["relative_l2_error"] for row in measured["gradient_differences"]
                        ),
                    }
                ),
                flush=True,
            )
            del cpu_gradients, metal_gradients
            cpu.zero_grad(set_to_none=True)
            gpu.zero_grad(set_to_none=True)
            torch.mps.empty_cache()
        check_fixed_hashes(cpu)
        check_fixed_hashes(gpu)
        if state_digest(gpu.trainable_parameters()) != initial_visual:
            raise RuntimeError("zero-update qualification changed the fresh visual model")
        (output / "qualification.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "numerics_passed": True,
                    "plan_sha256": sha256(path),
                    "initial_visual_sha256": initial_visual,
                    "fixed_backbones": fixed,
                    "comparisons_sha256": sha256(output / "qualification-comparisons.jsonl"),
                    "inputs_sha256": sha256(output / "qualification-inputs.jsonl"),
                    "judge_response_sha256": judge["response_sha256"],
                    "real_cases": comparisons,
                    "optimizer_updates": 0,
                    "caption_weight_during_proof": 0,
                    "actual_terra_scores_drive_all_90_gradient_tensors": True,
                    "resource_receipt": duty.receipt(),
                },
                indent=2,
            )
            + "\n"
        )
    if qualification_only:
        print(json.dumps({"stage": "qualification_complete", "optimizer_updates": 0}), flush=True)
        return
    cpu.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    (output / "steps.jsonl").touch(exist_ok=False)
    versions = {name: p._version for name, p in gpu.named_parameters() if not p.requires_grad}
    examples, optimizer_updates, decisions = 0, 0, []
    if resume_paths is not None:

        def restore_rng(cpu_rng, mps_rng):
            torch.mps.set_rng_state(mps_rng)
            torch.set_rng_state(cpu_rng)

        examples, optimizer_updates = duty.run(
            lambda: restore_checkpoint(
                plan,
                resume_paths,
                gpu.trainable_parameters(),
                optimizer,
                initial_visual=initial_visual,
                fixed=fixed,
                restore_rng=restore_rng,
            )
        )
        # Fresh fixed tensors remain unchanged; only qualified Terra-trained vision resumes.
        check_fixed_hashes(gpu)
        if "source_decisions" in resume_paths:
            decisions = json.loads(resume_paths["source_decisions"].read_text())
            (output / "open-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
        restored = {
            "stage": "checkpoint_restored",
            "training_examples": examples,
            "optimizer_updates": optimizer_updates,
            "visual_sha256": state_digest(gpu.trainable_parameters()),
            "source_visual_file_sha256": plan["resume"]["visual"]["sha256"],
            "source_optimizer_file_sha256": plan["resume"]["optimizer"]["sha256"],
            "terra_requests_during_restore": 0,
            "restore_only": restore_only,
            "resource_receipt": duty.receipt(),
        }
        (output / "resume-restored.json").write_text(json.dumps(restored, indent=2) + "\n")
        print(json.dumps(restored), flush=True)
        if restore_only:
            return
    skip_examples = examples
    for epoch in range(1, RECIPE["epochs"] + 1):
        ordered = sorted(pairs, key=lambda pair: order_key(pair, epoch))
        offset, skip_examples = skip_epoch(skip_examples, len(ordered))
        while offset < len(ordered):
            next_check = next(value for value in RECIPE["checks_after_updates"] if value > examples)
            count = min(RECIPE["batch_clips"], len(ordered) - offset, next_check - examples)
            batch = ordered[offset : offset + count]
            synchronize_visual_policy(gpu, cpu)
            policy_sha = state_digest(gpu.trainable_parameters())
            if (
                state_digest(
                    {
                        name: dict(cpu.named_parameters())[name]
                        for name in gpu.trainable_parameters()
                    }
                )
                != policy_sha
            ):
                raise RuntimeError("CPU rollout and GPU optimizer policies differ")
            data = [training_pixels(pair, RECIPE) for pair in batch]
            pools = [
                candidate_pool(
                    cpu,
                    tokenizer,
                    video,
                    mask,
                    pair["raw"]["reference"],
                    seed=RECIPE["seed"] + examples + index,
                )
                for index, (pair, (video, mask)) in enumerate(zip(batch, data, strict=True))
            ]
            receipt_path = output / "terra" / f"batch-{optimizer_updates + 1:06d}"
            try:
                qualities, judge = score_pools(batch, pools, receipt_path)
            except RuntimeError as error:
                # Scoring precedes zero_grad/backward/step. Save only completed updates,
                # then stop without retrying or touching the gateway's configuration.
                failure = {
                    "training_complete": False,
                    "training_examples": examples,
                    "optimizer_updates": optimizer_updates,
                    "failed_batch": optimizer_updates + 1,
                    "error": str(error),
                    "automatic_retries": 0,
                    "failed_batch_optimizer_applied": False,
                }
                (output / "scoring-stop.json").write_text(json.dumps(failure, indent=2) + "\n")
                if optimizer_updates:
                    checkpoint = save_checkpoint(
                        gpu,
                        optimizer,
                        output,
                        path,
                        reference["translator_checkpoint_sha256"],
                        examples,
                        optimizer_updates,
                        recovery=True,
                    )
                    failure["recovery_checkpoint"] = str(checkpoint.relative_to(ROOT))
                (output / "scoring-stop.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise
            append(
                output,
                "candidate-pools.jsonl",
                {
                    "optimizer_update": optimizer_updates + 1,
                    "policy_sha256": policy_sha,
                    "identities": [
                        {
                            key: pair["raw"][key]
                            for key in ("source", "example_id", "role", "training_allowed")
                        }
                        for pair in batch
                    ],
                    "pools": pools,
                    "qualities": qualities,
                    "judge": judge,
                },
            )
            optimizer.zero_grad(set_to_none=True)
            rate = RECIPE["learning_rate"] * lr_factor(examples, plan["planned_training_examples"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            measurements = []
            for (video, mask), pool, quality in zip(data, pools, qualities, strict=True):
                labels = candidate_labels(video, pool)
                result = duty.run(
                    lambda video=video,
                    mask=mask,
                    labels=labels,
                    quality=quality,
                    count=count: backward_candidate_risk(
                        gpu,
                        video,
                        mask,
                        labels,
                        quality,
                        alpha=RECIPE["risk_alpha"],
                        caption_weight=RECIPE["caption_weight"],
                        divisor=count,
                    )
                )
                result.pop("gold_logits")
                measurements.append(result)
                torch.mps.empty_cache()

            def step():
                norms = [
                    float(value.grad.norm()) if value.grad is not None else math.nan
                    for value in gpu.trainable_parameters().values()
                ]
                if len(norms) != 90 or any(not math.isfinite(value) for value in norms):
                    raise RuntimeError("Terra optimizer lacks all 90 finite gradients")
                check_fixed(gpu)
                if any(
                    p._version != versions[name]
                    for name, p in gpu.named_parameters()
                    if not p.requires_grad
                ):
                    raise RuntimeError("Terra optimizer changed a fixed-backbone tensor")
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        gpu.trainable_parameters().values(), RECIPE["gradient_clip"]
                    )
                )
                if not math.isfinite(norm):
                    raise RuntimeError("Terra gradient norm is nonfinite")
                optimizer.step()
                return {
                    "gradient_norm": norm,
                    "gradient_tensors": len(norms),
                    "nonzero_gradient_tensors": sum(value > 0 for value in norms),
                }

            gradient_receipt = duty.run(step)
            examples += count
            optimizer_updates += 1
            offset += count
            append(
                output,
                "steps.jsonl",
                {
                    "training_examples": examples,
                    "optimizer_update": optimizer_updates,
                    "epoch": epoch,
                    "batch_clips": count,
                    "learning_rate": rate,
                    "measurements": measurements,
                    "terra_receipt": str(receipt_path.relative_to(ROOT)),
                    **gradient_receipt,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "terra_training",
                        "training_examples": examples,
                        "optimizer_updates": optimizer_updates,
                        "mean_expected_terra_error": sum(
                            row["expected_terra_error"] for row in measurements
                        )
                        / count,
                        "mean_caption_loss": sum(row["caption_loss"] for row in measurements)
                        / count,
                        **gradient_receipt,
                        **duty.receipt(),
                    }
                ),
                flush=True,
            )
            storage(215000000, output=output)
            if examples not in RECIPE["checks_after_updates"]:
                continue
            checkpoint = save_checkpoint(
                gpu,
                optimizer,
                output,
                path,
                reference["translator_checkpoint_sha256"],
                examples,
                optimizer_updates,
            )
            check_fixed_hashes(gpu)
            import subprocess

            evaluation = output / f"cpu-evaluation-{examples:06d}"
            with output.joinpath(f"cpu-evaluation-{examples:06d}.log").open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools/evaluate_terra_visual.py"),
                        "--plan",
                        str(path),
                        "--checkpoint",
                        str(checkpoint),
                        "--update",
                        str(examples),
                        "--output-dir",
                        str(evaluation),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            decision = json.loads((evaluation / "decision.json").read_text())
            if decision["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError("Terra evaluation loaded a different policy")
            decisions.append(decision)
            (output / "open-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    checked_plan(path)
    check_fixed_hashes(gpu)
    (output / "decision.json").write_text(
        json.dumps(
            {
                "training_complete": True,
                "training_examples": examples,
                "optimizer_updates": optimizer_updates,
                "evaluations": decisions,
                "miner_modified": False,
                "promotion_authorized": False,
                "complete_composite_verified": False,
                "confirmation_decoded": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--restore-only", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--qualification-only", action="store_true")
    stage.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(
            args.freeze,
            resume_plan=args.resume_plan,
            resume_checkpoint=args.resume_checkpoint,
            qualification=args.qualification,
        )
    elif args.plan and args.output_dir:
        run(
            args.plan,
            args.output_dir,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
            restore_only=args.restore_only,
        )
    else:
        parser.error("provide --freeze or --plan with --output-dir")
