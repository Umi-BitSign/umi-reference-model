"""Fresh small-input vision trained against Terra medium's semantic error."""

from __future__ import annotations

import argparse
import copy
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
    return plan, pairs


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from tiny_rgb_feature_winner_protocol import select_cases
    from train_tiny_rgb_capacity_cpu import storage
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


def run(path, output):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import compare, state_digest
    from probe_tiny_rgb_feature_winner import gradients
    from safetensors.torch import save_file
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
    from train_tiny_rgb_capacity_cpu import storage
    from train_tiny_rgb_feature_winner_full import lr_factor, optimizer_payload, order_key
    from transformers import ByT5Tokenizer

    plan, pairs = checked_plan(path)
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
    gpu = duty.run(lambda: copy.deepcopy(cpu).to("mps"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "initial_visual_sha256": initial_visual,
                "fixed_backbones": fixed,
                "fresh_visual_weights": True,
                "old_visual_weights_loaded": False,
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
            lambda video=video, mask=mask, labels=labels, quality=quality: backward_candidate_risk(
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
    cpu.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    versions = {name: p._version for name, p in gpu.named_parameters() if not p.requires_grad}
    examples, optimizer_updates, decisions = 0, 0, []
    for epoch in range(1, RECIPE["epochs"] + 1):
        ordered = sorted(pairs, key=lambda pair: order_key(pair, epoch))
        offset = 0
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
            qualities, judge = score_pools(batch, pools, receipt_path)
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
            storage(215000000)
            if examples not in RECIPE["checks_after_updates"]:
                continue
            checkpoint = output / f"visual-after-{examples:06d}.safetensors"
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in gpu.trainable_parameters().items()
                },
                str(checkpoint),
                metadata={
                    "plan_sha256": sha256(path),
                    "update": str(examples),
                    "optimizer_updates": str(optimizer_updates),
                    "fixed_winner_checkpoint": reference["translator_checkpoint_sha256"],
                },
            )
            values, metadata = optimizer_payload(
                optimizer,
                gpu.trainable_parameters(),
                {"cpu": torch.get_rng_state(), "mps": torch.mps.get_rng_state()},
            )
            optimizer_path = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
            save_file(
                values,
                str(optimizer_path),
                metadata={"plan_sha256": sha256(path), "update": str(examples), **metadata},
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
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output_dir:
        run(args.plan, args.output_dir)
    else:
        parser.error("provide --freeze or --plan with --output-dir")
