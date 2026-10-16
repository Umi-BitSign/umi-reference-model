"""Fresh-C5 fixed-candidate Terra risk, with more local steps and fresh semantic gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_terra_short_curriculum import ROOT
from prepare_terra_short_curriculum import checked as checked_data
from terra_frozen_short import (
    RECIPE,
    REWARDS,
    batches,
    checked_rewards,
    key,
    pair,
    proof_cases,
    rate,
)
from train_terra_short_curriculum import append, artifact, evaluate
from train_terra_short_curriculum import specification as short_spec

DATA = ROOT / "runs/terra-short-curriculum-v1/plan.json"
CONTROL = ROOT / "runs/terra-short-curriculum-C5-control-v1/receipt.json"


def specification():
    data = checked_data(DATA)
    rewards = checked_rewards(data)
    result = short_spec(DATA, CONTROL)
    schedule = list(batches(data, rewards))
    repeats = RECIPE["local_steps_per_outer_batch"]
    result.update(
        {
            "schema": "umi-studio-terra-frozen-short-training/1",
            "recipe": RECIPE,
            "historical_training_rewards": artifact(REWARDS),
            "outer_training_exposures": result["planned_training_examples"],
            "planned_training_examples": result["planned_training_examples"] * repeats,
            "planned_optimizer_updates": len(schedule) * repeats,
            "checks_after_examples": [n * repeats for n in result["checks_after_examples"]],
            "integration_proof_identities": [
                list(key(r["data_row"])) for r in proof_cases(rewards)
            ],
            "qualification_scope": (
                "existing six-case component proof plus two authenticated replay cases"
            ),
            "frozen_training_rewards_are_not_fresh_on_policy_judgments": True,
        }
    )
    for name in ("terra_frozen_short.py", "train_terra_frozen_short.py"):
        result["source_sha256"][f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("frozen short recipe, source, data or baseline changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("frozen short resource policy changed")
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("frozen short plan requires a fresh owned runs directory")
    storage()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan_sha256": sha256(path), "updates": plan["planned_optimizer_updates"]}),
        flush=True,
    )


def checked_proof(receipt, path, initial, fixed, rewards):
    from tiny_rgb_capacity_protocol import verified_comparison

    result = json.loads(receipt.read_text())
    terminal = json.loads(receipt.with_name("exit.json").read_text())
    worker = json.loads(receipt.with_name("worker.json").read_text())
    comparisons = receipt.with_name("qualification-comparisons.jsonl")
    rows = [json.loads(s) for s in comparisons.read_text().splitlines()]
    expected = proof_cases(rewards)
    if (
        result.get("complete") is not True
        or result.get("actual_Terra_only_gradients") is not True
        or result.get("plan_sha256") != sha256(path)
        or result.get("initial_policy_digest") != initial
        or result.get("fixed_backbones") != fixed
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or result.get("caption_weight_during_proof") != 0
        or result.get("inputs") != expected
        or result.get("comparisons_sha256") != sha256(comparisons)
        or terminal.get("exit_code") != 0
        or terminal.get("pid") != worker.get("pid")
        or len(rows) != len(expected)
    ):
        raise ValueError("frozen short actual gradient integration proof differs")
    for row, case in zip(rows, expected, strict=True):
        if (
            row["identity"] != list(key(case["data_row"]))
            or row["quality"] != case["quality"]
            or row["caption_weight"] != 0
            or row["terra_gradient_coefficient_l1"] <= 0
        ):
            raise ValueError("frozen short proof changed real inputs or lacks semantic gradients")
        verified_comparison(row["comparison"], 8)
    return artifact(receipt)


def run(path, output, *, qualification_only=False, qualification=None):
    from terra_training_storage import storage
    from train_terra_c5_tail import gpu_lease

    plan = checked(path)
    if qualification_only == (qualification is not None):
        raise ValueError("choose a new qualification or an existing completed qualification")
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("frozen short training needs a fresh owned checkpoint directory")
    storage(215000000)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            fit(
                path,
                output,
                plan,
                qualification_only=qualification_only,
                qualification=qualification,
            )
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def fit(path, output, plan, *, qualification_only, qualification):
    import torch
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import (
        candidate_labels,
        fixed_hashes,
        gradients,
        load_model,
        native_memory,
        padded_memory,
        synchronize_policy,
    )
    from probe_tiny_rgb_capacity import compare, state_digest
    from terra_short_resume import RecoveryPacer
    from terra_training_recovery import CheckpointRequest, save_checkpoint
    from terra_training_storage import storage
    from terra_visual_learning import backward_candidate_risk
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_terra_c5_tail import checked_qualification
    from transformers import ByT5Tokenizer

    data = checked_data(DATA)
    rewards = checked_rewards(data)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if budget.duty_fraction() != 0.5 or budget.cpu_threads() != 2:
        raise ValueError("frozen short training requires 50% GPU duty and two CPU threads")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(RECIPE["seed"])
    duty, stop = WinnerDutyCycle(torch, budget), CheckpointRequest()
    cpu, reference = load_model()
    encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
    initial, fixed = state_digest(cpu.policy_parameters()), fixed_hashes(cpu, encoder)
    actual = {
        "language": state_digest(cpu.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    if reference != plan["reference"] or actual != plan["initial_reference_hashes"]:
        raise ValueError("frozen short experiment must start at original C5")
    component_path = ROOT / plan["qualified_component_plan"]["path"]
    checked_qualification(
        ROOT / plan["qualified_component_proof"]["path"],
        component_path,
        initial,
        fixed,
        json.loads(component_path.read_text()),
    )
    gpu, gpu_reference = load_model()
    if (
        gpu_reference != reference
        or state_digest(gpu.policy_parameters()) != initial
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("frozen short CPU/Metal initialization differs or Metal unavailable")
    gpu = duty.run(lambda: gpu.to("mps"))
    if state_digest(gpu.policy_parameters()) != initial or fixed_hashes(gpu, encoder) != fixed:
        raise ValueError("frozen short Metal transfer changed baseline weights")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    for row in rewards:
        candidate_labels(row["pool"])
        if any(
            tokenizer.decode(
                c["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            != c["text"]
            for c in row["pool"]
        ):
            raise ValueError("frozen reward tokens no longer decode to the graded text")
    cache, cache_bytes = {}, 0

    def memory(row):
        nonlocal cache_bytes
        identity_key = key(row["data_row"])
        if identity_key not in cache:
            value = native_memory(encoder, pair(row), reference["translator_checkpoint_sha256"])
            cost = value.numel() * value.element_size()
            if cache_bytes + cost > RECIPE["native_memory_CPU_cache_budget_bytes"]:
                raise RuntimeError("frozen native-memory CPU cache reached its 128-MiB bound")
            cache[identity_key] = padded_memory(value)
            cache_bytes += sum(v.numel() * v.element_size() for v in cache[identity_key])
            if cache_bytes > RECIPE["native_memory_CPU_cache_budget_bytes"]:
                raise RuntimeError("padded frozen memory exceeded the CPU cache bound")
        return cache[identity_key]

    if qualification_only:
        cases = proof_cases(rewards)
        for case in cases:
            features, mask = memory(case)
            labels = candidate_labels(case["pool"])
            cpu.zero_grad(set_to_none=True)
            gpu.zero_grad(set_to_none=True)
            before = backward_candidate_risk(
                cpu,
                features,
                mask,
                labels,
                case["quality"],
                alpha=RECIPE["risk_alpha"],
                caption_weight=0,
            )
            old = gradients(cpu)

            def metal_case(features=features, mask=mask, labels=labels, case=case):
                result = backward_candidate_risk(
                    gpu,
                    features,
                    mask,
                    labels,
                    case["quality"],
                    alpha=RECIPE["risk_alpha"],
                    caption_weight=0,
                )
                return result, gradients(gpu)

            after, new = duty.run(metal_case)
            comparison = compare(
                old, new, before["loss"], after["loss"], before["gold_logits"], after["gold_logits"]
            )
            worst = verified_comparison(comparison, 8)
            if before["terra_gradient_coefficient_l1"] <= 0:
                raise RuntimeError("frozen short actual semantic risk has no gradient")
            append(
                output,
                "qualification-comparisons.jsonl",
                {
                    "identity": key(case["data_row"]),
                    "quality": case["quality"],
                    "caption_weight": 0,
                    "terra_gradient_coefficient_l1": before["terra_gradient_coefficient_l1"],
                    "comparison": comparison,
                },
            )
            print(
                json.dumps(
                    {"stage": "frozen_short_qualification", "worst_relative_gradient_error": worst}
                ),
                flush=True,
            )
            del old, new, before, after
            cpu.zero_grad(set_to_none=True)
            gpu.zero_grad(set_to_none=True)
            torch.mps.empty_cache()
        if (
            state_digest(cpu.policy_parameters()) != initial
            or state_digest(gpu.policy_parameters()) != initial
            or fixed_hashes(cpu, encoder) != fixed
            or fixed_hashes(gpu, encoder) != fixed
        ):
            raise RuntimeError("frozen short qualification changed C5 weights")
        checked(path)
        write(
            output / "qualification.json",
            {
                "complete": True,
                "actual_Terra_only_gradients": True,
                "plan_sha256": sha256(path),
                "initial_policy_digest": initial,
                "fixed_backbones": fixed,
                "inputs": cases,
                "comparisons_sha256": sha256(output / "qualification-comparisons.jsonl"),
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "caption_weight_during_proof": 0,
                "resource_receipt": duty.receipt(),
            },
        )
        return

    proof = checked_proof(qualification, path, initial, fixed, rewards)
    cpu.requires_grad_(False)
    versions = {name: p._version for name, p in gpu.frozen_parameters().items()}
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "initial_visual_sha256": initial,
            "fixed_backbones": fixed,
            "initial_full_C5_hashes": actual,
            "integration_proof": proof,
            "historical_training_rewards": plan["historical_training_rewards"],
            "on_policy_training_claimed": False,
            "training_Terra_requests": 0,
            "promotion_authorized": False,
            "miner_or_codex_lb_inspected": False,
        },
    )
    control = json.loads(CONTROL.read_text())["predictions"]
    development = [r for r in data["records"] if r["role"] == "development"]
    examples = updates = outer = inner = 0
    step_inflight = False
    decisions, saved = [], {}
    reason = "planned_endpoint"
    pacer = RecoveryPacer()

    def save():
        if not updates:
            return None
        if updates not in saved:
            checkpoint = save_checkpoint(
                gpu,
                optimizer,
                output,
                path,
                reference["translator_checkpoint_sha256"],
                examples,
                updates,
            )
            write(
                checkpoint.with_name(checkpoint.stem + "-sampler.json"),
                {
                    "plan_sha256": sha256(path),
                    "policy": artifact(checkpoint),
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    "completed_outer_exposures": outer,
                    "completed_inner_steps_in_current_batch": inner,
                },
            )
            saved[updates] = checkpoint
        return saved[updates]

    def step():
        if any(
            p.requires_grad or p.grad is not None or p._version != versions[name]
            for name, p in gpu.frozen_parameters().items()
        ):
            raise RuntimeError("frozen short training crossed the frozen parameter boundary")
        norms = [
            float(p.grad.norm()) if p.grad is not None else math.nan
            for p in gpu.trainable_parameters().values()
        ]
        if len(norms) != 8 or any(not math.isfinite(v) for v in norms):
            raise RuntimeError("frozen short training lacks eight finite gradients")
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                gpu.trainable_parameters().values(), RECIPE["gradient_clip"]
            )
        )
        if not math.isfinite(norm):
            raise RuntimeError("frozen short gradient norm is nonfinite")
        optimizer.step()
        return {
            "gradient_norm": norm,
            "gradient_tensors": len(norms),
            "nonzero_gradient_tensors": sum(v > 0 for v in norms),
        }

    try:
        for offset, rows in batches(data, rewards):
            if offset != outer:
                raise RuntimeError("frozen short outer sampler position differs")
            count = len(rows)
            prepared = [(memory(r), candidate_labels(r["pool"]), r["quality"]) for r in rows]
            for group in optimizer.param_groups:
                group["lr"] = rate(offset, count)
            inner = 0
            for local_step in range(RECIPE["local_steps_per_outer_batch"]):
                if stop.requested:
                    checkpoint = save()
                    write(
                        output / "requested-stop.json",
                        {
                            "checkpoint": str(checkpoint),
                            "training_examples": examples,
                            "optimizer_updates": updates,
                        },
                    )
                    return
                optimizer.zero_grad(set_to_none=True)
                measured = []
                for (features, mask), labels, quality in prepared:
                    result = duty.run(
                        lambda features=features,
                        mask=mask,
                        labels=labels,
                        quality=quality,
                        count=count: backward_candidate_risk(
                            gpu,
                            features,
                            mask,
                            labels,
                            quality,
                            alpha=RECIPE["risk_alpha"],
                            caption_weight=RECIPE["caption_weight"],
                            divisor=count,
                        )
                    )
                    result.pop("gold_logits")
                    measured.append(result)
                    torch.mps.empty_cache()
                step_inflight = True
                gradient = duty.run(step)
                examples, updates, inner = examples + count, updates + 1, local_step + 1
                step_inflight = False
                append(
                    output,
                    "steps.jsonl",
                    {
                        "training_examples": examples,
                        "optimizer_update": updates,
                        "outer_offset": outer,
                        "local_step": inner,
                        "batch_clips": count,
                        "learning_rate": rate(offset, count),
                        "measurements": measured,
                        "historical_training_rewards": True,
                        **gradient,
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": "frozen_short_training",
                            "training_examples": examples,
                            "optimizer_updates": updates,
                            "outer_offset": outer,
                            **gradient,
                        }
                    ),
                    flush=True,
                )
            outer += count
            inner = 0
            storage(215000000, output=output)
            if outer in RECIPE["checks_after_outer_exposures"]:
                checkpoint = save()
                if fixed_hashes(gpu, encoder) != fixed:
                    raise RuntimeError("frozen short checkpoint changed frozen C5 weights")
                synchronize_policy(gpu, cpu)
                result = evaluate(
                    cpu,
                    encoder,
                    tokenizer,
                    reference,
                    development,
                    control,
                    output / f"semantic-{examples:06d}",
                    pacer,
                )
                result.update(
                    {
                        "training_examples": examples,
                        "optimizer_updates": updates,
                        "outer_exposures": outer,
                        "checkpoint_sha256": sha256(checkpoint),
                    }
                )
                decisions.append(result)
                append(output, "semantic-decisions.jsonl", result)
                if result["stop_for_semantic_decline"] or result["stop_for_failed_calibration"]:
                    reason = (
                        "significant_semantic_decline"
                        if result["stop_for_semantic_decline"]
                        else "failed_semantic_calibration"
                    )
                    break
                if result["control_stage_mastered"] or result["candidate_stage_mastered"]:
                    reason = "semantic_stage_mastery"
                    break
    except Exception as error:
        # A thrown optimizer step may have mutated some parameters. Never label
        # those values as a complete committed update or offer automatic recovery.
        checkpoint = saved[max(saved)] if step_inflight and saved else None
        if not step_inflight:
            checkpoint = save()
        write(
            output / "failed-stop.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "checkpoint": str(checkpoint),
                "training_examples": examples,
                "optimizer_updates": updates,
                "completed_outer_exposures": outer,
                "completed_inner_steps_in_current_batch": inner,
                "automatic_retry": False,
                "optimizer_step_outcome_ambiguous": step_inflight,
            },
        )
        raise
    checked(path)
    if fixed_hashes(gpu, encoder) != fixed or fixed_hashes(cpu, encoder) != fixed:
        raise RuntimeError("frozen short final C5 frozen weights changed")
    write(
        output / "decision.json",
        {
            "training_complete": examples == plan["planned_training_examples"],
            "training_examples": examples,
            "optimizer_updates": updates,
            "outer_exposures": outer,
            "semantic_decisions": decisions,
            "stop_reason": reason,
            "training_Terra_requests": 0,
            "on_policy_training_claimed": False,
            "WER_veto_used": False,
            "full_composite_or_competitive_win_claimed": False,
            "promotion_authorized": False,
            "resource_receipt": duty.receipt(),
            "CPU_memory_cache_bytes": cache_bytes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(
            args.plan,
            args.output,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
        )
    else:
        parser.error("provide --freeze, or --plan and --output with a qualification mode")


if __name__ == "__main__":
    main()
