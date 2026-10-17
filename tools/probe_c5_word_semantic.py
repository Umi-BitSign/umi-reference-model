"""Qualify actual Terra risk through original C5 vision on complete verified word clips."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_word_semantic_model import PARAMETERS, TENSORS, load_clip, load_student, require_model
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import identity, now, write
from probe_c5_linear_patch import checked as checked_component
from probe_c5_linear_patch import require_result
from probe_roi_semantic_learning import fixed_hashes
from probe_tiny_rgb_capacity import compare, state_digest
from roi_semantic_student import visual_gradients
from semantic_quality_judge import parse_terra_scores
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_word_semantic_protocol import POLICY as CANDIDATE_POLICY
from terra_word_semantic_protocol import paired_inputs, prepare_pools, qualities, select_cases
from terra_word_verified_inputs import checked_result, expected_header
from tiny_rgb_capacity_protocol import LIMITS, verified_comparison

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "runs/c5-native112-linear-qualification-v1/plan.json"
COMPONENT_RESULT = ROOT / "checkpoints/c5-native112-linear-qualification-v1/qualification.json"
INPUT_PLAN = ROOT / "runs/terra-word-native112-data-v1/plan.json"
INPUT_RESULT = ROOT / "checkpoints/terra-word-native112-data-v1/data.json"
INPUT_BINDING = INPUT_PLAN.parent / "verified-inputs.json"
CASE_BINDING = ROOT / "runs/c5-word-semantic-proposal-v1/verified-cases.json"
SCOPE = ROOT / "runs/c5-native112-gradient-scope-v1/scope.json"
ORIGINAL = ROOT / "runs/terra-c5-tail-training-v1/plan.json"
POLICY = {
    **CANDIDATE_POLICY,
    "scope": "all three complete predeclared word training clips; no development inference",
    "initialization": "original public C5 only; no locally trained initializer",
    "visual_chunk_frames": 2,
    "non_reentrant_visual_checkpointing": True,
    "trainable_parameters": PARAMETERS,
    "trainable_tensors": TENSORS,
    "GPU_duty_fraction": 0.5,
    "frozen_components": "complete original C5 sequence encoder/language and video-derived14D pose",
    "held_out_quality_evaluated": False,
    "model_win_claimed": False,
    "physical_free_floor_bytes": 80 * 1024**3,
    "maximum_receipt_bytes_per_phase": 32 * 1024**2,
    "headroom_bytes": 128 * 1024**2,
}


def input_evidence():
    binding = checked_result(INPUT_PLAN, INPUT_RESULT)
    if json.loads(INPUT_BINDING.read_text()) != binding:
        raise ValueError("word risk proof lost its verified complete input inventory")
    cases = select_cases(binding["records"])
    rebound = json.loads(CASE_BINDING.read_text())
    proposal_path = ROOT / rebound["metadata_proposal"]["path"]
    proposal = json.loads(proposal_path.read_text())
    if (
        rebound.get("verified_native_inputs") != artifact(INPUT_BINDING)
        or rebound.get("selected_training_cases") != cases
        or rebound.get("original_metadata_case_ids_preserved") is not True
        or rebound.get("actual_preflight_receipt_rebinding_complete") is not True
        or artifact(proposal_path) != rebound["metadata_proposal"]
        or [case["record"]["original"]["row"]["example_id"] for case in cases]
        != [case["example_id"] for case in proposal["selected_training_cases"]]
    ):
        raise ValueError("word risk proof changed its predeclared complete training cases")
    return binding, cases


def specification(component_path=COMPONENT, component_result=COMPONENT_RESULT):
    from tiny_rgb_full_run import runtime_binding

    component = checked_component(component_path)
    terminal(component_result)
    result = json.loads(component_result.read_text())
    require_result(result, component, sha256(component_path), component_result.parent)
    _, cases = input_evidence()
    initial = json.loads(ORIGINAL.read_text())
    if initial["reference"] != component["reference"]:
        raise ValueError("word risk proof changed the original C5 backbone-hash reference")
    scope = json.loads(SCOPE.read_text())
    if (
        scope["trainable_tensors"] != TENSORS
        or scope["trainable_parameters"] != PARAMETERS
        or any(
            component["assets"].get(name) != digest
            for name, digest in scope["weights_sha256"].items()
        )
    ):
        raise ValueError("word risk proof changed the original public visual parameter scope")
    data = json.loads(INPUT_PLAN.read_text())
    sources = {**component["source_sha256"], **data["source_sha256"]}
    for name in (
        "probe_c5_word_semantic.py",
        "c5_word_semantic_model.py",
        "c5_small_input_semantic.py",
        "terra_word_semantic_protocol.py",
        "terra_word_verified_inputs.py",
        "probe_roi_semantic_learning.py",
        "roi_semantic_student.py",
        "c5_terra_tail.py",
        "c5_winner_language_decoder.py",
        "c5_winner_mps_safe.py",
        "tiny_rgb_winner_student.py",
        "tiny_rgb_feature_winner_student.py",
        "tiny_rgb_mps_safe.py",
        "terra_visual_learning.py",
        "terra_teacher_candidates.py",
        "terra_quality_training.py",
        "semantic_quality_judge.py",
        "probe_jeff_terra_quality.py",
        "terra_short_resume.py",
        "terra_short_curriculum_learning.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-C5-word-semantic-gradient-proof/1",
        "policy": POLICY,
        "limits": LIMITS,
        "component_plan": artifact(component_path),
        "component_result": artifact(component_result),
        "input_plan": artifact(INPUT_PLAN),
        "input_result": artifact(INPUT_RESULT),
        "verified_inputs": artifact(INPUT_BINDING),
        "verified_cases": artifact(CASE_BINDING),
        "visual_scope": artifact(SCOPE),
        "original_backbone_hash_plan": artifact(ORIGINAL),
        "records": cases,
        "reference": component["reference"],
        "patch_projection": component["policy"]["patch_projection"],
        "pretrained_state_sha256": result["pretrained_state_sha256"],
        "initial_reference_hashes": initial["initial_reference_hashes"],
        "expected_visual_gradient_names": [
            "vision." + row["parameter"] for row in scope["parameters"]
        ],
        "source_sha256": sources,
        "assets": {**component["assets"], **data["assets"]},
        "runtime_binding": runtime_binding(),
        "resource_policy": component["resource_policy"],
        "resource_policy_sha256": component["resource_policy_sha256"],
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["component_plan"]["path"], ROOT / plan["component_result"]["path"]
    ):
        raise ValueError(
            "word semantic proof changed actual qualification, cases, recipe or sources"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from prepare_terra_word_native import guard

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word semantic proof requires a fresh owned plan")
    guard()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "training_cases": len(plan["records"])}), flush=True)


def shuffled_indices(paired):
    indices = list(range(len(paired["items"])))
    random.Random(POLICY["training_rollout_seed"]).shuffle(indices)
    return indices


def prepare(path, output, plan):
    from prepare_terra_word_native import guard

    model, tokenizer = load_student(plan)
    pools = prepare_pools(model, tokenizer, plan["records"])
    for index, (case, pool) in enumerate(zip(plan["records"], pools, strict=True)):
        write(output / f"pool-{index:02d}.json", {"case": case, "candidates": pool})
    paired = paired_inputs(plan["records"], pools, tokenizer)
    semantic = output / "semantic"
    semantic.mkdir()
    write(semantic / "blinded-inputs.json", paired)
    scores = [None] * len(paired["items"])
    indices, pacer = shuffled_indices(paired), RecoveryPacer()
    for offset in range(0, len(indices), 16):
        guard(output)
        chosen = indices[offset : offset + 16]
        response = pacer.score(
            [paired["items"][i] for i in chosen], semantic / f"batch-{offset // 16:03d}"
        )
        for position, value in zip(chosen, response["scores"], strict=True):
            scores[position] = value
    judged = qualities(paired, scores)
    require_model(model, plan)
    checked(path)
    guard(output)
    write(output / "scores.json", scores)
    write(
        output / "candidates.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "pools": pools,
            "comparison": paired,
            "qualities": judged,
            "scores": artifact(output / "scores.json"),
            "vision_initialization_sha256": state_digest(model.vision.state_dict()),
            "initial_reference_hashes": fixed_hashes(model),
            "optimizer_updates": 0,
            "new_Terra_requests": math.ceil(len(indices) / 16),
            "calibration_passed": True,
            "training_only": True,
            "promotion_authorized": False,
        },
    )
    print(json.dumps({"actual_training_rewards": judged, "optimizer_updates": 0}), flush=True)


def checked_candidates(path, plan_path, plan, tokenizer):
    from probe_jeff_terra_quality import terra_payload

    terminal(path)
    result = json.loads(path.read_text())
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or result.get("vision_initialization_sha256") != plan["pretrained_state_sha256"]
        or result.get("initial_reference_hashes") != plan["initial_reference_hashes"]
        or result.get("training_only") is not True
        or result.get("calibration_passed") is not True
        or result.get("optimizer_updates") != 0
        or result.get("promotion_authorized") is not False
        or artifact(ROOT / result["scores"]["path"]) != result["scores"]
    ):
        raise ValueError("word risk candidates are incomplete or changed their model/plan binding")
    paired = paired_inputs(plan["records"], result["pools"], tokenizer)
    if (
        paired != result["comparison"]
        or json.loads((path.parent / "semantic/blinded-inputs.json").read_text()) != paired
    ):
        raise ValueError("word risk changed the actual blind candidate-to-input mapping")
    for index, (case, pool) in enumerate(zip(plan["records"], result["pools"], strict=True)):
        if json.loads(path.with_name(f"pool-{index:02d}.json").read_text()) != {
            "case": case,
            "candidates": pool,
        }:
            raise ValueError("word risk candidates differ from actual per-case rollout files")
    indices, actual = shuffled_indices(paired), [None] * len(paired["items"])
    if result.get("new_Terra_requests") != math.ceil(len(indices) / 16):
        raise ValueError("word risk candidate request count changed")
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        directory = path.parent / "semantic" / f"batch-{offset // 16:03d}"
        request = json.loads((directory / "request.json").read_text())
        response = json.loads((directory / "response.json").read_text())
        scored = json.loads((directory / "scores.json").read_text())
        if (
            request != terra_payload([paired["items"][i] for i in chosen])
            or scored["response_sha256"] != sha256(directory / "response.json")
            or response.get("model") != POLICY["judge"]
            or response.get("reasoning", {}).get("effort") != POLICY["reasoning_effort"]
        ):
            raise ValueError("word risk rewards lack authentic model, prompt or response bindings")
        parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(chosen))])
        for position, index in enumerate(chosen):
            actual[index] = parsed[f"pair_{position:02d}"]
    if (
        actual != json.loads((path.parent / "scores.json").read_text())
        or qualities(paired, actual) != result["qualities"]
    ):
        raise ValueError("word risk changed actual calibrated reward values or identity mapping")
    return result


def qualify(path, output, plan, candidates):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import candidate_labels
    from prepare_terra_word_native import guard
    from terra_visual_learning import backward_candidate_risk

    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("word risk proof requires CPU2 and native Metal50%")
    duty = WinnerDutyCycle(torch, budget)
    model, tokenizer = load_student(plan)
    judged = checked_candidates(candidates, path, plan, tokenizer)
    reports = []
    for index, (case, pool, quality) in enumerate(
        zip(plan["records"], judged["pools"], judged["qualities"], strict=True)
    ):
        guard(output)
        clip, mask = load_clip(case)
        labels = candidate_labels(pool)
        model.zero_grad(set_to_none=True)
        cpu = backward_candidate_risk(
            model, clip, mask, labels, quality, alpha=POLICY["risk_alpha"], caption_weight=0
        )
        cpu_gradients = visual_gradients(model, expected_tensors=TENSORS)
        if set(cpu_gradients) != set(plan["expected_visual_gradient_names"]):
            raise ValueError("word risk CPU gradients changed original public parameter names")
        model.zero_grad(set_to_none=True)
        duty.run(lambda: model.to("mps"))

        def metal(clip=clip, mask=mask, labels=labels, quality=quality):
            measured = backward_candidate_risk(
                model, clip, mask, labels, quality, alpha=POLICY["risk_alpha"], caption_weight=0
            )
            return measured, visual_gradients(model, expected_tensors=TENSORS)

        measured, metal_gradients = duty.run(metal)
        comparison = compare(
            cpu_gradients,
            metal_gradients,
            cpu["loss"],
            measured["loss"],
            cpu["gold_logits"],
            measured["gold_logits"],
        )
        report = {
            "input_receipt": case["receipt"],
            "frames": len(clip.pixels),
            "actual_Terra_qualities": quality,
            "CPU_risk": cpu["loss"],
            "Metal_risk": measured["loss"],
            "comparison": comparison,
            "resources": duty.receipt(),
        }
        target = output / f"comparison-{index:02d}.json"
        write(target, report)
        worst = verified_comparison(comparison, TENSORS)
        model.zero_grad(set_to_none=True)
        duty.run(lambda: model.cpu())
        require_model(model, plan)
        reports.append(artifact(target))
        print(
            json.dumps(
                {
                    "qualified_word_case": index,
                    "frames": len(clip.pixels),
                    "worst_relative_gradient_error": worst,
                }
            ),
            flush=True,
        )
        del cpu, measured, cpu_gradients, metal_gradients, clip, mask, labels
        gc.collect()
        torch.mps.empty_cache()
    checked(path)
    write(
        output / "qualification.json",
        {
            "complete": True,
            "full_model_numerics_passed": True,
            "plan_sha256": sha256(path),
            "candidates": artifact(candidates),
            "comparisons": reports,
            "vision_initialization_sha256": state_digest(model.vision.state_dict()),
            "initial_reference_hashes": fixed_hashes(model),
            "trainable_tensors": TENSORS,
            "trainable_parameters": PARAMETERS,
            "pure_Terra_objective": True,
            "all_fixed_values_unchanged": True,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "resources": duty.receipt(),
            "model_win_claimed": False,
            "held_out_quality_evaluated": False,
            "promotion_authorized": False,
        },
    )


def require_resources(resources):
    if (
        resources.get("current_allowed_gpu_duty_fraction") != 0.5
        or resources.get("current_cpu_threads") != 2
        or resources.get("memory_policy") != "framework_default_no_artificial_six_GiB_cap"
        or any(
            not math.isfinite(resources[key]) or resources[key] <= 0
            for key in ("mps_active_seconds", "mps_idle_seconds")
        )
        or any(
            type(resources.get(key)) is not int or resources[key] <= 0
            for key in ("maximum_allocated_bytes", "maximum_driver_allocated_bytes")
        )
        or not math.isclose(
            resources["mps_active_seconds"], resources["mps_idle_seconds"], rel_tol=0, abs_tol=1e-8
        )
    ):
        raise ValueError("word risk qualification changed the50%/CPU2/default allocator policy")


def checked_qualification(result_path, plan_path, plan):
    from transformers import ByT5Tokenizer

    terminal(result_path)
    result = json.loads(result_path.read_text())
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    candidates = ROOT / result["candidates"]["path"]
    judged = checked_candidates(candidates, plan_path, plan, tokenizer)
    if (
        result.get("complete") is not True
        or result.get("full_model_numerics_passed") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or artifact(candidates) != result["candidates"]
        or result.get("vision_initialization_sha256") != plan["pretrained_state_sha256"]
        or result.get("initial_reference_hashes") != plan["initial_reference_hashes"]
        or result.get("trainable_tensors") != TENSORS
        or result.get("trainable_parameters") != PARAMETERS
        or result.get("pure_Terra_objective") is not True
        or result.get("all_fixed_values_unchanged") is not True
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or result.get("model_win_claimed") is not False
        or result.get("held_out_quality_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or len(result.get("comparisons", [])) != len(plan["records"])
    ):
        raise ValueError("word risk qualification is incomplete or changed actual proof invariants")
    for index, (case, descriptor, quality) in enumerate(
        zip(plan["records"], result["comparisons"], judged["qualities"], strict=True)
    ):
        actual = result_path.with_name(f"comparison-{index:02d}.json")
        if artifact(actual) != descriptor:
            raise ValueError("word risk qualification differs from actual comparison files")
        report = json.loads(actual.read_text())
        if (
            report.get("input_receipt") != case["receipt"]
            or report.get("frames") != expected_header(case["record"])["sampled_frames"]
            or report.get("actual_Terra_qualities") != quality
        ):
            raise ValueError("word risk qualification changed actual inputs, timelines or rewards")
        comparison = report["comparison"]
        require_resources(report["resources"])
        if (
            not math.isfinite(report["CPU_risk"])
            or not math.isfinite(report["Metal_risk"])
            or abs(report["CPU_risk"] - report["Metal_risk"])
            != comparison["loss_absolute_difference"]
        ):
            raise ValueError("word risk comparison differs from its actual finite risk values")
        verified_comparison(comparison, TENSORS)
        gradients = comparison["gradient_differences"]
        if set(row["parameter"] for row in gradients) != set(
            plan["expected_visual_gradient_names"]
        ) or any(
            not math.isfinite(row[key]) or row[key] <= 0
            for row in gradients
            for key in ("cpu_norm", "actual_norm")
        ):
            raise ValueError(
                "word risk qualification lacks the exact finite nonzero gradient scope"
            )
    require_resources(result["resources"])
    return result


def run(path, output, mode, candidates=None):
    import cv2
    from prepare_terra_word_native import guard

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("word risk proof requires the bound two-thread decoder")
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word risk proof requires a fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if mode == "candidates":
            prepare(path, output, plan)
        elif mode == "qualify" and candidates is not None:
            from train_terra_c5_tail import gpu_lease

            with gpu_lease():
                qualify(path, output, plan, candidates.resolve())
        else:
            raise ValueError(
                "word risk proof needs candidates or an actual rewarded qualification input"
            )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def pipeline(path, output):
    from prepare_terra_word_native import guard

    path, output = path.resolve(), output.resolve()
    plan = checked(path)
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word risk pipeline requires a fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for mode in ("candidates", "qualify"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plan",
                str(path),
                "--output",
                str(output / mode),
                "--mode",
                mode,
            ]
            if mode == "qualify":
                command += ["--candidates", str(output / "candidates/candidates.json")]
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise RuntimeError(f"word risk {mode} child failed; no retry or fallback")
        qualification = output / "qualify/qualification.json"
        checked_qualification(qualification, path, plan)
        checked(path)
        write(
            output / "complete.json",
            {
                "complete": True,
                "plan": artifact(path),
                "qualification": artifact(qualification),
                "optimizer_updates": 0,
                "automatic_retries": 0,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output", "candidates"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("candidates", "qualify"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    if args.pipeline and args.plan and args.output:
        pipeline(args.plan, args.output)
    elif args.plan and args.output and args.mode:
        run(args.plan, args.output, args.mode, args.candidates)
    elif args.plan or args.output or not args.freeze:
        parser.error("choose --freeze, --plan/--output/--mode, or --pipeline")
