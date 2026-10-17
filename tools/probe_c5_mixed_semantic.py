"""Qualify complete mixed-input Terra risk from authenticated word-final weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType

import probe_c5_word_semantic as word
import torch
from benchmark_fleurs_dev import sha256
from c5_mixed_risk_evidence import PROPOSAL, require_terminal
from c5_mixed_risk_evidence import specification as evidence_specification
from c5_mixed_semantic_model import PARAMETERS, TENSORS, load_clip, load_student, require_model
from launch_terra_visual_job import identity, now, write
from prepare_terra_word_native import guard
from probe_roi_semantic_learning import fixed_hashes
from probe_tiny_rgb_capacity import state_digest
from terra_mixed_native_input import expected_header
from terra_mixed_semantic_protocol import POLICY as CANDIDATE_POLICY
from terra_mixed_semantic_protocol import (
    paired_inputs,
    prepare_pools,
    shuffled_indices,
    verified_proof_rewards,
)
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PLAN = ROOT / "runs/c5-word-semantic-training-v1/plan.json"
TRAINING_RESULT = ROOT / "checkpoints/c5-word-semantic-training-v1/result.json"
INPUT_PLAN = ROOT / "runs/c5-mixed-native112-data-v1/plan.json"
INPUT_RESULT = ROOT / "checkpoints/c5-mixed-native112-data-v1/data.json"
WORD_PARENT = ROOT / "checkpoints/c5-word-semantic-evaluation-v1/complete.json"
SENTENCE_PARENT = ROOT / "checkpoints/c5-word-sentence-evaluation-v1/complete.json"
POLICY = {
    **word.POLICY,
    **CANDIDATE_POLICY,
    "scope": "all three predeclared complete mixed training clips; no development inference",
    "initialization": "complete authenticated word-final visual; original frozen C5 backbones",
    "previous_evaluations": "complete word96 and sentence34; original semantic stops enforced",
    "research_gain_failure": "no automatic stage advance; zero-update numerical research only",
    "fresh_optimizer_training_authorized": False,
}
require_resources = word.require_resources


def specification(
    training_path=TRAINING_PLAN,
    result_path=TRAINING_RESULT,
    input_plan=INPUT_PLAN,
    input_result=INPUT_RESULT,
    word_parent=WORD_PARENT,
    sentence_parent=SENTENCE_PARENT,
    proposal_path=PROPOSAL,
):
    from tiny_rgb_full_run import runtime_binding

    evidence = evidence_specification(
        training_path,
        result_path,
        input_plan,
        input_result,
        word_parent,
        sentence_parent,
        proposal_path,
    )
    return {
        **evidence,
        "schema": "umi-studio-C5-mixed-semantic-gradient-proof/1",
        "policy": POLICY,
        "runtime_binding": runtime_binding(),
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    binding = plan["initialization"]
    if plan != specification(
        ROOT / binding["word_training_plan"]["path"],
        ROOT / binding["word_training_result"]["path"],
        ROOT / plan["input_plan"]["path"],
        ROOT / plan["input_result"]["path"],
        ROOT / plan["preceding_evaluations"]["word"]["parent_complete"]["path"],
        ROOT / plan["preceding_evaluations"]["sentence"]["parent_complete"]["path"],
        ROOT / plan["metadata_proposal"]["path"],
    ):
        raise ValueError("mixed risk changed complete predecessors, inputs, initializer or sources")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("mixed risk requires a fresh owned plan")
    plan = specification()
    guard()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "training_cases": len(plan["records"])}), flush=True)


UTC = timezone.utc  # noqa: UP017 - pinned model runtime is Python3.10


def wait_for_predecessor(plan, *, clock=lambda: datetime.now(UTC), sleep=time.sleep):
    boundary = datetime.fromisoformat(plan["API_start_not_before"])
    if boundary.tzinfo is None:
        raise ValueError("mixed risk API boundary needs a timezone-aware timestamp")
    while (remaining := (boundary - clock()).total_seconds()) > 0:
        sleep(min(30, remaining))
    return clock().isoformat()


def prepare(path, output, plan):
    model, tokenizer = load_student(plan)
    pools = prepare_pools(model, tokenizer, plan["records"])
    for index, (case, pool) in enumerate(zip(plan["records"], pools, strict=True)):
        write(output / f"pool-{index:02d}.json", {"case": case, "candidates": pool})
    paired = paired_inputs(plan["records"], pools, tokenizer)
    semantic = output / "semantic"
    semantic.mkdir()
    write(semantic / "blinded-inputs.json", paired)
    indices = shuffled_indices(paired, POLICY["training_rollout_seed"])
    pacer = RecoveryPacer()
    write(
        semantic / "predecessor-spacing.json",
        {
            "not_before": plan["API_start_not_before"],
            "ready_at": wait_for_predecessor(plan),
        },
    )
    for offset in range(0, len(indices), 16):
        guard(output)
        chosen = indices[offset : offset + 16]
        pacer.score([paired["items"][i] for i in chosen], semantic / f"request-{offset // 16:03d}")
    actual = verified_proof_rewards(paired, semantic, POLICY["training_rollout_seed"])
    require_model(model, plan)
    checked(path)
    guard(output)
    write(output / "scores.json", actual["scores"])
    write(
        output / "candidates.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "pools": pools,
            "comparison": paired,
            "qualities": actual["qualities"],
            "scores": artifact(output / "scores.json"),
            "initialization": plan["initialization"],
            "vision_initialization_sha256": state_digest(model.vision.state_dict()),
            "initial_reference_hashes": fixed_hashes(model),
            "optimizer_updates": 0,
            "new_Terra_requests": math.ceil(len(indices) / 16),
            "calibration_passed": True,
            "training_only": True,
            "promotion_authorized": False,
        },
    )


def checked_candidates(path, plan_path, plan, tokenizer):
    require_terminal(path)
    result = json.loads(path.read_text())
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or result.get("initialization") != plan["initialization"]
        or result.get("vision_initialization_sha256")
        != plan["initialization"]["initial_visual_state_sha256"]
        or result.get("initial_reference_hashes") != plan["initial_reference_hashes"]
        or result.get("training_only") is not True
        or result.get("calibration_passed") is not True
        or result.get("optimizer_updates") != 0
        or result.get("promotion_authorized") is not False
        or result.get("scores") != artifact(path.with_name("scores.json"))
    ):
        raise ValueError("mixed candidates changed terminal initializer, model or plan evidence")
    paired = paired_inputs(plan["records"], result["pools"], tokenizer)
    if paired != result["comparison"] or set(path.parent.glob("pool-*.json")) != {
        path.with_name(f"pool-{index:02d}.json") for index in range(len(plan["records"]))
    }:
        raise ValueError("mixed candidates changed actual pairings or complete pool inventory")
    for index, (case, pool) in enumerate(zip(plan["records"], result["pools"], strict=True)):
        if json.loads(path.with_name(f"pool-{index:02d}.json").read_text()) != {
            "case": case,
            "candidates": pool,
        }:
            raise ValueError("mixed candidates differ from actual per-case rollout files")
    directory = path.parent / "semantic"
    spacing = json.loads((directory / "predecessor-spacing.json").read_text())
    ready = datetime.fromisoformat(spacing["ready_at"])
    boundary = datetime.fromisoformat(plan["API_start_not_before"])
    if (
        spacing["not_before"] != plan["API_start_not_before"]
        or ready.tzinfo is None
        or boundary.tzinfo is None
        or ready < boundary
    ):
        raise ValueError("mixed candidates started before predecessor API cooldown")
    actual = verified_proof_rewards(paired, directory, POLICY["training_rollout_seed"])
    if (
        result.get("new_Terra_requests") != math.ceil(len(paired["items"]) / 16)
        or actual["scores"] != json.loads(path.with_name("scores.json").read_text())
        or actual["qualities"] != result["qualities"]
    ):
        raise ValueError("mixed candidates changed authentic raw scores or complete request count")
    return result


def bound_qualification():
    original = word.qualify
    result = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "POLICY": POLICY,
            "load_student": load_student,
            "load_clip": load_clip,
            "require_model": require_model,
            "checked_candidates": checked_candidates,
            "checked": checked,
        },
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = (
        None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    )
    return result


def qualify(path, output, plan, candidates):
    return bound_qualification()(path, output, plan, candidates)


def candidate_tokenizer(plan):
    from transformers import ByT5Tokenizer

    return ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )


def checked_qualification(result_path, plan_path, plan):
    require_terminal(result_path)
    result = json.loads(result_path.read_text())
    tokenizer = candidate_tokenizer(plan)
    candidates = ROOT / result["candidates"]["path"]
    judged = checked_candidates(candidates, plan_path, plan, tokenizer)
    if (
        result.get("complete") is not True
        or result.get("full_model_numerics_passed") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or artifact(candidates) != result["candidates"]
        or result.get("vision_initialization_sha256")
        != plan["initialization"]["initial_visual_state_sha256"]
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
        or set(result_path.parent.glob("comparison-*.json"))
        != {
            result_path.with_name(f"comparison-{index:02d}.json")
            for index in range(len(plan["records"]))
        }
    ):
        raise ValueError("mixed qualification changed full proof, initializer or numerical scope")
    for index, (case, descriptor, quality) in enumerate(
        zip(plan["records"], result["comparisons"], judged["qualities"], strict=True)
    ):
        target = result_path.with_name(f"comparison-{index:02d}.json")
        if artifact(target) != descriptor:
            raise ValueError("mixed qualification differs from actual comparison files")
        report = json.loads(target.read_text())
        if (
            report.get("input_receipt") != case["receipt"]
            or report.get("frames") != expected_header(case["case"])["sampled_frames"]
            or report.get("actual_Terra_qualities") != quality
        ):
            raise ValueError("mixed qualification changed actual inputs, full timeline or rewards")
        comparison = report["comparison"]
        require_resources(report["resources"])
        if (
            not math.isfinite(report["CPU_risk"])
            or not math.isfinite(report["Metal_risk"])
            or abs(report["CPU_risk"] - report["Metal_risk"])
            != comparison["loss_absolute_difference"]
        ):
            raise ValueError("mixed comparison differs from its actual finite risk values")
        word.verified_comparison(comparison, TENSORS)
        gradients = comparison["gradient_differences"]
        if set(row["parameter"] for row in gradients) != set(
            plan["expected_visual_gradient_names"]
        ) or any(
            not math.isfinite(row[key]) or row[key] <= 0
            for row in gradients
            for key in ("cpu_norm", "actual_norm")
        ):
            raise ValueError("mixed qualification lacks exact finite nonzero350-gradient scope")
    require_resources(result["resources"])
    return result


def run(path, output, mode, candidates=None):
    import cv2
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("mixed risk requires the bound two-thread decoder")
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("mixed risk requires a fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            if mode == "candidates":
                prepare(path, output, plan)
            elif mode == "qualify" and candidates is not None:
                qualify(path, output, plan, candidates.resolve())
            else:
                raise ValueError("mixed risk needs candidates or an actual rewarded proof input")
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def pipeline(path, output):
    path, output = path.resolve(), output.resolve()
    plan = checked(path)
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("mixed risk pipeline requires a fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for mode, filename in (
            ("candidates", "candidates.json"),
            ("qualify", "qualification.json"),
        ):
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
            child = subprocess.run(command, check=False)
            if child.returncode:
                raise RuntimeError(f"mixed risk {mode} child failed; no retry or fallback")
            receipt = output / mode / filename
            require_terminal(receipt)
            if json.loads(receipt.read_text()).get("complete") is not True:
                raise ValueError("mixed risk child lacks its actual complete receipt")
            if mode == "candidates":
                checked_candidates(receipt, path, plan, candidate_tokenizer(plan))
        qualification = output / "qualify/qualification.json"
        checked_qualification(qualification, path, plan)
        checked(path)
        guard(output / "candidates")
        guard(output / "qualify")
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
