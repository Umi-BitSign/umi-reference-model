"""Evaluate only the completed broad ROI curriculum through the unchanged C5 gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType

import evaluate_roi_semantic as shared
import torch
from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import ROOT, terminal
from launch_terra_visual_job import identity, now, write
from probe_roi_seen_translation import freeze
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from train_roi_broad_semantic import checked as checked_training
from train_roi_broad_semantic import verify_coverage

POLICY = {
    **shared.POLICY,
    "candidate": "fixed final1668-exposure318-clip curriculum checkpoint only",
    "stage_boundary_checkpoints": "recovery only, never quality-selected",
}


def require_final(result, plan, training_plan_sha):
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != training_plan_sha
        or result.get("training_examples") != plan["planned_training_examples"]
        or result.get("unique_training_clips") != plan["unique_training_clips"]
        or type(result.get("optimizer_updates")) is not int
        or not 0 < result["optimizer_updates"] <= plan["maximum_optimizer_updates"]
        or result.get("maximum_optimizer_updates") != plan["maximum_optimizer_updates"]
        or result.get("all_fixed_backbones_unchanged") is not True
        or result.get("semantic_translation_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or not result.get("visual")
        or not result.get("checkpoints")
        or result["checkpoints"][-1] != result["visual"]
        or result.get("visual_initialization_sha256") != plan["vision_initialization_sha256"]
    ):
        raise ValueError("broad ROI evaluation requires the completed fixed-final curriculum")


def verify_steps(plan, steps, result):
    if len(steps) != len(plan["batches"]):
        raise ValueError("broad ROI evaluation cannot omit unfinished curriculum batches")
    examples = updates = 0
    observed = []
    for number, (step, batch) in enumerate(zip(steps, plan["batches"], strict=True)):
        quality = step["Terra_qualities"]
        indices = batch["training_record_indices"]
        if (
            any(step.get(key) != value for key, value in batch.items())
            or step.get("batch") != number
            or len(quality) != len(indices)
            or any(
                not q or q[0] != 10 or any(type(v) is not int or not 0 <= v <= 10 for v in q)
                for q in quality
            )
        ):
            raise ValueError("broad ROI completion lost planned order or actual reward rows")
        contrasting = sum(len(set(q)) > 1 for q in quality)
        if contrasting:
            updates += 1
            norm = step.get("gradient_norm_before_clip")
            if type(norm) not in (float, int) or not math.isfinite(norm) or norm <= 0:
                raise ValueError("broad ROI applied updates need finite nonzero gradients")
        elif step.get("gradient_norm_before_clip") is not None:
            raise ValueError("broad ROI tied rewards cannot claim an applied gradient")
        examples += len(indices)
        observed.extend(indices)
        if (
            step.get("contrasting_clips") != contrasting
            or step.get("optimizer_updates") != updates
            or step.get("training_examples") != examples
        ):
            raise ValueError("broad ROI actual update/exposure counters differ from completion")
    verify_coverage(plan, examples, observed, result["checkpoints"])
    if updates != result["optimizer_updates"]:
        raise ValueError("broad ROI final optimizer counter differs from complete training steps")


def specification(training_plan_path, result_path):
    from safetensors import safe_open

    training_plan_path, result_path = training_plan_path.resolve(), result_path.resolve()
    fit = checked_training(training_plan_path)
    inputs = shared.checked_inputs(shared.INPUT_PLAN)
    result = json.loads(result_path.read_text())
    terminal(result_path)
    require_final(result, fit, sha256(training_plan_path))
    run_path, steps_path = result_path.with_name("run.json"), result_path.with_name("steps.jsonl")
    run = json.loads(run_path.read_text())
    if (
        run.get("plan_sha256") != sha256(training_plan_path)
        or run.get("initial_visual_sha256") != fit["vision_initialization_sha256"]
        or run.get("fixed_backbones") != inputs["initial_reference_hashes"]
        or run.get("recipe") != fit["recipe"]
        or run.get("unique_training_clips") != fit["unique_training_clips"]
    ):
        raise ValueError("broad ROI result has a different actual training run")
    verify_steps(fit, [json.loads(line) for line in steps_path.read_text().splitlines()], result)
    visual = ROOT / result["visual"]["path"]
    commit = json.loads(visual.with_suffix(".json").read_text())
    if (
        artifact(visual) != result["visual"]
        or commit.get("complete") is not True
        or commit.get("visual") != result["visual"]
        or commit.get("training_examples") != fit["planned_training_examples"]
        or commit.get("optimizer_updates") != result["optimizer_updates"]
    ):
        raise ValueError("broad ROI fixed-final checkpoint hash/counters lack a complete commit")
    with safe_open(visual, framework="pt") as saved:
        metadata = saved.metadata()
    if (
        metadata.get("plan_sha256") != sha256(training_plan_path)
        or metadata.get("update") != str(fit["planned_training_examples"])
        or metadata.get("optimizer_updates") != str(result["optimizer_updates"])
        or metadata.get("fixed_winner_checkpoint")
        != inputs["reference"]["translator_checkpoint_sha256"]
    ):
        raise ValueError("broad ROI final checkpoint has different training provenance")
    source = {**inputs["source_sha256"], **fit["source_sha256"]}
    for name in (
        "evaluate_roi_semantic.py",
        "evaluate_roi_broad_semantic.py",
        "score_roi_short_translation.py",
        "terra_semantic_selection.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-broad-semantic-evaluation/1",
        "policy": POLICY,
        "fit_plan": artifact(training_plan_path),
        "fit_result": artifact(result_path),
        "fit_run": artifact(run_path),
        "fit_steps": artifact(steps_path),
        "visual": artifact(visual),
        "visual_commit": artifact(visual.with_suffix(".json")),
        "visual_state_sha256": result["visual_final_sha256"],
        "model_plan": fit["model_plan"],
        "input_protocol": artifact(shared.INPUT_PLAN),
        "records": inputs["records"],
        "control": inputs["control"],
        "reference": inputs["reference"],
        "assets": inputs["assets"],
        "resource_policy": inputs["resource_policy"],
        "initial_reference_hashes": inputs["initial_reference_hashes"],
        "source_sha256": source,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(ROOT / plan["fit_plan"]["path"], ROOT / plan["fit_result"]["path"]):
        raise ValueError("broad ROI evaluation changed fixed weights, complete training or inputs")
    verify_snapshot(path, plan)
    return plan


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != bound_phase("score_specification")(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("broad ROI scoring changed actual outputs, rubric or selection gate")
    verify_snapshot(path, plan)
    return plan


def bound_phase(name):
    """Preserve numerical/scoring bytecode; privately bind only control callbacks."""
    function = {
        "infer": shared.infer,
        "score_specification": shared.score_specification,
        "run": shared.run,
    }[name]
    replacements = {"checked": checked}
    if name == "run":
        replacements.update(checked_scores=checked_scores, infer=bound_phase("infer"))
    value = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    value.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    return value


def pipeline(path, output):
    path, output = path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("broad ROI evaluation pipeline needs a fresh owned output")
    plan = json.loads(path.read_text())
    verify_snapshot(path, plan)
    if plan["source_sha256"]["tools/evaluate_roi_broad_semantic.py"] != sha256(Path(__file__)):
        raise ValueError("broad ROI evaluation source differs from its frozen plan")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(path),
            "--output",
            str(output / "inference"),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError("broad ROI inference stopped; scoring not started, no retry")
        score_plan = path.parent.with_name(path.parent.name + "-scores") / "plan.json"
        freeze(
            score_plan, bound_phase("score_specification")(path, output / "inference/receipt.json")
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(score_plan),
            "--output",
            str(output / "scoring"),
            "--score",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError("broad ROI scoring stopped; no retry or model fallback")
        write(
            output / "complete.json",
            {
                "complete": True,
                "inference_plan": artifact(path),
                "scoring_plan": artifact(score_plan),
                "decision": artifact(output / "scoring/decision.json"),
                "optimizer_updates": 0,
                "automatic_retries": 0,
                "promotion_authorized": False,
            },
        )
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--freeze-scores", type=Path)
    parser.add_argument("--inference-plan", type=Path)
    parser.add_argument("--decoded", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze and args.training_plan and args.training_result:
        torch.set_num_threads(2)
        freeze(args.freeze, specification(args.training_plan, args.training_result))
        if args.pipeline:
            if not args.output:
                parser.error("--pipeline requires a fresh --output")
            pipeline(args.freeze, args.output)
    elif args.freeze_scores and args.inference_plan and args.decoded:
        torch.set_num_threads(2)
        freeze(
            args.freeze_scores,
            bound_phase("score_specification")(args.inference_plan, args.decoded),
        )
    elif args.plan and args.output:
        bound_phase("run")(args.plan, args.output, scoring=args.score)
    else:
        parser.error("choose completed-training/score freezing, or --plan --output [--score]")


if __name__ == "__main__":
    main()
