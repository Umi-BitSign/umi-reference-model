"""Fixed-final sentence gate for the complete original-C5 word curriculum."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType

import evaluate_c5_small_input as native
import evaluate_roi_semantic as scoring
import torch
from benchmark_fleurs_dev import sha256
from c5_word_evaluation_io import storage
from c5_word_final import checked_final
from launch_terra_visual_job import identity, now, write
from probe_tiny_rgb_capacity import state_digest
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PLAN = ROOT / "runs/c5-native112-evaluation-v1/plan.json"
POLICY = {
    **native.POLICY,
    "scope": "same34 reserved sentences; complete fixed-final word weights only",
    "appearance": "qualified native112 projection with all350 final word-trained visual tensors",
    "word_improvement_is_not_continuous_translation_evidence": True,
}


def specification(training_path, result_path):
    fit, result, evidence = checked_final(training_path, result_path)
    inputs = native.checked(NATIVE_PLAN)
    if fit["model_plan"]["reference"] != inputs["reference"]:
        raise ValueError("word training and sentence controls use different original C5 references")
    sources = {**inputs["source_sha256"], **fit["source_sha256"]}
    for name in ("c5_word_final.py", "c5_word_evaluation_io.py", "evaluate_c5_word_sentences.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-C5-word-final-sentence-evaluation/1",
        "policy": POLICY,
        "training_plan": artifact(training_path),
        "training_result": artifact(result_path),
        "training_evidence": evidence,
        "input_protocol": artifact(NATIVE_PLAN),
        "records": inputs["records"],
        "reference": inputs["reference"],
        "control": inputs["control"],
        "assets": inputs["assets"],
        "initial_reference_hashes": inputs["initial_reference_hashes"],
        "model_plan": fit["model_plan"],
        "candidate_visual": result["visual"],
        "candidate_visual_state_sha256": result["visual_final_sha256"],
        "resource_policy": inputs["resource_policy"],
        "source_sha256": sources,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["training_plan"]["path"], ROOT / plan["training_result"]["path"]
    ):
        raise ValueError(
            "word sentence evaluation changed full training, final weights or controls"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path, training_path, result_path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word sentence evaluation requires a fresh owned plan")
    plan = specification(training_path.resolve(), result_path.resolve())
    storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "fixed_final_only": True}), flush=True)


def candidate_appearance(plan, hands, face):
    from safetensors.torch import load_file

    appearance = native.appearance_model(plan["model_plan"], hands, face)
    target = ROOT / plan["candidate_visual"]["path"]
    if artifact(target) != plan["candidate_visual"]:
        raise ValueError("word candidate's actual final visual file changed")
    incoming = load_file(str(target), device="cpu")
    names = set(plan["model_plan"]["expected_visual_gradient_names"])
    if set(incoming) != names:
        raise ValueError("word candidate cannot replace an incomplete or expanded visual scope")
    existing = appearance.state_dict()
    for name, value in incoming.items():
        if not name.startswith("vision."):
            raise ValueError("word candidate cannot replace frozen sequence or language values")
        key = name.removeprefix("vision.")
        if (
            key not in existing
            or value.shape != existing[key].shape
            or value.dtype != existing[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("word candidate has altered shapes, dtypes or nonfinite visual values")
        existing[key] = value
    appearance.load_state_dict(existing, strict=True)
    if state_digest(appearance.state_dict()) != plan["candidate_visual_state_sha256"]:
        raise ValueError("word candidate model does not reproduce the actual final visual state")
    return appearance


def bound_inference():
    original = native.infer
    result = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "appearance_model": candidate_appearance,
            "checked": checked,
            "storage": storage,
        },
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = (
        None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    )
    return result


def infer(path, output, plan):
    # This private view adapts the old invariant's internal field name. The saved
    # plan distinguishes the original initializer from the actual trained state.
    view = {**plan, "pretrained_state_sha256": plan["candidate_visual_state_sha256"]}
    return bound_inference()(path, output, view)


def run_inference(path, output):
    original = native.run
    result = FunctionType(
        original.__code__,
        {**original.__globals__, "checked": checked, "infer": infer, "storage": storage},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = (
        None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    )
    return result(path, output)


def score_specification(inference_path, decoded_path):
    original = scoring.score_specification
    result = FunctionType(
        original.__code__,
        {**original.__globals__, "checked": checked},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = (
        None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    )
    return result(inference_path, decoded_path)


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != score_specification(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("word sentence scoring changed actual tokens, roles or semantic gate")
    verify_snapshot(path, plan)
    return plan


def run_scores(path, output):
    original = scoring.run
    result = FunctionType(
        original.__code__,
        {**original.__globals__, "checked_scores": checked_scores, "storage": storage},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = (
        None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    )
    return result(path, output, scoring=True)


def pipeline(path, output):
    from probe_roi_seen_translation import freeze as freeze_scoring

    path, output = path.resolve(), output.resolve()
    plan = checked(path)
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word final sentence pipeline needs a fresh owned output")
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
            "--mode",
            "inference",
        ]
        child = subprocess.run(command, check=False)
        if child.returncode:
            raise RuntimeError("word sentence inference stopped; no scoring or retry")
        score_path = path.parent.with_name(path.parent.name + "-scores") / "plan.json"
        storage(output=output)
        freeze_scoring(score_path, score_specification(path, output / "inference/receipt.json"))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(score_path),
            "--output",
            str(output / "scoring"),
            "--mode",
            "scores",
        ]
        child = subprocess.run(command, check=False)
        if child.returncode:
            raise RuntimeError("word sentence scoring stopped; no retry or fallback")
        checked(path)
        write(
            output / "complete.json",
            {
                "complete": True,
                "inference_plan": artifact(path),
                "scoring_plan": artifact(score_path),
                "decision": artifact(output / "scoring/decision.json"),
                "training_result": plan["training_result"],
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


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "training-plan", "training-result", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("inference", "scores"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze and args.training_plan and args.training_result:
        freeze(args.freeze, args.training_plan, args.training_result)
        if args.pipeline and args.output:
            pipeline(args.freeze, args.output)
    elif args.plan and args.output:
        if args.pipeline:
            pipeline(args.plan, args.output)
        elif args.mode == "inference":
            run_inference(args.plan, args.output)
        elif args.mode == "scores":
            run_scores(args.plan, args.output)
        else:
            parser.error("choose --pipeline or a child --mode")
    else:
        parser.error("freeze only with actual complete training, or run an existing plan")


if __name__ == "__main__":
    main()
