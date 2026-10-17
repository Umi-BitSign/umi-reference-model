"""Compare fixed-final Terra-trained ROI vision with C5 on the unchanged34-clip gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import (
    CONTROL,
    ROOT,
    crops,
    decode,
    predicted_streams,
    prediction,
    require_control,
    terminal,
)
from evaluate_roi_short_translation import checked as checked_inputs
from launch_terra_visual_job import identity, now, write
from probe_roi_seen_translation import freeze
from probe_roi_semantic_learning import fixed_hashes, load_student
from probe_tiny_rgb_capacity import state_digest
from score_roi_short_translation import POLICY as SCORE_POLICY
from score_roi_short_translation import verified_prediction
from terra_short_curriculum_learning import comparison_items, score_comparison, semantic_decision
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_visual_teacher_data import load_targets
from train_roi_semantic import checked as checked_training

INPUT_PLAN = ROOT / "runs/roi-short-translation-evaluation-v1/plan.json"
POLICY = {
    "scope": "same34 reserved development clips, all C5 controls reproduced first",
    "candidate": "fixed final one-pass88-clip semantic training checkpoint only",
    "pixels": "complete native-parity64px face/hand crops and video-derived14D pose",
    "generation": "original published C5 policy and CPU budget, no reduced horizon",
    "backbones": "original frozen C5 sequence/language, CPU2",
    "appearance_GPU_average_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_downloads": 0,
    "primary_metric": "paired Terra medium semantic quality0-10, two calibrated shuffled passes",
    "WER_is_diagnostic_only": True,
    "automatic_retries": 0,
    "promotion_authorized": False,
}


def require_final(result, training_plan_sha):
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != training_plan_sha
        or result.get("training_examples") != 88
        or type(result.get("optimizer_updates")) is not int
        or not 0 < result["optimizer_updates"] <= 22
        or result.get("all_fixed_backbones_unchanged") is not True
        or result.get("semantic_translation_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or not result.get("visual")
    ):
        raise ValueError("ROI semantic evaluation requires the completed fixed-final88-clip fit")


def appearance_state(values, expected):
    if set(values) != {f"vision.{name}" for name in expected}:
        raise ValueError("semantic checkpoint must contain exactly the116 qualified visual tensors")
    result = {name: values[f"vision.{name}"] for name in expected}
    if any(
        value.shape != expected[name].shape
        or value.dtype != expected[name].dtype
        or not torch.isfinite(value).all()
        for name, value in result.items()
    ):
        raise ValueError("semantic checkpoint tensor shape/dtype/finite values changed")
    return result


def specification(training_plan_path, result_path):
    from safetensors import safe_open

    training_plan_path, result_path = training_plan_path.resolve(), result_path.resolve()
    fit = checked_training(training_plan_path)
    inputs = checked_inputs(INPUT_PLAN)
    result = json.loads(result_path.read_text())
    terminal(result_path)
    require_final(result, sha256(training_plan_path))
    visual = ROOT / result["visual"]["path"]
    commit = json.loads(visual.with_suffix(".json").read_text())
    if (
        artifact(visual) != result["visual"]
        or commit.get("complete") is not True
        or commit.get("visual") != result["visual"]
        or commit.get("training_examples") != 88
        or commit.get("optimizer_updates") != result["optimizer_updates"]
    ):
        raise ValueError("semantic fixed-final checkpoint lacks matching complete hash/counters")
    with safe_open(visual, framework="pt") as saved:
        metadata = saved.metadata()
    if (
        metadata.get("plan_sha256") != sha256(training_plan_path)
        or metadata.get("update") != "88"
        or metadata.get("optimizer_updates") != str(result["optimizer_updates"])
        or metadata.get("fixed_winner_checkpoint")
        != inputs["reference"]["translator_checkpoint_sha256"]
    ):
        raise ValueError("semantic fixed-final checkpoint provenance differs")
    source = {**inputs["source_sha256"], **fit["source_sha256"]}
    for name in (
        "evaluate_roi_semantic.py",
        "score_roi_short_translation.py",
        "terra_semantic_selection.py",
    ):
        source[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-roi-semantic-evaluation/1",
        "policy": POLICY,
        "fit_plan": artifact(training_plan_path),
        "fit_result": artifact(result_path),
        "visual": artifact(visual),
        "visual_commit": artifact(visual.with_suffix(".json")),
        "visual_state_sha256": result["visual_final_sha256"],
        "model_plan": fit["model_plan"],
        "input_protocol": artifact(INPUT_PLAN),
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
        raise ValueError("semantic evaluation changed fixed weights, inputs or generation policy")
    verify_snapshot(path, plan)
    return plan


def infer(path, output, plan):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import native_memory
    from safetensors.torch import load_file

    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("semantic evaluation requires50% appearance Metal duty and CPU2")
    duty = WinnerDutyCycle(torch, budget)
    model, tokenizer = load_student(plan["model_plan"])
    model.requires_grad_(False).eval()
    model.vision.load_state_dict(
        appearance_state(load_file(ROOT / plan["visual"]["path"]), model.vision.state_dict()),
        strict=True,
    )
    if (
        state_digest(model.vision.state_dict()) != plan["visual_state_sha256"]
        or fixed_hashes(model) != plan["initial_reference_hashes"]
    ):
        raise ValueError("semantic evaluation loaded another visual or C5 value")
    duty.run(lambda: model.vision.to("mps"))
    baseline = json.loads(CONTROL.read_text())["predictions"]
    controls = {(r["source"], r["example_id"]): r for r in baseline}
    predictions = []
    with torch.no_grad():
        for index, record in enumerate(plan["records"]):
            row = record["development"]
            memory = native_memory(
                model.semantic_encoder,
                row["original_pair"],
                plan["reference"]["translator_checkpoint_sha256"],
            )
            text, tokens = decode(
                model.language, tokenizer, plan["reference"]["generation"], memory
            )
            expected = controls[(row["source"], row["example_id"])]["hypothesis"]
            write(
                output / f"control-{index:04d}.json",
                {
                    **prediction(row, text),
                    "tokens": tokens,
                    "matches_frozen_control": text == expected,
                },
            )
            require_control(text, expected)
            print(json.dumps({"stage": "C5_control_reproduced", "clips": index + 1}), flush=True)
        for index, record in enumerate(plan["records"]):
            storage(output=output)
            row = record["development"]
            pixels = crops(record)
            features = predicted_streams(
                model.vision, pixels, load_targets(row["original_pair"]), duty
            )
            memory = model.semantic_encoder(
                **{name: values[None] for name, values in features.items()}, return_dict=True
            ).last_hidden_state
            if memory.shape != (1, len(pixels), 1536) or not torch.isfinite(memory).all():
                raise ValueError("semantic evaluation cannot drop original frame coverage")
            text, tokens = decode(
                model.language, tokenizer, plan["reference"]["generation"], memory
            )
            result = prediction(row, text)
            predictions.append(result)
            write(
                output / f"candidate-{index:04d}.json",
                {
                    **result,
                    "tokens": tokens,
                    "native_crop_parity": True,
                    "predicted_streams_sha256": state_digest(features),
                    "semantic_memory_sha256": state_digest({"memory": memory}),
                },
            )
            print(json.dumps({"stage": "semantic_ROI_decoded", "clips": index + 1}), flush=True)
    if (
        state_digest(model.vision.state_dict()) != plan["visual_state_sha256"]
        or fixed_hashes(model) != plan["initial_reference_hashes"]
    ):
        raise ValueError("semantic evaluation changed fixed weights")
    checked(path)
    paired = comparison_items([r["development"] for r in plan["records"]], baseline, predictions)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "clips": 34,
            "all34_frozen_C5_controls_reproduced": True,
            "all_fixed_weights_unchanged": True,
            "predictions": predictions,
            "comparison": paired,
            "mean_WER_diagnostic": sum(r["WER_diagnostic"] for r in predictions) / 34,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": duty.receipt(),
        },
    )


def verified_control(row, case, expected, tokenizer):
    tokens = case.get("tokens")
    if (
        row["role"] != "development"
        or row["training_allowed"] is not False
        or not isinstance(tokens, list)
        or not tokens
        or len(tokens) > 2048
        or any(type(v) is not int or not 0 <= v < 384 for v in tokens)
        or case.get("matches_frozen_control") is not True
        or tokenizer.decode(tokens, skip_special_tokens=True) != case["hypothesis"]
    ):
        raise ValueError("semantic scoring lost actual reserved C5 token/text evidence")
    if any(case.get(k) != v for k, v in prediction(row, case["hypothesis"]).items()):
        raise ValueError("semantic scoring changed a control identity or diagnostic")
    require_control(case["hypothesis"], expected)


def score_specification(inference_plan_path, decoded_path):
    from transformers import ByT5Tokenizer

    inference_plan_path, decoded_path = inference_plan_path.resolve(), decoded_path.resolve()
    plan = checked(inference_plan_path)
    decoded = json.loads(decoded_path.read_text())
    terminal(decoded_path)
    if (
        decoded.get("complete") is not True
        or decoded.get("plan_sha256") != sha256(inference_plan_path)
        or decoded.get("clips") != 34
        or decoded.get("all34_frozen_C5_controls_reproduced") is not True
        or decoded.get("all_fixed_weights_unchanged") is not True
        or decoded.get("optimizer_updates") != 0
        or decoded.get("new_Terra_requests") != 0
        or decoded.get("promotion_authorized") is not False
    ):
        raise ValueError("semantic scoring requires complete unchanged-weight actual inference")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    baseline = json.loads(CONTROL.read_text())["predictions"]
    indexed = {(r["source"], r["example_id"]): r for r in baseline}
    predictions, assets = [], {}
    for index, record in enumerate(plan["records"]):
        row = record["development"]
        control_path = decoded_path.with_name(f"control-{index:04d}.json")
        control = json.loads(control_path.read_text())
        verified_control(
            row, control, indexed[(row["source"], row["example_id"])]["hypothesis"], tokenizer
        )
        candidate_path = decoded_path.with_name(f"candidate-{index:04d}.json")
        predictions.append(
            verified_prediction(row, json.loads(candidate_path.read_text()), tokenizer)
        )
        for asset in (control_path, candidate_path):
            assets[str(asset.relative_to(ROOT))] = sha256(asset)
    paired = comparison_items([r["development"] for r in plan["records"]], baseline, predictions)
    if paired != decoded["comparison"] or predictions != decoded["predictions"]:
        raise ValueError("semantic scoring paired coverage differs from actual decoded cases")
    return {
        "schema": "umi-studio-roi-semantic-scoring/1",
        "policy": SCORE_POLICY,
        "inference_plan": artifact(inference_plan_path),
        "decoded": artifact(decoded_path),
        "assets": assets,
        "comparison": paired,
        "baseline": baseline,
        "predictions": predictions,
        "source_sha256": plan["source_sha256"],
        "planned_Terra_requests": 2 * math.ceil(len(paired["items"]) / 16),
        "promotion_authorized": False,
    }


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != score_specification(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("semantic scoring changed actual outputs, rubric, or selection gate")
    verify_snapshot(path, plan)
    return plan


def run(path, output, *, scoring=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    validate = checked_scores if scoring else checked
    plan = validate(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("semantic evaluation requires a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if scoring:
            scores = score_comparison(plan["comparison"], output / "semantic", RecoveryPacer())
            result = semantic_decision(
                plan["comparison"], scores, plan["baseline"], plan["predictions"]
            )
            validate(path)
            write(
                output / "decision.json",
                {
                    **result,
                    "plan_sha256": sha256(path),
                    "new_Terra_requests": plan["planned_Terra_requests"],
                    "optimizer_updates": 0,
                },
            )
            print(json.dumps(result), flush=True)
        else:
            from train_terra_c5_tail import gpu_lease

            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
            if os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] != "threads;2":
                raise ValueError("semantic ROI inference requires two-thread FFmpeg")
            import cv2

            cv2.setNumThreads(2)
            with gpu_lease():
                infer(path, output, plan)
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
        freeze(args.freeze_scores, score_specification(args.inference_plan, args.decoded))
    elif args.plan and args.output:
        run(args.plan, args.output, scoring=args.score)
    else:
        parser.error(
            "choose inference/score freezing with dependencies, or --plan --output [--score]"
        )


def pipeline(path, output):
    """Block on each own child; score once only after complete inference exit0."""
    path, output = path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("semantic pipeline requires a fresh owned output")
    plan = json.loads(path.read_text())
    verify_snapshot(path, plan)
    if plan["source_sha256"]["tools/evaluate_roi_semantic.py"] != sha256(Path(__file__)):
        raise ValueError("semantic pipeline source differs from its frozen plan")
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
            raise RuntimeError("semantic inference stopped; scoring not started, no retry")
        score_plan = path.parent.with_name(path.parent.name + "-scores") / "plan.json"
        freeze(score_plan, score_specification(path, output / "inference/receipt.json"))
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
            raise RuntimeError("semantic scoring stopped; no retry or model fallback")
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


if __name__ == "__main__":
    main()
