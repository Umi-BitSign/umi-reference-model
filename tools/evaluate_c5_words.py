"""Complete96-word comparison of original224px C5 and fixed-final native112 weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import sha256
from c5_small_input_vision import STREAMS, SmallInputDino, input_tensor
from c5_word_evaluation_io import (
    checked_streams,
    development_records,
    evaluation_row,
    original_control_input,
    save_streams,
    storage,
)
from c5_word_final import checked_final
from evaluate_c5_small_input import predicted_streams
from evaluate_c5_word_sentences import candidate_appearance
from evaluate_roi_semantic import verified_control
from evaluate_roi_short_translation import decode, prediction, terminal
from launch_terra_visual_job import identity, now, write
from probe_c5_small_input import legacy_models
from probe_c5_word_semantic import input_evidence, require_resources
from probe_jeff_terra_quality import terra_payload
from probe_tiny_rgb_capacity import state_digest
from score_roi_short_translation import POLICY as SCORE_POLICY
from score_roi_short_translation import verified_prediction
from semantic_quality_judge import parse_terra_scores
from terra_short_curriculum_learning import (
    RECIPE as SCORE_RECIPE,
)
from terra_short_curriculum_learning import (
    comparison_items,
    score_comparison,
    semantic_decision,
)
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot
from terra_word_verified_inputs import load_verified_input

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "scope": "all96 original word-development clips, all8 reserved signers and60 known labels",
    "baseline": "original public C5; independently decoded native224px CPU appearance",
    "candidate": "complete fixed-final word curriculum; qualified linear native112px appearance",
    "timeline": "complete source0,2,4,... with independently verified native crop pixels",
    "pose": "same authenticated original video-derived14D stream and missing-detection behavior",
    "control_order": (
        "inference child reproduces all96 original224px control tokens before trained candidate "
        "loading, input reads or decoding"
    ),
    "baseline_input_verification": (
        "reproduce existing112px input-receipt hashes without model inference, then independently "
        "crop original source frames at224px for baseline inference"
    ),
    "cache": "private original224px features only; one complete clip resident at a time",
    "generation": "unchanged original C5 sequence/language/generation and CPU budget",
    "CPU_threads": 2,
    "candidate_appearance_GPU_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_downloads": 0,
    "WER_is_diagnostic_only": True,
    "C5_upstream_training_inventory_known": False,
    "word_labels_shared_with_training_intentionally": True,
    "word_learning_is_not_continuous_translation_evidence": True,
    "promotion_authorized": False,
}


def specification(training_path, result_path):
    fit, result, evidence = checked_final(training_path, result_path)
    binding, _ = input_evidence()
    records = development_records(binding)
    sources = dict(fit["source_sha256"])
    for name in (
        "evaluate_c5_words.py",
        "c5_word_final.py",
        "c5_word_evaluation_io.py",
        "evaluate_c5_word_sentences.py",
        "score_roi_short_translation.py",
        "evaluate_roi_semantic.py",
        "terra_semantic_selection.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-C5-complete-word-evaluation/1",
        "policy": POLICY,
        "training_plan": artifact(training_path),
        "training_result": artifact(result_path),
        "training_evidence": evidence,
        "records": records,
        "verified_inputs": fit["verified_inputs"],
        "reference": fit["model_plan"]["reference"],
        "model_plan": fit["model_plan"],
        "initial_reference_hashes": fit["model_plan"]["initial_reference_hashes"],
        "candidate_visual": result["visual"],
        "candidate_visual_state_sha256": result["visual_final_sha256"],
        "resource_policy": fit["model_plan"]["resource_policy"],
        "source_sha256": sources,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["training_plan"]["path"], ROOT / plan["training_result"]["path"]
    ):
        raise ValueError(
            "word evaluation changed complete training, final values or reserved inputs"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path, training_path, result_path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word evaluation requires a fresh owned plan")
    plan = specification(training_path.resolve(), result_path.resolve())
    storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "development_clips": 96}), flush=True)


def backbone_hashes(encoder, language):
    return {
        "encoder": state_digest(encoder.state_dict()),
        "language": state_digest(language.state_dict()),
    }


def models(plan):
    from c5_terra_tail import load_model
    from prepare_roi_appearance import native_helpers
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer

    native_helpers()
    original, reference = load_model()
    language = original.language.requires_grad_(False).eval()
    encoder = load_frozen_winner_encoder(reference)
    if (
        reference != plan["reference"]
        or backbone_hashes(encoder, language) != plan["initial_reference_hashes"]
        or any(
            parameter.device.type != "cpu" or parameter.requires_grad
            for model in (encoder, language)
            for parameter in model.parameters()
        )
    ):
        raise ValueError("word evaluation changed original fixed CPU C5 sequence/language values")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    return encoder, language, tokenizer


def memory_for(encoder, streams):
    features = {}
    for name in (*STREAMS, "pose_features"):
        value = streams[name]
        features[name] = (torch.from_numpy(value) if isinstance(value, np.ndarray) else value)[None]
    memory = encoder(**features, return_dict=True).last_hidden_state
    frames = len(streams["pose_features"])
    if memory.shape != (1, frames, 1536) or not torch.isfinite(memory).all():
        raise ValueError("word evaluation changed complete native memory timeline")
    return memory


def original_streams(appearance, values):
    frames = len(values["pixels"])
    parts = {name: [] for name in STREAMS}
    for start in range(0, frames, 2):
        pixels = input_tensor(values["pixels"][start : start + 2], side=224)
        features = appearance(pixels)
        for name in STREAMS:
            parts[name].append(features[name].detach().cpu().numpy().copy())
    return {
        **{name: np.concatenate(parts[name]) for name in STREAMS},
        "pose_features": values["pose_features"],
        "frame_indices": values["frame_indices"],
    }


@torch.no_grad()
def baseline(path, output, plan):
    from evaluate_tiny_rgb_feature_winner import peak_bytes

    encoder, language, tokenizer = models(plan)
    hands, face = legacy_models(plan["model_plan"])
    appearance = SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
    expected = plan["model_plan"]["pretrained_state_sha256"]
    if state_digest(appearance.state_dict()) != expected:
        raise ValueError("word224 controls did not load original public visual values")
    predictions, files = [], []
    for index, case in enumerate(plan["records"]):
        storage(output=output)
        row = evaluation_row(case)
        values = original_control_input(case)
        streams = original_streams(appearance, values)
        del values
        descriptor = save_streams(output / f"streams-{index:04d}.npz", streams, case)
        memory = memory_for(encoder, streams)
        text, tokens = decode(language, tokenizer, plan["reference"]["generation"], memory)
        result = prediction(row, text)
        filename = output / f"baseline-{index:04d}.json"
        write(
            filename,
            {
                **result,
                "tokens": tokens,
                "native_crop_parity": True,
                "native_input_side": 224,
                "input_receipt": case["receipt"],
                "original_streams": descriptor,
            },
        )
        predictions.append(result)
        files.append(artifact(filename))
        del streams, memory
        print(json.dumps({"stage": "word224_baseline", "clips": index + 1}), flush=True)
    if (
        state_digest(appearance.state_dict()) != expected
        or backbone_hashes(encoder, language) != plan["initial_reference_hashes"]
    ):
        raise ValueError("word original224 baseline changed a frozen C5 value")
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "clips": 96,
            "all_original_C5_values_unchanged": True,
            "original_visual_state_sha256": expected,
            "predictions": predictions,
            "baseline_files": files,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": {
                "GPU_use": False,
                "CPU_threads": 2,
                "memory_policy": "framework_default_no_artificial_six_GiB_cap",
                "process_peak_bytes": peak_bytes(),
            },
        },
    )


def checked_baseline(receipt_path, plan_path, plan, tokenizer):
    terminal(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("complete") is not True
        or receipt.get("plan_sha256") != sha256(plan_path)
        or receipt.get("clips") != 96
        or receipt.get("all_original_C5_values_unchanged") is not True
        or receipt.get("original_visual_state_sha256")
        != plan["model_plan"]["pretrained_state_sha256"]
        or receipt.get("optimizer_updates") != 0
        or receipt.get("new_Terra_requests") != 0
        or receipt.get("promotion_authorized") is not False
        or receipt.get("resources", {}).get("GPU_use") is not False
        or receipt.get("resources", {}).get("CPU_threads") != 2
        or receipt.get("resources", {}).get("memory_policy")
        != "framework_default_no_artificial_six_GiB_cap"
        or type(receipt.get("resources", {}).get("process_peak_bytes")) is not int
        or receipt["resources"]["process_peak_bytes"] <= 0
    ):
        raise ValueError("word comparison requires complete original224 C5 controls")
    directory = receipt_path.parent
    if {file.name for file in directory.glob("baseline-*.json")} != {
        f"baseline-{i:04d}.json" for i in range(96)
    } or {file.name for file in directory.glob("streams-*.npz")} != {
        f"streams-{i:04d}.npz" for i in range(96)
    }:
        raise ValueError("word original224 baseline changed complete actual file coverage")
    predictions, files, streams, evidence = [], [], [], [artifact(receipt_path)]
    for index, case in enumerate(plan["records"]):
        filename = directory / f"baseline-{index:04d}.json"
        saved = json.loads(filename.read_text())
        predictions.append(verified_prediction(evaluation_row(case), saved, tokenizer))
        descriptor = saved["original_streams"]
        if (
            saved.get("input_receipt") != case["receipt"]
            or saved.get("native_input_side") != 224
            or descriptor != artifact(directory / f"streams-{index:04d}.npz")
        ):
            raise ValueError("word original224 controls changed actual input or feature bindings")
        checked_streams(descriptor, case)
        files.append(artifact(filename))
        streams.append(descriptor)
        evidence.extend((artifact(filename), descriptor))
    if predictions != receipt["predictions"] or files != receipt["baseline_files"]:
        raise ValueError("word original224 receipt differs from actual token/stream files")
    return {"predictions": predictions, "streams": streams, "evidence": evidence}


@torch.no_grad()
def infer(path, output, plan, baseline_path):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle

    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.cpu_threads() != 2
        or budget.duty_fraction() != 0.5
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("word candidate inference requires CPU2 and appearance Metal50%")
    duty = WinnerDutyCycle(torch, budget)
    encoder, language, tokenizer = models(plan)
    original = checked_baseline(baseline_path, path, plan, tokenizer)
    for index, (case, descriptor, expected) in enumerate(
        zip(plan["records"], original["streams"], original["predictions"], strict=True)
    ):
        storage(output=output)
        streams = checked_streams(descriptor, case)
        memory = memory_for(encoder, streams)
        text, tokens = decode(language, tokenizer, plan["reference"]["generation"], memory)
        saved = json.loads(baseline_path.with_name(f"baseline-{index:04d}.json").read_text())
        if text != expected["hypothesis"] or tokens != saved["tokens"]:
            raise ValueError("word control did not reproduce complete frozen original C5 tokens")
        write(
            output / f"control-{index:04d}.json",
            {
                **prediction(evaluation_row(case), text),
                "tokens": tokens,
                "matches_frozen_control": True,
                "input_receipt": case["receipt"],
                "original_streams": descriptor,
            },
        )
        del streams, memory
        print(json.dumps({"stage": "word224_control_reproduced", "clips": index + 1}), flush=True)
    hands, face = legacy_models(plan["model_plan"])
    appearance = candidate_appearance(plan, hands.model, face.model).requires_grad_(False).eval()
    appearance = duty.run(lambda: appearance.to("mps"))
    predictions = []
    for index, (case, descriptor) in enumerate(
        zip(plan["records"], original["streams"], strict=True)
    ):
        storage(output=output)
        values = load_verified_input(case, purpose="development")
        cached = checked_streams(descriptor, case)
        if not np.array_equal(
            values["pose_features"], cached["pose_features"]
        ) or not np.array_equal(values["frame_indices"], cached["frame_indices"]):
            raise ValueError("word candidate changed the original fixed pose or source timeline")
        streams = predicted_streams(
            appearance,
            values["pixels"],
            {name: cached[name] for name in (*STREAMS, "pose_features")},
            duty,
        )
        memory = memory_for(encoder, streams)
        text, tokens = decode(language, tokenizer, plan["reference"]["generation"], memory)
        result = prediction(evaluation_row(case), text)
        write(
            output / f"candidate-{index:04d}.json",
            {
                **result,
                "tokens": tokens,
                "native_crop_parity": True,
                "input_receipt": case["receipt"],
                "predicted_streams_sha256": state_digest(streams),
                "semantic_memory_sha256": state_digest({"memory": memory}),
            },
        )
        predictions.append(result)
        del values, cached, streams, memory
        print(json.dumps({"stage": "word112_candidate", "clips": index + 1}), flush=True)
    if (
        duty.run(lambda: state_digest(appearance.state_dict()))
        != plan["candidate_visual_state_sha256"]
        or backbone_hashes(encoder, language) != plan["initial_reference_hashes"]
    ):
        raise ValueError("word comparison changed final visual or original frozen C5 values")
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "clips": 96,
            "all96_original_C5_control_tokens_reproduced": True,
            "all_fixed_values_unchanged": True,
            "baseline": artifact(baseline_path),
            "predictions": predictions,
            "comparison": comparison_items(
                [evaluation_row(case) for case in plan["records"]],
                original["predictions"],
                predictions,
            ),
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": duty.receipt(),
        },
    )


def score_specification(plan_path, decoded_path):
    from transformers import ByT5Tokenizer

    plan = checked(plan_path)
    terminal(decoded_path)
    receipt = json.loads(decoded_path.read_text())
    if (
        receipt.get("complete") is not True
        or receipt.get("plan_sha256") != sha256(plan_path)
        or receipt.get("clips") != 96
        or receipt.get("all96_original_C5_control_tokens_reproduced") is not True
        or receipt.get("all_fixed_values_unchanged") is not True
        or receipt.get("optimizer_updates") != 0
        or receipt.get("new_Terra_requests") != 0
        or receipt.get("promotion_authorized") is not False
    ):
        raise ValueError("word scoring requires complete reproduced-control final-model inference")
    require_resources(receipt["resources"])
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["reference"]["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    baseline_path = ROOT / receipt["baseline"]["path"]
    if artifact(baseline_path) != receipt["baseline"]:
        raise ValueError("word scoring lost actual original224 baseline receipt")
    original = checked_baseline(baseline_path, plan_path, plan, tokenizer)
    assets, predictions = original["evidence"], []
    for prefix in ("control", "candidate"):
        if {file.name for file in decoded_path.parent.glob(prefix + "-*.json")} != {
            f"{prefix}-{index:04d}.json" for index in range(96)
        }:
            raise ValueError("word scoring cannot omit or add a decoded identity")
    for index, (case, expected, streams) in enumerate(
        zip(plan["records"], original["predictions"], original["streams"], strict=True)
    ):
        row = evaluation_row(case)
        control_path = decoded_path.with_name(f"control-{index:04d}.json")
        candidate_path = decoded_path.with_name(f"candidate-{index:04d}.json")
        control, candidate = [
            json.loads(file.read_text()) for file in (control_path, candidate_path)
        ]
        verified_control(row, control, expected["hypothesis"], tokenizer)
        baseline = json.loads(baseline_path.with_name(f"baseline-{index:04d}.json").read_text())
        if (
            control.get("input_receipt") != case["receipt"]
            or control["tokens"] != baseline["tokens"]
            or control.get("original_streams") != streams
            or candidate.get("input_receipt") != case["receipt"]
            or any(
                re.fullmatch(r"[0-9a-f]{64}", candidate.get(key, "")) is None
                for key in ("predicted_streams_sha256", "semantic_memory_sha256")
            )
        ):
            raise ValueError(
                "word scoring changed actual input, tokens or complete feature evidence"
            )
        predictions.append(verified_prediction(row, candidate, tokenizer))
        assets.extend((artifact(control_path), artifact(candidate_path)))
    comparison = comparison_items(
        [evaluation_row(case) for case in plan["records"]], original["predictions"], predictions
    )
    if comparison != receipt["comparison"] or predictions != receipt["predictions"]:
        raise ValueError("word paired scoring differs from actual decoded identities")
    return {
        "schema": "umi-studio-C5-complete-word-semantic-scoring/1",
        "policy": SCORE_POLICY,
        "inference_plan": artifact(plan_path),
        "decoded": artifact(decoded_path),
        "assets": assets,
        "comparison": comparison,
        "baseline": original["predictions"],
        "predictions": predictions,
        "source_sha256": plan["source_sha256"],
        "planned_Terra_requests": 2 * math.ceil(len(comparison["items"]) / 16),
        "promotion_authorized": False,
    }


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != score_specification(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("word scoring changed actual outputs, original controls or semantic gate")
    verify_snapshot(path, plan)
    return plan


def authenticated_passes(comparison, directory):
    if json.loads((directory / "blinded-inputs.json").read_text()) != comparison:
        raise ValueError("word semantic scoring changed actual blinded pairings")
    passes = []
    for repeat in range(2):
        indices = list(range(len(comparison["items"])))
        random.Random(SCORE_RECIPE["seed"] + repeat).shuffle(indices)
        joined = [None] * len(indices)
        expected = {
            f"pass-{repeat + 1}-batch-{offset // 16:03d}" for offset in range(0, len(indices), 16)
        }
        if {path.name for path in directory.glob(f"pass-{repeat + 1}-batch-*")} != expected:
            raise ValueError("word semantic scoring changed actual request coverage")
        for offset in range(0, len(indices), 16):
            chosen = indices[offset : offset + 16]
            path = directory / f"pass-{repeat + 1}-batch-{offset // 16:03d}"
            request, started, response, scored = [
                json.loads((path / name).read_text())
                for name in ("request.json", "started.json", "response.json", "scores.json")
            ]
            if (
                request != terra_payload([comparison["items"][i] for i in chosen])
                or started["request_sha256"] != sha256(path / "request.json")
                or started["automatic_retries"] != 0
                or started["pairs"] != len(chosen)
                or started["model"] != "gpt-5.6-terra"
                or started["reasoning_effort"] != "medium"
                or response.get("model") != "gpt-5.6-terra"
                or response.get("reasoning", {}).get("effort") != "medium"
                or response.get("status") != "completed"
                or scored["model"] != "gpt-5.6-terra"
                or scored["reasoning_effort"] != "medium"
                or scored["response_sha256"] != sha256(path / "response.json")
            ):
                raise ValueError("word semantic scores lack authentic requests, model or responses")
            parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(chosen))])
            scores = [parsed[f"pair_{i:02d}"] for i in range(len(chosen))]
            if scores != scored["scores"]:
                raise ValueError("word semantic scores differ from the actual raw response")
            for index, score in zip(chosen, scores, strict=True):
                joined[index] = score
        if joined != json.loads((directory / f"pass-{repeat + 1}-scores.json").read_text()):
            raise ValueError("word complete semantic pass changed response-to-item mapping")
        passes.append(joined)
    return passes


def score(path, output, plan):
    measured = score_comparison(plan["comparison"], output / "semantic", RecoveryPacer())
    actual = authenticated_passes(plan["comparison"], output / "semantic")
    if actual != measured:
        raise ValueError("word semantic runner differs from its actual raw score evidence")
    result = semantic_decision(plan["comparison"], actual, plan["baseline"], plan["predictions"])
    result.update(
        {
            "scope": (
                "all96 reserved complete word-caption clips;60 intentionally shared known labels"
            ),
            "all8_local_development_signers_reserved": True,
            "word_recognition_curriculum_only": True,
            "continuous_translation_win_claimed": False,
            "plan_sha256": sha256(path),
            "optimizer_updates": 0,
            "new_Terra_requests": plan["planned_Terra_requests"],
        }
    )
    checked_scores(path)
    write(output / "decision.json", result)
    print(json.dumps(result), flush=True)


def run(path, output, mode, baseline_path=None):
    import cv2
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("word evaluation requires its bound two-thread decoder")
    plan = checked_scores(path) if mode == "scores" else checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word evaluation child needs fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if mode == "scores":
            score(path, output, plan)
        else:
            with gpu_lease():
                if mode == "baseline":
                    baseline(path, output, plan)
                elif mode == "inference" and baseline_path:
                    infer(path, output, plan, baseline_path.resolve())
                else:
                    raise ValueError("word inference requires actual original224 baseline receipts")
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def pipeline(path, output):
    from probe_roi_seen_translation import freeze as freeze_scoring

    path, output = path.resolve(), output.resolve()
    plan = checked(path)
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word comparison pipeline needs a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for mode in ("baseline", "inference"):
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
            if mode == "inference":
                command += ["--baseline", str(output / "baseline/receipt.json")]
            child = subprocess.run(command, check=False)
            if child.returncode:
                raise RuntimeError(f"word {mode} child stopped; no retry or scoring")
            receipt_path = output / mode / "receipt.json"
            terminal(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("complete") is not True
                or receipt.get("clips") != 96
                or receipt.get("plan_sha256") != sha256(path)
            ):
                raise ValueError("word child lacks its actual complete same-plan96-input receipt")
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
            raise RuntimeError("word scoring child stopped; no retry or fallback")
        checked(path)
        write(
            output / "complete.json",
            {
                "complete": True,
                "inference_plan": artifact(path),
                "scoring_plan": artifact(score_path),
                "baseline": artifact(output / "baseline/receipt.json"),
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
    for name in ("freeze", "training-plan", "training-result", "plan", "output", "baseline"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("baseline", "inference", "scores"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze and args.training_plan and args.training_result:
        freeze(args.freeze, args.training_plan, args.training_result)
        if args.pipeline and args.output:
            pipeline(args.freeze, args.output)
    elif args.plan and args.output and args.pipeline:
        pipeline(args.plan, args.output)
    elif args.plan and args.output and args.mode:
        run(args.plan, args.output, args.mode, args.baseline)
    else:
        parser.error("freeze only after complete training, or run an existing comparison plan")


if __name__ == "__main__":
    main()
