"""Compare qualified original112px C5 vision with the unchanged34 native controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from benchmark_fleurs_dev import sha256
from c5_small_input_vision import STREAMS, SmallInputDino, input_tensor
from evaluate_roi_semantic import INPUT_PLAN
from evaluate_roi_short_translation import (
    CONTROL,
    ROOT,
    decode,
    prediction,
    require_control,
    terminal,
)
from evaluate_roi_short_translation import checked as checked_inputs
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from probe_c5_small_input import checked as checked_qualification
from probe_c5_small_input import legacy_models, pack_rgb
from probe_tiny_rgb_capacity import state_digest
from terra_short_curriculum_learning import comparison_items
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from tiny_rgb_capacity_protocol import verified_comparison
from tiny_rgb_visual_teacher_data import load_targets

POLICY = {
    "scope": "same34 reserved development identities; all C5 control texts reproduced first",
    "appearance": "original C5 hand/face weights, direct native112px crops, no training",
    "native_input_side": 112,
    "appearance_batch_frames": 2,
    "sampling": "complete existing stride2 timeline, independently checked native crop pixels",
    "pose": "unchanged video-derived14D stream; no caption-conditioned inputs",
    "sequence_and_language": "original C5 weights and published generation/CPU budget",
    "CPU_threads": 2,
    "appearance_GPU_duty_fraction": 0.5,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "new_downloads": 0,
    "WER_is_diagnostic_only": True,
    "promotion_authorized": False,
}


def require_qualification(result, plan, plan_sha):
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != plan_sha
        or result.get("component_numerics_passed") is not True
        or result.get("translation_quality_evaluated") is not False
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or result.get("promotion_authorized") is not False
        or result.get("trainable_parameters") != 43260672
        or result.get("trainable_tensors") != 350
        or plan["policy"]["native_side"] != POLICY["native_input_side"]
        or len(result.get("comparisons", [])) != len(plan["records"])
    ):
        raise ValueError("native112 evaluation requires complete unchanged-weight qualification")
    for record, comparison in zip(plan["records"], result["comparisons"], strict=True):
        original = record["original_row"]
        frames = original["original_pair"]["header"]["sampled_frames"]
        if (
            comparison.get("source") != original["source"]
            or comparison.get("example_id") != original["example_id"]
            or comparison.get("complete_crop_frames") != frames
            or comparison.get("numerical_sample_positions") != [0, frames - 1]
            or comparison.get("both_sides_native_pixel_parity") is not True
            or not 0
            <= comparison["control224_maximum_absolute_error"]
            <= plan["policy"]["control_maximum_absolute_error"]
        ):
            raise ValueError("native112 qualification changed actual input/control coverage")
        verified_comparison(comparison["comparison"], 350)


def specification(qualification_plan, qualification_result):
    qualification_plan, qualification_result = (
        qualification_plan.resolve(),
        qualification_result.resolve(),
    )
    if not qualification_result.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("native112 qualification must remain in owned checkpoints")
    qualified = checked_qualification(qualification_plan)
    result = json.loads(qualification_result.read_text())
    terminal(qualification_result)
    require_qualification(result, qualified, sha256(qualification_plan))
    for i, comparison in enumerate(result["comparisons"]):
        if (
            json.loads(qualification_result.with_name(f"comparison-{i:02d}.json").read_text())
            != comparison
        ):
            raise ValueError("native112 qualification differs from its actual comparison files")
    inputs = checked_inputs(INPUT_PLAN)
    if inputs["reference"] != qualified["reference"]:
        raise ValueError("native112 qualification and evaluation reference models differ")
    sources = {**inputs["source_sha256"], **qualified["source_sha256"]}
    for name in ("evaluate_c5_small_input.py", "evaluate_roi_semantic.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-c5-native112-translation-evaluation/1",
        "policy": POLICY,
        "qualification_plan": artifact(qualification_plan),
        "qualification_result": artifact(qualification_result),
        "pretrained_state_sha256": result["pretrained_state_sha256"],
        "input_protocol": artifact(INPUT_PLAN),
        "records": inputs["records"],
        "control": inputs["control"],
        "reference": inputs["reference"],
        "assets": {**inputs["assets"], **qualified["assets"]},
        "resource_policy": qualified["resource_policy"],
        "initial_reference_hashes": inputs["initial_reference_hashes"],
        "source_sha256": sources,
        "C5_upstream_training_inventory_known": False,
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["qualification_plan"]["path"], ROOT / plan["qualification_result"]["path"]
    ):
        raise ValueError(
            "native112 evaluation changed qualification, inputs, weights or generation"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path, qualification_plan, qualification_result):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("native112 evaluation requires a fresh owned plan")
    torch.set_num_threads(2)
    storage()
    plan = specification(qualification_plan, qualification_result)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan": artifact(path), "development_clips": len(plan["records"])}), flush=True
    )


def crops(record):
    from extract_mediapipe_landmarks import unpack_landmarks

    pair = record["development"]["original_pair"]
    video, header = ROOT / pair["raw"]["video"], pair["header"]
    landmark = ROOT / record["landmarks"]["path"]
    if sha256(video) != pair["raw"]["video_sha256"] or artifact(landmark) != record["landmarks"]:
        raise ValueError("native112 evaluation video or landmark bytes changed")
    landmarks, metadata = unpack_landmarks(landmark)
    if (
        metadata["source_sha256"] != pair["raw"]["video_sha256"]
        or metadata["frames"] != header["source_frames"]
        or metadata["fps"] != 25
    ):
        raise ValueError("native112 evaluation source timeline changed")
    sampled = {j: landmarks[i] for j, i in enumerate(range(0, header["source_frames"], 2))}
    if any(
        not np.isfinite(np.asarray(points)).all()
        for value in sampled.values()
        if value is not None
        for points in value.values()
        if points is not None
    ):
        raise ValueError("native112 evaluation active landmarks must be finite")
    hands, face, native_reader = native_helpers()
    side, count = POLICY["native_input_side"], header["sampled_frames"]
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(side, side)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(side, side)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(v, count, side) for v in (facial, left, right)], axis=1)
    del facial, left, right
    reader = native_reader(video)
    try:
        left, right = hands(output_size=(side, side)).extract_hand_frames(
            NativeSampledReader(reader, header), sampled
        )
        facial = face(output_size=(side, side)).extract_face_frames(
            NativeSampledReader(reader, header), sampled
        )
    finally:
        del reader
    reference = np.stack([pack_rgb(v, count, side) for v in (facial, left, right)], axis=1)
    if not np.array_equal(pixels, reference):
        raise ValueError("native112 evaluation streamed/native crop pixels differ")
    return pixels


def predicted_streams(appearance, pixels, original, duty):
    if set(original) != {*STREAMS, "pose_features"} or original["pose_features"].shape != (
        len(pixels),
        14,
    ):
        raise ValueError("native112 appearance cannot drop frames or replace video-derived pose")
    pose = torch.from_numpy(original["pose_features"].copy())
    if pose.dtype != torch.float32 or not torch.isfinite(pose).all():
        raise ValueError("native112 evaluation pose must be finite native FP32")
    parts = {name: [] for name in STREAMS}
    chunk = POLICY["appearance_batch_frames"]
    for start in range(0, len(pixels), chunk):
        batch = input_tensor(pixels[start : start + chunk])

        def forward(batch=batch):
            return {
                name: value.detach().cpu() for name, value in appearance(batch.to("mps")).items()
            }

        for name, values in duty.run(forward).items():
            parts[name].append(values)
    result = {name: torch.cat(parts[name]) for name in STREAMS}
    if any(
        value.shape != (len(pixels), 384) or not torch.isfinite(value).all()
        for value in result.values()
    ):
        raise ValueError(
            "native112 appearance changed frame coverage, feature width or finite values"
        )
    return {**result, "pose_features": pose}


def infer(path, output, plan):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import load_model, native_memory
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer

    native_helpers()
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if (
        budget.duty_fraction() != 0.5
        or budget.cpu_threads() != 2
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("native112 inference requires50% appearance Metal duty/CPU2")
    duty = WinnerDutyCycle(torch, budget)
    model, reference = load_model()
    model.requires_grad_(False).eval()
    encoder = load_frozen_winner_encoder(reference)
    if reference != plan["reference"]:
        raise ValueError("native112 inference loaded another language reference")
    hashes = {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    if hashes != plan["initial_reference_hashes"]:
        raise ValueError("native112 inference changed original C5 sequence/language weights")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    hands, face = legacy_models(plan)
    appearance = SmallInputDino(hands.model, face.model).requires_grad_(False).eval()
    if state_digest(appearance.state_dict()) != plan["pretrained_state_sha256"]:
        raise ValueError("native112 inference loaded another hand/face state")
    appearance = duty.run(lambda: appearance.to("mps"))
    baseline = json.loads(CONTROL.read_text())["predictions"]
    controls = {(row["source"], row["example_id"]): row for row in baseline}
    predictions = []
    with torch.no_grad():
        for index, record in enumerate(plan["records"]):
            storage(output=output)
            row = record["development"]
            memory = native_memory(
                encoder, row["original_pair"], reference["translator_checkpoint_sha256"]
            )
            text, tokens = decode(model.language, tokenizer, reference["generation"], memory)
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
                appearance, pixels, load_targets(row["original_pair"]), duty
            )
            memory = encoder(
                **{name: value[None] for name, value in features.items()}, return_dict=True
            ).last_hidden_state
            if memory.shape != (1, len(pixels), 1536) or not torch.isfinite(memory).all():
                raise ValueError("native112 semantic memory lost complete frame coverage")
            text, tokens = decode(model.language, tokenizer, reference["generation"], memory)
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
            print(
                json.dumps({"stage": "native112_translation_decoded", "clips": index + 1}),
                flush=True,
            )
    if state_digest(appearance.state_dict()) != plan["pretrained_state_sha256"] or hashes != {
        "language": state_digest(model.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }:
        raise ValueError("native112 inference changed a fixed model value")
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "clips": 34,
            "all34_frozen_C5_controls_reproduced": True,
            "all_fixed_weights_unchanged": True,
            "predictions": predictions,
            "comparison": comparison_items(
                [r["development"] for r in plan["records"]], baseline, predictions
            ),
            "mean_WER_diagnostic": sum(row["WER_diagnostic"] for row in predictions) / 34,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "semantic_scores_pending": True,
            "resources": duty.receipt(),
        },
    )


def run(path, output):
    import cv2
    from train_terra_c5_tail import gpu_lease

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("native112 evaluation requires the two-thread FFmpeg policy")
    cv2.setNumThreads(2)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("native112 inference requires fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            infer(path, output, plan)
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--qualification-plan", type=Path)
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        if not (args.qualification_plan and args.qualification_result):
            parser.error("--freeze requires --qualification-plan and --qualification-result")
        freeze(args.freeze, args.qualification_plan, args.qualification_result)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan/--output")


if __name__ == "__main__":
    main()
