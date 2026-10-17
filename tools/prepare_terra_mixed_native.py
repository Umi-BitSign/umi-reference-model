"""Freeze and authenticate all214 easier word plus318 fresh native112 phrase inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from audit_terra_word_landmarks import terminal
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import DEMO
from prepare_roi_broad_corpus import checked as checked_phrases
from terra_mixed_native_input import (
    expected_header,
    input_receipt,
    read_input,
    require_receipt,
    select_inputs,
    training_row,
)
from terra_training_recovery import artifact, verify_snapshot
from terra_word_verified_inputs import checked_result as checked_words
from train_terra_c5_tail import gpu_lease

ROOT = Path(__file__).resolve().parents[1]
WORD_PLAN = ROOT / "runs/terra-word-native112-data-v1/plan.json"
WORD_RESULT = ROOT / "checkpoints/terra-word-native112-data-v1/data.json"
WORD_BINDING = ROOT / "runs/terra-word-native112-data-v1/verified-inputs.json"
PHRASE_PLAN = ROOT / "runs/roi-broad-short-data-v1/plan.json"
PHRASE_RESULT = ROOT / "checkpoints/roi-broad-short-data-v1/data.json"
SENTENCE_PLAN = ROOT / "runs/c5-native112-evaluation-v1/plan.json"
NATIVE_PLAN = ROOT / "runs/c5-winner-full-v2-early/plan.json"
POLICY = {
    "CPU_threads": 2,
    "GPU_use": False,
    "exclusive_existing_C5_lease": True,
    "physical_free_floor_bytes": 80 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "maximum_output_bytes": 32 * 1024**2,
    "native_side": 112,
    "training_clips": {"word": 214, "phrase": 318},
    "sampled_frames": 38876,
    "retain_easier_words": (
        "all original214 training identities and exact verified RGB/pose/index bytes"
    ),
    "phrase_inputs": (
        "fresh original full-frame112px native crops, never resize existing64px caches"
    ),
    "sampling": (
        "complete original indices0,2,4,...; no caption cuts, invented word boundaries "
        "or dropped frames"
    ),
    "pose": (
        "recompute original14D PoseProcessor on sampled landmarks; "
        "require original C5 pose equality"
    ),
    "persisted_inputs": "hashes, shapes and receipts only; no RGB, pose or feature caches",
    "access": "one complete clip resident; native reference decodes one source frame at a time",
    "failure_policy": "stop at first failure; no retry, substitution or quality filtering",
    "new_downloads": 0,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "model_training_ready": False,
    "fresh_full_model_and_Terra_risk_qualification_required": True,
    "optimizer_recipe_selected": False,
    "translation_quality_evaluated": False,
    "promotion_authorized": False,
}


def guard(output=None, *, disk_usage=shutil.disk_usage):
    used = 0
    if output is not None:
        if output.absolute() != output.resolve() or not output.is_relative_to(ROOT / "checkpoints"):
            raise ValueError("mixed native output must remain canonical in owned checkpoints/")
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("mixed native output cannot follow aliases")
            if path.is_file():
                used += path.stat().st_size
    remaining = POLICY["maximum_output_bytes"] - used
    if remaining < 0 or disk_usage(ROOT).free < (
        POLICY["physical_free_floor_bytes"] + POLICY["headroom_bytes"] + remaining
    ):
        raise RuntimeError("mixed native preflight exceeded32MiB or unchanged80GiB floor/headroom")


def specification():
    from tiny_rgb_full_run import runtime_binding

    words = checked_words(WORD_PLAN, WORD_RESULT)
    if json.loads(WORD_BINDING.read_text()) != words:
        raise ValueError("mixed native words differ from the accepted complete input binding")
    phrase_plan = checked_phrases(PHRASE_PLAN)
    terminal(PHRASE_RESULT.parent)
    phrase_data = json.loads(PHRASE_RESULT.read_text())
    if (
        phrase_data.get("plan_sha256") != sha256(PHRASE_PLAN)
        or phrase_data.get("all_streamed_pixels_equal_native_reference") is not True
        or phrase_data.get("coverage_counts") != phrase_plan["coverage_counts"]
        or phrase_data.get("GPU_use") is not False
        or phrase_data.get("new_Terra_requests") != 0
        or phrase_data.get("optimizer_updates") != 0
        or phrase_data.get("translation_promotion_authorized") is not False
    ):
        raise ValueError("mixed native phrases lost their original complete preparation")
    sentence_plan = json.loads(SENTENCE_PLAN.read_text())
    native_rows = json.loads(NATIVE_PLAN.read_text())["evaluation"]
    cases, boundaries = select_inputs(words, phrase_plan, phrase_data, sentence_plan, native_rows)
    sources = {**json.loads(WORD_PLAN.read_text())["source_sha256"], **phrase_plan["source_sha256"]}
    for name in (
        "terra_mixed_native_input.py",
        "prepare_terra_mixed_native.py",
        "terra_word_verified_inputs.py",
        "c5_word_training.py",
        "c5_word_semantic_model.py",
        "c5_small_input_semantic.py",
        "terra_word_semantic_protocol.py",
        "train_terra_c5_tail.py",
        "terra_training_recovery.py",
        "training_media_artifact.py",
        "probe_c5_small_input.py",
        "tiny_rgb_full_run.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-complete-mixed-native112-input-preflight/1",
        "policy": POLICY,
        "word_plan": artifact(WORD_PLAN),
        "word_result": artifact(WORD_RESULT),
        "word_binding": artifact(WORD_BINDING),
        "phrase_plan": artifact(PHRASE_PLAN),
        "phrase_result": artifact(PHRASE_RESULT),
        "phrase_authority": terminal(PHRASE_RESULT.parent),
        "sentence_boundary": artifact(SENTENCE_PLAN),
        "native_boundary": artifact(NATIVE_PLAN),
        "records": cases,
        "metadata_boundaries": boundaries,
        "counts": dict(Counter(case["kind"] for case in cases)),
        "sampled_frames_by_kind": {
            kind: sum(
                expected_header(case)["sampled_frames"] for case in cases if case["kind"] == kind
            )
            for kind in ("word", "phrase")
        },
        "source_sha256": sources,
        "assets": {
            **phrase_plan["assets"],
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    DEMO / name
                    for name in (
                        "crop_hands.py",
                        "crop_face.py",
                        "video_reader.py",
                        "body_features.py",
                    )
                )
            },
        },
        "runtime_binding": runtime_binding(),
        "resource_policy": phrase_plan["resource_policy"],
        "resource_policy_sha256": phrase_plan["resource_policy_sha256"],
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("mixed native plan changed complete training inputs, boundaries or policy")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("mixed native preflight requires a fresh owned plan")
    guard()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "counts": plan["counts"]}), flush=True)


def prepare(path, output):
    plan = checked(path)
    original = output.absolute()
    output = output.resolve()
    if original != output or output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("mixed native preflight requires fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        receipts, frames = [], Counter()
        for index, case in enumerate(plan["records"]):
            guard(output)
            values = read_input(case)
            target = output / f"input-{index:04d}.json"
            write(target, input_receipt(case, values))
            receipts.append(artifact(target))
            frames[case["kind"]] += len(values["frame_indices"])
            del values
            print(json.dumps({"verified_inputs": index + 1, "kind": case["kind"]}), flush=True)
        if dict(frames) != plan["sampled_frames_by_kind"]:
            raise ValueError("mixed native preparation lost complete word/phrase frame coverage")
        checked(path)
        guard(output)
        write(
            output / "data.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "receipts": receipts,
                "counts": plan["counts"],
                "sampled_frames_by_kind": dict(frames),
                "all_streamed_pixels_equal_native_reference": True,
                "all_original_pose_and_easier_word_inputs_preserved": True,
                "CPU_threads": 2,
                "GPU_use": False,
                "model_training_ready": False,
                "optimizer_recipe_selected": False,
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "translation_quality_evaluated": False,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def run(path, output):
    import cv2

    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("mixed native preflight requires its two-thread FFmpeg policy")
    cv2.setNumThreads(2)
    # Serialize this CPU decoder with owned C5 training/inference before any
    # heavy validation, new output directory or source-video access.
    with gpu_lease():
        prepare(path, output)


def checked_result(plan_path, result_path):
    original = result_path.absolute()
    result_path = result_path.resolve()
    if original != result_path or not result_path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("mixed native result must be a canonical owned receipt")
    plan = checked(plan_path)
    authority = terminal(result_path.parent)
    result = json.loads(result_path.read_text())
    expected = {
        "complete": True,
        "plan_sha256": sha256(plan_path),
        "counts": plan["counts"],
        "sampled_frames_by_kind": plan["sampled_frames_by_kind"],
        "all_streamed_pixels_equal_native_reference": True,
        "all_original_pose_and_easier_word_inputs_preserved": True,
        "CPU_threads": 2,
        "GPU_use": False,
        "model_training_ready": False,
        "optimizer_recipe_selected": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "translation_quality_evaluated": False,
        "promotion_authorized": False,
    }
    if any(result.get(key) != value for key, value in expected.items()) or len(
        result.get("receipts", [])
    ) != len(plan["records"]):
        raise ValueError("mixed native result lacks complete fixed word/phrase preparation")
    paths = [
        result_path.with_name(f"input-{index:04d}.json") for index in range(len(plan["records"]))
    ]
    if set(result_path.parent.glob("input-*.json")) != set(paths):
        raise ValueError("mixed native result changed exact per-input receipt inventory")
    records = []
    for case, path, descriptor in zip(plan["records"], paths, result["receipts"], strict=True):
        if artifact(path) != descriptor:
            raise ValueError("mixed native per-input receipt bytes changed")
        saved = json.loads(path.read_text())
        require_receipt(case, saved)
        records.append({"case": case, "receipt": descriptor, "verified_input": saved})
    return {
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "authority": authority,
        "records": records,
        "counts": plan["counts"],
        "model_training_qualification_still_required": True,
        "verification_source_sha256": sha256(Path(__file__)),
    }


def load_verified_input(bound):
    case = bound["case"]
    training_row(case)  # Reject forbidden roles before receipt or media access.
    path = ROOT / bound["receipt"]["path"]
    if artifact(path) != bound["receipt"]:
        raise ValueError("mixed native input lost its actual preflight receipt bytes")
    saved = json.loads(path.read_text())
    if saved != bound["verified_input"]:
        raise ValueError("mixed native input changed its frozen preflight binding")
    require_receipt(case, saved)
    values = read_input(case)
    if input_receipt(case, values) != saved:
        raise ValueError("mixed native RGB, pose or complete timeline changed on reread")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze and not args.plan and not args.output:
        freeze(args.freeze)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("choose --freeze or --plan/--output")


if __name__ == "__main__":
    main()
