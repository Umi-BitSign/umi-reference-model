"""Check a complete trained timing final against fresh original native224 video.

Four fixed training clips only. No development or independent confirmation input,
judge call or optimizer update is permitted by this controller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from c5_timing_feature_data import (
    ROOT,
    WIDTHS,
    TimingFeatureDataset,
    artifact,
    bound,
    owned,
    read,
    require,
    sha,
    terminal,
)
from launch_terra_visual_job import identity, now, write

INDICES = [0, 1, 214, 215]
POLICY = {
    "indices": INDICES,
    "input": "fresh complete native224 pixels with independent streamed/native parity",
    "appearance": "original frozen public face/hand values; no trained visual parameters",
    "comparison": "fresh native and stored complete streams through the trained adapter",
    "absolute_tolerance": 1e-4,
    "relative_tolerance": 1e-4,
    "original_full_generation_tokens_must_match": True,
    "CPU_threads": 2,
    "GPU_use": False,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "development_or_confirmation_inputs": 0,
    "quality_gain_established": False,
    "promotion_authorized": False,
}


def specification(final_audit_path):
    from c5_timing_final import checked_final

    final = checked_final(final_audit_path)
    training = read(bound(final["plan"]))
    data = read(bound(training["data_audit"]))
    feature_plan = read(bound(data["plan"]))
    sources = dict(final["source_sha256"])
    for name in ("qualify_c5_timing_native.py", "c5_timing_candidate.py", "c5_timing_final.py"):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-trained-timing-native-qualification-plan/1",
        "final_audit": artifact(final_audit_path),
        "adapter": final["adapter"],
        "model_plan": final["model_plan"],
        "data_audit": training["data_audit"],
        "cases": [feature_plan["records"][index] for index in INDICES],
        "records": [data["records"][index] for index in INDICES],
        "policy": POLICY,
        "source_sha256": sources,
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(plan == specification(bound(plan["final_audit"])), "trained native timing plan changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path, final_audit_path):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "trained native timing requires a fresh plan")
    plan = specification(final_audit_path)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def compare_array(left, right, message):
    require(
        left.shape == right.shape
        and left.dtype == right.dtype == np.float32
        and np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.allclose(
            left, right, atol=POLICY["absolute_tolerance"], rtol=POLICY["relative_tolerance"]
        ),
        message,
    )
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def compare_clip(model, tokenizer, cached, fresh, generation):
    import torch
    from c5_terra_tail import candidate_labels
    from c5_timing_semantic_model import clip_input
    from evaluate_roi_short_translation import decode
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList

    require(
        cached["row"] == fresh["row"]
        and cached["row"]["role"] == "training"
        and cached["row"]["training_allowed"] is True
        and cached["header"] == fresh["header"]
        and set(cached["streams"]) == set(fresh["streams"]) == set(WIDTHS)
        and cached["frame_indices"].dtype == fresh["frame_indices"].dtype == np.int64
        and cached["valid_intervals"].dtype == fresh["valid_intervals"].dtype == np.bool_
        and cached["frame_indices"].tobytes() == fresh["frame_indices"].tobytes()
        and cached["valid_intervals"].tobytes() == fresh["valid_intervals"].tobytes()
        and cached["streams"]["pose_features"].tobytes()
        == fresh["streams"]["pose_features"].tobytes(),
        "trained native timing changed original pose, intervals or complete source grid",
    )
    before = {
        name: compare_array(
            cached["streams"][name], fresh["streams"][name], "fresh native timing streams differ"
        )
        for name in cached["streams"]
    }
    clip, native = clip_input(cached), clip_input(fresh)
    mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
    anchor = {
        "tokens": tokenizer(cached["row"]["reference"], truncation=False).input_ids,
        "truncated": False,
        "termination": "eos",
    }
    labels = candidate_labels([anchor])[0]
    with torch.no_grad():
        features, _, timing = model.timed_features(clip, mask)
        native_features, _, native_timing = model.timed_features(native, mask)
        warped = {
            name: compare_array(
                features[name][0].numpy(),
                native_features[name][0].numpy(),
                "trained native warped features differ",
            )
            for name in features
        }
        require(
            torch.allclose(timing["positions"], native_timing["positions"], rtol=1e-4, atol=1e-4),
            "trained native timing positions differ",
        )
        cached_memory, _ = model.encode(clip, mask)
        native_memory, _ = model.encode(native, mask)
        memory_difference = compare_array(
            cached_memory[0].numpy(),
            native_memory[0].numpy(),
            "trained native semantic memory differs",
        )
        cached_output = model(clip, mask, labels)["language"]
        native_output = model(native, mask, labels)["language"]
        logits_difference = compare_array(
            cached_output.logits.numpy(),
            native_output.logits.numpy(),
            "trained native teacher-forcing logits differ",
        )
        require(
            torch.allclose(cached_output.loss, native_output.loss, rtol=1e-4, atol=1e-4),
            "trained native caption loss differs",
        )
        _, cached_tokens = decode(model.language, tokenizer, generation, cached_memory)
        native_tokens = model.generate(
            native,
            mask,
            **generation,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([CpuBudget()]),
        )[0].tolist()
        require(cached_tokens == native_tokens, "trained native full generation tokens differ")
    return {
        "row": cached["row"],
        "frames": mask.shape[1],
        "source_feature_maximum_differences": before,
        "warped_feature_maximum_differences": warped,
        "semantic_memory_maximum_difference": memory_difference,
        "teacher_forcing_logits_maximum_difference": logits_difference,
        "native_generation_tokens_equal": True,
        "tokens": native_tokens,
        "generation": generation,
    }


def qualify(path, output, plan):
    import torch
    from c5_small_input_vision import SmallInputDino
    from c5_timing_candidate import load_candidate, require_candidate
    from c5_timing_native_input import input_receipt, read_training224
    from c5_timing_semantic_model import require_model
    from evaluate_c5_words import original_streams
    from prepare_c5_timing_features import compare_features, guard, native_reference
    from probe_c5_small_input import legacy_models
    from probe_tiny_rgb_capacity import state_digest

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    model, tokenizer, binding = load_candidate(bound(plan["final_audit"]))
    hands, face = legacy_models(plan["model_plan"])
    appearance = SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
    visual = state_digest(appearance.state_dict())
    require(
        visual == plan["model_plan"]["pretrained_state_sha256"]
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
            for p in appearance.parameters()
        ),
        "trained native timing loaded different visual values",
    )
    comparisons = []
    for position, (index, case) in enumerate(zip(INDICES, plan["cases"], strict=True)):
        guard(output)
        cached = dataset.load(index)
        require(
            cached["row"] == plan["records"][position]["row"],
            "trained native timing case join changed",
        )
        pixels = read_training224(case)
        receipt = input_receipt(pixels)
        require(
            receipt == read(bound(plan["records"][position]["receipt"]))["input"],
            "trained native source pixels or original input metadata changed",
        )
        with torch.no_grad():
            streams = original_streams(appearance, pixels)
            indices = np.arange(len(streams["pose_features"]), dtype=np.int64)
            native = native_reference(hands, face, pixels["pixels"], indices)
        visual_comparison = compare_features(streams, native, indices)
        fresh = {
            "row": cached["row"],
            "header": pixels["header"],
            "frame_indices": pixels["frame_indices"],
            "valid_intervals": pixels["valid_intervals"],
            "streams": {k: streams[k] for k in cached["streams"]},
        }
        measured = compare_clip(
            model, tokenizer, cached, fresh, plan["model_plan"]["reference"]["generation"]
        )
        filename = output / f"case-{position:02d}.json"
        write(
            filename,
            {
                "index": index,
                "record": plan["records"][position],
                "input": receipt,
                "complete_native_visual_comparison": visual_comparison,
                "comparison": measured,
                "adapter": plan["adapter"],
                "optimizer_updates": 0,
            },
        )
        comparisons.append(artifact(filename))
        require_candidate(model, binding)
        print(
            json.dumps({"stage": "trained_timing_native", "clips": position + 1, "of": 4}),
            flush=True,
        )
    require_model(model, plan["model_plan"])
    require_candidate(model, binding)
    require(
        state_digest(appearance.state_dict()) == visual,
        "native timing changed original visual values",
    )
    require(plan == checked(path), "trained native timing source or candidate changed")
    write(
        output / "qualification.json",
        {
            "schema": "umi-studio-trained-timing-native-qualification/1",
            "complete": True,
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "adapter": plan["adapter"],
            "comparisons": comparisons,
            "clips": 4,
            "trained_native_runtime_qualified": True,
            "all_original_model_and_trained_adapter_values_unchanged": True,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "CPU_threads": 2,
            "GPU_use": False,
            "development_or_confirmation_inputs": 0,
            "quality_gain_established": False,
            "promotion_authorized": False,
        },
    )


def checked_qualification(plan_path, result_path, poll_path):
    import math

    owned(plan_path, "runs")
    owned(result_path, "checkpoints")
    owned(poll_path, "runs")
    plan = checked(plan_path)
    result, poll = read(result_path), read(poll_path)
    directory = result_path.parent
    authority = terminal(directory)
    worker = read(bound(authority["worker.json"]))
    argv = worker["argv"]
    flag = "--plan" if "--plan" in argv else "--freeze"
    require(
        poll["tool"] == "write_stdin"
        and type(poll["requested_session_id"]) is int
        and poll["actual_owned_pid"] == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None
        and Path(argv[1]).absolute() == ROOT / "tools/qualify_c5_timing_native.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == directory,
        "trained native timing needs actual terminal worker/program/plan evidence",
    )
    require(
        result_path.name == "qualification.json"
        and result["schema"] == "umi-studio-trained-timing-native-qualification/1"
        and result["complete"] is True
        and result["plan"] == artifact(plan_path)
        and result["worker"] == authority["worker.json"]
        and result["adapter"] == plan["adapter"]
        and len(result["comparisons"]) == result["clips"] == 4
        and result["trained_native_runtime_qualified"] is True
        and result["all_original_model_and_trained_adapter_values_unchanged"] is True
        and result["optimizer_updates"] == result["new_Terra_requests"] == 0
        and result["CPU_threads"] == 2
        and result["GPU_use"] is False
        and result["development_or_confirmation_inputs"] == 0
        and result["quality_gain_established"] is result["promotion_authorized"] is False,
        "trained native timing qualification is incomplete or changed its exact candidate/scope",
    )
    expected = {result_path, directory / "worker.json", directory / "exit.json"}
    for position, (index, descriptor, record) in enumerate(
        zip(INDICES, result["comparisons"], plan["records"], strict=True)
    ):
        path = bound(descriptor)
        require(
            path == directory / f"case-{position:02d}.json",
            "trained native timing case order changed",
        )
        report = read(path)
        measured, inputs = report["comparison"], report["input"]
        cached_input = read(bound(record["receipt"]))["input"]
        frames = record["header"]["sampled_frames"]
        native = report["complete_native_visual_comparison"]
        require(
            report["index"] == index
            and report["record"] == record
            and report["adapter"] == plan["adapter"]
            and report["optimizer_updates"] == 0
            and inputs == cached_input
            and inputs["header"] == record["header"]
            and inputs["native_side"] == 224
            and inputs["streamed_pixels_equal_native_reference"] is True
            and all(
                inputs["arrays"][name] == record["arrays"][name]
                for name in (
                    "pose_features",
                    "frame_indices",
                    "valid_intervals",
                    "source_hand_tip_visible",
                )
            )
            and inputs["arrays"]["pixels"]["shape"] == [frames, 3, 224, 224, 3]
            and inputs["arrays"]["pixels"]["dtype"] == "uint8"
            and measured["row"] == record["row"]
            and measured["frames"] == frames
            and measured["native_generation_tokens_equal"] is True
            and measured["generation"] == plan["model_plan"]["reference"]["generation"]
            and native["sample_indices"] == list(range(frames)),
            "trained native timing lost complete source, pose, pixels or generation evidence",
        )
        values = [
            measured["semantic_memory_maximum_difference"],
            measured["teacher_forcing_logits_maximum_difference"],
            *measured["source_feature_maximum_differences"].values(),
            *measured["warped_feature_maximum_differences"].values(),
            *(v["maximum_absolute_difference"] for v in native["streams"].values()),
        ]
        require(
            set(measured["source_feature_maximum_differences"]) == set(WIDTHS)
            and set(measured["warped_feature_maximum_differences"]) == set(WIDTHS)
            and set(native["streams"]) == set(WIDTHS) - {"pose_features"}
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "trained native timing numerical reports changed",
        )
        expected.add(path)
    actual = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in actual)
        and {p for p in actual if p.is_file()} == expected,
        "trained native timing has incomplete, extra or aliased output files",
    )
    return (
        plan,
        result,
        [
            artifact(plan_path),
            artifact(result_path),
            artifact(poll_path),
            *authority.values(),
            *result["comparisons"],
        ],
    )


def run(path, output):
    import cv2
    import torch
    from prepare_c5_timing_features import guard, require_environment
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    plan = checked(path)
    owned(output, "checkpoints")
    require(not output.exists(), "trained native timing needs a fresh output")
    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            qualify(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output", "final-audit"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if args.freeze and args.final_audit and not args.plan:
        freeze(args.freeze, args.final_audit)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("freeze from a complete final audit, or run a frozen native qualification")


if __name__ == "__main__":
    main()
