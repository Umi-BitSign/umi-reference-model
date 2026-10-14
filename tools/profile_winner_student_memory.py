"""Measure winner/student representation mismatch on six approved training cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def memory_statistics(student, teacher):
    import torch
    from torch.nn import functional as F

    if (
        student.shape != teacher.shape
        or student.ndim != 3
        or not all(torch.isfinite(value).all() for value in (student, teacher))
    ):
        raise ValueError("memory diagnostic requires finite matching complete timelines")
    s, t = student.float(), teacher.float()
    student_rms, teacher_rms = float(s.square().mean().sqrt()), float(t.square().mean().sqrt())
    if not student_rms or not teacher_rms:
        raise ValueError("memory diagnostic cannot compare zero representations")
    scale = float((s * t).sum() / s.square().sum())
    return {
        "student_rms": student_rms,
        "teacher_rms": teacher_rms,
        "student_to_teacher_rms_ratio": student_rms / teacher_rms,
        "mean_frame_cosine": float(F.cosine_similarity(s, t, dim=-1).mean()),
        "unscaled_rmse": float((s - t).square().mean().sqrt()),
        "least_squares_global_scale": scale,
        "scaled_rmse": float((s * scale - t).square().mean().sqrt()),
        "teacher_mean": float(t.mean()),
        "student_mean": float(s.mean()),
        "quality_verified": False,
    }


def run(output):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from c5_winner_language_decoder import load_winner_language
    from evaluate_tiny_rgb_winner_cpu import CHECKPOINT, PARENT, install_trainable_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_run import runtime_binding
    from tiny_rgb_full_training_tools import decode_rgb_video_fast
    from tiny_rgb_visual_teacher_data import load_targets
    from tiny_rgb_winner_student import RawRgbWinnerStudent, load_frozen_winner_encoder
    from train_tiny_rgb_capacity_cpu import storage
    from train_tiny_rgb_winner_full import PROOF_PLAN

    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists() or not output.is_relative_to(root / "runs"):
        raise ValueError("memory profile needs fresh private runs/ output")
    proof = json.loads(PROOF_PLAN.read_text())
    if sha256(PROOF_PLAN) != "f45ae74305292428254820d6517b4002c6fc78bf8f5b661cbd882e49297d4564":
        raise ValueError("memory profile actual training-case selection changed")
    sources = {
        **proof["source_sha256"],
        "tools/profile_winner_student_memory.py": sha256(Path(__file__)),
    }
    if any(sha256(root / name) != digest for name, digest in sources.items()):
        raise ValueError("memory profile qualified implementation changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if runtime_binding() != proof["runtime_binding"]:
        raise ValueError("memory profile qualified runtime changed")
    storage()
    output.mkdir(parents=True)
    plan = {
        "schema": "umi-studio-winner-student-training-memory-profile/1",
        "source_sha256": sources,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "parent_plan_sha256": sha256(PARENT),
        "proof_plan_sha256": sha256(PROOF_PLAN),
        "runtime_binding": proof["runtime_binding"],
        "cases": proof["cases"],
        "training_only": True,
        "optimizer_updates": 0,
        "device": "cpu",
        "cpu_threads": 2,
        "held_out_labels_used": False,
        "miner_modified": False,
    }
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, output / "source-snapshot")
    language, reference = load_winner_language()
    student = RawRgbWinnerStudent(language).eval()
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as file:
        metadata = file.metadata()
    install_trainable_weights(
        student,
        load_file(CHECKPOINT, device="cpu"),
        metadata,
        plan["parent_plan_sha256"],
        reference["translator_checkpoint_sha256"],
    )
    teacher = load_frozen_winner_encoder(reference).cpu().eval()
    records = []
    for pair in plan["cases"]:
        raw = pair["raw"]
        if (
            raw["role"] != "training"
            or raw["training_allowed"] is not True
            or sha256(root / raw["video"]) != raw["video_sha256"]
        ):
            raise ValueError("memory profile cannot use protected/changed raw data")
        pixels, indices, info = decode_rgb_video_fast(raw["video"], 192, 12.5, 512)
        if (
            len(pixels) != pair["header"]["sampled_frames"]
            or hashlib.sha256(indices.tobytes()).hexdigest()
            != pair["header"]["frame_indices_sha256"]
            or info["source_shape"] != pair["header"]["source_shape"]
        ):
            raise ValueError("memory profile native frame timeline changed")
        video = pixel_tensor(pixels)[None]
        targets = {
            name: torch.from_numpy(value)[None] for name, value in load_targets(pair).items()
        }
        with torch.no_grad():
            expected = teacher(**targets, return_dict=True).last_hidden_state
            actual, _ = student.encode(video, torch.ones(1, len(pixels), dtype=torch.bool))
        record = {
            "source": raw["source"],
            "example_id": raw["example_id"],
            "frames": len(pixels),
            **memory_statistics(actual, expected),
        }
        records.append(record)
        with output.joinpath("cases.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    if any(
        parameter.grad is not None
        for module in (student, teacher)
        for parameter in module.parameters()
    ):
        raise ValueError("read-only memory profile unexpectedly received gradients")
    if sha256(CHECKPOINT) != plan["checkpoint_sha256"] or any(
        sha256(root / name) != digest for name, digest in sources.items()
    ):
        raise ValueError("read-only memory profile fixed bytes changed")
    receipt = {
        "complete": True,
        "plan_sha256": sha256(plan_path),
        "cases": records,
        "optimizer_updates": 0,
        "quality_verified": False,
        "miner_modified": False,
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "receipt_sha256": sha256(output / "receipt.json")}),
        flush=True,
    )


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[name] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
