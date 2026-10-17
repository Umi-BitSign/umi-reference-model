"""Freeze and verify every eligible word input without model inference or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import DEMO
from terra_training_recovery import artifact, verify_snapshot
from terra_word_native_input import DIAGNOSTIC, RESERVED, bind_records, read_input, require_role

ROOT = Path(__file__).resolve().parents[1]
LANDMARK_PLAN = ROOT / "runs/terra-word-landmarks-v1/plan.json"
LANDMARK_AUDIT = LANDMARK_PLAN.parent / "complete-audit.json"
POLICY = {
    "CPU_threads": 2,
    "GPU_use": False,
    "new_downloads": 0,
    "new_Terra_requests": 0,
    "optimizer_updates": 0,
    "physical_free_floor_bytes": 80 * 1024**3,
    "maximum_output_bytes": 32 * 1024**2,
    "headroom_bytes": 128 * 1024**2,
    "native_side": 112,
    "sampling": "complete source indices0,2,4,... with original native crop pixel equality",
    "pose": "original PoseProcessor on the same sampled dictionary, converted to FP32",
    "persisted_inputs": "hashes, shapes and receipts only; no RGB or pose cache",
    "failure_policy": "stop at first failure; no retry or quality filtering",
    "model_training_ready": False,
    "translation_quality_evaluated": False,
    "promotion_authorized": False,
}


def eligible_records(records):
    counts, seen, labels = Counter(), set(), defaultdict(set)
    eligible, excluded = [], []
    for record in records:
        row = record["original"]["row"]
        if row["example_id"] in seen:
            raise ValueError("word input preflight cannot repeat an identity")
        seen.add(row["example_id"])
        if row["role"] == "training" and row["participant"] in DIAGNOSTIC:
            if (
                row["training_allowed"] is not True
                or record["ROI_lineage_training_allowed"] is not False
            ):
                raise ValueError(
                    "word input preflight cannot change original diagnostic permissions"
                )
            excluded.append(record)
            continue
        require_role(record, row["role"])
        counts[row["role"]] += 1
        if row["role"] == "training":
            labels[row["label"]].add(row["participant"])
        eligible.append(record)
    if (
        counts != {"training": 214, "development": 96}
        or len(excluded) != 14
        or len(labels) != 60
        or min(map(len, labels.values())) < 2
    ):
        raise ValueError("word input preflight changed the fixed214/96/14 inventory or60 labels")
    return eligible, excluded


def guard(output=None):
    used = 0
    if output is not None:
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("word native preflight cannot follow output aliases")
            if path.is_file():
                used += path.stat().st_size
    if used > POLICY["maximum_output_bytes"]:
        raise RuntimeError("word native preflight exceeded its32MiB receipt budget")
    required = (
        POLICY["physical_free_floor_bytes"]
        + POLICY["headroom_bytes"]
        + POLICY["maximum_output_bytes"]
        - used
    )
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError("word native preflight requires the unchanged80GiB floor and headroom")


def specification():
    from tiny_rgb_full_run import runtime_binding

    records = bind_records(LANDMARK_PLAN, LANDMARK_AUDIT)
    eligible, excluded = eligible_records(records)
    parent = json.loads(LANDMARK_PLAN.read_text())
    sources = dict(parent["source_sha256"])
    for name in (
        "prepare_terra_word_native.py",
        "terra_word_native_input.py",
        "audit_terra_word_landmarks.py",
        "prepare_roi_appearance.py",
        "prepare_roi_short_corpus.py",
        "probe_c5_small_input.py",
        "tiny_rgb_full_run.py",
        "training_media_artifact.py",
        "terra_training_recovery.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-word-native112-input-preflight/1",
        "policy": POLICY,
        "landmark_plan": artifact(LANDMARK_PLAN),
        "landmark_audit": artifact(LANDMARK_AUDIT),
        "records": records,
        "eligible_counts": dict(Counter(r["original"]["row"]["role"] for r in eligible)),
        "excluded_original_training_identities": [
            r["original"]["row"]["example_id"] for r in excluded
        ],
        "reserved_STEM_participants": sorted(RESERVED),
        "ROI_diagnostic_participants": sorted(DIAGNOSTIC),
        "source_sha256": sources,
        "assets": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                DEMO / name
                for name in ("crop_hands.py", "crop_face.py", "video_reader.py", "body_features.py")
            )
        },
        "runtime_binding": runtime_binding(),
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("word native preflight roles, sources, native helpers or policy changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word native preflight requires a fresh owned plan")
    guard()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "eligible": plan["eligible_counts"]}), flush=True)


def run(path, output):
    import cv2

    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") != "threads;2":
        raise ValueError("word native preflight requires the two-thread FFmpeg policy")
    cv2.setNumThreads(2)
    plan = checked(path)
    eligible, excluded = eligible_records(plan["records"])
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word native preflight requires fresh owned output")
    guard()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        receipts, frames = [], Counter()
        for index, record in enumerate(eligible):
            guard(output)
            row = record["original"]["row"]
            values = read_input(record, purpose=row["role"])
            receipt = {
                "record": record,
                "header": values["header"],
                "arrays": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    }
                    for name, value in values.items()
                    if name != "header"
                },
                "streamed_pixels_equal_native_reference": True,
            }
            target = output / f"input-{index:04d}.json"
            write(target, receipt)
            receipts.append(artifact(target))
            frames[row["role"]] += len(values["pixels"])
            del values
            print(json.dumps({"verified_inputs": index + 1, "role": row["role"]}), flush=True)
        checked(path)
        guard(output)
        write(
            output / "data.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "receipts": receipts,
                "counts": plan["eligible_counts"],
                "sampled_frames_by_role": dict(frames),
                "excluded_original_training_identities": [
                    r["original"]["row"]["example_id"] for r in excluded
                ],
                "all_streamed_pixels_equal_native_reference": True,
                "model_training_ready": False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    if args.plan and args.output:
        run(args.plan, args.output)
    elif args.plan or args.output or not args.freeze:
        parser.error("choose --freeze or --plan/--output")
