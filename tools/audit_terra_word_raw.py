"""Audit complete word media and screen cross-role duplicates before model preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import write
from prepare_terra_word_data import ROOT, checked, paths_for, verified

POLICY = {
    "CPU_threads": 2,
    "GPU_use": False,
    "duplicate_decode_fps": 4,
    "duplicate_decode_side": 32,
    "duplicate_temporal_samples": 12,
    "maximum_signature_RAM_bytes": 32 * 1024**2,
    "maximum_cross_role_duration_fractional_difference": 0.05,
    "maximum_cross_role_RGB_mean_absolute_difference": 2.0,
    "perceptual_flags_require_review": True,
    "thumbnails_are_not_training_or_evaluation_inputs": True,
    "model_training_ready": False,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
}


def signature(path, duration):
    side, fps = POLICY["duplicate_decode_side"], POLICY["duplicate_decode_fps"]
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "2",
            "-filter_threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={fps},scale={side}:{side}:flags=area,format=rgb24",
            "-f",
            "rawvideo",
            "-threads",
            "2",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if raw.size % (side * side * 3):
        raise ValueError("word duplicate decode has incomplete RGB frames")
    frames = raw.reshape(-1, side, side, 3)
    if not max(1, math.floor(duration * fps) - 2) <= len(frames) <= math.ceil(duration * fps) + 2:
        raise ValueError("word duplicate decode frame count differs from complete source duration")
    indices = np.rint(np.linspace(0, len(frames) - 1, POLICY["duplicate_temporal_samples"])).astype(
        np.int64
    )
    return frames[indices].copy()


def duplicate_screen(records, signatures):
    if len(records) != len(signatures) or not records:
        raise ValueError("word duplicate screen requires a signature for every original clip")
    if any(record["row"]["role"] not in ("training", "development") for record in records):
        raise ValueError("word duplicate screen requires explicit training/development roles")
    expected = (
        POLICY["duplicate_temporal_samples"],
        POLICY["duplicate_decode_side"],
        POLICY["duplicate_decode_side"],
        3,
    )
    if any(value.dtype != np.uint8 or value.shape != expected for value in signatures):
        raise ValueError("word duplicate signature shape/dtype changed")
    if sum(value.nbytes for value in signatures) > POLICY["maximum_signature_RAM_bytes"]:
        raise ValueError("word duplicate signatures exceed their memory bound")
    by_hash = defaultdict(list)
    for i, row in enumerate(records):
        by_hash[row["video_sha256"]].append(i)
    exact = [indices for indices in by_hash.values() if len(indices) > 1]
    near = []
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            if left["row"]["role"] == right["row"]["role"]:
                continue
            durations = [left["probe"]["duration_sec"], right["probe"]["duration_sec"]]
            if (
                abs(durations[0] - durations[1]) / max(durations)
                > POLICY["maximum_cross_role_duration_fractional_difference"]
            ):
                continue
            difference = float(
                np.abs(signatures[i].astype(np.int16) - signatures[j].astype(np.int16)).mean()
            )
            if difference <= POLICY["maximum_cross_role_RGB_mean_absolute_difference"]:
                near.append({"indices": [i, j], "RGB_mean_absolute_difference": difference})
    return {
        "exact_duplicate_groups": exact,
        "cross_role_perceptual_review_flags": near,
        "cross_role_screen_passed": not near
        and not any(len({records[i]["row"]["role"] for i in group}) > 1 for group in exact),
        "limitations": (
            "A low-resolution equal-time color screen can flag close re-encodes. "
            "No flags does not prove all transformed, cropped, mirrored "
            "or overlapping clips absent. "
            "Review flags before training; never silently drop or relabel rows."
        ),
    }


def audit(plan_path, data, receipts, output):
    plan_path, data, receipts, output = (p.resolve() for p in (plan_path, data, receipts, output))
    if (
        not data.is_relative_to(ROOT / "data")
        or not receipts.is_relative_to(ROOT / "runs")
        or not output.is_relative_to(ROOT / "runs")
    ):
        raise ValueError("word raw audit must use owned data/runs paths")
    if output.exists():
        raise ValueError("word raw audit requires a fresh result path")
    plan, _ = checked(plan_path)
    plan_sha = sha256(plan_path)
    if (
        len(plan["records"])
        * POLICY["duplicate_temporal_samples"]
        * POLICY["duplicate_decode_side"] ** 2
        * 3
        > POLICY["maximum_signature_RAM_bytes"]
    ):
        raise ValueError("word duplicate audit cannot fit its fixed signature memory budget")
    records, signatures = [], []
    for i, row in enumerate(plan["records"]):
        if not verified(row, data, receipts, plan_sha):
            raise ValueError("word raw audit requires every complete source receipt")
        video, receipt = paths_for(row, data, receipts)
        record = json.loads(receipt.read_text())
        signatures.append(signature(video, record["probe"]["duration_sec"]))
        records.append(record)
        print(
            json.dumps({"complete_media_audited": i + 1, "planned": len(plan["records"])}),
            flush=True,
        )
    participants, groups = {}, {}
    for role in ("training", "development"):
        participants[role] = {r["participant"] for r in plan["records"] if r["role"] == role}
        groups[role] = {r["source_group"] for r in plan["records"] if r["role"] == role}
    if (
        participants["training"] & participants["development"]
        or groups["training"] & groups["development"]
    ):
        raise ValueError("word acquisition has participant/recording overlap across roles")
    if set(plan["heldout_participants"]) != participants["development"]:
        raise ValueError("word acquisition changed reserved development participants")
    screen = duplicate_screen(records, signatures)
    result = {
        "schema": "umi-studio-word-raw-audit/1",
        "complete_raw_media_audit": True,
        "plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": plan_sha},
        "policy": POLICY,
        "by_role": dict(Counter(r["role"] for r in plan["records"])),
        "raw_bytes": sum(r["archive"]["file_size"] for r in plan["records"]),
        "participants_by_role": {role: sorted(values) for role, values in participants.items()},
        "participant_and_recording_groups_disjoint": True,
        "label_overlap_train_development_intentional": True,
        "duplicate_screen": screen,
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            **plan["source_sha256"],
        },
        "ffmpeg_executable_sha256": sha256(Path(shutil.which("ffmpeg")).resolve()),
        "runtime": {
            "numpy": np.__version__,
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "records": [
            {
                "receipt": str(paths_for(row, data, receipts)[1].relative_to(ROOT)),
                "receipt_sha256": sha256(paths_for(row, data, receipts)[1]),
                "signature_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for row, value in zip(plan["records"], signatures, strict=True)
        ],
        "model_training_ready": False,
        "label_semantics_reviewed": False,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
    }
    checked(plan_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(
        json.dumps(
            {
                "complete": True,
                "receipt_sha256": sha256(output),
                "cross_role_screen_passed": screen["cross_role_screen_passed"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan", "data", "receipts", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    audit(args.plan, args.data, args.receipts, args.output)


if __name__ == "__main__":
    main()
