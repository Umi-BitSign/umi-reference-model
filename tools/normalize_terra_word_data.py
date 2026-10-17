"""Preserve complete word clips while binding a bounded, role-separated25fps derivative."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import write
from prepare_terra_word_data import POLICY as RAW_POLICY
from prepare_terra_word_data import ROOT, paths_for, probe, verified
from prepare_terra_word_data import checked as checked_raw
from terra_training_recovery import artifact, verify_snapshot
from training_media_artifact import media_artifact

RAW_PLAN = ROOT / "runs/terra-word-data-v1/plan.json"
RAW_AUDIT = ROOT / "runs/terra-word-data-v1/raw-audit.json"
RAW_DATA = ROOT / "data/asl-stem-wiki/word-pilot-v1/raw"
RAW_RECEIPTS = ROOT / "runs/terra-word-data-v1/raw-receipts"
POLICY = {
    "target_fps": 25,
    "whole_source_interval": True,
    "temporal_cuts_spatial_crops_and_resize": False,
    "codec": "libx264 ultrafast CRF18 yuv420p",
    "maximum_duration_difference_seconds": 2 / 25,
    "maximum_batch_clips": 16,
    "maximum_total_derivative_bytes": 2 * 1024**3,
    "maximum_derivative_clip_bytes": 64 * 1024**2,
    "physical_free_floor_bytes": RAW_POLICY["physical_free_floor_bytes"],
    "CPU_threads": 2,
    "GPU_use": False,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
    "model_training_ready": False,
    "ROI_lineage_future_training_excluded_participants": ["P0", "P19"],
}


def command(source, output):
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        "2",
        "-filter_threads",
        "1",
        "-noautorotate",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        "fps=25",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]


def validate_output(original, actual):
    if (
        actual["fps"] != 25
        or any(actual[k] != original[k] for k in ("width", "height"))
        or abs(actual["duration_sec"] - original["duration_sec"])
        > POLICY["maximum_duration_difference_seconds"] + 1e-6
        or abs(actual["decoded_frames"] - 25 * original["duration_sec"]) > 2.000001
        or not math.isfinite(actual["duration_sec"])
    ):
        raise ValueError("word normalization changed spatial geometry or complete source coverage")


def specification():
    parent, _ = checked_raw(RAW_PLAN)
    audited = json.loads(RAW_AUDIT.read_text())
    if (
        audited.get("complete_raw_media_audit") is not True
        or audited.get("plan") != artifact(RAW_PLAN)
        or audited.get("participant_and_recording_groups_disjoint") is not True
        or audited["duplicate_screen"].get("cross_role_screen_passed") is not True
        or audited["duplicate_screen"]["exact_duplicate_groups"]
        or len(audited["records"]) != len(parent["records"])
    ):
        raise ValueError("word normalization requires complete audited original media")
    records = []
    for row, audited_row in zip(parent["records"], audited["records"], strict=True):
        if not verified(row, RAW_DATA, RAW_RECEIPTS, sha256(RAW_PLAN)):
            raise ValueError("word normalization lost its verified original media")
        video, receipt = paths_for(row, RAW_DATA, RAW_RECEIPTS)
        if artifact(receipt) != {
            "path": audited_row["receipt"],
            "sha256": audited_row["receipt_sha256"],
        }:
            raise ValueError("word normalization raw receipt differs from completed audit")
        raw = json.loads(receipt.read_text())
        records.append(
            {
                "row": row,
                "source": media_artifact(video),
                "source_receipt": artifact(receipt),
                "source_probe": raw["probe"],
            }
        )
    return {
        "schema": "umi-studio-word-normalization/1",
        "policy": POLICY,
        "raw_plan": artifact(RAW_PLAN),
        "raw_audit": artifact(RAW_AUDIT),
        "records": records,
        "source_sha256": {
            **parent["source_sha256"],
            **audited["source_sha256"],
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            "tools/training_media_artifact.py": sha256(ROOT / "tools/training_media_artifact.py"),
        },
        "ffmpeg_executable_sha256": sha256(Path(shutil.which("ffmpeg")).resolve()),
        "ffprobe_executable_sha256": sha256(Path(shutil.which("ffprobe")).resolve()),
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("word normalization source, role or transform binding changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word normalization requires a fresh owned plan")
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan_sha256": sha256(path), "clips": len(plan["records"])}), flush=True)


def guard(data, prospective=0):
    files = list(data.rglob("*")) if data.exists() else []
    if any(p.is_symlink() for p in files):
        raise ValueError("word derivative budget cannot follow symlinks")
    used = sum(p.stat().st_size for p in files if p.is_file())
    remaining = POLICY["maximum_total_derivative_bytes"] - used
    if prospective > remaining or remaining < 0:
        raise RuntimeError("word derivatives exceeded the declared two-GiB budget")
    required = (
        POLICY["physical_free_floor_bytes"]
        + RAW_POLICY["temporary_headroom_bytes"]
        + RAW_POLICY["checkpoint_reserve_bytes"]
        + remaining
    )
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError("word derivatives lack their80GiB floor and remaining output reserve")


def prepare(path, data, receipts, limit):
    path, data, receipts = (p.resolve() for p in (path, data, receipts))
    if (
        not data.is_relative_to(ROOT / "data")
        or not receipts.is_relative_to(ROOT / "runs")
        or not 1 <= limit <= POLICY["maximum_batch_clips"]
    ):
        raise ValueError("word normalization requires owned paths and a batch of1-16")
    plan = checked(path)
    plan_sha = sha256(path)
    pending = []
    for record in plan["records"]:
        target, receipt = paths_for(record["row"], data, receipts)
        if target.parent.is_symlink() or receipt.parent.is_symlink():
            raise ValueError("word derivative roles cannot follow aliased directories")
        if target.exists() or receipt.exists():
            if (
                not target.is_file()
                or not receipt.is_file()
                or target.is_symlink()
                or receipt.is_symlink()
            ):
                raise ValueError("word derivative or receipt is orphaned or aliased")
            saved = json.loads(receipt.read_text())
            if (
                saved.get("plan_sha256") != plan_sha
                or saved.get("record") != record
                or saved.get("derivative") != media_artifact(target)
            ):
                raise ValueError("word derivative receipt/source binding changed")
            validate_output(record["source_probe"], saved["probe"])
        else:
            if target.with_suffix(".partial.mp4").exists():
                raise ValueError("inspect unfinished word derivative before explicit recovery")
            pending.append(record)
    for record in pending[:limit]:
        guard(data, POLICY["maximum_derivative_clip_bytes"])
        if media_artifact(ROOT / record["source"]["path"]) != record["source"]:
            raise ValueError("word source changed before normalization")
        target, receipt = paths_for(record["row"], data, receipts)
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.mp4")
        subprocess.run(
            command(ROOT / record["source"]["path"], partial),
            check=True,
            capture_output=True,
            timeout=120,
        )
        actual = probe(partial, record["source_probe"]["duration_sec"])
        validate_output(record["source_probe"], actual)
        if not 0 < partial.stat().st_size <= POLICY["maximum_derivative_clip_bytes"]:
            raise ValueError("word derivative file exceeded its per-clip bound")
        os.link(partial, target)
        partial.unlink()
        write(
            receipt,
            {
                "schema": "umi-studio-word-normalized-video/1",
                "plan_sha256": plan_sha,
                "record": record,
                "derivative": media_artifact(target),
                "probe": actual,
                "model_training_ready": False,
                "ROI_lineage_training_allowed": record["row"]["training_allowed"]
                and record["row"]["participant"]
                not in POLICY["ROI_lineage_future_training_excluded_participants"],
            },
        )
        print(
            json.dumps(
                {
                    "example_id": record["row"]["example_id"],
                    "role": record["row"]["role"],
                    "complete_source_25fps": True,
                }
            ),
            flush=True,
        )
    guard(data)
    checked(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.data and args.receipts:
        prepare(args.plan, args.data, args.receipts, args.limit)
    else:
        parser.error("provide --freeze or --plan/--data/--receipts")


if __name__ == "__main__":
    main()
