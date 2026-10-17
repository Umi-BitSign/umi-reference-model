"""Bind complete word derivatives to existing offline CPU MediaPipe in small batches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from launch_terra_visual_job import identity, now, write
from normalize_terra_word_data import checked as checked_normalization
from normalize_terra_word_data import validate_output
from prepare_asl_stem_wiki_landmarks import (
    DOCKER_SOCKET,
    DOCKERFILE,
    IMAGE,
    container_command,
    docker_image_id,
)
from prepare_terra_word_data import ROOT, paths_for
from terra_training_recovery import artifact, verify_snapshot
from training_media_artifact import media_artifact

NORMALIZATION = ROOT / "runs/terra-word-normalization-v1/plan.json"
CLIPS = ROOT / "data/asl-stem-wiki/word-pilot-v1/clips-25fps"
RECEIPTS = ROOT / "runs/terra-word-normalization-v1/clip-receipts"
POLICY = {
    "CPU_threads": 2,
    "container_memory_bytes": 3 * 1024**3,
    "network": "none",
    "maximum_batch_clips": 16,
    "maximum_total_landmark_bytes": 1024**3,
    "maximum_landmark_clip_bytes": 16 * 1024**2,
    "physical_free_floor_bytes": 80 * 1024**3,
    "headroom_bytes": 128 * 1024**2,
    "complete25fps_frames": True,
    "mediapipe_version": "0.10.14",
    "service_checks_performed": False,
    "model_training_ready": False,
    "optimizer_updates": 0,
    "new_Terra_requests": 0,
}


def checked_landmarks(path, record):
    rows, metadata = unpack_landmarks(path)
    expected = record["normalized_probe"]
    if (
        metadata.get("source_sha256") != record["clip"]["sha256"]
        or metadata.get("frames") != expected["decoded_frames"]
        or metadata.get("fps") != 25
        or metadata.get("width") != expected["width"]
        or metadata.get("height") != expected["height"]
        or metadata.get("mediapipe_version") != POLICY["mediapipe_version"]
        or metadata.get("delegate") != "cpu"
        or set(rows) != set(range(expected["decoded_frames"]))
        or any(
            not np.isfinite(np.asarray(points)).all()
            for frame in rows.values()
            if frame is not None
            for points in frame.values()
            if points is not None
        )
    ):
        raise ValueError("word landmark source, complete timeline or active coordinates changed")
    return metadata


def specification():
    parent = checked_normalization(NORMALIZATION)
    records = []
    for original in parent["records"]:
        clip, receipt = paths_for(original["row"], CLIPS, RECEIPTS)
        saved = json.loads(receipt.read_text())
        expected_allowed = (
            original["row"]["training_allowed"]
            and original["row"]["participant"]
            not in parent["policy"]["ROI_lineage_future_training_excluded_participants"]
        )
        if (
            saved.get("plan_sha256") != sha256(NORMALIZATION)
            or saved.get("record") != original
            or saved.get("derivative") != media_artifact(clip)
            or saved.get("ROI_lineage_training_allowed") is not expected_allowed
        ):
            raise ValueError("word landmarks require every bound complete normalized source")
        validate_output(original["source_probe"], saved["probe"])
        records.append(
            {
                "original": original,
                "clip": media_artifact(clip),
                "normalized_receipt": artifact(receipt),
                "normalized_probe": saved["probe"],
                "ROI_lineage_training_allowed": saved["ROI_lineage_training_allowed"],
            }
        )
    sources = dict(parent["source_sha256"])
    for name in (
        "prepare_terra_word_landmarks.py",
        "prepare_asl_stem_wiki_landmarks.py",
        "extract_mediapipe_landmarks.py",
        "launch_terra_visual_job.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-word-landmarks/1",
        "normalization_plan": artifact(NORMALIZATION),
        "policy": POLICY,
        "records": records,
        "source_sha256": sources,
        "docker_image_id": docker_image_id(DOCKER_SOCKET),
        "dockerfile_sha256": sha256(DOCKERFILE),
        "model_assets": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "artifacts/baseline-6fe8df59/models" / name
                for name in ("face_landmarker_v2_with_blendshapes.task", "hand_landmarker.task")
            )
        },
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("word landmarks changed normalized media, models or extraction policy")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word landmarks require a fresh owned plan")
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan_sha256": sha256(path), "clips": len(plan["records"])}), flush=True)


def guard(data):
    paths = list(data.rglob("*")) if data.exists() else []
    if any(path.is_symlink() for path in paths):
        raise ValueError("word landmark outputs cannot follow role aliases")
    used = sum(path.stat().st_size for path in paths if path.is_file())
    remaining = POLICY["maximum_total_landmark_bytes"] - used
    if remaining < POLICY["maximum_landmark_clip_bytes"]:
        raise RuntimeError("word landmarks exhausted their one-GiB output reservation")
    if (
        shutil.disk_usage(ROOT).free
        < POLICY["physical_free_floor_bytes"] + POLICY["headroom_bytes"] + remaining
    ):
        raise RuntimeError("word landmarks lack their80GiB floor and remaining output reserve")


def prepare(path, data, receipts, job, limit):
    path, data, receipts, job = (p.resolve() for p in (path, data, receipts, job))
    if (
        not data.is_relative_to(ROOT / "data")
        or not all(p.is_relative_to(ROOT / "runs") for p in (receipts, job))
        or job.exists()
        or not 1 <= limit <= POLICY["maximum_batch_clips"]
    ):
        raise ValueError("word landmark batch needs owned paths, a fresh job and1-16 clips")
    plan = checked(path)
    plan_sha = sha256(path)
    pending = []
    for record in plan["records"]:
        row = record["original"]["row"]
        target = data / row["role"] / (row["example_id"] + ".npz")
        receipt = receipts / row["role"] / (row["example_id"] + ".json")
        if target.parent.is_symlink() or receipt.parent.is_symlink():
            raise ValueError("word landmark role directories cannot be aliases")
        if target.exists() or receipt.exists():
            saved = json.loads(receipt.read_text())
            if (
                saved.get("plan_sha256") != plan_sha
                or saved.get("record") != record
                or saved.get("landmarks") != media_artifact(target)
            ):
                raise ValueError("word landmark receipt changed or output is orphaned")
            checked_landmarks(target, record)
        else:
            if target.with_suffix(".partial.npz").exists():
                raise ValueError("inspect unfinished word landmarks before explicit recovery")
            pending.append((record, target, receipt))
    guard(data)
    job.mkdir(parents=True)
    write(job / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for number, (record, target, receipt) in enumerate(pending[:limit]):
            guard(data)
            if media_artifact(ROOT / record["clip"]["path"]) != record["clip"]:
                raise ValueError("word landmark source changed before extraction")
            target.parent.mkdir(parents=True, exist_ok=True)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            command = container_command(
                docker_socket=DOCKER_SOCKET,
                clip=ROOT / record["clip"]["path"],
                output_name=target.name,
                landmark_dir=target.parent,
            )
            command[command.index(IMAGE)] = plan["docker_image_id"]
            write(
                job / f"container-{number:02d}.json", {"command": command, "automatic_retries": 0}
            )
            result = subprocess.run(
                command,
                env={**os.environ, "DOCKER_HOST": DOCKER_SOCKET},
                capture_output=True,
                text=True,
            )
            (job / f"container-{number:02d}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"word landmark container failed with exit{result.returncode}; no retry"
                )
            metadata = checked_landmarks(target, record)
            if target.stat().st_size > POLICY["maximum_landmark_clip_bytes"]:
                raise RuntimeError("word landmark output exceeded the per-clip bound")
            write(
                receipt,
                {
                    "schema": "umi-studio-word-landmark-receipt/1",
                    "plan_sha256": plan_sha,
                    "record": record,
                    "landmarks": media_artifact(target),
                    "metadata": metadata,
                    "model_training_ready": False,
                    "service_checks_performed": False,
                },
            )
            print(
                json.dumps(
                    {
                        "example_id": record["original"]["row"]["example_id"],
                        "role": record["original"]["row"]["role"],
                        "complete_landmarks": True,
                    }
                ),
                flush=True,
            )
        checked(path)
        code = 0
    except BaseException as error:
        write(job / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(job / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("freeze", "plan", "data", "receipts", "job"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif all((args.plan, args.data, args.receipts, args.job)):
        prepare(args.plan, args.data, args.receipts, args.job, args.limit)
    else:
        parser.error("provide --freeze or --plan/--data/--receipts/--job")


if __name__ == "__main__":
    main()
