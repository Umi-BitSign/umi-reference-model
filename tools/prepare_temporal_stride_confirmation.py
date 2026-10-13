"""Prepare fixed confirmation clips and C5-compatible landmarks on Linux CPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import extract_clip, sha256
from cache_how2sign_encoder import resample
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_asl_stem_wiki_landmarks import (
    DOCKERFILE,
    EXTRACTOR,
    MIN_FREE_BYTES,
    container_command,
    docker_image_id,
)

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checked_completed(
    receipts_path: Path,
    *,
    plan_hash: str,
    docker_image: str,
    clip_dir: Path,
    landmark_dir: Path,
) -> set[str]:
    completed = set()
    for row in read_jsonl(receipts_path):
        example_id = row.get("example_id")
        clip = clip_dir / f"{example_id}.mp4"
        landmarks = landmark_dir / f"{example_id}.npz"
        if (
            row.get("schema") != "umi-studio-temporal-stride-confirmation-landmarks/1"
            or row.get("confirmation_plan_sha256") != plan_hash
            or row.get("docker_image_id") != docker_image
            or row.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or row.get("extractor_sha256") != sha256(EXTRACTOR)
            or example_id in completed
            or not clip.is_file()
            or not landmarks.is_file()
            or row.get("clip_sha256") != sha256(clip)
            or row.get("landmarks_sha256") != sha256(landmarks)
        ):
            raise ValueError(f"confirmation landmark receipt differs: {example_id}")
        _, metadata = unpack_landmarks(landmarks)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or metadata.get("frames") != row.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"confirmation landmark metadata differs: {example_id}")
        completed.add(example_id)
    return completed


def prepare_clip(record: dict, destination: Path) -> None:
    source = ROOT / record["source_path"]
    if record["domain"] == "how2sign":
        if record["start_seconds"] != 0.0 or record["end_seconds"] is not None:
            raise ValueError("How2Sign confirmation clip boundaries differ")
        resample(source, destination)
    elif record["domain"] == "fleurs":
        start = record["start_seconds"]
        end = record["end_seconds"]
        if not 0 <= start < end:
            raise ValueError("FLEURS confirmation clip boundaries differ")
        extract_clip(source, destination, start, end, "25")
    else:
        raise ValueError("confirmation record uses another domain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--docker-socket", required=True)
    parser.add_argument("--limit", type=int, default=88)
    args = parser.parse_args()
    if not 1 <= args.limit <= 88 or not args.plan.is_file():
        parser.error("confirmation plan and limit are invalid")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-temporal-stride-confirmation-plan/1"
        or plan.get("candidate_temporal_stride") != 2
        or plan.get("confirmation_examples") != {"fleurs": 46, "how2sign": 42}
        or len(plan.get("records", [])) != 88
    ):
        raise ValueError("confirmation plan scope differs")
    plan_hash = sha256(args.plan)
    image_id = docker_image_id(args.docker_socket)
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.landmark_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_completed(
        args.receipts,
        plan_hash=plan_hash,
        docker_image=image_id,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
    )
    pending = [row for row in plan["records"] if row["example_id"] not in completed][
        : args.limit
    ]
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    for record in pending:
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("confirmation landmark extraction reached the disk reserve")
        example_id = record["example_id"]
        source = ROOT / record["source_path"]
        if not source.is_file() or sha256(source) != record["source_sha256"]:
            raise ValueError(f"confirmation source changed: {example_id}")
        clip = args.clip_dir / f"{example_id}.mp4"
        if not clip.exists():
            prepare_clip(record, clip)
        landmark_path = args.landmark_dir / f"{example_id}.npz"
        if landmark_path.exists():
            raise ValueError(f"unreceipted confirmation landmarks exist: {landmark_path}")
        result = subprocess.run(
            container_command(
                docker_socket=args.docker_socket,
                clip=clip,
                output_name=landmark_path.name,
                landmark_dir=args.landmark_dir,
            ),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"container landmark extraction failed for {example_id}: {result.stderr[-4000:]}"
            )
        landmarks, metadata = unpack_landmarks(landmark_path)
        clip_hash = sha256(clip)
        if (
            len(landmarks) != metadata["frames"]
            or metadata["source_sha256"] != clip_hash
            or metadata["fps"] != 25.0
            or metadata["mediapipe_version"] != "0.10.14"
        ):
            raise ValueError(f"confirmation landmark output differs: {example_id}")
        receipt = {
            "schema": "umi-studio-temporal-stride-confirmation-landmarks/1",
            "example_id": example_id,
            "domain": record["domain"],
            "confirmation_plan_sha256": plan_hash,
            "source_sha256": record["source_sha256"],
            "clip_sha256": clip_hash,
            "landmarks_sha256": sha256(landmark_path),
            "frames": metadata["frames"],
            "docker_image_id": image_id,
            "dockerfile_sha256": sha256(DOCKERFILE),
            "extractor_sha256": sha256(EXTRACTOR),
            "miner_health_policy": "user_authorized_maintenance",
        }
        with args.receipts.open("a", encoding="utf-8") as output:
            output.write(json.dumps(receipt, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(example_id)
        print(f"prepared {len(completed)}/88 {record['domain']} {example_id}", flush=True)


if __name__ == "__main__":
    main()
