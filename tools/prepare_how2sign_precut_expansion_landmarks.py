"""Resample pre-cut How2Sign expansion clips and extract Linux CPU landmarks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from cache_how2sign_encoder import resample
from cache_how2sign_precut_expansion_streams import checked_receipted_prefix, read_jsonl
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_asl_stem_wiki_landmarks import (
    DOCKERFILE,
    DOCKER_SOCKET,
    EXTRACTOR,
    IMAGE,
    container_command,
    docker_image_id,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODELS = ROOT / "artifacts/baseline-6fe8df59/models"
SCHEMA = "umi-studio-how2sign-precut-expansion-landmarks/1"
# The remaining resampled clips plus their stride-2 C5 caches are projected to
# use under 3 GiB. Keep a large reserve without stopping at APFS's rounded
# 100-GiB boundary.
MIN_FREE_BYTES = 96 * 1024**3


def checked_completed(
    path: Path,
    *,
    selection_hash: str,
    source_receipts_hash: str,
    image_id: str,
    clip_dir: Path,
    landmark_dir: Path,
) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        name = row.get("sentence_name")
        clip = clip_dir / f"{name}.mp4"
        landmarks = landmark_dir / f"{name}.npz"
        if (
            row.get("schema") != SCHEMA
            or row.get("selection_sha256") != selection_hash
            or row.get("source_receipts_sha256") != source_receipts_hash
            or row.get("docker_image_id") != image_id
            or row.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or row.get("extractor_sha256") != sha256(EXTRACTOR)
            or name in completed
            or not clip.is_file()
            or not landmarks.is_file()
            or row.get("clip_sha256") != sha256(clip)
            or row.get("landmarks_sha256") != sha256(landmarks)
        ):
            raise ValueError(f"pre-cut How2Sign landmark receipt differs: {name}")
        _, metadata = unpack_landmarks(landmarks)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or metadata.get("frames") != row.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"pre-cut How2Sign landmark metadata differs: {name}")
        completed.add(name)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--docker-socket", default=DOCKER_SOCKET)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("landmark extraction requires miner maintenance mode")
    if not 1 <= args.limit <= 1650:
        parser.error("limit must be in [1, 1650]")
    if (
        not args.selection.is_file()
        or not args.source_receipts.is_file()
        or not args.source_dir.is_dir()
        or not all(
            path.is_file()
            for path in (
                DOCKERFILE,
                EXTRACTOR,
                BASELINE_MODELS / "face_landmarker_v2_with_blendshapes.task",
                BASELINE_MODELS / "hand_landmarker.task",
            )
        )
    ):
        parser.error("pre-cut source data, models or container sources are missing")
    selection_hash = sha256(args.selection)
    source_receipts_hash = sha256(args.source_receipts)
    selection = json.loads(args.selection.read_text())
    source_receipts = read_jsonl(args.source_receipts)
    records = checked_receipted_prefix(selection, source_receipts, selection_hash)
    receipts_by_name = {row["sentence_name"]: row for row in source_receipts}
    image_id = docker_image_id(args.docker_socket)
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.landmark_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_completed(
        args.receipts,
        selection_hash=selection_hash,
        source_receipts_hash=source_receipts_hash,
        image_id=image_id,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
    )
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    new_count = 0
    for record in records:
        name = record["sentence_name"]
        if name in completed:
            continue
        if new_count >= args.limit:
            break
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("landmark extraction reached the disk reserve")
        source = args.source_dir / f"{name}.mp4"
        source_receipt = receipts_by_name[name]
        if (
            not source.is_file()
            or source.stat().st_size != source_receipt["clip_size_bytes"]
            or sha256(source) != source_receipt["clip_sha256"]
        ):
            raise ValueError(f"pre-cut How2Sign source changed: {name}")
        clip = args.clip_dir / f"{name}.mp4"
        resample(source, clip)
        landmark_path = args.landmark_dir / f"{name}.npz"
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
                f"container landmark extraction failed for {name}: {result.stderr[-4000:]}"
            )
        landmarks, metadata = unpack_landmarks(landmark_path)
        clip_hash = sha256(clip)
        if (
            len(landmarks) != metadata["frames"]
            or metadata["source_sha256"] != clip_hash
            or metadata["fps"] != 25.0
            or metadata["mediapipe_version"] != "0.10.14"
        ):
            raise ValueError(f"container landmark output differs: {name}")
        receipt = {
            "schema": SCHEMA,
            "sentence_name": name,
            "selection_sha256": selection_hash,
            "source_receipts_sha256": source_receipts_hash,
            "source_clip_sha256": source_receipt["clip_sha256"],
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
        completed.add(name)
        new_count += 1
        print(f"prepared {len(completed)}/{len(records)} {name}", flush=True)


if __name__ == "__main__":
    main()
