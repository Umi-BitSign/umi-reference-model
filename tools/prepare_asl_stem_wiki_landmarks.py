"""Resample selected ASL STEM clips and extract landmarks in bounded Linux CPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import extract_clip, sha256
from extract_mediapipe_landmarks import unpack_landmarks
from fetch_asl_stem_wiki_videos import verified_receipts

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODELS = ROOT / "artifacts/baseline-6fe8df59/models"
DOCKERFILE = ROOT / "tools/mediapipe_cpu.Dockerfile"
EXTRACTOR = ROOT / "tools/extract_mediapipe_landmarks.py"
DOCKER_SOCKET = "unix:///Users/sam/.colima/umi-training/docker.sock"
IMAGE = "umi-mediapipe-cpu:0.10.14"
MIN_FREE_BYTES = 100 * 1024**3


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checked_completed(
    receipts_path: Path,
    *,
    selection_hash: str,
    source_receipts_hash: str,
    docker_image_id: str,
    clip_dir: Path,
    landmark_dir: Path,
) -> set[str]:
    completed = set()
    for row in read_jsonl(receipts_path):
        example_id = row.get("example_id")
        clip = clip_dir / f"{example_id}.mp4"
        landmarks = landmark_dir / f"{example_id}.npz"
        if (
            row.get("schema") != "umi-studio-asl-stem-wiki-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            or row.get("source_receipts_sha256") != source_receipts_hash
            or row.get("docker_image_id") != docker_image_id
            or row.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or row.get("extractor_sha256") != sha256(EXTRACTOR)
            or example_id in completed
            or not clip.is_file()
            or not landmarks.is_file()
            or row.get("clip_sha256") != sha256(clip)
            or row.get("landmarks_sha256") != sha256(landmarks)
        ):
            raise ValueError(f"ASL STEM Wiki landmark receipt differs: {example_id}")
        _, metadata = unpack_landmarks(landmarks)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or metadata.get("frames") != row.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"ASL STEM Wiki landmark metadata differs: {example_id}")
        completed.add(example_id)
    return completed


def container_command(
    *,
    docker_socket: str,
    clip: Path,
    output_name: str,
    landmark_dir: Path,
) -> list[str]:
    clip_relative = clip.resolve().relative_to(ROOT.resolve())
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "3g",
        "--volume",
        f"{ROOT}:/workspace:ro",
        "--volume",
        f"{landmark_dir.resolve()}:/output:rw",
        IMAGE,
        "python",
        "/workspace/tools/extract_mediapipe_landmarks.py",
        "--input",
        f"/workspace/{clip_relative.as_posix()}",
        "--face-model",
        "/workspace/artifacts/baseline-6fe8df59/models/face_landmarker_v2_with_blendshapes.task",
        "--hand-model",
        "/workspace/artifacts/baseline-6fe8df59/models/hand_landmarker.task",
        "--output",
        f"/output/{output_name}",
    ]


def docker_image_id(docker_socket: str) -> str:
    environment = {**os.environ, "DOCKER_HOST": docker_socket}
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    identifier = result.stdout.strip()
    if not identifier.startswith("sha256:") or len(identifier) != 71:
        raise ValueError("unexpected MediaPipe container image identity")
    return identifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--docker-socket", default=DOCKER_SOCKET)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1024:
        parser.error("limit must be in [1, 1024]")
    if not all(path.is_file() for path in (args.selection, args.source_receipts)):
        parser.error("selection and source receipts must exist")
    if not args.source_dir.is_dir() or not all(
        path.is_file()
        for path in (
            DOCKERFILE,
            EXTRACTOR,
            BASELINE_MODELS / "face_landmarker_v2_with_blendshapes.task",
            BASELINE_MODELS / "hand_landmarker.task",
        )
    ):
        parser.error("source data, models or container sources are missing")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    source_receipts_hash = sha256(args.source_receipts)
    source_lines = args.source_receipts.read_text().splitlines()
    verified_receipts(selection, selection_hash, source_lines, args.source_dir)
    source_receipts = {row["example_id"]: row for row in map(json.loads, source_lines)}
    image_id = docker_image_id(args.docker_socket)
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.landmark_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_completed(
        args.receipts,
        selection_hash=selection_hash,
        source_receipts_hash=source_receipts_hash,
        docker_image_id=image_id,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
    )
    pending = [row for row in selection["records"] if row["example_id"] not in completed][
        : args.limit
    ]
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    for row in pending:
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("landmark extraction reached the disk reserve")
        example_id = row["example_id"]
        source = args.source_dir / f"{example_id}.mp4"
        if sha256(source) != source_receipts[example_id]["video_sha256"]:
            raise ValueError(f"ASL STEM Wiki source changed: {example_id}")
        clip = args.clip_dir / f"{example_id}.mp4"
        extract_clip(source, clip, 0, row["duration_sec"], "25")
        landmark_path = args.landmark_dir / f"{example_id}.npz"
        command = container_command(
            docker_socket=args.docker_socket,
            clip=clip,
            output_name=landmark_path.name,
            landmark_dir=args.landmark_dir,
        )
        result = subprocess.run(
            command,
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
            raise ValueError(f"container landmark output differs: {example_id}")
        receipt = {
            "schema": "umi-studio-asl-stem-wiki-landmarks/1",
            "example_id": example_id,
            "participant": row["participant"],
            "article": row["article"],
            "selection_sha256": selection_hash,
            "source_receipts_sha256": source_receipts_hash,
            "source_video_sha256": source_receipts[example_id]["video_sha256"],
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
        print(
            f"prepared {len(completed)}/{selection['selected_examples']} {example_id}",
            flush=True,
        )


if __name__ == "__main__":
    main()
