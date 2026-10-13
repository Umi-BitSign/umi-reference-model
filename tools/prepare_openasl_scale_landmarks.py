"""Normalize frozen OpenASL scale clips to 25 fps and extract MediaPipe landmarks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from fetch_openasl_scale_clips import clip_stem
from prepare_asl_stem_wiki_landmarks import (
    BASELINE_MODELS,
    DOCKER_SOCKET,
    DOCKERFILE,
    EXTRACTOR,
    container_command,
    docker_image_id,
)
from prepare_openasl_landmarks import FULL_FRAME_POLICY, read_jsonl
from openasl_scale_storage import checked_storage

ROOT = Path(__file__).resolve().parents[1]


def normalize_scale_clip(source: Path, clip: Path, duration: float) -> None:
    if clip.is_file():
        return
    temporary = clip.with_suffix(".partial.mp4")
    if temporary.exists():
        raise RuntimeError(f"unfinished OpenASL normalization must be inspected: {temporary}")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-an", "-vf", "setpts=PTS-STARTPTS,fps=25",
            "-t", str(round(duration, 3)), "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(temporary),
        ],
        check=True,
    )
    temporary.rename(clip)


def checked_source_clips(
    selection: dict,
    selection_hash: str,
    source_dir: Path,
    receipt_rows: list[dict],
) -> dict[str, dict]:
    records = selection["records"]
    by_example = {row["example_id"]: row for row in records}
    sources = {}
    for receipt in receipt_rows:
        example_id = receipt.get("example_id")
        record = by_example.get(example_id)
        path = source_dir / receipt.get("filename", "")
        if (
            receipt.get("schema") != "umi-studio-openasl-scale-clip-receipt/1"
            or receipt.get("selection_sha256") != selection_hash
            or record is None
            or receipt.get("youtube_id") != record["youtube_id"]
            or example_id in sources
            or not path.is_file()
            or receipt.get("video_sha256") != sha256(path)
        ):
            raise ValueError(f"OpenASL scale source clip differs: {example_id}")
        sources[example_id] = {**receipt, "path": path}
    if set(sources) != set(by_example):
        raise ValueError("OpenASL scale source clip receipts are incomplete")
    return sources


def checked_completed(
    path: Path,
    *,
    selection_hash: str,
    sources: dict[str, dict],
    image_id: str,
    clip_dir: Path,
    landmark_dir: Path,
) -> set[str]:
    completed = set()
    for row in read_jsonl(path):
        identifier = row.get("clip_id")
        source = sources.get(row.get("example_id"))
        clip = clip_dir / f"{identifier}.mp4"
        landmarks = landmark_dir / f"{identifier}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            or source is None
            or row.get("source_video_sha256") != source["video_sha256"]
            or row.get("docker_image_id") != image_id
            or row.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or row.get("extractor_sha256") != sha256(EXTRACTOR)
            or row.get("spatial_policy") != FULL_FRAME_POLICY
            or identifier in completed
            or not clip.is_file()
            or not landmarks.is_file()
            or row.get("clip_sha256") != sha256(clip)
            or row.get("landmarks_sha256") != sha256(landmarks)
        ):
            raise ValueError(f"OpenASL scale landmark receipt differs: {identifier}")
        _, metadata = unpack_landmarks(landmarks)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or metadata.get("frames") != row.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"OpenASL scale landmark metadata differs: {identifier}")
        completed.add(identifier)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--docker-socket", default=DOCKER_SOCKET)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not 1 <= args.limit <= 64
        or not all(path.is_file() for path in (args.selection, args.source_receipts))
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
        parser.error("OpenASL scale preparation inputs are incomplete")
    selection = json.loads(args.selection.read_text())
    records = selection.get("records", [])
    if (
        selection.get("schema") != "umi-studio-openasl-scale-selection/1"
        or selection.get("selected_new_clips") != 360
        or len(records) != 360
    ):
        raise ValueError("OpenASL scale selection differs")
    selection_hash = sha256(args.selection)
    sources = checked_source_clips(
        selection,
        selection_hash,
        args.source_dir,
        read_jsonl(args.source_receipts),
    )
    image_id = docker_image_id(args.docker_socket)
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.landmark_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_completed(
        args.receipts,
        selection_hash=selection_hash,
        sources=sources,
        image_id=image_id,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
    )
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    new_count = 0
    for position, row in enumerate(records):
        identifier = clip_stem(position, row)
        if identifier in completed:
            continue
        if new_count >= args.limit:
            break
        checked_storage()
        source = sources[row["example_id"]]
        clip = args.clip_dir / f"{identifier}.mp4"
        normalize_scale_clip(source["path"], clip, row["duration_sec"])
        landmark_path = args.landmark_dir / f"{identifier}.npz"
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
        )
        if result.returncode:
            raise RuntimeError(
                f"container landmark extraction failed for {identifier}: {result.stderr[-4000:]}"
            )
        landmarks, metadata = unpack_landmarks(landmark_path)
        checked_storage(reserve_item=False)
        clip_hash = sha256(clip)
        if (
            len(landmarks) != metadata["frames"]
            or metadata["source_sha256"] != clip_hash
            or metadata["fps"] != 25.0
            or metadata["mediapipe_version"] != "0.10.14"
        ):
            raise ValueError(f"OpenASL scale landmark output differs: {identifier}")
        receipt = {
            "schema": "umi-studio-openasl-landmarks/1",
            "clip_id": identifier,
            "example_id": row["example_id"],
            "youtube_id": row["youtube_id"],
            "source_group": row["source_group"],
            "selection_sha256": selection_hash,
            "source_video_sha256": source["video_sha256"],
            "source_clip_receipt_sha256": sha256(args.source_receipts),
            "clip_sha256": clip_hash,
            "landmarks_sha256": sha256(landmark_path),
            "frames": metadata["frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "clip_fps": 25,
            "timestamp_policy": "video_pts_reset_before_fps25_v1",
            "spatial_policy": FULL_FRAME_POLICY,
            "normalized_bbox": None,
            "bbox_crop_geometry": None,
            "docker_image_id": image_id,
            "dockerfile_sha256": sha256(DOCKERFILE),
            "extractor_sha256": sha256(EXTRACTOR),
            "reference": row["reference"],
            "miner_health_policy": "user_authorized_maintenance",
            "storage_policy": "clip_only_80gib_floor_4gib_budget_v1",
        }
        with args.receipts.open("a", encoding="utf-8") as output:
            output.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(identifier)
        new_count += 1
        print(f"prepared {len(completed)}/360 {identifier}", flush=True)


if __name__ == "__main__":
    main()
