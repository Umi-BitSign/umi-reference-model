"""Prepare aspect-preserving OpenASL clips and C5 MediaPipe sidecars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import extract_clip, sha256
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_asl_stem_wiki_landmarks import (
    BASELINE_MODELS,
    DOCKER_SOCKET,
    DOCKERFILE,
    EXTRACTOR,
    container_command,
    docker_image_id,
)

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 100 * 1024**3
FULL_FRAME_POLICY = "full_source_frame_preserve_aspect_ratio"
BBOX_POLICY = "annotated_bbox_preserve_aspect_ratio"


def clip_id(position: int, youtube_id: str) -> str:
    safe_id = youtube_id.replace("-", "").replace("_", "")
    if position < 0 or len(youtube_id) != 11 or not safe_id.isalnum():
        raise ValueError("invalid OpenASL clip identity")
    return f"openasl-{position:04d}-{youtube_id}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def bbox_crop_geometry(bbox: list[float], width: int, height: int) -> dict[str, int]:
    if (
        len(bbox) != 4
        or width < 2
        or height < 2
        or not bbox[0] < bbox[2]
        or not bbox[1] < bbox[3]
    ):
        raise ValueError("invalid OpenASL bounding box")
    x0, y0, x1, y1 = (
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height),
    )
    if (x1 - x0) % 2:
        x1 += 1
    if (y1 - y0) % 2:
        y1 += 1
    left = max(-x0, 0)
    top = max(-y0, 0)
    right = max(x1 - width, 0)
    bottom = max(y1 - height, 0)
    return {
        "crop_x": x0 + left,
        "crop_y": y0 + top,
        "crop_width": x1 - x0,
        "crop_height": y1 - y0,
        "pad_left": left,
        "pad_top": top,
        "pad_right": right,
        "pad_bottom": bottom,
    }


def extract_bbox_clip(
    source: Path,
    clip: Path,
    start: float,
    end: float,
    bbox: list[float],
    width: int,
    height: int,
) -> dict[str, int]:
    geometry = bbox_crop_geometry(bbox, width, height)
    if clip.is_file():
        return geometry
    temporary = clip.with_suffix(".partial.mp4")
    if temporary.exists():
        raise RuntimeError(f"unfinished bbox clip must be inspected: {temporary}")
    pad_width = width + geometry["pad_left"] + geometry["pad_right"]
    pad_height = height + geometry["pad_top"] + geometry["pad_bottom"]
    filters = (
        f"pad={pad_width}:{pad_height}:{geometry['pad_left']}:"
        f"{geometry['pad_top']}:black,"
        f"crop={geometry['crop_width']}:{geometry['crop_height']}:"
        f"{geometry['crop_x']}:{geometry['crop_y']},fps=25"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-t",
                str(round(end - start, 3)),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                filters,
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
                str(temporary),
            ],
            check=True,
        )
        temporary.rename(clip)
    finally:
        temporary.unlink(missing_ok=True)
    return geometry


def verified_sources(
    selection: dict,
    selection_hash: str,
    source_dir: Path,
    receipt_rows: list[dict],
) -> dict[str, dict]:
    expected = {row["youtube_id"] for row in selection["records"]}
    sources = {}
    for row in receipt_rows:
        youtube_id = row.get("youtube_id")
        path = source_dir / row.get("filename", "")
        if (
            row.get("schema") != "umi-studio-openasl-source-receipt/1"
            or row.get("selection_sha256") != selection_hash
            or youtube_id not in expected
            or youtube_id in sources
            or not path.is_file()
            or sha256(path) != row.get("video_sha256")
        ):
            raise ValueError(f"OpenASL source receipt differs: {youtube_id}")
        sources[youtube_id] = {**row, "path": path}
    if set(sources) != expected or len(sources) != selection["source_videos"]:
        raise ValueError("OpenASL source receipts are incomplete")
    return sources


def checked_completed(
    receipts_path: Path,
    *,
    selection_hash: str,
    sources: dict[str, dict],
    docker_image_id_value: str,
    clip_dir: Path,
    landmark_dir: Path,
    spatial_policy: str,
) -> set[str]:
    completed = set()
    for row in read_jsonl(receipts_path):
        identifier = row.get("clip_id")
        source = sources.get(row.get("youtube_id"))
        clip = clip_dir / f"{identifier}.mp4"
        landmarks = landmark_dir / f"{identifier}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-landmarks/1"
            or row.get("selection_sha256") != selection_hash
            or source is None
            or row.get("source_video_sha256") != source["video_sha256"]
            or row.get("docker_image_id") != docker_image_id_value
            or row.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or row.get("extractor_sha256") != sha256(EXTRACTOR)
            or row.get("spatial_policy") != spatial_policy
            or identifier in completed
            or not clip.is_file()
            or not landmarks.is_file()
            or row.get("clip_sha256") != sha256(clip)
            or row.get("landmarks_sha256") != sha256(landmarks)
        ):
            raise ValueError(f"OpenASL landmark receipt differs: {identifier}")
        _, metadata = unpack_landmarks(landmarks)
        if (
            metadata.get("source_sha256") != row["clip_sha256"]
            or metadata.get("frames") != row.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"OpenASL landmark metadata differs: {identifier}")
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
    parser.add_argument(
        "--spatial-policy",
        choices=(FULL_FRAME_POLICY, BBOX_POLICY),
        default=FULL_FRAME_POLICY,
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--docker-socket", default=DOCKER_SOCKET)
    args = parser.parse_args()
    if not 1 <= args.limit <= 120:
        parser.error("limit must be in [1, 120]")
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
    if (
        selection.get("schema") != "umi-studio-openasl-pilot-selection/1"
        or selection.get("clips") != len(selection.get("records", []))
        or selection.get("source_videos")
        != len({row["youtube_id"] for row in selection["records"]})
    ):
        raise ValueError("OpenASL pilot selection differs")
    sources = verified_sources(
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
        docker_image_id_value=image_id,
        clip_dir=args.clip_dir,
        landmark_dir=args.landmark_dir,
        spatial_policy=args.spatial_policy,
    )
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    new_count = 0
    for position, row in enumerate(selection["records"]):
        identifier = clip_id(position, row["youtube_id"])
        if identifier in completed:
            continue
        if new_count >= args.limit:
            break
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("OpenASL preparation reached the 100 GiB disk reserve")
        source = sources[row["youtube_id"]]
        clip = args.clip_dir / f"{identifier}.mp4"
        crop_geometry = None
        if args.spatial_policy == FULL_FRAME_POLICY:
            extract_clip(
                source["path"], clip, row["start_sec"], row["end_sec"], "25"
            )
        else:
            crop_geometry = extract_bbox_clip(
                source["path"],
                clip,
                row["start_sec"],
                row["end_sec"],
                row["bbox"],
                source["width"],
                source["height"],
            )
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
        clip_hash = sha256(clip)
        if (
            len(landmarks) != metadata["frames"]
            or metadata["source_sha256"] != clip_hash
            or metadata["fps"] != 25.0
            or metadata["mediapipe_version"] != "0.10.14"
        ):
            raise ValueError(f"OpenASL landmark output differs: {identifier}")
        receipt = {
            "schema": "umi-studio-openasl-landmarks/1",
            "clip_id": identifier,
            "example_id": row["example_id"],
            "youtube_id": row["youtube_id"],
            "source_group": row["source_group"],
            "selection_sha256": selection_hash,
            "source_video_sha256": source["video_sha256"],
            "clip_sha256": clip_hash,
            "landmarks_sha256": sha256(landmark_path),
            "frames": metadata["frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "clip_fps": 25,
            "spatial_policy": args.spatial_policy,
            "normalized_bbox": row["bbox"] if crop_geometry is not None else None,
            "bbox_crop_geometry": crop_geometry,
            "docker_image_id": image_id,
            "dockerfile_sha256": sha256(DOCKERFILE),
            "extractor_sha256": sha256(EXTRACTOR),
            "reference": row["reference"],
            "miner_health_policy": "user_authorized_maintenance",
        }
        with args.receipts.open("a", encoding="utf-8") as output:
            output.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(identifier)
        new_count += 1
        print(f"prepared {len(completed)}/{selection['clips']} {identifier}", flush=True)


if __name__ == "__main__":
    main()
