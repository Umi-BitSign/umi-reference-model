"""Extract pinned C5 landmarks for the frozen resolution-tuning inventory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from evaluate_fresh_lowres_tuning import load_tuning
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_asl_stem_wiki_landmarks import (
    DOCKERFILE,
    EXTRACTOR,
    container_command,
    docker_image_id,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-c5-resolution-tuning-landmarks/1"
MIN_FREE_BYTES = 90 * 1024**3


def landmark_filename(row: dict) -> str:
    return f"{row['source']}--{row['example_id']}.npz"


def checked_completed(
    receipts: Path,
    *,
    inventory_hash: str,
    image_id: str,
    rows: list[dict],
    landmark_dir: Path,
) -> set[tuple[str, str]]:
    expected = {(row["source"], row["example_id"]): row for row in rows}
    completed = set()
    if not receipts.exists():
        return completed
    for line in receipts.read_text().splitlines():
        record = json.loads(line)
        identity = (record.get("source"), record.get("example_id"))
        row = expected.get(identity)
        path = landmark_dir / record.get("landmark_filename", "")
        if (
            record.get("schema") != SCHEMA
            or row is None
            or identity in completed
            or record.get("tuning_inventory_sha256") != inventory_hash
            or record.get("video_sha256") != row["video_sha256"]
            or record.get("docker_image_id") != image_id
            or record.get("dockerfile_sha256") != sha256(DOCKERFILE)
            or record.get("extractor_sha256") != sha256(EXTRACTOR)
            or record.get("landmark_filename") != landmark_filename(row)
            or not path.is_file()
            or record.get("landmarks_sha256") != sha256(path)
        ):
            raise ValueError(f"resolution landmark receipt differs: {identity}")
        _, metadata = unpack_landmarks(path)
        if (
            metadata.get("source_sha256") != row["video_sha256"]
            or metadata.get("frames") != record.get("frames")
            or metadata.get("fps") != 25.0
            or metadata.get("mediapipe_version") != "0.10.14"
        ):
            raise ValueError(f"resolution landmark metadata differs: {identity}")
        completed.add(identity)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--landmark-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--docker-socket", required=True)
    parser.add_argument("--limit", type=int, default=23)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode:
        parser.error("landmark preparation requires explicit miner maintenance mode")
    if not 1 <= args.limit <= 23 or not args.inventory.is_file():
        parser.error("inventory or limit is invalid")
    rows = load_tuning(args.inventory, ROOT)
    inventory_hash = sha256(args.inventory)
    image_id = docker_image_id(args.docker_socket)
    args.landmark_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    completed = checked_completed(
        args.receipts,
        inventory_hash=inventory_hash,
        image_id=image_id,
        rows=rows,
        landmark_dir=args.landmark_dir,
    )
    environment = {**os.environ, "DOCKER_HOST": args.docker_socket}
    pending = [
        row for row in rows if (row["source"], row["example_id"]) not in completed
    ][: args.limit]
    for row in pending:
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("resolution landmark extraction reached the disk reserve")
        path = args.landmark_dir / landmark_filename(row)
        if path.exists():
            raise ValueError(f"unreceipted resolution landmarks exist: {path}")
        result = subprocess.run(
            container_command(
                docker_socket=args.docker_socket,
                clip=row["video_path"],
                output_name=path.name,
                landmark_dir=args.landmark_dir,
            ),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "container landmark extraction failed for "
                f"{row['example_id']}: {result.stderr[-4000:]}"
            )
        landmarks, metadata = unpack_landmarks(path)
        if (
            len(landmarks) != metadata["frames"]
            or metadata["source_sha256"] != row["video_sha256"]
            or metadata["fps"] != 25.0
            or metadata["mediapipe_version"] != "0.10.14"
        ):
            raise ValueError(f"resolution landmark output differs: {row['example_id']}")
        record = {
            "schema": SCHEMA,
            "source": row["source"],
            "example_id": row["example_id"],
            "tuning_inventory_sha256": inventory_hash,
            "video_sha256": row["video_sha256"],
            "landmark_filename": path.name,
            "landmarks_sha256": sha256(path),
            "frames": metadata["frames"],
            "docker_image_id": image_id,
            "dockerfile_sha256": sha256(DOCKERFILE),
            "extractor_sha256": sha256(EXTRACTOR),
            "miner_health_policy": "user_authorized_maintenance",
        }
        with args.receipts.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add((row["source"], row["example_id"]))
        print(
            f"prepared {len(completed)}/23 {row['source']} {row['example_id']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
