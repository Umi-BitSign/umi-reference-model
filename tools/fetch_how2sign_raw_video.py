"""Fetch one official How2Sign raw frontal training video by exact ZIP ranges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from how2sign_split_remote_zip import SplitRemoteZip, SplitZipEntry


def checked_training_video(
    training_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    video_name: str,
    entry: SplitZipEntry,
) -> tuple[str, int]:
    if Path(video_name).name != video_name:
        raise ValueError("video name contains a path separator")
    match = re.fullmatch(r"(.+)-(\d+)-rgb_front", video_name)
    if match is None or entry.name != f"raw_videos/{video_name}.mp4":
        raise ValueError("raw archive member is not a frontal training video")
    matching = [row for row in training_rows if row["VIDEO_NAME"] == video_name]
    if not matching or {row["VIDEO_ID"] for row in matching} != {match.group(1)}:
        raise ValueError("raw video is absent from official training metadata")
    if any(row["VIDEO_ID"] == match.group(1) for row in validation_rows):
        raise ValueError("raw video shares an ID with official validation")
    return match.group(1), int(match.group(2))


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START",
            "END",
            "SENTENCE",
        ]:
            raise ValueError("unexpected original-timestamp transcript columns")
        return list(reader)


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    duration = float(payload.get("format", {}).get("duration", 0))
    if (
        len(streams) != 1
        or streams[0]["width"] <= 0
        or streams[0]["height"] <= 0
        or not 0 < duration < 3600
    ):
        raise ValueError("raw training video has invalid stream or duration")
    return {"stream": streams[0], "duration_sec": duration}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--max-compressed-mib", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_compressed_mib <= 512:
        parser.error("compressed download limit must be between 1 and 512 MiB")
    manifest = json.loads(args.manifest.read_text())
    training_rows = read_metadata(args.metadata)
    validation_rows = read_metadata(args.validation_metadata)
    if not args.license.is_file():
        parser.error("official source license file is missing")
    revision = miner_revision(args.miner_health_url, None)
    archive = SplitRemoteZip(manifest)
    member = f"raw_videos/{args.video_name}.mp4"
    entries = archive.load_directory()
    if member not in entries:
        raise ValueError("training video is absent from frozen raw archive")
    entry = entries[member]
    video_id, signer = checked_training_video(
        training_rows, validation_rows, args.video_name, entry
    )
    revision = miner_revision(args.miner_health_url, revision)
    destination = args.output_dir / f"{args.video_name}.mp4"
    temporary = destination.with_suffix(".partial.mp4")
    if destination.exists() or temporary.exists():
        raise ValueError("existing raw video or partial download needs inspection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        prior = [json.loads(line) for line in args.receipt.read_text().splitlines()]
        if any(row["video_name"] == args.video_name for row in prior):
            raise ValueError("raw video already has a download receipt")

    def check_health() -> None:
        nonlocal revision
        revision = miner_revision(args.miner_health_url, revision)

    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("xb") as output:
            for chunk in archive.iter_member(
                member,
                max_compressed_bytes=args.max_compressed_mib * 1024 * 1024,
                check_health=check_health,
            ):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != entry.file_size:
            raise RuntimeError("raw video size differs from publisher archive")
        probe = probe_video(temporary)
        revision = miner_revision(args.miner_health_url, revision)
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    record = {
        "schema": "umi-studio-how2sign-raw-training-video/1",
        "source_url": manifest["source_url"],
        "archive_manifest_sha256": sha256(args.manifest),
        "archive_member": member,
        "archive_disk": entry.disk,
        "archive_header_offset": entry.header_offset,
        "archive_compressed_size": entry.compressed_size,
        "archive_crc32": entry.crc32,
        "training_metadata_sha256": sha256(args.metadata),
        "validation_metadata_sha256": sha256(args.validation_metadata),
        "license_sha256": sha256(args.license),
        "video_name": args.video_name,
        "video_id": video_id,
        "signer": signer,
        "video_sha256": digest.hexdigest(),
        "video_size_bytes": size,
        "video_probe": probe,
        "miner_model_revision": revision,
    }
    with args.receipt.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    print(f"fetched {args.video_name} ({size} bytes)")


if __name__ == "__main__":
    main()
