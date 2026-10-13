"""Fetch a bounded number of videos from the frozen raw training expansion."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256

MIN_FREE_BYTES_AFTER_BATCH = 100 * 1024 * 1024 * 1024


def verified_downloads(
    selection: dict,
    receipt_lines: list[str],
    output_dir: Path,
) -> set[str]:
    videos = selection["videos"]
    by_name = {video["video_name"]: video for video in videos}
    if (
        selection.get("schema")
        not in {
            "umi-studio-how2sign-raw-training-expansion/1",
            "umi-studio-how2sign-raw-training-expansion/2",
        }
        or selection.get("training_only") is not True
        or len(by_name) != len(videos)
        or selection["selected_video_count"] != len(videos)
        or selection["selected_clip_count"] != sum(video["eligible_clip_count"] for video in videos)
        or selection["selected_compressed_bytes"]
        != sum(video["archive_compressed_size"] for video in videos)
    ):
        raise ValueError("raw training expansion is incomplete")
    completed = set()
    for line in receipt_lines:
        record = json.loads(line)
        name = record["video_name"]
        video = by_name.get(name)
        path = output_dir / f"{name}.mp4"
        if (
            video is None
            or name in completed
            or record["schema"] != "umi-studio-how2sign-raw-training-video/1"
            or record["source_url"] != selection["source_url"]
            or record["archive_manifest_sha256"] != selection["archive_manifest_sha256"]
            or record["training_metadata_sha256"] != selection["training_metadata_sha256"]
            or record["validation_metadata_sha256"] != selection["validation_metadata_sha256"]
            or record["license_sha256"] != selection["license_sha256"]
            or record["video_id"] != video["video_id"]
            or record["signer"] != video["signer"]
            or record["archive_member"] != video["archive_member"]
            or record["archive_disk"] != video["archive_disk"]
            or record["archive_header_offset"] != video["archive_header_offset"]
            or record["archive_compressed_size"] != video["archive_compressed_size"]
            or record["archive_crc32"] != video["archive_crc32"]
            or record["video_size_bytes"] != video["archive_file_size"]
            or (
                selection["schema"] == "umi-studio-how2sign-raw-training-expansion/2"
                and (
                    video.get("verified_video_sha256") != record["video_sha256"]
                    or video.get("verified_video_duration_sec")
                    != record.get("video_probe", {}).get("duration_sec")
                )
            )
            or not path.is_file()
            or sha256(path) != record["video_sha256"]
        ):
            raise ValueError(f"raw source receipt differs from expansion plan: {name}")
        completed.add(name)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--realigned-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--precut-selection", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        parser.error("a raw download batch must contain 1-8 videos")
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_text())
    expected = {
        "archive_manifest_sha256": args.manifest,
        "training_metadata_sha256": args.training_metadata,
        "realigned_metadata_sha256": args.realigned_metadata,
        "validation_metadata_sha256": args.validation_metadata,
        "precut_selection_sha256": args.precut_selection,
        "license_sha256": args.license,
    }
    if any(selection.get(key) != sha256(path) for key, path in expected.items()):
        raise ValueError("raw expansion source hashes differ from frozen selection")
    manifest = json.loads(args.manifest.read_text())
    if (
        selection["archive_central_directory_sha256"]
        != manifest["zip64_central_directory"]["sha256"]
    ):
        raise ValueError("raw expansion uses another split-archive directory")
    prior_lines = args.receipt.read_text().splitlines() if args.receipt.exists() else []
    completed = verified_downloads(selection, prior_lines, args.output_dir)
    pending = [video for video in selection["videos"] if video["video_name"] not in completed][
        : args.limit
    ]
    reserve_target = args.output_dir if args.output_dir.exists() else args.output_dir.parent
    expected_download_bytes = sum(video["archive_file_size"] for video in pending)
    projected_free = shutil.disk_usage(reserve_target).free - expected_download_bytes
    if projected_free < MIN_FREE_BYTES_AFTER_BATCH:
        raise RuntimeError("raw video batch would breach the miner disk reserve")
    revision = miner_revision(args.miner_health_url, None)
    for video in pending:
        revision = miner_revision(args.miner_health_url, revision)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("fetch_how2sign_raw_video.py")),
                "--manifest",
                str(args.manifest),
                "--metadata",
                str(args.training_metadata),
                "--validation-metadata",
                str(args.validation_metadata),
                "--license",
                str(args.license),
                f"--video-name={video['video_name']}",
                "--output-dir",
                str(args.output_dir),
                "--receipt",
                str(args.receipt),
                "--miner-health-url",
                args.miner_health_url,
                "--max-compressed-mib",
                str(selection["max_compressed_bytes_per_video"] // (1024 * 1024)),
            ],
            check=True,
        )
        revision = miner_revision(args.miner_health_url, revision)
        completed = verified_downloads(
            selection,
            args.receipt.read_text().splitlines(),
            args.output_dir,
        )
        if video["video_name"] not in completed or sha256(args.selection) != selection_hash:
            raise RuntimeError("raw expansion receipt or frozen selection changed during download")
    print(f"verified {len(completed)}/{selection['selected_video_count']} raw training videos")


if __name__ == "__main__":
    main()
