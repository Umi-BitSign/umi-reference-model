"""Fetch a bounded batch from the frozen ASL STEM Wiki selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from fetch_asl_stem_wiki_metadata import (
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    AslStemWikiRemoteZip,
    inventory_record,
    validate_inventory,
)
from fetch_how2sign_raw_video import probe_video

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "runs/asl-stem-wiki-v1/source-manifest.json"
LICENSE = ROOT / "data/asl-stem-wiki/source/use.txt"
MIN_FREE_BYTES_AFTER_BATCH = 100 * 1024 * 1024 * 1024


def verified_receipts(
    selection: dict,
    selection_hash: str,
    receipt_lines: list[str],
    output_dir: Path,
) -> set[str]:
    records = selection["records"]
    by_id = {row["example_id"]: row for row in records}
    if (
        selection.get("schema")
        != "umi-studio-asl-stem-wiki-training-selection/1"
        or selection.get("training_only") is not True
        or len(by_id) != len(records)
        or selection.get("selected_examples") != len(records)
    ):
        raise ValueError("ASL STEM Wiki selection is incomplete")
    completed = set()
    for line in receipt_lines:
        receipt = json.loads(line)
        example_id = receipt.get("example_id")
        row = by_id.get(example_id)
        path = output_dir / f"{example_id}.mp4"
        if (
            row is None
            or example_id in completed
            or receipt.get("schema")
            != "umi-studio-asl-stem-wiki-training-video/1"
            or receipt.get("selection_sha256") != selection_hash
            or receipt.get("license_sha256") != selection["license_sha256"]
            or any(
                receipt.get(key) != row[key]
                for key in (
                    "participant",
                    "article",
                    "archive_member",
                    "archive_crc32",
                    "archive_compressed_size",
                    "archive_file_size",
                    "archive_header_offset",
                )
            )
            or receipt.get("video_size_bytes") != row["archive_file_size"]
            or not path.is_file()
            or sha256(path) != receipt.get("video_sha256")
        ):
            raise ValueError(f"ASL STEM Wiki receipt differs: {example_id}")
        completed.add(example_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if args.miner_maintenance_mode == bool(args.miner_health_url):
        parser.error(
            "choose exactly one of --miner-health-url or --miner-maintenance-mode"
        )
    if not 1 <= args.limit <= 64:
        parser.error("download batch must contain 1-64 videos")
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_text())
    source = json.loads(SOURCE_MANIFEST.read_text())
    if (
        source.get("schema") != "umi-studio-asl-stem-wiki-source/1"
        or selection.get("source_manifest_sha256") != sha256(SOURCE_MANIFEST)
        or selection.get("archive_inventory_sha256")
        != source["archive_inventory_sha256"]
        or selection.get("license_sha256") != sha256(LICENSE)
    ):
        raise ValueError("ASL STEM Wiki source differs from frozen selection")
    prior_lines = args.receipts.read_text().splitlines() if args.receipts.exists() else []
    completed = verified_receipts(
        selection, selection_hash, prior_lines, args.output_dir
    )
    pending = [
        row for row in selection["records"] if row["example_id"] not in completed
    ][: args.limit]
    reserve_target = args.output_dir if args.output_dir.exists() else args.output_dir.parent
    required = sum(row["archive_file_size"] for row in pending)
    if shutil.disk_usage(reserve_target).free - required < MIN_FREE_BYTES_AFTER_BATCH:
        raise RuntimeError("ASL STEM Wiki batch would breach the disk reserve")
    if not pending:
        print(f"verified {len(completed)}/{selection['selected_examples']} videos")
        return

    revision = (
        None
        if args.miner_maintenance_mode
        else miner_revision(args.miner_health_url, None)
    )

    def check_miner() -> None:
        nonlocal revision
        if not args.miner_maintenance_mode:
            revision = miner_revision(args.miner_health_url, revision)

    reader = AslStemWikiRemoteZip(
        file_id="direct-download",
        archive_name=ARCHIVE_NAME,
        archive_size=ARCHIVE_SIZE,
    )
    with zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        validate_inventory(infos)
        canonical_inventory = json.dumps(
            [inventory_record(info) for info in infos],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if (
            hashlib.sha256(canonical_inventory).hexdigest()
            != selection["archive_inventory_sha256"]
        ):
            raise ValueError("ASL STEM Wiki archive inventory changed")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.receipts.parent.mkdir(parents=True, exist_ok=True)
        for row in pending:
            check_miner()
            info = archive.getinfo(row["archive_member"])
            if inventory_record(info) != {
                "name": row["archive_member"],
                "crc32": row["archive_crc32"],
                "compressed_size": row["archive_compressed_size"],
                "file_size": row["archive_file_size"],
                "compression": row["archive_compression"],
                "header_offset": row["archive_header_offset"],
            }:
                raise ValueError("selected archive member changed")
            payload = archive.read(info)
            if len(payload) != row["archive_file_size"]:
                raise ValueError("downloaded ASL STEM Wiki video has wrong size")
            target = args.output_dir / f"{row['example_id']}.mp4"
            partial = target.with_suffix(".partial.mp4")
            try:
                with partial.open("xb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                video_probe = probe_video(partial)
                if abs(video_probe["duration_sec"] - row["duration_sec"]) > max(
                    0.5, row["duration_sec"] * 0.02
                ):
                    raise ValueError("video duration differs from publisher metadata")
                partial.rename(target)
            finally:
                partial.unlink(missing_ok=True)
            receipt = {
                "schema": "umi-studio-asl-stem-wiki-training-video/1",
                "example_id": row["example_id"],
                "participant": row["participant"],
                "article": row["article"],
                "archive_member": row["archive_member"],
                "archive_crc32": row["archive_crc32"],
                "archive_compressed_size": row["archive_compressed_size"],
                "archive_file_size": row["archive_file_size"],
                "archive_header_offset": row["archive_header_offset"],
                "selection_sha256": selection_hash,
                "license_sha256": selection["license_sha256"],
                "video_size_bytes": target.stat().st_size,
                "video_sha256": sha256(target),
                "video_probe": video_probe,
                "miner_model_revision": revision,
                "miner_health_policy": (
                    "user_authorized_maintenance"
                    if args.miner_maintenance_mode
                    else "required_stable_revision"
                ),
            }
            with args.receipts.open("a", encoding="utf-8") as output:
                output.write(json.dumps(receipt, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            completed.add(row["example_id"])
            print(
                f"verified {len(completed)}/{selection['selected_examples']} "
                f"{row['example_id']}",
                flush=True,
            )
            check_miner()
    if sha256(args.selection) != selection_hash:
        raise RuntimeError("frozen ASL STEM Wiki selection changed during download")


if __name__ == "__main__":
    main()
