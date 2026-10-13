"""Fetch one publisher FSboard train landmark shard with a pinned receipt.

The downloaded file is not eligible for training until
audit_fsboard_train_arrow.py produces a passing audit and frozen row index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.feather as feather

from benchmark_fleurs_dev import miner_revision, sha256

ROOT = Path(__file__).resolve().parents[1]
DATASET = "googleai/fsboard"


def frozen_entry(manifest: dict, number: int) -> dict:
    rows = manifest.get("shards")
    if (
        manifest.get("schema") != "umi-studio-fsboard-train-landmark-manifest/1"
        or manifest.get("dataset") != DATASET
        or manifest.get("source_split") != "train"
        or manifest.get("test_used") is not False
        or not isinstance(rows, list)
        or len(rows) != 100
    ):
        raise ValueError("FSboard train shard manifest is invalid")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest.get("canonical_listing_sha256"):
        raise ValueError("FSboard train shard manifest listing differs from its digest")
    row = rows[number]
    expected_path = f"daun_v3/landmarks/daun_v3-train.arrow-{number:05d}-of-00100"
    if row.get("shard_number") != number or row.get("source_path") != expected_path:
        raise ValueError("FSboard train shard manifest entry is out of order")
    return row


def publisher_size(listing: str, source_path: str) -> int:
    matches = []
    for row in csv.reader(io.StringIO(listing)):
        if len(row) == 3 and row[0] == source_path:
            matches.append(int(row[1]))
    if len(matches) != 1 or not 100_000_000 <= matches[0] <= 1_500_000_000:
        raise ValueError("publisher listing lacks one plausible training shard size")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-number", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_number < 100:
        parser.error("shard number must be between 0 and 99")
    if (
        args.output_dir.resolve() != ROOT / "data/fsboard/source"
        or not args.receipt.resolve().is_relative_to(ROOT / "runs")
        or args.receipt.exists()
    ):
        parser.error("output must be the FSboard source directory and receipt must be new")
    manifest = json.loads(args.manifest.read_text())
    entry = frozen_entry(manifest, args.shard_number)
    source_path = entry["source_path"]
    shard_name = Path(source_path).name
    target = args.output_dir / shard_name
    if target.exists():
        raise ValueError("training shard already exists; will not overwrite it")
    revision = miner_revision(args.miner_health_url, None)
    listing = subprocess.run(
        ["kaggle", "datasets", "files", DATASET, "--page-size", "200", "-v"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if listing.returncode:
        raise RuntimeError(f"Kaggle file listing failed with exit {listing.returncode}")
    expected_size = publisher_size(listing.stdout, source_path)
    if expected_size != entry["size_bytes"]:
        raise ValueError("publisher listing differs from frozen training shard manifest")
    revision = miner_revision(args.miner_health_url, revision)
    if shutil.disk_usage(ROOT).free < expected_size + 100 * 1024**3:
        raise RuntimeError("less than 100 GiB would remain after the download")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsboard-train-", dir=args.output_dir) as temp:
        fetched = subprocess.run(
            ["kaggle", "datasets", "download", DATASET, "-f", source_path,
             "-p", temp, "-q"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if fetched.returncode:
            raise RuntimeError(f"Kaggle shard download failed with exit {fetched.returncode}")
        source = Path(temp) / shard_name
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError("downloaded FSboard shard differs in size from publisher listing")
        examples = len(feather.read_table(source, columns=["serialized"], memory_map=True))
        if not 100 <= examples <= 2000:
            raise ValueError("downloaded FSboard shard has an unexpected number of examples")
        revision = miner_revision(args.miner_health_url, revision)
        digest = sha256(source)
        os.replace(source, target)
    revision = miner_revision(args.miner_health_url, revision)
    receipt = {
        "schema": "umi-studio-fsboard-train-shard-receipt/1",
        "dataset": DATASET,
        "source_split": "train",
        "source_path": source_path,
        "shard_name": shard_name,
        "size_bytes": expected_size,
        "examples": examples,
        "sha256": digest,
        "manifest_sha256": sha256(args.manifest),
        "miner_model_revision": revision,
        "model_training_started_from_this_shard": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
