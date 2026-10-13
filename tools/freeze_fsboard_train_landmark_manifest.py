"""Freeze publisher filenames and sizes for the 100 FSboard train landmark shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
DATASET = "googleai/fsboard"
SHARD = re.compile(r"daun_v3/landmarks/daun_v3-train\.arrow-(\d{5})-of-00100\Z")


def manifest_rows(listing: str) -> list[dict]:
    shards = {}
    for row in csv.reader(io.StringIO(listing)):
        if len(row) != 3:
            continue
        match = SHARD.fullmatch(row[0])
        if match is None:
            continue
        number = int(match.group(1))
        size = int(row[1])
        if (
            number in shards or not 0 <= number < 100
            or not 100_000_000 <= size <= 1_500_000_000 or not row[2]
        ):
            raise ValueError("publisher training shard listing has invalid or duplicated rows")
        shards[number] = {
            "shard_number": number,
            "source_path": row[0],
            "size_bytes": size,
            "publisher_creation_date": row[2],
        }
    if set(shards) != set(range(100)):
        raise ValueError("publisher listing does not cover all 100 train landmark shards")
    return [shards[number] for number in range(100)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-shard", type=Path, required=True)
    parser.add_argument("--first-shard-audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("manifest output must be new and under runs/")
    listed = subprocess.run(
        ["kaggle", "datasets", "files", DATASET, "--page-size", "200", "-v"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if listed.returncode:
        raise RuntimeError(f"Kaggle file listing failed with exit {listed.returncode}")
    rows = manifest_rows(listed.stdout)
    first_audit = json.loads(args.first_shard_audit.read_text())
    if (
        args.first_shard.name != Path(rows[0]["source_path"]).name
        or args.first_shard.stat().st_size != rows[0]["size_bytes"]
        or first_audit.get("shard_sha256") != sha256(args.first_shard)
    ):
        raise ValueError("publisher listing disagrees with the frozen first training shard")
    canonical_listing = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema": "umi-studio-fsboard-train-landmark-manifest/1",
        "dataset": DATASET,
        "source_split": "train",
        "test_used": False,
        "publisher_url": "https://www.kaggle.com/datasets/googleai/fsboard",
        "publisher_license": "CC BY 4.0",
        "shards": rows,
        "shard_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "canonical_listing_sha256": hashlib.sha256(canonical_listing).hexdigest(),
        "first_shard_sha256": first_audit["shard_sha256"],
        "first_shard_audit_sha256": sha256(args.first_shard_audit),
        "videos_downloaded": False,
        "remaining_shards_downloaded_by_this_command": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "shard_count": manifest["shard_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "canonical_listing_sha256": manifest["canonical_listing_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
