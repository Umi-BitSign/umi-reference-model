"""Fetch a frozen FSboard validation sample for offline fingerspelling tests.

Only exact selected clips are downloaded. The test split and live miner are not
queried for data or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from select_fsboard_validation import CLIP_NAME, MD5, SOURCE_PREFIX


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selection(selection: dict) -> list[dict]:
    rows = selection.get("rows")
    if (
        selection.get("schema") != "umi-studio-fsboard-validation-pilot/1"
        or selection.get("dataset") != "googleai/fsboard"
        or selection.get("source_split") != "validation"
        or selection.get("test_used") is not False
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("unexpected FSboard validation selection")
    names = set()
    for row in rows:
        name = row["clip_filename"]
        if (
            not CLIP_NAME.fullmatch(name)
            or row["source_path"] != f"{SOURCE_PREFIX}/{name}"
            or not MD5.fullmatch(row["md5"])
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 1
            or name in names
        ):
            raise ValueError("invalid or duplicated FSboard validation clip")
        names.add(name)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    selection = json.loads(args.selection.read_text())
    rows = validate_selection(selection)
    selection_hash = sha256(args.selection)
    by_name = {row["clip_filename"]: row for row in rows}
    completed = {}
    if args.receipts.exists():
        for line in args.receipts.read_text().splitlines():
            receipt = json.loads(line)
            name = receipt["clip_filename"]
            if (
                name not in by_name
                or name in completed
                or receipt["selection_sha256"] != selection_hash
                or receipt["source_path"] != by_name[name]["source_path"]
                or receipt["source_md5"] != by_name[name]["md5"]
                or receipt["size_bytes"] != by_name[name]["size_bytes"]
            ):
                raise ValueError("existing FSboard receipt differs from selection")
            clip = args.output_dir / name
            if not clip.is_file() or sha256(clip) != receipt["sha256"]:
                raise ValueError(f"downloaded clip missing or changed: {clip}")
            completed[name] = receipt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    revision = miner_revision(args.miner_health_url, None)
    fetched = 0
    for row in rows:
        name = row["clip_filename"]
        if name in completed:
            continue
        if fetched >= args.limit:
            break
        target = args.output_dir / name
        if target.exists():
            raise ValueError(f"unreceipted FSboard clip needs inspection: {target}")
        revision = miner_revision(args.miner_health_url, revision)
        with tempfile.TemporaryDirectory(prefix="fsboard-", dir=args.output_dir) as temp:
            completed_download = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "googleai/fsboard",
                    "-f",
                    row["source_path"],
                    "-p",
                    temp,
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed_download.returncode:
                raise RuntimeError(
                    f"Kaggle clip download failed: exit {completed_download.returncode}"
                )
            source = Path(temp) / name
            if (
                not source.is_file()
                or source.stat().st_size != row["size_bytes"]
                or md5(source) != row["md5"]
            ):
                raise ValueError(f"FSboard clip differs from publisher metadata: {name}")
            os.replace(source, target)
        revision = miner_revision(args.miner_health_url, revision)
        receipt = {
            "schema": "umi-studio-fsboard-validation-clip/1",
            "selection_sha256": selection_hash,
            "clip_filename": name,
            "source_path": row["source_path"],
            "source_md5": row["md5"],
            "size_bytes": row["size_bytes"],
            "sha256": sha256(target),
            "miner_model_revision": revision,
        }
        with args.receipts.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fetched += 1
        print(f"{len(completed) + fetched}: fetched {name}", flush=True)


if __name__ == "__main__":
    main()
