"""Fetch only the selected public FLEURS-ASL development videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

EXPECTED_INVENTORY_SHA256 = "7eca25dd64ed35a8e94c2732a59e5e0e27ae1d5e359fe1e817689ed8fd38f07c"
DATASET = "googleai/fleurs-asl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _miner_health(url: str, expected_revision: str | None) -> str:
    with urllib.request.urlopen(url, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("miner health endpoint returned a non-200 status")
        document = json.load(response)
    revision = document.get("model_revision")
    if document.get("ok") is not True or not isinstance(revision, str):
        raise RuntimeError("miner health endpoint is not healthy")
    if expected_revision is not None and revision != expected_revision:
        raise RuntimeError("miner model revision changed during source acquisition")
    return revision


def _selected_files(selection: dict[str, Any], split: str) -> list[str]:
    if selection.get("schema") != "umi-studio-fleurs-dev-split/1":
        raise ValueError("selection schema differs")
    groups = ("training", "validation") if split == "both" else (split,)
    files = {row["source_video"] for group in groups for row in selection[group]}
    if any(
        re.fullmatch(r"fleurs-asl-v1/videos/dev_document_[0-9]+_signer_[34]\.mp4", name) is None
        for name in files
    ):
        raise ValueError("selection contains a non-development source video")
    return sorted(files)


def _expected_files(inventory: dict[str, Any], names: list[str]) -> dict[str, dict[str, Any]]:
    if (
        inventory.get("schema") != "umi-fleurs-asl-source-inventory/1"
        or inventory.get("kaggle_handle") != DATASET
        or inventory.get("kaggle_version") != 2
    ):
        raise ValueError("source inventory differs from FLEURS-ASL Kaggle version 2")
    by_name = {record["path"]: record for record in inventory["files"]}
    if len(by_name) != len(inventory["files"]):
        raise ValueError("source inventory contains duplicate paths")
    missing = set(names) - by_name.keys()
    if missing:
        raise ValueError(f"source inventory is missing {len(missing)} selected videos")
    selected = {name: by_name[name] for name in names}
    for name, record in selected.items():
        if (
            record.get("path") != name
            or not isinstance(record.get("byte_count"), int)
            or record["byte_count"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) is None
        ):
            raise ValueError(f"source inventory has invalid file metadata: {name}")
    return selected


def _download(name: str, destination: Path, health_url: str, revision: str) -> None:
    process = subprocess.Popen(
        ["kaggle", "datasets", "download", DATASET, "-f", name, "-p", str(destination), "-q"],
        stdout=subprocess.DEVNULL,
    )
    try:
        while True:
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _miner_health(health_url, revision)
                continue
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, process.args)
            return
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation", "both"), required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.minimum_free_gib < 1 or (args.limit is not None and args.limit < 1):
        parser.error("free-space floor and file limit must be positive")
    if args.receipt is not None and args.limit is not None:
        parser.error("a limited fetch cannot produce a complete receipt")
    if args.receipt is not None and args.receipt.exists():
        parser.error("receipt already exists")
    if not args.destination.is_dir():
        parser.error("destination must be an existing directory")

    inventory_bytes = args.inventory.read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != EXPECTED_INVENTORY_SHA256:
        raise ValueError("source inventory differs from the pinned public-file inventory")
    selection = json.loads(args.selection.read_bytes())
    names = _selected_files(selection, args.split)
    expected = _expected_files(json.loads(inventory_bytes), names)
    if args.limit is not None:
        names = names[: args.limit]
    revision = _miner_health(args.miner_health_url, None)
    records = []
    for index, name in enumerate(names, start=1):
        revision = _miner_health(args.miner_health_url, revision)
        if shutil.disk_usage(args.destination).free < args.minimum_free_gib * 1024**3:
            raise RuntimeError("free disk space dropped below the acquisition floor")
        target = args.destination / Path(name).name
        record = expected[name]
        if not target.exists():
            _download(name, args.destination, args.miner_health_url, revision)
        if target.stat().st_size != record["byte_count"] or _sha256(target) != record["sha256"]:
            raise RuntimeError(f"downloaded source video differs from inventory: {name}")
        revision = _miner_health(args.miner_health_url, revision)
        records.append(
            {"path": name, "sha256": record["sha256"], "size_bytes": record["byte_count"]}
        )
        print(f"verified {index}/{len(names)} {target.name}", flush=True)
    if args.receipt is not None:
        payload = {
            "schema": "umi-studio-fleurs-dev-source-files/1",
            "dataset": DATASET,
            "inventory_sha256": EXPECTED_INVENTORY_SHA256,
            "selection_sha256": _sha256(args.selection),
            "split": args.split,
            "miner_model_revision_during_acquisition": revision,
            "files": records,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")


if __name__ == "__main__":
    main()
