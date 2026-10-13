"""Probe bounded OpenASL source availability without downloading media."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    if (
        not args.selection.is_file()
        or args.limit < 1
        or not args.selection.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.receipts.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL probe needs a frozen selection, run ledger and positive limit")
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("schema") != "umi-studio-openasl-source-candidates/1"
        or selection.get("selected_clips") != 512
        or len(selection.get("selected_sources", [])) != 256
    ):
        raise ValueError("OpenASL source candidates differ")
    completed = {}
    if args.receipts.exists():
        for line in args.receipts.read_text().splitlines():
            row = json.loads(line)
            if row.get("schema") != "umi-studio-openasl-source-probe/1":
                raise ValueError("OpenASL probe receipt schema differs")
            completed[row["youtube_id"]] = row
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    selected = selection["selected_sources"][: args.limit]
    for position, youtube_id in enumerate(selected):
        if youtube_id in completed:
            continue
        command = [
            "uvx",
            "yt-dlp",
            "--dump-single-json",
            "--skip-download",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            f"https://www.youtube.com/watch?v={youtube_id}",
        ]
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
            )
            metadata = json.loads(process.stdout)
            receipt = {
                "schema": "umi-studio-openasl-source-probe/1",
                "selection_sha256": sha256(args.selection),
                "position": position,
                "youtube_id": youtube_id,
                "available": True,
                "duration_sec": metadata.get("duration"),
                "filesize_approx": metadata.get("filesize_approx"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "fps": metadata.get("fps"),
                "extractor": metadata.get("extractor_key"),
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            receipt = {
                "schema": "umi-studio-openasl-source-probe/1",
                "selection_sha256": sha256(args.selection),
                "position": position,
                "youtube_id": youtube_id,
                "available": False,
                "duration_sec": None,
                "filesize_approx": None,
                "width": None,
                "height": None,
                "fps": None,
                "extractor": None,
            }
        with args.receipts.open("a", encoding="utf-8") as output:
            output.write(json.dumps(receipt, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        print(
            f"{position + 1}/{len(selected)} {youtube_id} available={receipt['available']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
