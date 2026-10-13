"""Download the bounded OpenASL pilot source videos with receipts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 100 * 1024**3
DOWNLOAD_STRATEGIES = (
    (
        "https",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        (),
    ),
    (
        "hls",
        (
            "bestvideo[height<=720][protocol^=m3u8]+"
            "bestaudio[protocol^=m3u8]/best[height<=720][protocol^=m3u8]"
        ),
        (),
    ),
    (
        "android-vr",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        ("--extractor-args", "youtube:player_client=android_vr"),
    ),
)


def media_probe(path: Path) -> dict:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    stream = payload["streams"][0]
    return {
        "duration_sec": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
    }


def download_source(output_dir: Path, youtube_id: str) -> tuple[str | None, list[dict]]:
    template = output_dir / f"{youtube_id}.%(ext)s"
    attempts = []
    for strategy, format_selector, extra_args in DOWNLOAD_STRATEGIES:
        process = subprocess.run(
            [
                "uvx",
                "yt-dlp",
                "--no-playlist",
                "--no-warnings",
                "--quiet",
                "--no-progress",
                "--format",
                format_selector,
                "--merge-output-format",
                "mp4",
                "--output",
                str(template),
                *extra_args,
                f"https://www.youtube.com/watch?v={youtube_id}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        attempts.append(
            {
                "strategy": strategy,
                "returncode": process.returncode,
                "error_tail": process.stderr.strip()[-1000:],
            }
        )
        if process.returncode == 0:
            return strategy, attempts
    return None, attempts


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.selection.is_file()
        or not args.output_dir.resolve().is_relative_to((ROOT / "data/openasl").resolve())
        or not args.receipts.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.failures.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL fetch needs a pilot selection and workspace outputs")
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("schema") != "umi-studio-openasl-pilot-selection/1"
        or selection.get("source_videos") != 60
        or selection.get("clips") != 120
    ):
        raise ValueError("OpenASL pilot selection differs")
    source_ids = list(dict.fromkeys(row["youtube_id"] for row in selection["records"]))
    completed = {}
    if args.receipts.exists():
        for line in args.receipts.read_text().splitlines():
            row = json.loads(line)
            path = args.output_dir / row["filename"]
            if (
                row.get("schema") != "umi-studio-openasl-source-receipt/1"
                or row.get("selection_sha256") != sha256(args.selection)
                or not path.is_file()
                or sha256(path) != row["video_sha256"]
            ):
                raise ValueError("OpenASL source receipt differs")
            completed[row["youtube_id"]] = row
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    for position, youtube_id in enumerate(source_ids):
        if youtube_id in completed:
            continue
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise RuntimeError("OpenASL source fetch reached the 100 GiB disk reserve")
        strategy, attempts = download_source(args.output_dir, youtube_id)
        if strategy is None:
            failure = {
                "schema": "umi-studio-openasl-source-fetch-failure/1",
                "selection_sha256": sha256(args.selection),
                "position": position,
                "youtube_id": youtube_id,
                "attempts": attempts,
                "miner_health_policy": "user_authorized_maintenance",
            }
            append_jsonl(args.failures, failure)
            print(
                f"{position + 1}/{len(source_ids)} {youtube_id} fetch_failed",
                flush=True,
            )
            continue
        files = [
            path
            for path in args.output_dir.glob(f"{youtube_id}.*")
            if path.stem == youtube_id and path.suffix not in {".part", ".ytdl"}
        ]
        if len(files) != 1:
            raise ValueError(f"OpenASL source output differs: {youtube_id}")
        path = files[0]
        probe = media_probe(path)
        receipt = {
            "schema": "umi-studio-openasl-source-receipt/1",
            "selection_sha256": sha256(args.selection),
            "position": position,
            "youtube_id": youtube_id,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "video_sha256": sha256(path),
            "download_strategy": strategy,
            **probe,
            "miner_health_policy": "user_authorized_maintenance",
        }
        append_jsonl(args.receipts, receipt)
        print(
            f"{position + 1}/{len(source_ids)} {youtube_id} bytes={receipt['bytes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
