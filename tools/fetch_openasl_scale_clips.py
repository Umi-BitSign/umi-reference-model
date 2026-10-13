"""Fetch only the frozen OpenASL scale time ranges, with immutable receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256
from fetch_openasl_pilot_sources import DOWNLOAD_STRATEGIES
from openasl_scale_storage import checked_storage

ROOT = Path(__file__).resolve().parents[1]


def clip_stem(position: int, row: dict) -> str:
    return f"openasl-scale-{position:04d}-{row['youtube_id']}"


def media_probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration,start_time,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    duration = stream.get("duration")
    if duration in (None, "N/A"):
        numerator, denominator = stream["avg_frame_rate"].split("/")
        duration = int(stream["nb_frames"]) * float(denominator) / float(numerator)
    return {
        "duration_sec": float(duration),
        "container_duration_sec": float(payload["format"]["duration"]),
        "video_start_time_sec": float(stream.get("start_time", 0)),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
    }


def download_clip(
    output_dir: Path, position: int, row: dict
) -> tuple[str | None, list[dict]]:
    template = output_dir / f"{clip_stem(position, row)}.%(ext)s"
    section = f"*{row['start_sec']:.3f}-{row['end_sec']:.3f}"
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
                "--concurrent-fragments",
                "1",
                "--format",
                format_selector,
                "--merge-output-format",
                "mp4",
                "--download-sections",
                section,
                "--force-keyframes-at-cuts",
                "--downloader-args",
                "ffmpeg_o:-threads 2 -filter_threads 1",
                "--output",
                str(template),
                *extra_args,
                f"https://www.youtube.com/watch?v={row['youtube_id']}",
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
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not args.selection.is_file()
        or args.limit < 1
        or not args.output_dir.resolve().is_relative_to((ROOT / "data/openasl-scale").resolve())
        or not args.receipts.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.failures.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL scale fetch requires maintenance mode and workspace outputs")
    selection = json.loads(args.selection.read_text())
    rows = selection.get("records", [])
    if (
        selection.get("schema") != "umi-studio-openasl-scale-selection/1"
        or selection.get("selected_new_sources") != 180
        or selection.get("selected_new_clips") != 360
        or len(rows) != 360
        or selection.get("acquisition_policy", {}).get("required_free_disk_reserve_gib") != 100
    ):
        raise ValueError("OpenASL scale selection differs")
    selection_hash = sha256(args.selection)
    completed = {}
    if args.receipts.exists():
        for line in args.receipts.read_text().splitlines():
            receipt = json.loads(line)
            path = args.output_dir / receipt["filename"]
            if (
                receipt.get("schema") != "umi-studio-openasl-scale-clip-receipt/1"
                or receipt.get("selection_sha256") != selection_hash
                or not path.is_file()
                or sha256(path) != receipt["video_sha256"]
                or receipt["example_id"] in completed
            ):
                raise ValueError("OpenASL scale clip receipt differs")
            completed[receipt["example_id"]] = receipt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(rows[: args.limit]):
        if row["example_id"] in completed:
            continue
        checked_storage()
        strategy, attempts = download_clip(args.output_dir, position, row)
        if strategy is None:
            append_jsonl(
                args.failures,
                {
                    "schema": "umi-studio-openasl-scale-clip-fetch-failure/1",
                    "selection_sha256": selection_hash,
                    "position": position,
                    "example_id": row["example_id"],
                    "youtube_id": row["youtube_id"],
                    "attempts": attempts,
                    "miner_health_policy": "user_authorized_maintenance",
                },
            )
            print(f"{position + 1}/{args.limit} {row['example_id']} fetch_failed", flush=True)
            continue
        stem = clip_stem(position, row)
        files = [
            path
            for path in args.output_dir.glob(f"{stem}.*")
            if path.stem == stem and path.suffix not in {".part", ".ytdl"}
        ]
        if len(files) != 1:
            raise ValueError(f"OpenASL scale clip output differs: {row['example_id']}")
        path = files[0]
        probe = media_probe(path)
        checked_storage(reserve_item=False)
        if (
            probe["duration_sec"] <= 0
            or abs(probe["duration_sec"] - row["duration_sec"]) > 1.0
            or probe["width"] < 1
            or probe["height"] < 1
        ):
            raise ValueError(f"OpenASL scale clip media differs: {row['example_id']}")
        receipt = {
            "schema": "umi-studio-openasl-scale-clip-receipt/1",
            "selection_sha256": selection_hash,
            "position": position,
            "example_id": row["example_id"],
            "youtube_id": row["youtube_id"],
            "source_url": f"https://www.youtube.com/watch?v={row['youtube_id']}",
            "source_start_sec": row["start_sec"],
            "source_end_sec": row["end_sec"],
            "expected_duration_sec": row["duration_sec"],
            "filename": path.name,
            "bytes": path.stat().st_size,
            "video_sha256": sha256(path),
            "download_strategy": strategy,
            "storage_policy": "clip_only_80gib_floor_4gib_budget_v1",
            **probe,
            "miner_health_policy": "user_authorized_maintenance",
        }
        append_jsonl(args.receipts, receipt)
        print(
            f"{position + 1}/{args.limit} {row['example_id']} bytes={receipt['bytes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
