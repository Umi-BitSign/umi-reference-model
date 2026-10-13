"""Encode bounded raw-video batches from the verified How2Sign expansion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts, verified_index
from crop_how2sign_raw_expansion import receipt_paths, verified_crops
from fetch_how2sign_raw_expansion import verified_downloads


def existing_indexes(ledger_dir: Path, selections: dict[str, tuple[Path, Path]]) -> dict[str, Path]:
    by_hash = {sha256(selection): name for name, (selection, _) in selections.items()}
    if len(by_hash) != len(selections):
        raise ValueError("two raw videos share a selection hash")
    indexes = {}
    for path in ledger_dir.glob("*encoder-cache.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        hashes = {row["selection_sha256"] for row in rows}
        if len(hashes) != 1 or next(iter(hashes)) not in by_hash:
            raise ValueError(f"raw encoder index has an unknown selection: {path}")
        name = by_hash[next(iter(hashes))]
        if name in indexes:
            raise ValueError(f"multiple raw encoder indexes for {name}")
        indexes[name] = path
    return indexes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--start-video", type=int, default=0)
    parser.add_argument("--stop-video", type=int)
    parser.add_argument("--limit-videos", type=int, default=1)
    parser.add_argument("--save-stream-features", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if args.baseline.resolve() != workspace / "artifacts/baseline-6fe8df59":
        parser.error("requires the independent frozen baseline")
    if not 1 <= args.limit_videos <= 8:
        parser.error("an encoder batch must contain 1-8 source videos")
    plan = json.loads(args.selection.read_text())
    if plan.get("schema") != "umi-studio-how2sign-raw-training-expansion/2":
        raise ValueError("raw encoder requires the source-duration-checked selection")
    if sha256(args.video_receipts) != plan["video_receipts_sha256"]:
        raise ValueError("raw source receipt ledger differs from the frozen selection")
    stop = len(plan["videos"]) if args.stop_video is None else args.stop_video
    if not 0 <= args.start_video < stop <= len(plan["videos"]):
        parser.error("source video window is outside the frozen selection")
    revision = miner_revision(args.miner_health_url, None)
    video_lines = args.video_receipts.read_text().splitlines()
    if len(verified_downloads(plan, video_lines, args.video_dir)) != len(plan["videos"]):
        raise ValueError("all selected raw source videos must be verified")
    video_receipts = {row["video_name"]: row for row in map(json.loads, video_lines)}
    selections = receipt_paths(args.ledger_dir)
    if len(verified_crops(plan["videos"], video_receipts, selections, args.clip_dir)) != len(
        plan["videos"]
    ):
        raise ValueError("all selected raw source videos must have verified crops")
    indexes = existing_indexes(args.ledger_dir, selections)
    revision = miner_revision(args.miner_health_url, revision)
    processed = 0
    for video in plan["videos"][args.start_video : stop]:
        if processed >= args.limit_videos:
            break
        name = video["video_name"]
        selection_path, clip_receipt_path = selections[name]
        selection_hash = sha256(selection_path)
        crop_selection = json.loads(selection_path.read_text())
        receipts = checked_receipts(
            crop_selection, selection_hash, clip_receipt_path.read_text().splitlines()
        )
        index = indexes.get(name, args.ledger_dir / f"{name}-encoder-cache.jsonl")
        records = verified_index(index, selection_hash, None, receipts, args.cache_dir)
        if len(records) == video["eligible_clip_count"]:
            continue
        if len(records) > video["eligible_clip_count"]:
            raise ValueError(f"raw encoder index exceeds planned clips: {name}")
        revision = miner_revision(args.miner_health_url, revision)
        command = [
            sys.executable,
            str(Path(__file__).with_name("cache_how2sign_encoder.py")),
            "--selection",
            str(selection_path),
            "--clip-receipts",
            str(clip_receipt_path),
            "--source-dir",
            str(args.clip_dir),
            "--resampled-dir",
            str(args.resampled_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--index",
            str(index),
            "--baseline",
            str(args.baseline),
            "--miner-health-url",
            args.miner_health_url,
            "--limit",
            str(video["eligible_clip_count"]),
        ]
        if args.save_stream_features:
            command.append("--save-stream-features")
        subprocess.run(command, check=True)
        revision = miner_revision(args.miner_health_url, revision)
        records = verified_index(index, selection_hash, None, receipts, args.cache_dir)
        if len(records) != video["eligible_clip_count"]:
            raise RuntimeError(f"raw encoder batch did not complete its planned video: {name}")
        indexes[name] = index
        processed += 1
        print(f"verified {name}: {len(records)} encoder states", flush=True)
    print(f"encoded {processed} new raw source videos in this batch")


if __name__ == "__main__":
    main()
