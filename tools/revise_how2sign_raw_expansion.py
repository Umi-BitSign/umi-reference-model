"""Freeze a source-duration-checked revision of the raw training selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from fetch_how2sign_raw_expansion import verified_downloads


def revised_selection(
    original: dict,
    video_receipts: list[dict],
    original_hash: str,
    receipt_hash: str,
) -> dict:
    if original.get("schema") != "umi-studio-how2sign-raw-training-expansion/1":
        raise ValueError("source selection is not the frozen raw expansion")
    by_name = {row["video_name"]: row for row in video_receipts}
    if len(by_name) != len(video_receipts) or set(by_name) != {
        video["video_name"] for video in original["videos"]
    }:
        raise ValueError("every selected raw video needs one verified source receipt")
    videos = []
    excluded = []
    for video in original["videos"]:
        name = video["video_name"]
        receipt = by_name[name]
        duration = receipt["video_probe"]["duration_sec"]
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"raw video has no finite duration: {name}")
        clips = []
        for clip in video["clips"]:
            start = clip["start_sec"]
            end = clip["end_sec"]
            if not 0 <= start < end:
                raise ValueError(f"raw expansion has an invalid interval: {clip['sentence_name']}")
            if end > duration + 0.01:
                excluded.append(
                    {
                        "video_name": name,
                        "sentence_name": clip["sentence_name"],
                        "start_sec": start,
                        "end_sec": end,
                        "verified_video_duration_sec": duration,
                        "reason": "publisher interval extends past verified source video end",
                    }
                )
            else:
                clips.append(clip)
        if len(clips) < original["min_clips_per_video"]:
            raise ValueError(f"raw video has too few available intervals: {name}")
        videos.append(
            {
                **video,
                "clips": clips,
                "eligible_clip_count": len(clips),
                "eligible_duration_sec": sum(clip["end_sec"] - clip["start_sec"] for clip in clips),
                "verified_video_duration_sec": duration,
                "verified_video_sha256": receipt["video_sha256"],
            }
        )
    return {
        **original,
        "schema": "umi-studio-how2sign-raw-training-expansion/2",
        "previous_selection_sha256": original_hash,
        "video_receipts_sha256": receipt_hash,
        "excluded_unavailable_clips": excluded,
        "excluded_unavailable_clip_count": len(excluded),
        "selected_clip_count": sum(video["eligible_clip_count"] for video in videos),
        "selected_duration_sec": sum(video["eligible_duration_sec"] for video in videos),
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.selection.resolve():
        parser.error("revised selection must not replace the original")
    original_hash = sha256(args.selection)
    receipt_hash = sha256(args.video_receipts)
    original = json.loads(args.selection.read_text())
    receipt_lines = args.video_receipts.read_text().splitlines()
    revision = miner_revision(args.miner_health_url, None)
    downloaded = verified_downloads(original, receipt_lines, args.source_dir)
    if len(downloaded) != original["selected_video_count"]:
        raise ValueError("all raw sources must be downloaded before revising the selection")
    revision = miner_revision(args.miner_health_url, revision)
    selection = revised_selection(
        original,
        [json.loads(line) for line in receipt_lines],
        original_hash,
        receipt_hash,
    )
    if sha256(args.selection) != original_hash or sha256(args.video_receipts) != receipt_hash:
        raise ValueError("source selection or receipts changed during revision")
    if args.output.exists():
        if json.loads(args.output.read_text()) != selection:
            raise ValueError("frozen source-duration selection differs from current sources")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
    miner_revision(args.miner_health_url, revision)
    print(
        f"selected {selection['selected_clip_count']} source-verified clips; "
        f"excluded {selection['excluded_unavailable_clip_count']} unavailable intervals"
    )


if __name__ == "__main__":
    main()
