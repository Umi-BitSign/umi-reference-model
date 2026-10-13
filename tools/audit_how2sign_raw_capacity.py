"""Audit unused How2Sign raw training intervals without fetching source videos."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from crop_how2sign_raw_video import read_aligned, select_rows
from fetch_how2sign_raw_video import read_metadata

ROOT = Path(__file__).resolve().parents[1]
MIN_CLIPS = 4


def audit(
    training_rows: list[dict[str, str]],
    aligned_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    precut: dict,
    current_raw: dict,
) -> dict:
    if (
        precut.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or current_raw.get("schema") != "umi-studio-how2sign-raw-training-expansion/2"
    ):
        raise ValueError("raw-capacity audit needs the frozen training selections")
    source_ids = {}
    for row in training_rows:
        name, video_id = row["VIDEO_NAME"], row["VIDEO_ID"]
        if name in source_ids and source_ids[name] != video_id:
            raise ValueError("official training video has inconsistent source IDs")
        source_ids[name] = video_id
    excluded_ids = {row["VIDEO_ID"] for row in validation_rows}
    if excluded_ids & set(source_ids.values()):
        raise ValueError("official training and validation source IDs overlap")
    excluded_ids.update(row["video_id"] for row in precut["rows"])
    selected_names = {video["video_name"] for video in current_raw["videos"]}
    if len(selected_names) != current_raw["selected_video_count"]:
        raise ValueError("current raw selection repeats a video name")
    grouped = defaultdict(list)
    for row in aligned_rows:
        grouped[row["VIDEO_NAME"]].append(row)
    by_signer = defaultdict(lambda: {
        "candidate_videos": 0,
        "candidate_clips": 0,
        "candidate_duration_sec": 0.0,
        "already_selected_videos": 0,
        "remaining_videos": 0,
        "remaining_clips": 0,
        "remaining_duration_sec": 0.0,
    })
    invalid_videos = 0
    too_few_clips = 0
    candidate_names = set()
    for name, rows in grouped.items():
        match = re.fullmatch(r"(.+)-(\d+)-rgb_front", name)
        if match is None or Path(name).name != name:
            raise ValueError("publisher realignment has an invalid frontal video name")
        video_id, signer_text = match.groups()
        if source_ids.get(name) != video_id or {row["VIDEO_ID"] for row in rows} != {video_id}:
            raise ValueError("publisher realignment differs from training source identity")
        if video_id in excluded_ids:
            continue
        receipt = {
            "video_name": name,
            "video_id": video_id,
            "signer": int(signer_text),
            "video_sha256": "not-downloaded",
            "video_probe": {"duration_sec": float("inf")},
        }
        try:
            clips = select_rows(rows, receipt, set())
        except ValueError:
            invalid_videos += 1
            continue
        if len(clips) < MIN_CLIPS:
            too_few_clips += 1
            continue
        candidate_names.add(name)
        values = by_signer[str(receipt["signer"])]
        duration = sum(clip["end_sec"] - clip["start_sec"] for clip in clips)
        values["candidate_videos"] += 1
        values["candidate_clips"] += len(clips)
        values["candidate_duration_sec"] += duration
        if name in selected_names:
            values["already_selected_videos"] += 1
        else:
            values["remaining_videos"] += 1
            values["remaining_clips"] += len(clips)
            values["remaining_duration_sec"] += duration
    if not selected_names <= candidate_names:
        raise ValueError("current raw selection is not contained in metadata candidates")
    signers = {key: by_signer[key] for key in sorted(by_signer, key=int)}
    return {
        "schema": "umi-studio-how2sign-raw-capacity-audit/1",
        "minimum_eligible_clips_per_video": MIN_CLIPS,
        "source_video_bytes_checked": False,
        "source_video_duration_checked": False,
        "archive_members_checked": False,
        "invalid_or_overlapping_video_metadata_count": invalid_videos,
        "too_few_eligible_clips_video_count": too_few_clips,
        "totals": {
            field: sum(values[field] for values in signers.values())
            for field in next(iter(signers.values()))
        },
        "by_signer": signers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--realigned-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--precut-selection", type=Path, required=True)
    parser.add_argument("--current-raw-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("raw-capacity audit needs a new run-ledger output")
    precut = json.loads(args.precut_selection.read_text())
    current_raw = json.loads(args.current_raw_selection.read_text())
    hashes = {
        "training_metadata_sha256": sha256(args.training_metadata),
        "realigned_metadata_sha256": sha256(args.realigned_metadata),
        "validation_metadata_sha256": sha256(args.validation_metadata),
        "precut_selection_sha256": sha256(args.precut_selection),
        "current_raw_selection_sha256": sha256(args.current_raw_selection),
    }
    if any(current_raw.get(key) != value for key, value in hashes.items() if key != "current_raw_selection_sha256"):
        raise ValueError("raw-capacity inputs differ from the frozen raw selection")
    revision = miner_revision(args.miner_health_url, None)
    result = audit(
        read_metadata(args.training_metadata),
        read_aligned(args.realigned_metadata),
        read_metadata(args.validation_metadata),
        precut,
        current_raw,
    )
    result.update(hashes)
    result["miner_model_revision"] = miner_revision(args.miner_health_url, revision)
    result["audit_source_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished raw-capacity audit needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    totals = result["totals"]
    print(
        f"metadata candidates={totals['candidate_videos']} "
        f"remaining={totals['remaining_videos']} "
        f"remaining_hours={totals['remaining_duration_sec'] / 3600:.2f}"
    )


if __name__ == "__main__":
    main()
